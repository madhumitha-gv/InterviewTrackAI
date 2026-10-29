"""Stages D & E — hybrid match scoring and decisions.

The contract this module enforces:

  * Semantic similarity is ONE feature in a transparent weighted sum. It is never
    treated as a calibrated probability.
  * **High semantic similarity alone can never merge two applications.** When no
    explicit identity evidence (requisition, employer, native thread, or shared
    recruiter) is present, the score is hard-capped below the auto-match threshold.
  * Hard blockers veto a merge outright: conflicting requisition IDs, conflicting
    employers. A clear role-family conflict blocks auto-matching unless an exact
    requisition match resolves it.
  * Every decision carries its feature vector, reasons, and blockers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .config import MatchWeights, Settings, get_settings
from .embeddings import (
    EmbeddingBackend,
    build_application_document,
    cosine_similarity,
    get_backend,
)
from .enums import ClassificationSource, ContactType, MatchDecisionKind
from .models import Application, Classification, MatchDecision, MatchFeatures, ParsedMessage
from .normalize import (
    canonical_company,
    canonical_person,
    company_similarity,
    requisitions_conflict,
    role_similarity,
)

#: Company names scoring at or above this are treated as the same employer.
EMPLOYER_SAME_THRESHOLD = 0.82
#: Below this, two known employer names are treated as a hard conflict.
EMPLOYER_CONFLICT_THRESHOLD = 0.55
#: Role similarity at or above this counts as the same role.
ROLE_SAME_THRESHOLD = 0.70
#: Half-life (days) for temporal proximity decay.
TEMPORAL_HALFLIFE_DAYS = 21.0


@dataclass
class ApplicationContext:
    """An application plus the cached state needed to score candidates against it."""

    application: Application
    recruiter_names: set[str] = field(default_factory=set)
    recruiter_emails: set[str] = field(default_factory=set)
    thread_ids: set[str] = field(default_factory=set)
    rfc_message_ids: set[str] = field(default_factory=set)
    recent_subjects: list[str] = field(default_factory=list)
    #: Contacts learned for this application (contact learning).
    known_contact_emails: set[str] = field(default_factory=set)
    #: Messages the user explicitly said do NOT belong here.
    separated_gmail_ids: set[str] = field(default_factory=set)
    embedding: np.ndarray | None = field(default=None, repr=False)
    last_activity: datetime | None = None

    @property
    def key(self) -> str:
        return self.application.key

    def document(self) -> str:
        app = self.application
        return build_application_document(
            employer=app.employer,
            job_title=app.job_title,
            requisition_id=app.requisition_id,
            recruiters=sorted(self.recruiter_names),
            recent_subjects=self.recent_subjects[-4:],
        )

    def invalidate_embedding(self) -> None:
        self.embedding = None

    def ensure_embedding(self, backend: EmbeddingBackend) -> np.ndarray:
        if self.embedding is None:
            self.embedding = backend.encode_one(self.document())
        return self.embedding


def _temporal_proximity(days: float) -> float:
    """Exponential decay in [0, 1]. Recent activity nudges a match; it never drives one."""
    if days < 0:
        days = abs(days)
    return float(np.exp(-days / TEMPORAL_HALFLIFE_DAYS))


def compute_features(
    msg: ParsedMessage,
    cls: Classification,
    ctx: ApplicationContext,
    msg_embedding: np.ndarray | None = None,
    backend: EmbeddingBackend | None = None,
    contact_type: ContactType = ContactType.UNKNOWN,
    counterparties: set[str] | None = None,
) -> MatchFeatures:
    """Build the full feature vector for (message, application) — no thresholds applied."""
    app = ctx.application
    ent = cls.entities
    feats = MatchFeatures()

    # --- Stage A: native Gmail threading ---
    feats.same_gmail_thread = msg.thread_id in ctx.thread_ids
    refs = set(msg.raw.references or [])
    if msg.raw.in_reply_to:
        refs.add(msg.raw.in_reply_to)
    feats.thread_reference_link = bool(refs & ctx.rfc_message_ids)

    # --- requisition ---
    if ent.requisition_id and app.requisition_id:
        if ent.requisition_id == app.requisition_id:
            feats.requisition_match = True
        else:
            feats.requisition_conflict = True

    # --- employer ---
    if ent.employer_canonical and app.employer_canonical:
        sim = company_similarity(ent.employer_canonical, app.employer_canonical)
        if sim >= EMPLOYER_SAME_THRESHOLD:
            feats.employer_match = True
        elif sim < EMPLOYER_CONFLICT_THRESHOLD:
            feats.employer_conflict = True

    # --- role ---
    if ent.job_title and app.job_title:
        feats.role_similarity = role_similarity(ent.job_title, app.job_title)
        if ent.role_family and app.role_family and ent.role_family != app.role_family:
            feats.role_conflict = True
        elif feats.role_similarity < 0.35:
            # Different enough lexically to count as a conflict even without families.
            feats.role_conflict = True

    # --- recruiter / contact overlap ---
    name = canonical_person(ent.recruiter_name or "")
    email = (ent.recruiter_email or "").lower()
    feats.shared_recruiter = bool(
        (name and name in ctx.recruiter_names) or (email and email in ctx.recruiter_emails)
    )
    # Who the message is "with". For mail the candidate SENT, that is the
    # recipient, not the sender — otherwise every reply the candidate writes looks
    # like it came from a stranger and starts a new application.
    parties = set(counterparties or ())
    sender = (msg.raw.from_addr or "").lower()
    if sender:
        parties.add(sender)
    feats.known_contact = bool(parties & ctx.known_contact_emails)
    feats.automated_sender = contact_type in {
        ContactType.ATS_NOREPLY, ContactType.SCHEDULING_TOOL
    }

    # --- a user-confirmed separation is a hard constraint ---
    feats.user_separated = msg.gmail_id in ctx.separated_gmail_ids

    # --- temporal ---
    msg_dt = msg.raw.date
    if msg_dt and ctx.last_activity:
        feats.days_apart = abs((msg_dt - ctx.last_activity).total_seconds()) / 86400.0
        feats.temporal_proximity = _temporal_proximity(feats.days_apart)

    # --- Stage C: semantic similarity ---
    if backend is not None and msg_embedding is not None:
        feats.semantic_similarity = round(
            cosine_similarity(msg_embedding, ctx.ensure_embedding(backend)), 4
        )

    return feats


def score_features(feats: MatchFeatures, weights: MatchWeights) -> tuple[float, list[str]]:
    """Transparent weighted sum. Returns (score, human-readable reasons).

    NOT a probability — a tunable additive score compared against configured
    thresholds.
    """
    score = 0.0
    reasons: list[str] = []

    if feats.same_gmail_thread:
        score += weights.same_gmail_thread
        reasons.append(f"native Gmail thread (+{weights.same_gmail_thread:.2f})")
    if feats.thread_reference_link:
        score += weights.thread_reference_link
        reasons.append(f"RFC reply-header link (+{weights.thread_reference_link:.2f})")
    if feats.requisition_match:
        score += weights.requisition_match
        reasons.append(f"exact requisition ID match (+{weights.requisition_match:.2f})")
    if feats.employer_match:
        score += weights.employer_match
        reasons.append(f"employer identity match (+{weights.employer_match:.2f})")
    if feats.role_similarity > 0:
        contrib = weights.role_similarity * feats.role_similarity
        score += contrib
        reasons.append(f"job title similarity {feats.role_similarity:.2f} (+{contrib:.2f})")
    if feats.known_contact:
        score += weights.known_contact
        reasons.append(
            f"sender or recipient is a known contact of this application "
            f"(+{weights.known_contact:.2f})"
        )
    if feats.shared_recruiter:
        score += weights.shared_recruiter
        reasons.append(f"shared recruiter contact (+{weights.shared_recruiter:.2f})")

    # Semantic contribution: floored, then rescaled. Below the floor it is noise.
    if feats.semantic_similarity >= weights.semantic_floor:
        span = max(1e-6, 1.0 - weights.semantic_floor)
        scaled = (feats.semantic_similarity - weights.semantic_floor) / span
        contrib = weights.semantic_similarity * scaled
        score += contrib
        reasons.append(
            f"semantic similarity {feats.semantic_similarity:.2f} (+{contrib:.2f})"
        )

    if feats.temporal_proximity > 0:
        contrib = weights.temporal_proximity * feats.temporal_proximity
        score += contrib
        reasons.append(
            f"temporal proximity {feats.days_apart:.0f}d (+{contrib:.2f})"
        )

    return round(min(1.0, score), 4), reasons


def find_blockers(feats: MatchFeatures) -> list[str]:
    """Hard constraints that forbid an automatic merge, in priority order.

    Nothing — no score, no model, no native Gmail thread — overrides these.
    """
    blockers: list[str] = []
    if feats.user_separated:
        blockers.append("user confirmed these are separate")
    if feats.requisition_conflict:
        blockers.append("conflicting requisition IDs")
    if feats.employer_conflict:
        blockers.append("conflicting employer identity")
    if feats.role_conflict and not feats.requisition_match:
        blockers.append("different role / job family")
    return blockers


def decide(
    msg: ParsedMessage,
    cls: Classification,
    ctx: ApplicationContext,
    settings: Settings,
    msg_embedding: np.ndarray | None = None,
    backend: EmbeddingBackend | None = None,
    contact_type: ContactType = ContactType.UNKNOWN,
    counterparties: set[str] | None = None,
) -> MatchDecision:
    """Score one (message, application) pair and return a decision."""
    feats = compute_features(msg, cls, ctx, msg_embedding=msg_embedding,
                             backend=backend, contact_type=contact_type,
                             counterparties=counterparties)
    score, reasons = score_features(feats, settings.weights)
    blockers = find_blockers(feats)

    # --- A native Gmail thread is STRONG EVIDENCE, not unconditional proof. -------
    # A thread can genuinely span two roles (a recruiter replying about a second
    # opening on the same chain). So the thread short-circuits to AUTO_MATCH only when
    # no hard constraint is violated; otherwise it goes to review and the application
    # is allowed to split.
    if feats.same_gmail_thread:
        if blockers:
            return MatchDecision(
                gmail_id=msg.gmail_id, application_key=ctx.key,
                candidate_keys=[ctx.key],
                kind=MatchDecisionKind.REVIEW_REQUIRED,
                rule_fired="R1_native_thread_conflicted",
                score=max(score, 0.5), features=feats,
                reasons=reasons + [
                    "same Gmail thread, but a hard constraint is violated — the thread "
                    "appears to span more than one hiring process, so it was not merged"
                ],
                blockers=blockers, source=ClassificationSource.HYBRID,
            )
        return MatchDecision(
            gmail_id=msg.gmail_id, application_key=ctx.key, candidate_keys=[ctx.key],
            kind=MatchDecisionKind.AUTO_MATCH, rule_fired="R1_native_thread",
            score=1.0, features=feats,
            reasons=["native Gmail thread, with no conflicting identity evidence"]
                    + reasons,
            blockers=[], source=ClassificationSource.DETERMINISTIC,
        )

    # --- Exact requisition match is decisive when nothing contradicts it. ---------
    if feats.requisition_match and not blockers:
        return MatchDecision(
            gmail_id=msg.gmail_id, application_key=ctx.key, candidate_keys=[ctx.key],
            kind=MatchDecisionKind.AUTO_MATCH, rule_fired="R2_requisition",
            score=max(score, 0.9), features=feats,
            reasons=["exact requisition ID match"] + reasons,
            blockers=[], source=ClassificationSource.DETERMINISTIC,
        )

    # --- Hard vetoes.
    if feats.requisition_conflict or feats.employer_conflict or feats.user_separated:
        return MatchDecision(
            gmail_id=msg.gmail_id, application_key=ctx.key, candidate_keys=[ctx.key],
            kind=MatchDecisionKind.NO_MATCH, rule_fired="HARD_CONSTRAINT",
            score=score, features=feats,
            reasons=reasons, blockers=blockers, source=ClassificationSource.HYBRID,
        )

    # --- The central safety rule -------------------------------------------------
    # Explicit identity evidence = requisition match, employer match, native thread
    # link, or a shared named recruiter. Without ANY of it, semantic similarity is
    # capped below the auto-match threshold, so it can only ever ask for review.
    has_identity_evidence = (
        feats.requisition_match
        or feats.employer_match
        or feats.thread_reference_link
        or feats.shared_recruiter
        or feats.known_contact
    )
    # An automated sender (ATS / scheduling tool) carries no identity of its own: it
    # may join only when BOTH employer and role match.
    if feats.automated_sender and not (
        feats.employer_match and feats.role_similarity >= ROLE_SAME_THRESHOLD
    ) and not feats.requisition_match:
        blockers.append(
            "automated sender requires both employer and role to match"
        )
    if not has_identity_evidence:
        capped = min(score, settings.weights.semantic_only_cap)
        if capped < score:
            reasons.append(
                f"capped at {settings.weights.semantic_only_cap:.2f}: no explicit identity "
                "evidence — semantic similarity alone cannot merge applications"
            )
        score = capped
        blockers.append("no explicit employer/requisition/thread evidence")

    # --- Role-family conflict blocks auto-match but still allows review.
    if blockers:
        kind = (
            MatchDecisionKind.REVIEW_REQUIRED
            if score >= settings.review_threshold
            else MatchDecisionKind.NO_MATCH
        )
        return MatchDecision(
            gmail_id=msg.gmail_id, application_key=ctx.key, candidate_keys=[ctx.key],
            kind=kind, rule_fired="BLOCKED", score=score,
            features=feats, reasons=reasons, blockers=blockers,
            source=ClassificationSource.HYBRID,
        )

    if score >= settings.auto_match_threshold:
        kind = MatchDecisionKind.AUTO_MATCH
    elif score >= settings.review_threshold:
        kind = MatchDecisionKind.REVIEW_REQUIRED
    else:
        kind = MatchDecisionKind.NO_MATCH

    rule = "R3_known_contact" if feats.known_contact else (
        "R4_employer_role" if feats.employer_match else "R6_weighted_evidence"
    )
    return MatchDecision(
        gmail_id=msg.gmail_id, application_key=ctx.key, candidate_keys=[ctx.key],
        kind=kind, rule_fired=rule, score=score,
        features=feats, reasons=reasons, blockers=blockers,
        source=(
            ClassificationSource.HYBRID
            if feats.semantic_similarity > 0
            else ClassificationSource.DETERMINISTIC
        ),
    )


def best_decision(
    msg: ParsedMessage,
    cls: Classification,
    contexts: list[ApplicationContext],
    settings: Settings | None = None,
    msg_embedding: np.ndarray | None = None,
    backend: EmbeddingBackend | None = None,
    contact_type: ContactType = ContactType.UNKNOWN,
    counterparties: set[str] | None = None,
) -> tuple[MatchDecision | None, list[MatchDecision]]:
    """Score a message against every existing application context.

    Returns (winning decision or None, all decisions sorted best-first). Comparison is
    against application *contexts*, not against every individual email.
    """
    settings = settings or get_settings()
    backend = backend or get_backend()
    if msg_embedding is None:
        backend = backend
    all_decisions = [
        decide(msg, cls, ctx, settings, msg_embedding=msg_embedding, backend=backend,
               contact_type=contact_type, counterparties=counterparties)
        for ctx in contexts
    ]
    ranked = sorted(
        all_decisions,
        key=lambda d: (
            {MatchDecisionKind.AUTO_MATCH: 3, MatchDecisionKind.NEW_APPLICATION: 2,
             MatchDecisionKind.REVIEW_REQUIRED: 1, MatchDecisionKind.NO_MATCH: 0}[d.kind],
            d.score,
        ),
        reverse=True,
    )
    if not ranked:
        return None, []
    top = ranked[0]
    return (top if top.kind != MatchDecisionKind.NO_MATCH else None), ranked
