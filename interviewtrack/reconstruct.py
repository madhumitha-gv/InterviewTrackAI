"""The cross-thread reconstruction pipeline (Stages A-F orchestration).

Messages are processed in chronological order and matched incrementally against
existing *application contexts*. This keeps the comparison count at
O(messages x applications) rather than O(messages^2), and means an application's
accumulated identity (employer, role, requisition, recruiters) is what later
messages are judged against.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from . import status as status_engine
from .classify import classify_message, is_probably_recruitment
from .contacts import (
    ContactRegistry,
    contact_from_message,
    participants,
    resolve_contact_types,
)
from .config import Settings, get_settings
from .embeddings import EmbeddingBackend, build_message_document, get_backend
from .enums import (
    LEARNABLE_CONTACT_TYPES,
    ApplicationStatus,
    ClassificationSource,
    ContactType,
    EventType,
    MatchDecisionKind,
)
from .matching import ApplicationContext, best_decision
from .models import (
    Application,
    Contact,
    Classification,
    Correction,
    EvidenceReference,
    InterviewEvent,
    MatchDecision,
    ParsedMessage,
    RawMessage,
    RecruiterContact,
)
from .normalize import (
    canonical_company,
    canonical_person,
    canonical_role,
    company_similarity,
)
from .parsing import parse_message

logger = logging.getLogger(__name__)

UNKNOWN_EMPLOYER = "Unidentified employer"


@dataclass
class ReconstructionResult:
    """Everything a sync produced, including why each decision was made."""

    applications: list[Application] = field(default_factory=list)
    decisions: list[MatchDecision] = field(default_factory=list)
    skipped_non_recruitment: list[str] = field(default_factory=list)
    parse_error_ids: list[str] = field(default_factory=list)
    embedding_backend: str = ""
    semantic_enabled: bool = False
    processed_count: int = 0
    window_start: datetime | None = None
    registry: "ContactRegistry | None" = None

    @property
    def review_queue(self) -> list[Application]:
        """Applications awaiting a human decision."""
        return [a for a in self.applications if a.needs_review]

    def by_key(self, key: str) -> Application | None:
        return next((a for a in self.applications if a.key == key), None)


def _application_key(cls: Classification, fallback: str) -> str:
    """Stable, deterministic key for an employer+role hiring process.

    Stability matters: the same application must get the same key on every sync so
    that stored corrections keep applying.
    """
    ent = cls.entities
    emp = ent.employer_canonical or ""
    if emp and ent.requisition_id:
        basis = f"{emp}|req={ent.requisition_id}"
    elif emp:
        role = ent.job_title_canonical or ent.role_family or ""
        basis = f"{emp}|role={role}" if role else f"{emp}|role=?"
    else:
        # No identifiable employer: never collapse these together.
        basis = f"unknown|{fallback}"
    digest = hashlib.blake2b(basis.encode("utf-8"), digest_size=6).hexdigest()
    return f"app_{digest}"


def _event_from(msg: ParsedMessage, cls: Classification) -> InterviewEvent:
    from .gmail.links import best_message_link  # local import avoids a cycle

    preview = (msg.clean_text or msg.raw.snippet or "").strip()
    evidence_ref = EvidenceReference(
        message_id=msg.gmail_id,
        thread_id=msg.thread_id,
        rfc_message_id=msg.raw.rfc_message_id,
        quote=cls.evidence[:200],
        gmail_url=best_message_link(msg.raw.rfc_message_id, msg.thread_id),
    )
    return InterviewEvent(
        evidence_ref=evidence_ref,
        extraction_source=cls.source,
        gmail_id=msg.gmail_id,
        thread_id=msg.thread_id,
        rfc_message_id=msg.raw.rfc_message_id,
        occurred_at=msg.raw.date,
        event_type=cls.event_type,
        occurrence=cls.occurrence,
        sender=msg.raw.from_addr,
        sender_name=msg.raw.from_name,
        subject=msg.raw.subject,
        preview=preview[:280],
        evidence=cls.evidence,
        confidence=cls.confidence,
        event_datetime=cls.event_datetime,
    )


def _new_context(cls: Classification, msg: ParsedMessage, key: str,
                 user_id: str) -> ApplicationContext:
    ent = cls.entities
    app = Application(
        key=key,
        user_id=user_id,
        employer=ent.employer or UNKNOWN_EMPLOYER,
        employer_canonical=ent.employer_canonical or "",
        job_title=ent.job_title,
        job_title_canonical=ent.job_title_canonical,
        role_family=ent.role_family,
        requisition_id=ent.requisition_id,
    )
    return ApplicationContext(application=app)


def _is_better_employer_name(candidate: str, current: str) -> bool:
    """Prefer a written-out employer name over one derived from a domain.

    A domain yields "Vantagesystems"; the body yields "Vantage Systems". Both
    canonicalize the same, so only the display form differs — show the readable one.
    """
    if not candidate or not current:
        return False
    if canonical_company(candidate).replace(" ", "") != canonical_company(
        current
    ).replace(" ", ""):
        return False  # different companies: never silently rename
    return candidate.count(" ") > current.count(" ")


def _absorb(ctx: ApplicationContext, msg: ParsedMessage, cls: Classification,
            decision: MatchDecision | None,
            contact: Contact | None = None,
            registry: ContactRegistry | None = None,
            user_email: str | None = None,
            learn: bool = False) -> None:
    """Attach a message to an application and enrich the application's identity."""
    app = ctx.application
    ent = cls.entities

    if any(e.gmail_id == msg.gmail_id for e in app.events):
        return  # idempotent: duplicate sync must not duplicate events

    app.events.append(_event_from(msg, cls))
    if msg.thread_id not in app.thread_ids:
        app.thread_ids.append(msg.thread_id)
    ctx.thread_ids.add(msg.thread_id)
    if msg.raw.rfc_message_id:
        ctx.rfc_message_ids.add(msg.raw.rfc_message_id)
    if msg.raw.subject:
        ctx.recent_subjects.append(msg.raw.subject)

    # --- enrich identity: fill gaps, never overwrite a confident value with a guess
    if ent.employer and (
        app.employer == UNKNOWN_EMPLOYER
        or (not app.employer_canonical and ent.employer_canonical)
        or _is_better_employer_name(ent.employer, app.employer)
    ):
        app.employer = ent.employer
        app.employer_canonical = ent.employer_canonical or canonical_company(ent.employer)
    if ent.job_title and not app.job_title:
        app.job_title = ent.job_title
        app.job_title_canonical = ent.job_title_canonical or canonical_role(ent.job_title)
        app.role_family = ent.role_family or app.role_family
    if ent.role_family and not app.role_family:
        app.role_family = ent.role_family
    if ent.requisition_id and not app.requisition_id:
        app.requisition_id = ent.requisition_id

    # --- recruiter contacts
    if ent.recruiter_email:
        email = ent.recruiter_email.lower()
        if email not in ctx.recruiter_emails:
            ctx.recruiter_emails.add(email)
            app.recruiters.append(
                RecruiterContact(
                    name=(msg.raw.from_name or ent.recruiter_name or "").strip(),
                    email=email,
                    organization=ent.sender_domain,
                )
            )
    if ent.recruiter_name:
        ctx.recruiter_names.add(canonical_person(ent.recruiter_name))

    # --- typed contact record ---
    if contact is not None and contact.email:
        if not any(c.email == contact.email for c in app.contacts):
            app.contacts.append(contact)
        ctx.known_contact_emails.add(contact.email)
        if registry is not None:
            registry.register(contact)
            # Contact learning: a sender becomes a known contact of this application
            # once one of their messages matched on identity evidence. This is what
            # carries a recruiter handoff or a coordinator's first email.
            if learn and contact.contact_type in LEARNABLE_CONTACT_TYPES:
                registry.learn(ctx.key, contact.email)
                # Everyone else on the thread is a candidate contact too, which is
                # how a CC'd agency recruiter links a handoff to the new sender.
                for addr in participants(msg, user_email):
                    ctx.known_contact_emails.add(addr)

    if msg.raw.date and (ctx.last_activity is None or msg.raw.date > ctx.last_activity):
        ctx.last_activity = msg.raw.date

    if decision is not None:
        app.decisions.append(decision)
        app.match_confidence = round(
            max(app.match_confidence, decision.score) if app.events else decision.score, 4
        )

    # The application's semantic identity changed, so its cached vector is stale.
    ctx.invalidate_embedding()


def _rank_candidates(siblings, vec, backend) -> list[str]:
    """Rank candidate applications for a role-less message.

    Embeddings are used here and only here, purely to order candidates for a human.
    They never cause a merge.
    """
    if vec is None or backend is None:
        return [c.key for c in siblings]
    from .embeddings import cosine_similarity

    scored = [
        (cosine_similarity(vec, c.ensure_embedding(backend)), c.key) for c in siblings
    ]
    return [key for _, key in sorted(scored, reverse=True)]


def reconstruct(
    messages: Sequence[RawMessage],
    user_email: str | None = None,
    user_id: str = "demo",
    settings: Settings | None = None,
    backend: EmbeddingBackend | None = None,
    corrections: Iterable[Correction] = (),
    now: datetime | None = None,
) -> ReconstructionResult:
    """Run the full pipeline over a batch of messages.

    Idempotent with respect to `gmail_id`: re-running on the same input (or a
    superset) yields the same applications and no duplicated events.
    """
    settings = settings or get_settings()
    backend = backend or get_backend()
    now = now or datetime.now(timezone.utc)

    result = ReconstructionResult(
        embedding_backend=backend.describe(),
        semantic_enabled=backend.is_semantic,
    )

    # --- parse + classify ---
    parsed: list[tuple[ParsedMessage, Classification]] = []
    borderline: list[tuple[ParsedMessage, Classification]] = []
    seen_ids: set[str] = set()
    for raw in messages:
        if raw.gmail_id in seen_ids:
            continue  # dedupe at the source
        seen_ids.add(raw.gmail_id)

        msg = parse_message(raw)
        if msg.parse_errors:
            result.parse_error_ids.append(msg.gmail_id)
        # A message with no usable sender AND no usable content cannot be placed.
        if "missing_or_invalid_sender" in msg.parse_errors and "empty_body" in msg.parse_errors:
            result.skipped_non_recruitment.append(msg.gmail_id)
            continue

        cls = classify_message(msg, user_email=user_email)
        if "excluded:bulk_opportunity" in cls.rules_fired:
            result.skipped_non_recruitment.append(msg.gmail_id)
            continue
        # A specific rule firing is stronger evidence than the keyword gate. Short
        # confirmations ("Confirming our screening call for Wednesday") score low on
        # keyword density but are unambiguous recruitment mail, so the gate must not
        # veto them — the spec explicitly warns against an over-restrictive filter.
        rule_fired = cls.event_type not in {
            EventType.OTHER_RECRUITMENT, EventType.NOT_RECRUITMENT
        }
        if not cls.is_recruitment or not (rule_fired or is_probably_recruitment(msg)):
            borderline.append((msg, cls))
            continue
        parsed.append((msg, cls))

    # Rescue borderline messages on two grounds, both conservative:
    #   * thread continuity — the Gmail thread is already a recruitment conversation;
    #   * known sender — this address already sent recruitment mail in this batch.
    # A terse note from a recruiter ("send me your latest resume") carries almost no
    # recruiting vocabulary, so a keyword gate alone would silently drop it and break
    # the process it belongs to.
    recruitment_threads = {m.thread_id for m, _ in parsed}
    recruitment_senders = {
        (m.raw.from_addr or "").lower() for m, _ in parsed if m.raw.from_addr
    }
    for msg, cls in borderline:
        sender = (msg.raw.from_addr or "").lower()
        on_known_thread = msg.thread_id in recruitment_threads
        from_known_sender = sender in recruitment_senders

        if on_known_thread:
            # A message on a thread already known to be a recruitment conversation is
            # part of that process, even when its own content is too thin to classify
            # (a bare "yes, Tuesday works"). Coerce it to recruitment rather than drop
            # it: a candidate reply becomes a candidate follow-up, anything else
            # generic correspondence. Neither can advance a stage on its own.
            if not cls.is_recruitment:
                from_candidate = (
                    (user_email and sender == user_email.lower())
                    or "SENT" in msg.raw.labels
                )
                new_type = (
                    EventType.CANDIDATE_FOLLOWUP if from_candidate
                    else EventType.OTHER_RECRUITMENT
                )
                cls = cls.model_copy(update={
                    "event_type": new_type, "is_recruitment": True,
                    "rules_fired": sorted(set(cls.rules_fired) | {"rescued:known_thread"}),
                })
            parsed.append((msg, cls))
        elif cls.is_recruitment and from_known_sender:
            parsed.append((msg, cls))
        elif from_known_sender and sender:
            # The keyword gate said "not recruitment", but this address has already
            # sent recruitment mail in this batch. A recruiter asking "send me your
            # latest resume" carries almost no recruiting vocabulary yet is plainly
            # part of the process, so the sender overrides the gate. It enters as
            # generic correspondence, which cannot advance a stage on its own.
            cls = cls.model_copy(update={
                "event_type": EventType.OTHER_RECRUITMENT,
                "is_recruitment": True,
                "rules_fired": sorted(set(cls.rules_fired) | {"rescued:known_sender"}),
            })
            parsed.append((msg, cls))
        else:
            result.skipped_non_recruitment.append(msg.gmail_id)

    # --- chronological order drives incremental clustering ---
    parsed.sort(key=lambda pair: pair[0].raw.sort_key)
    result.processed_count = len(parsed)
    if parsed:
        result.window_start = min(
            (m.raw.sort_key for m, _ in parsed), default=None
        )

    # --- embed all candidate messages in one batch (CPU-efficient) ---
    documents = [
        build_message_document(
            subject=m.raw.subject,
            clean_text=m.clean_text,
            employer=c.entities.employer,
            job_title=c.entities.job_title,
            event_type=c.event_type.value,
            recruiter=c.entities.recruiter_name,
        )
        for m, c in parsed
    ]
    embeddings = backend.encode(documents) if documents else None

    # --- incremental matching ---
    registry = ContactRegistry()
    result.registry = registry
    # Decide each sender's contact type once, across every message they sent.
    resolved_types = resolve_contact_types([m for m, _ in parsed], user_email)
    contexts: list[ApplicationContext] = []

    for idx, (msg, cls) in enumerate(parsed):
        vec = embeddings[idx] if embeddings is not None else None
        contact = contact_from_message(msg, user_email=user_email,
                                       resolved=resolved_types)
        registry.register(contact)

        # A sender already known to an application is a strong narrowing signal —
        # but employer and role still decide, so it is passed to the matcher as a
        # feature rather than used as a short-circuit.
        # The people this message is "with": the sender, plus every recipient when
        # the candidate wrote it themselves.
        counterparties = participants(msg, user_email)
        counterparties.add(contact.email)
        counterparties.discard("")

        known_apps: set[str] = set()
        for addr in counterparties:
            known_apps |= registry.applications_for(addr)
        for ctx in contexts:
            if ctx.key in known_apps:
                ctx.known_contact_emails |= counterparties

        winner, ranked = best_decision(
            msg, cls, contexts, settings=settings, msg_embedding=vec, backend=backend,
            contact_type=contact.contact_type, counterparties=counterparties,
        )
        result.decisions.extend(ranked)

        # --- Rule 3 guard, checked BEFORE accepting any automatic match. ---------
        # A known contact who runs several roles at one employer cannot resolve a
        # message that names no role. One recruiter, two openings, a bare "send me
        # your latest resume" — only a human can say which it belongs to.
        roleless_ambiguous: list[ApplicationContext] = []
        if not cls.entities.job_title:
            siblings = [
                c for c in contexts
                if c.application.employer_canonical
                and cls.entities.employer_canonical
                and company_similarity(
                    c.application.employer_canonical, cls.entities.employer_canonical
                ) >= 0.82
            ]
            if len(siblings) > 1:
                roleless_ambiguous = siblings

        if (
            winner is not None
            and winner.kind == MatchDecisionKind.AUTO_MATCH
            and not roleless_ambiguous
        ):
            ctx = next(c for c in contexts if c.key == winner.application_key)
            _absorb(ctx, msg, cls, winner, contact=contact, registry=registry,
                    user_email=user_email, learn=True)
            continue

        # REVIEW_REQUIRED or NO_MATCH: start a new application. Nothing is merged
        # on weak evidence, and nothing is discarded.
        # --- role missing: the employer is known but the role is not -------------
        # One open application at that employer -> attach with reduced confidence and
        # a "role inferred" reason. Several -> rank the candidates (this is the one
        # place embeddings earn their keep) and ask for review. Never guess.
        if cls.entities.employer and not cls.entities.job_title:
            siblings = roleless_ambiguous or [
                c for c in contexts
                if c.application.employer_canonical
                and company_similarity(
                    c.application.employer_canonical, cls.entities.employer_canonical
                ) >= 0.82
            ]
            if len(siblings) == 1:
                target = siblings[0]
                inferred = MatchDecision(
                    gmail_id=msg.gmail_id, application_key=target.key,
                    candidate_keys=[target.key], kind=MatchDecisionKind.AUTO_MATCH,
                    rule_fired="R6_role_inferred", score=0.55,
                    confidence_penalty=0.25,
                    reasons=[
                        "role not named, but this employer has exactly one open "
                        "application — attached with reduced confidence (role inferred)"
                    ],
                    source=ClassificationSource.DETERMINISTIC,
                )
                result.decisions.append(inferred)
                _absorb(target, msg, cls, inferred, contact=contact,
                        registry=registry, user_email=user_email, learn=False)
                continue
            if len(siblings) > 1:
                ranked_keys = _rank_candidates(siblings, vec, backend)
                review = MatchDecision(
                    gmail_id=msg.gmail_id, application_key=None,
                    candidate_keys=ranked_keys,
                    kind=MatchDecisionKind.REVIEW_REQUIRED,
                    rule_fired="R6_roleless_ambiguous", score=0.0,
                    reasons=[
                        f"role not named and this employer has {len(siblings)} open "
                        "applications; candidates ranked by semantic similarity for "
                        "human review — embeddings rank, they never merge"
                    ],
                    blockers=["role unknown with multiple candidate applications"],
                    source=ClassificationSource.HYBRID,
                )
                result.decisions.append(review)
                key = f"review_{msg.gmail_id}"
                ctx = _new_context(cls, msg, key, user_id)
                ctx.application.needs_review = True
                ctx.application.notes.append(
                    "Role not named; this employer has several open applications. "
                    "Ranked candidates: " + ", ".join(ranked_keys[:3])
                )
                contexts.append(ctx)
                _absorb(ctx, msg, cls, review, contact=contact, registry=registry,
                        user_email=user_email, learn=False)
                continue

        key = _application_key(cls, fallback=msg.thread_id)
        existing = next((c for c in contexts if c.key == key), None)
        if existing is not None:
            # Same deterministic identity (employer+role/req) — safe to join.
            _absorb(existing, msg, cls, winner, contact=contact, registry=registry,
                    user_email=user_email, learn=True)
            continue

        ctx = _new_context(cls, msg, key, user_id)
        # Record the outcome as NEW_APPLICATION when employer and role are both known.
        if winner is None and cls.entities.employer and cls.entities.job_title:
            result.decisions.append(MatchDecision(
                gmail_id=msg.gmail_id, application_key=key, candidate_keys=[key],
                kind=MatchDecisionKind.NEW_APPLICATION, rule_fired="R7_new_application",
                score=1.0,
                reasons=[
                    f"no existing application matched; employer "
                    f"({cls.entities.employer}) and role ({cls.entities.job_title}) "
                    "are both known, so a new application was opened"
                ],
                source=ClassificationSource.DETERMINISTIC,
            ))
        if winner is not None and winner.kind == MatchDecisionKind.REVIEW_REQUIRED:
            ctx.application.needs_review = True
            ctx.application.notes.append(
                f"Possible match to {winner.application_key} "
                f"(score {winner.score:.2f}) — needs human confirmation: "
                + "; ".join(winner.reasons[:3])
            )
        if not cls.entities.employer:
            ctx.application.needs_review = True
            ctx.application.notes.append(
                "Employer could not be identified from this correspondence; "
                "no association was invented."
            )
        contexts.append(ctx)
        _absorb(ctx, msg, cls, winner, contact=contact, registry=registry,
                user_email=user_email,
                learn=bool(cls.entities.employer and cls.entities.job_title))

    # --- sticky manual corrections ---
    contexts = apply_corrections(contexts, corrections, user_id=user_id)

    # --- display-name normalization across applications ---
    # One application may know the employer only as a domain ("Vantagesystems")
    # while another has the written-out name ("Vantage Systems"). They canonicalize
    # identically, so adopt the readable form. Display only — identity is untouched.
    _harmonize_employer_names(contexts)

    # A later message in a process can supply the employer that the first one lacked.
    # When that happens, the original "employer unknown" review reason no longer
    # holds, so clear it (any other review reason keeps the flag).
    _EMP_UNKNOWN_NOTE = (
        "Employer could not be identified from this correspondence; "
        "no association was invented."
    )
    for ctx in contexts:
        app = ctx.application
        if app.needs_review and app.employer_canonical and _EMP_UNKNOWN_NOTE in app.notes:
            app.notes = [n for n in app.notes if n != _EMP_UNKNOWN_NOTE]
            if not app.notes:
                app.needs_review = False

    # --- status resolution ---
    window_start = now - timedelta(days=settings.lookback_days)
    for ctx in contexts:
        status_engine.resolve(ctx.application, now=now, window_start=window_start)

    result.applications = [c.application for c in contexts]
    result.applications.sort(
        key=lambda a: (a.latest_activity_at or datetime(1970, 1, 1, tzinfo=timezone.utc)),
        reverse=True,
    )
    return result


# --- manual corrections --------------------------------------------------------


def _harmonize_employer_names(contexts: list[ApplicationContext]) -> None:
    """Give every application of one employer the best-written form of its name."""
    best: dict[str, str] = {}
    for ctx in contexts:
        name = ctx.application.employer
        if not name or name == UNKNOWN_EMPLOYER:
            continue
        key = canonical_company(name).replace(" ", "")
        if not key:
            continue
        current = best.get(key)
        if current is None or _is_better_employer_name(name, current):
            best[key] = name
    for ctx in contexts:
        name = ctx.application.employer
        key = canonical_company(name).replace(" ", "") if name else ""
        preferred = best.get(key)
        if preferred and preferred != name:
            ctx.application.employer = preferred


def apply_corrections(
    contexts: list[ApplicationContext],
    corrections: Iterable[Correction],
    user_id: str = "demo",
) -> list[ApplicationContext]:
    """Re-apply stored manual corrections on top of automatic reconstruction.

    Corrections are authoritative and survive every later synchronization.
    """
    by_key = {c.key: c for c in contexts}

    for corr in corrections:
        if corr.user_id != user_id:
            continue
        try:
            if corr.kind == "merge":
                _merge(by_key, corr.application_key, corr.target_key)
            elif corr.kind == "assign":
                _assign(by_key, corr.gmail_id, corr.target_key or corr.application_key)
            elif corr.kind == "detach":
                _detach(by_key, corr.gmail_id, corr.application_key, user_id)
            elif corr.kind == "split":
                _detach(by_key, corr.gmail_id, corr.application_key, user_id)
            elif corr.kind == "field":
                _set_fields(by_key, corr.application_key, corr.payload)
            elif corr.kind == "resolve_review":
                ctx = by_key.get(corr.application_key or "")
                if ctx is not None:
                    ctx.application.needs_review = False
                    ctx.application.notes.append("Reviewed and confirmed manually.")
        except Exception as exc:  # a bad stored correction must not break a sync
            logger.warning("Skipping correction %s: %s", corr.kind, exc)

    # Drop applications emptied by corrections.
    return [c for c in by_key.values() if c.application.events]


def _rebuild_context(ctx: ApplicationContext) -> None:
    """Recompute cached context state from the application's current events."""
    app = ctx.application
    ctx.thread_ids = {e.thread_id for e in app.events}
    ctx.rfc_message_ids = {e.rfc_message_id for e in app.events if e.rfc_message_id}
    ctx.recent_subjects = [e.subject for e in app.timeline if e.subject]
    ctx.recruiter_emails = {r.email for r in app.recruiters if r.email}
    ctx.recruiter_names = {canonical_person(r.name) for r in app.recruiters if r.name}
    app.thread_ids = sorted(ctx.thread_ids)
    ctx.last_activity = max((e.sort_key for e in app.events), default=None)
    ctx.invalidate_embedding()


def _merge(by_key: dict[str, ApplicationContext], source_key: str | None,
           target_key: str | None) -> None:
    if not source_key or not target_key or source_key == target_key:
        return
    src, dst = by_key.get(source_key), by_key.get(target_key)
    if src is None or dst is None:
        return
    existing = {e.gmail_id for e in dst.application.events}
    for event in src.application.events:
        if event.gmail_id not in existing:
            dst.application.events.append(event)
    for rec in src.application.recruiters:
        if rec.email not in {r.email for r in dst.application.recruiters}:
            dst.application.recruiters.append(rec)
    dst.application.notes.append(f"Manually merged from {source_key}.")
    dst.application.needs_review = False
    dst.application.decisions.append(
        MatchDecision(
            gmail_id="*", application_key=target_key, kind=MatchDecisionKind.AUTO_MATCH,
            score=1.0, reasons=[f"manual merge of {source_key}"],
            source=ClassificationSource.MANUAL,
        )
    )
    src.application.events = []
    _rebuild_context(dst)


def _assign(by_key: dict[str, ApplicationContext], gmail_id: str | None,
            target_key: str | None) -> None:
    if not gmail_id or not target_key:
        return
    dst = by_key.get(target_key)
    if dst is None:
        return
    moved = None
    for ctx in by_key.values():
        for event in list(ctx.application.events):
            if event.gmail_id == gmail_id:
                moved = event
                if ctx.key != target_key:
                    ctx.application.events.remove(event)
                    _rebuild_context(ctx)
    if moved is None:
        return
    if gmail_id not in {e.gmail_id for e in dst.application.events}:
        dst.application.events.append(moved)
    dst.application.notes.append(f"Message {gmail_id} manually assigned here.")
    dst.application.needs_review = False
    _rebuild_context(dst)


def _detach(by_key: dict[str, ApplicationContext], gmail_id: str | None,
            app_key: str | None, user_id: str) -> None:
    """Pull a message out of an application into its own standalone record."""
    if not gmail_id or not app_key:
        return
    ctx = by_key.get(app_key)
    if ctx is None:
        return
    event = next((e for e in ctx.application.events if e.gmail_id == gmail_id), None)
    if event is None:
        return
    ctx.application.events.remove(event)
    ctx.application.notes.append(f"Message {gmail_id} manually detached.")
    _rebuild_context(ctx)

    new_key = f"{app_key}_split_{gmail_id[:8]}"
    split = Application(
        key=new_key,
        user_id=user_id,
        employer=ctx.application.employer,
        employer_canonical=ctx.application.employer_canonical,
        job_title=ctx.application.job_title,
        job_title_canonical=ctx.application.job_title_canonical,
        role_family=ctx.application.role_family,
        events=[event],
        needs_review=True,
        notes=[f"Manually split out of {app_key}."],
    )
    new_ctx = ApplicationContext(application=split)
    _rebuild_context(new_ctx)
    by_key[new_key] = new_ctx


def _set_fields(by_key: dict[str, ApplicationContext], app_key: str | None,
                payload: dict) -> None:
    ctx = by_key.get(app_key or "")
    if ctx is None:
        return
    app = ctx.application
    if employer := payload.get("employer"):
        app.employer = str(employer)
        app.employer_canonical = canonical_company(employer)
    if title := payload.get("job_title"):
        app.job_title = str(title)
        app.job_title_canonical = canonical_role(title)
    if req := payload.get("requisition_id"):
        app.requisition_id = str(req)
    app.notes.append("Fields manually corrected.")
    app.needs_review = False
    ctx.invalidate_embedding()
