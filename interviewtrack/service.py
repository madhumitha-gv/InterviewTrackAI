"""Application service: ties retrieval, reconstruction and persistence together.

The UI talks only to this module, so the Streamlit layer stays thin and the whole
workflow stays testable without a browser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

from .config import Settings, get_settings
from .db import Database, get_db
from .embeddings import EmbeddingBackend, get_backend
from .enums import ApplicationStatus, Stage
from .fixtures.synthetic import ALL_MESSAGES, USER_EMAIL, gmail_messages_as_api_payloads
from .gmail.client import FixtureTransport, GmailTransport, SyncReport, fetch_messages
from .gmail.desktop_oauth import AuthError
from .sources import EmailSource, GmailEmailSource, SyntheticEmailSource
from .models import Application, Correction, RawMessage
from .reconstruct import ReconstructionResult, reconstruct

logger = logging.getLogger(__name__)

DEMO_USER_ID = "demo-user"


@dataclass
class SyncOutcome:
    """Result of a synchronize call — reported to the user verbatim."""

    new_messages: int = 0
    total_messages: int = 0
    applications: int = 0
    needs_review: int = 0
    report: SyncReport | None = None
    reconstruction: ReconstructionResult | None = None
    errors: list[str] = field(default_factory=list)


class InterviewTrackService:
    """Coordinates Gmail retrieval, reconstruction and SQLite persistence."""

    def __init__(self, db: Database | None = None, settings: Settings | None = None,
                 backend: EmbeddingBackend | None = None):
        self.settings = settings or get_settings()
        self.db = db or get_db()
        self._backend = backend

    @property
    def backend(self) -> EmbeddingBackend:
        if self._backend is None:
            self._backend = get_backend()
        return self._backend

    # --- users ---------------------------------------------------------------

    def register_user(self, user_id: str, email: str, name: str = "",
                      picture: str = "") -> None:
        self.db.upsert_user(user_id, email, name, picture)

    def ensure_demo_user(self) -> str:
        self.db.upsert_user(DEMO_USER_ID, USER_EMAIL, "Avery (Demo)")
        return DEMO_USER_ID

    # --- synchronization -----------------------------------------------------

    def synchronize(
        self,
        user_id: str,
        user_email: str,
        transport: GmailTransport | None = None,
        incremental: bool = True,
        progress: Callable[[int, int], None] | None = None,
        now: datetime | None = None,
    ) -> SyncOutcome:
        """Fetch new mail (if a transport is given), then rebuild applications.

        With no transport, nothing is fetched and the already-stored messages are
        re-reconstructed. That is how the demo-mode "Refresh" works.
        """
        outcome = SyncOutcome()

        if transport is not None:
            known = self.db.known_gmail_ids(user_id) if incremental else set()
            try:
                messages, report = fetch_messages(
                    transport, self.settings, known_ids=known, progress=progress
                )
            except Exception as exc:
                outcome.errors.append(f"Gmail retrieval failed: {type(exc).__name__}: {exc}")
                messages, report = [], SyncReport()
            outcome.report = report
            outcome.errors.extend(report.errors)
            if messages:
                self.db.upsert_messages(user_id, messages)
            outcome.new_messages = len(messages)

        return self._rebuild(user_id, user_email, outcome, now=now)

    def sync_from_source(
        self,
        user_id: str,
        user_email: str,
        source: EmailSource,
        incremental: bool = True,
        progress: Callable[[int, int], None] | None = None,
        now: datetime | None = None,
    ) -> SyncOutcome:
        """Fetch through an EmailSource adapter, then rebuild.

        A Gmail authorization failure is surfaced as an error on the outcome. It is
        never answered with synthetic data — the UI must say Gmail is unavailable.
        """
        outcome = SyncOutcome()
        known = self.db.known_gmail_ids(user_id) if incremental else set()
        if incremental:
            # Metadata retained after a failed body fetch is not a completed fetch.
            known -= {m.gmail_id for m in self.db.load_messages(user_id) if not m.body_fetched}
        try:
            messages, report = source.fetch(known_ids=known, progress=progress)
        except AuthError as exc:
            outcome.errors.append(f"Gmail not available: {exc}")
            return outcome
        except Exception as exc:
            outcome.errors.append(
                f"{source.name} fetch failed: {type(exc).__name__}: {exc}"
            )
            return outcome

        outcome.report = report
        outcome.errors.extend(report.errors)
        if messages:
            self.db.upsert_messages(user_id, messages)
        outcome.new_messages = len(messages)
        return self._rebuild(user_id, user_email, outcome, now=now)

    def load_demo_data(self, user_id: str | None = None, via_gmail_parser: bool = True,
                       now: datetime | None = None,
                       profile_key: str | None = None) -> SyncOutcome:
        """Seed a synthetic dataset through the SAME pipeline Gmail Mode uses.

        The fixtures are rendered as Gmail API payloads and run through the real
        two-pass retrieval + reconstruction path — only the transport is in-memory,
        so Demo Mode shows the exact behaviour a live account would.

        `profile_key` selects one of the cross-inbox mailbox profiles; omitted, it
        uses the default scenario dataset.
        """
        if profile_key:
            from .fixtures.mailboxes import ALL_PROFILES

            profile = next((p for p in ALL_PROFILES if p.key == profile_key), None)
            if profile is None:
                raise ValueError(f"Unknown demo profile: {profile_key!r}")
            uid = f"demo:{profile_key}"
            self.db.upsert_user(uid, profile.user_email, profile.label)
            transport = FixtureTransport(
                gmail_messages_as_api_payloads(profile.messages), page_size=4
            )
            return self.synchronize(
                uid, profile.user_email, transport=transport, incremental=True,
                now=now or datetime(2025, 11, 1, tzinfo=timezone.utc),
            )

        uid = user_id or self.ensure_demo_user()
        transport = FixtureTransport(gmail_messages_as_api_payloads(), page_size=8)
        return self.synchronize(
            uid, USER_EMAIL, transport=transport, incremental=True,
            now=now or datetime(2025, 10, 28, tzinfo=timezone.utc),
        )

    def reset_and_reload_demo(self, now: datetime | None = None) -> SyncOutcome:
        """Wipe demo data and reseed it (used by the demo Reset button)."""
        self.db.delete_user_data(DEMO_USER_ID)
        return self.load_demo_data(now=now)

    def _rebuild(self, user_id: str, user_email: str, outcome: SyncOutcome,
                 now: datetime | None = None) -> SyncOutcome:
        """Re-run reconstruction over all stored messages plus stored corrections."""
        messages = self.db.load_messages(user_id)
        outcome.total_messages = len(messages)
        corrections = self.db.load_corrections(user_id)

        result = reconstruct(
            messages,
            user_email=user_email,
            user_id=user_id,
            settings=self.settings,
            backend=self.backend,
            corrections=corrections,
            now=now,
        )
        self.db.replace_applications(user_id, result.applications)
        self.db.save_decisions(user_id, result.decisions)
        self.db.set_sync_state(
            user_id, len(messages),
            window_start=(now or datetime.now(timezone.utc))
            - timedelta(days=self.settings.lookback_days),
            notes=f"backend={self.backend.describe()}",
        )

        outcome.reconstruction = result
        outcome.applications = len(result.applications)
        outcome.needs_review = len(result.review_queue)
        return outcome

    # --- reads ---------------------------------------------------------------

    def applications(self, user_id: str) -> list[Application]:
        return self.db.load_applications(user_id)

    def interview_applications(self, user_id: str, user_email: str) -> list[Application]:
        """Only show processes with explicit incoming interview/advancement evidence.

        Keep the original records for future association and manual corrections;
        project the timeline from its first invitation for the interview-only UI.
        """
        from .interview_policy import interview_start_evidence, interview_bucket
        from .parsing import parse_message
        starts = set()
        for raw in self.db.load_messages(user_id):
            msg = parse_message(raw)
            if interview_start_evidence(raw.subject, msg.clean_text or raw.snippet,
                    from_candidate=raw.from_addr.lower() == user_email.lower() or "SENT" in raw.labels):
                starts.add(raw.gmail_id)
        visible = []
        for app in self.applications(user_id):
            events = app.timeline
            first = next((i for i, event in enumerate(events) if event.gmail_id in starts), None)
            if first is not None and interview_bucket(app):
                visible.append(app.model_copy(update={"events": events[first:],
                    "thread_ids": sorted({event.thread_id for event in events[first:]})}))
        return visible

    def sync_state(self, user_id: str) -> dict[str, Any] | None:
        return self.db.get_sync_state(user_id)

    def decisions_for(self, user_id: str, gmail_id: str) -> list[dict[str, Any]]:
        return self.db.load_decisions(user_id, gmail_id)

    # --- corrections (persisted, re-applied on every sync) -------------------

    def _record(self, user_id: str, user_email: str, correction: Correction,
                now: datetime | None = None) -> SyncOutcome:
        self.db.add_correction(correction)
        return self._rebuild(user_id, user_email, SyncOutcome(), now=now)

    def merge_applications(self, user_id: str, user_email: str, source_key: str,
                           target_key: str, now: datetime | None = None) -> SyncOutcome:
        return self._record(
            user_id, user_email,
            Correction(user_id=user_id, kind="merge", application_key=source_key,
                       target_key=target_key),
            now=now,
        )

    def split_message(self, user_id: str, user_email: str, app_key: str,
                      gmail_id: str, now: datetime | None = None) -> SyncOutcome:
        return self._record(
            user_id, user_email,
            Correction(user_id=user_id, kind="split", application_key=app_key,
                       gmail_id=gmail_id),
            now=now,
        )

    def assign_message(self, user_id: str, user_email: str, gmail_id: str,
                       target_key: str, now: datetime | None = None) -> SyncOutcome:
        return self._record(
            user_id, user_email,
            Correction(user_id=user_id, kind="assign", gmail_id=gmail_id,
                       target_key=target_key),
            now=now,
        )

    def correct_fields(self, user_id: str, user_email: str, app_key: str,
                       employer: str | None = None, job_title: str | None = None,
                       requisition_id: str | None = None,
                       now: datetime | None = None) -> SyncOutcome:
        payload = {
            k: v for k, v in {
                "employer": employer, "job_title": job_title,
                "requisition_id": requisition_id,
            }.items() if v
        }
        return self._record(
            user_id, user_email,
            Correction(user_id=user_id, kind="field", application_key=app_key,
                       payload=payload),
            now=now,
        )

    def confirm_review(self, user_id: str, user_email: str, app_key: str,
                       now: datetime | None = None) -> SyncOutcome:
        return self._record(
            user_id, user_email,
            Correction(user_id=user_id, kind="resolve_review", application_key=app_key),
            now=now,
        )

    # --- data deletion -------------------------------------------------------

    def delete_everything(self, user_id: str) -> dict[str, int]:
        return self.db.delete_user_data(user_id)


# --- filtering / search helpers (pure functions, unit-testable) ---------------

FILTERS: dict[str, Callable[[Application], bool]] = {
    "Active": lambda a: a.status in {
        ApplicationStatus.ACTION_REQUIRED, ApplicationStatus.SCHEDULED,
    },
    "Assessment": lambda a: a.current_stage == Stage.ASSESSMENT and a.status in {
        ApplicationStatus.ACTION_REQUIRED, ApplicationStatus.AWAITING_RESPONSE,
    },
    "Awaiting": lambda a: a.status == ApplicationStatus.AWAITING_RESPONSE,
    "Offers": lambda a: a.status == ApplicationStatus.OFFER,
    "Rejected": lambda a: a.status == ApplicationStatus.REJECTED,
    "Closed": lambda a: a.status in {
        ApplicationStatus.WITHDRAWN, ApplicationStatus.REJECTED,
    },
    "Needs review": lambda a: a.needs_review,
    "All": lambda a: True,
}


def search_applications(apps: Sequence[Application], query: str) -> list[Application]:
    """Case-insensitive search over employer, job title and recruiter names."""
    q = (query or "").strip().lower()
    if not q:
        return list(apps)
    out: list[Application] = []
    for a in apps:
        haystack = " ".join(
            filter(None, [
                a.employer, a.job_title or "", a.requisition_id or "",
                *[r.name for r in a.recruiters], *[r.email for r in a.recruiters],
            ])
        ).lower()
        if q in haystack:
            out.append(a)
    return out


#: Active processes sort first, but rejected applications are preserved, never hidden.
#: Active work sorts first; closed applications are preserved, never hidden.
_STATUS_PRIORITY = {
    ApplicationStatus.OFFER: 0,
    ApplicationStatus.ACTION_REQUIRED: 1,
    ApplicationStatus.SCHEDULED: 2,
    ApplicationStatus.AWAITING_RESPONSE: 3,
    ApplicationStatus.NEEDS_REVIEW: 4,
    ApplicationStatus.WITHDRAWN: 5,
    ApplicationStatus.REJECTED: 6,
}


def sort_applications(apps: Sequence[Application]) -> list[Application]:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return sorted(
        apps,
        key=lambda a: (
            _STATUS_PRIORITY.get(a.status, 9),
            -(a.latest_activity_at or epoch).timestamp(),
        ),
    )
