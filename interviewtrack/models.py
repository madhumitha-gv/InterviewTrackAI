"""Pydantic models for every structure that crosses a module boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import (
    ApplicationStatus,
    ClassificationSource,
    ContactType,
    EventType,
    MatchDecisionKind,
    Occurrence,
    Stage,
    StageState,
)


class RawMessage(BaseModel):
    """A Gmail message as retrieved, before parsing.

    `body_text`/`body_html` stay empty until the metadata-first filter decides this
    message is worth a full fetch.
    """

    model_config = ConfigDict(extra="forbid")

    gmail_id: str
    thread_id: str
    rfc_message_id: str | None = None
    in_reply_to: str | None = None
    references: list[str] = Field(default_factory=list)
    from_addr: str = ""
    from_name: str = ""
    to_addrs: list[str] = Field(default_factory=list)
    cc_addrs: list[str] = Field(default_factory=list)
    subject: str = ""
    date: datetime | None = None
    snippet: str = ""
    labels: list[str] = Field(default_factory=list)
    body_text: str = ""
    body_html: str = ""
    body_fetched: bool = False

    @field_validator("date")
    @classmethod
    def _tz_aware(cls, v: datetime | None) -> datetime | None:
        """Normalize to UTC so cross-message ordering is always well defined."""
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @property
    def sort_key(self) -> datetime:
        return self.date or datetime(1970, 1, 1, tzinfo=timezone.utc)


class ParsedMessage(BaseModel):
    """A RawMessage with clean, embedding-ready text extracted."""

    model_config = ConfigDict(extra="forbid")

    raw: RawMessage
    clean_text: str = ""
    quoted_text: str = ""
    parse_errors: list[str] = Field(default_factory=list)

    @property
    def gmail_id(self) -> str:
        return self.raw.gmail_id

    @property
    def thread_id(self) -> str:
        return self.raw.thread_id


class Entities(BaseModel):
    """Normalized entities extracted from one message."""

    model_config = ConfigDict(extra="forbid")

    employer: str | None = None
    employer_canonical: str | None = None
    employer_confidence: float = 0.0
    employer_source: str = ""
    job_title: str | None = None
    job_title_canonical: str | None = None
    role_family: str | None = None
    requisition_id: str | None = None
    recruiter_name: str | None = None
    recruiter_email: str | None = None
    sender_domain: str = ""
    is_staffing_sender: bool = False
    is_ats_sender: bool = False


class Classification(BaseModel):
    """Event typing + structured extraction for a single message."""

    model_config = ConfigDict(extra="forbid")

    gmail_id: str
    thread_id: str
    event_type: EventType
    occurrence: Occurrence = Occurrence.MENTIONED
    is_recruitment: bool = True
    entities: Entities = Field(default_factory=Entities)
    event_datetime: datetime | None = None
    evidence: str = ""
    confidence: float = 0.0
    source: ClassificationSource = ClassificationSource.DETERMINISTIC
    rules_fired: list[str] = Field(default_factory=list)


class MatchFeatures(BaseModel):
    """The feature vector behind a single match decision — fully inspectable."""

    model_config = ConfigDict(extra="forbid")

    requisition_match: bool = False
    requisition_conflict: bool = False
    employer_match: bool = False
    employer_conflict: bool = False
    role_similarity: float = 0.0
    role_conflict: bool = False
    shared_recruiter: bool = False
    known_contact: bool = False
    known_contact_ambiguous: bool = False
    automated_sender: bool = False
    same_gmail_thread: bool = False
    user_separated: bool = False
    thread_reference_link: bool = False
    semantic_similarity: float = 0.0
    temporal_proximity: float = 0.0
    days_apart: float = 0.0


class MatchDecision(BaseModel):
    """Why a message was (or was not) attached to an application."""

    model_config = ConfigDict(extra="forbid")

    gmail_id: str
    application_key: str | None = None
    candidate_keys: list[str] = Field(default_factory=list)
    kind: MatchDecisionKind
    rule_fired: str = ""
    score: float = 0.0
    confidence_penalty: float = 0.0
    features: MatchFeatures = Field(default_factory=MatchFeatures)
    reasons: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    source: ClassificationSource = ClassificationSource.HYBRID


class InterviewEvent(BaseModel):
    """One timeline entry inside a reconstructed application."""

    model_config = ConfigDict(extra="forbid")

    gmail_id: str
    thread_id: str
    rfc_message_id: str | None = None
    occurred_at: datetime | None = None
    event_type: EventType
    occurrence: Occurrence
    sender: str = ""
    sender_name: str = ""
    subject: str = ""
    preview: str = ""
    evidence: str = ""
    evidence_ref: EvidenceReference | None = None
    extraction_source: ClassificationSource = ClassificationSource.DETERMINISTIC
    confidence: float = 0.0
    event_datetime: datetime | None = None

    @property
    def sort_key(self) -> datetime:
        return self.occurred_at or datetime(1970, 1, 1, tzinfo=timezone.utc)


class EvidenceReference(BaseModel):
    """A pointer back to the message that supports a claim.

    Every event and every status carries one, so nothing the UI shows is unsourced.
    """

    model_config = ConfigDict(extra="forbid")

    message_id: str
    thread_id: str = ""
    rfc_message_id: str | None = None
    quote: str = ""
    gmail_url: str = ""


class Organization(BaseModel):
    """An employer or an intermediary. Intermediaries never establish employer identity."""

    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    display_name: str = ""
    domains: list[str] = Field(default_factory=list)
    is_employer: bool = True
    is_intermediary: bool = False


class Contact(BaseModel):
    """A participant in a hiring process.

    Supporting evidence for matching, never application identity.
    """

    model_config = ConfigDict(extra="forbid")

    email: str
    display_name: str = ""
    org_domain: str = ""
    contact_type: ContactType = ContactType.UNKNOWN
    first_seen: datetime | None = None

    @property
    def is_automated(self) -> bool:
        return self.contact_type in {ContactType.ATS_NOREPLY, ContactType.SCHEDULING_TOOL}

    @property
    def label(self) -> str:
        return self.display_name or self.email


class RecruiterContact(BaseModel):
    """Deprecated alias kept so older persisted rows still load."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    email: str = ""
    organization: str = ""


class Application(BaseModel):
    """A reconstructed hiring process: one employer + one role, many threads."""

    model_config = ConfigDict(extra="forbid")

    key: str
    user_id: str = "demo"
    employer: str
    employer_canonical: str
    job_title: str | None = None
    job_title_canonical: str | None = None
    role_family: str | None = None
    requisition_id: str | None = None

    events: list[InterviewEvent] = Field(default_factory=list)
    thread_ids: list[str] = Field(default_factory=list)
    contacts: list[Contact] = Field(default_factory=list)
    recruiters: list[RecruiterContact] = Field(default_factory=list)
    decisions: list[MatchDecision] = Field(default_factory=list)

    # --- status, kept as separate fields rather than one collapsed value ---
    current_stage: Stage = Stage.OUTREACH
    stage_state: StageState = StageState.UNKNOWN
    status: ApplicationStatus = ApplicationStatus.ACTION_REQUIRED
    evidence_message_id: str | None = None
    stage_note: str = ""
    latest_activity_at: datetime | None = None
    latest_activity_label: str = ""
    last_supported_milestone: str = ""
    match_confidence: float = 0.0
    needs_review: bool = False
    history_possibly_incomplete: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def timeline(self) -> list[InterviewEvent]:
        """Events in chronological order."""
        return sorted(self.events, key=lambda e: e.sort_key)

    @property
    def thread_count(self) -> int:
        return len(set(self.thread_ids))

    @property
    def display_title(self) -> str:
        return f"{self.employer} — {self.job_title or 'Unspecified role'}"

    @property
    def recruiter_contacts(self) -> list[Contact]:
        """Human contacts, most useful first."""
        order = {
            ContactType.INTERNAL_RECRUITER: 0, ContactType.AGENCY_RECRUITER: 1,
            ContactType.COORDINATOR: 2, ContactType.HIRING_MANAGER: 3,
        }
        return sorted(
            [c for c in self.contacts if not c.is_automated
             and c.contact_type != ContactType.SELF],
            key=lambda c: order.get(c.contact_type, 9),
        )

    def contact_for(self, email: str) -> Contact | None:
        email = (email or "").lower()
        return next((c for c in self.contacts if c.email.lower() == email), None)


class LLMAssociationVerdict(BaseModel):
    """Schema the optional LLM must conform to. Advisory only — cannot override blockers."""

    model_config = ConfigDict(extra="forbid")

    associate: bool
    application_key: str | None = None
    shared_evidence: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    rationale: str = ""
    uncertainty: float = Field(default=1.0, ge=0.0, le=1.0)
    recommend_human_review: bool = True


class Correction(BaseModel):
    """A manual override. Sticky: re-applied after every subsequent sync."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = "demo"
    kind: str  # "assign" | "detach" | "merge" | "split" | "field"
    gmail_id: str | None = None
    application_key: str | None = None
    target_key: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
