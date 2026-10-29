"""Controlled vocabularies shared across the pipeline."""

from __future__ import annotations

from enum import Enum


class EventType(str, Enum):
    """Recruitment event taxonomy (section 5 of the spec)."""

    RECRUITER_OUTREACH = "recruiter_outreach"
    SCREENING_INVITATION = "screening_invitation"
    SCREENING_CONFIRMATION = "screening_confirmation"
    TECHNICAL_INTERVIEW = "technical_interview"
    HIRING_MANAGER_INTERVIEW = "hiring_manager_interview"
    FINAL_INTERVIEW = "final_interview"
    SCHEDULING = "scheduling"
    ASSESSMENT_ASSIGNED = "assessment_assigned"
    ASSESSMENT_SUBMITTED = "assessment_submitted"
    RECRUITER_FOLLOWUP = "recruiter_followup"
    CANDIDATE_FOLLOWUP = "candidate_followup"
    OFFER = "offer"
    REJECTION = "rejection"
    WITHDRAWAL = "withdrawal"
    OTHER_RECRUITMENT = "other_recruitment"
    NOT_RECRUITMENT = "not_recruitment"


#: Event types that represent an actual interview round (used for stage resolution).
INTERVIEW_EVENTS = frozenset(
    {
        EventType.SCREENING_INVITATION,
        EventType.SCREENING_CONFIRMATION,
        EventType.TECHNICAL_INTERVIEW,
        EventType.HIRING_MANAGER_INTERVIEW,
        EventType.FINAL_INTERVIEW,
    }
)

#: Event types that terminate an application.
TERMINAL_EVENTS = frozenset({EventType.REJECTION, EventType.WITHDRAWAL})


class ContactType(str, Enum):
    """Who a sender is in a hiring process.

    Contact type is *supporting evidence* for matching, never application identity:
    one recruiter can run several roles, and several people work one role.
    """

    INTERNAL_RECRUITER = "internal_recruiter"
    AGENCY_RECRUITER = "agency_recruiter"
    COORDINATOR = "coordinator"
    HIRING_MANAGER = "hiring_manager"
    ATS_NOREPLY = "ats_noreply"
    SCHEDULING_TOOL = "scheduling_tool"
    SELF = "self"
    UNKNOWN = "unknown"


#: Senders that are machines or intermediaries. They may participate in a process but
#: can never establish employer identity on their own.
AUTOMATED_CONTACT_TYPES = frozenset(
    {ContactType.ATS_NOREPLY, ContactType.SCHEDULING_TOOL}
)

#: Human contacts that can be "learned" as belonging to an application.
LEARNABLE_CONTACT_TYPES = frozenset(
    {
        ContactType.INTERNAL_RECRUITER,
        ContactType.AGENCY_RECRUITER,
        ContactType.COORDINATOR,
        ContactType.HIRING_MANAGER,
    }
)


class StageState(str, Enum):
    """State of the current stage, kept separate from the stage itself."""

    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELED = "canceled"
    UNKNOWN = "unknown"


class Occurrence(str, Enum):
    """Did the event happen, or is it only planned/discussed?

    Nothing is promoted to COMPLETED by clock time alone — it requires explicit
    textual confirmation in the email.
    """

    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    CANCELED = "canceled"
    MENTIONED = "mentioned"


class Stage(str, Enum):
    """Ordered interview funnel stages. Order matters: used as a monotonic ratchet."""

    OUTREACH = "outreach"
    SCREENING = "screening"
    ASSESSMENT = "assessment"
    TECHNICAL = "technical_interview"
    HIRING_MANAGER = "hiring_manager_interview"
    FINAL = "final_interview"
    OFFER = "offer"
    CLOSED = "closed"


#: Monotonic funnel ordering. CLOSED is deliberately excluded — it is a status, not a rung.
STAGE_ORDER: tuple[Stage, ...] = (
    Stage.OUTREACH,
    Stage.SCREENING,
    Stage.ASSESSMENT,
    Stage.TECHNICAL,
    Stage.HIRING_MANAGER,
    Stage.FINAL,
    Stage.OFFER,
)

STAGE_RANK: dict[Stage, int] = {stage: i for i, stage in enumerate(STAGE_ORDER)}


class ApplicationStatus(str, Enum):
    """Status comes from the latest status-changing event, never from the clock."""

    ACTION_REQUIRED = "action_required"
    SCHEDULED = "scheduled"
    AWAITING_RESPONSE = "awaiting_response"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    NEEDS_REVIEW = "needs_review"


#: UI filter buckets, exactly as the spec defines them.
STATUS_FILTERS: dict[str, frozenset[ApplicationStatus]] = {
    "Active": frozenset({ApplicationStatus.ACTION_REQUIRED, ApplicationStatus.SCHEDULED}),
    "Assessment": frozenset({ApplicationStatus.ACTION_REQUIRED}),  # + stage == ASSESSMENT
    "Awaiting": frozenset({ApplicationStatus.AWAITING_RESPONSE}),
    "Offers": frozenset({ApplicationStatus.OFFER}),
    "Rejected": frozenset({ApplicationStatus.REJECTED}),
    "Closed": frozenset({ApplicationStatus.WITHDRAWN}),
}


class MatchDecisionKind(str, Enum):
    AUTO_MATCH = "AUTO_MATCH"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NEW_APPLICATION = "NEW_APPLICATION"
    NO_MATCH = "NO_MATCH"


class ClassificationSource(str, Enum):
    """Provenance of a classification or match — never silently blended."""

    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    LLM = "llm"
    MANUAL = "manual"


#: Which funnel stage each event type implies. Events absent from this map
#: (follow-ups, generic scheduling, correspondence) deliberately imply NO stage,
#: so they update "latest activity" without inventing interview progress.
InterviewStageMap: dict[EventType, Stage] = {
    EventType.RECRUITER_OUTREACH: Stage.OUTREACH,
    EventType.OTHER_RECRUITMENT: Stage.OUTREACH,
    EventType.SCREENING_INVITATION: Stage.SCREENING,
    EventType.SCREENING_CONFIRMATION: Stage.SCREENING,
    EventType.ASSESSMENT_ASSIGNED: Stage.ASSESSMENT,
    EventType.ASSESSMENT_SUBMITTED: Stage.ASSESSMENT,
    EventType.TECHNICAL_INTERVIEW: Stage.TECHNICAL,
    EventType.HIRING_MANAGER_INTERVIEW: Stage.HIRING_MANAGER,
    EventType.FINAL_INTERVIEW: Stage.FINAL,
    EventType.OFFER: Stage.OFFER,
}
