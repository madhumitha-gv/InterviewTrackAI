"""Timeline resolver: chronological stage, state and status from evidence only.

The rules this encodes, all from the spec's transition table:

  * **Stage only moves forward.** A later message about an earlier round never pulls it
    back.
  * **Status comes from the latest status-changing event.** A candidate follow-up is
    not status-changing; it moves `last_activity_at` and nothing else.
  * **A passed date never proves completion.** A scheduled interview whose date has
    passed with no confirming email renders "date passed, outcome unknown".
  * **A rejection closes only its own application**, and a message arriving afterwards
    for that same application goes to review rather than silently reopening it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .enums import (
    STAGE_RANK,
    ApplicationStatus,
    EventType,
    InterviewStageMap,
    Occurrence,
    Stage,
    StageState,
)
from .models import Application, InterviewEvent

#: Events that change the application's status. Everything else only updates activity.
_STATUS_CHANGING = frozenset(
    {
        EventType.RECRUITER_OUTREACH,
        EventType.SCREENING_INVITATION,
        EventType.SCREENING_CONFIRMATION,
        EventType.TECHNICAL_INTERVIEW,
        EventType.HIRING_MANAGER_INTERVIEW,
        EventType.FINAL_INTERVIEW,
        EventType.SCHEDULING,
        EventType.ASSESSMENT_ASSIGNED,
        EventType.ASSESSMENT_SUBMITTED,
        EventType.OFFER,
        EventType.REJECTION,
        EventType.WITHDRAWAL,
    }
)

#: Terminal events and the status each produces.
_TERMINAL = {
    EventType.OFFER: ApplicationStatus.OFFER,
    EventType.REJECTION: ApplicationStatus.REJECTED,
    EventType.WITHDRAWAL: ApplicationStatus.WITHDRAWN,
}


def _status_for(event: InterviewEvent) -> ApplicationStatus | None:
    """The status a single event implies, or None if it changes nothing."""
    etype, occ = event.event_type, event.occurrence

    if etype in _TERMINAL:
        return _TERMINAL[etype]
    if etype == EventType.ASSESSMENT_SUBMITTED:
        return ApplicationStatus.AWAITING_RESPONSE
    if etype == EventType.ASSESSMENT_ASSIGNED:
        return ApplicationStatus.ACTION_REQUIRED
    if etype not in _STATUS_CHANGING:
        return None

    # Interview-ish events: an explicitly completed round means we are now waiting.
    if occ == Occurrence.COMPLETED:
        return ApplicationStatus.AWAITING_RESPONSE
    if occ == Occurrence.SCHEDULED:
        return ApplicationStatus.SCHEDULED
    if occ == Occurrence.CANCELED:
        return ApplicationStatus.ACTION_REQUIRED
    # An invitation that has not been confirmed is the candidate's move.
    return ApplicationStatus.ACTION_REQUIRED


def resolve(app: Application, now: datetime | None = None,
            window_start: datetime | None = None) -> Application:
    """Recompute stage, stage_state, status, activity and evidence. Idempotent."""
    now = now or datetime.now(timezone.utc)
    timeline = app.timeline

    if not timeline:
        app.current_stage = Stage.OUTREACH
        app.stage_state = StageState.UNKNOWN
        app.status = (
            ApplicationStatus.NEEDS_REVIEW if app.needs_review
            else ApplicationStatus.ACTION_REQUIRED
        )
        app.latest_activity_at = None
        app.latest_activity_label = ""
        app.last_supported_milestone = ""
        app.evidence_message_id = None
        app.stage_note = ""
        return app

    latest = timeline[-1]
    app.latest_activity_at = latest.occurred_at
    app.latest_activity_label = _activity_label(latest)

    # --- stage: forward-only ratchet over non-canceled stage-bearing events ---
    best_rank, best_stage = -1, Stage.OUTREACH
    stage_event: InterviewEvent | None = None
    for event in timeline:
        if event.occurrence == Occurrence.CANCELED:
            continue
        stage = InterviewStageMap.get(event.event_type)
        if stage is None:
            continue
        rank = STAGE_RANK.get(stage, -1)
        if rank > best_rank:
            best_rank, best_stage, stage_event = rank, stage, event

    app.current_stage = best_stage
    app.last_supported_milestone = _milestone_label(stage_event) if stage_event else ""

    # --- stage_state, kept separate from the stage itself ---
    app.stage_state = StageState.UNKNOWN
    app.stage_note = ""
    if stage_event is not None:
        if stage_event.occurrence == Occurrence.COMPLETED:
            app.stage_state = StageState.COMPLETED
        elif stage_event.occurrence == Occurrence.CANCELED:
            app.stage_state = StageState.CANCELED
        elif stage_event.occurrence == Occurrence.SCHEDULED:
            app.stage_state = StageState.SCHEDULED
            # A passed date is NOT evidence the interview happened.
            when = stage_event.event_datetime or stage_event.occurred_at
            if when is not None and when < now:
                app.stage_state = StageState.UNKNOWN
                app.stage_note = "date passed, outcome unknown"

    # --- status: from the latest status-changing event ---
    status: ApplicationStatus | None = None
    evidence_id: str | None = None
    rejected_at: datetime | None = None
    for event in timeline:
        implied = _status_for(event)
        if implied is not None:
            status, evidence_id = implied, event.gmail_id
        if event.event_type in {EventType.REJECTION, EventType.WITHDRAWAL}:
            rejected_at = event.sort_key

    app.status = status or ApplicationStatus.ACTION_REQUIRED
    app.evidence_message_id = evidence_id

    # A message arriving after a rejection does not silently reopen the application.
    if rejected_at is not None:
        later = [
            e for e in timeline
            if e.sort_key > rejected_at
            and e.event_type not in {EventType.REJECTION, EventType.WITHDRAWAL}
        ]
        if later:
            app.needs_review = True
            note = (
                f"{len(later)} message(s) arrived after this application was closed; "
                "flagged for review rather than reopened."
            )
            if note not in app.notes:
                app.notes.append(note)

    if app.needs_review:
        app.status = ApplicationStatus.NEEDS_REVIEW

    # --- incomplete history ---
    if window_start is not None:
        first = timeline[0]
        earliest = first.occurred_at
        if earliest is not None and earliest <= window_start + timedelta(days=2):
            _flag_incomplete(
                app,
                "Earliest known email sits at the edge of the retrieval window; "
                "earlier events may exist outside it.",
            )
        first_stage = InterviewStageMap.get(first.event_type, Stage.OUTREACH)
        if first.event_type not in {
            EventType.RECRUITER_OUTREACH, EventType.OTHER_RECRUITMENT,
        } and STAGE_RANK.get(first_stage, 0) >= STAGE_RANK[Stage.TECHNICAL]:
            _flag_incomplete(
                app,
                "History starts mid-process (the first retrieved email is already a "
                "later round), so earlier events are likely missing.",
            )
    return app


def _flag_incomplete(app: Application, note: str) -> None:
    app.history_possibly_incomplete = True
    if note not in app.notes:
        app.notes.append(note)


_EVENT_LABELS: dict[EventType, str] = {
    EventType.RECRUITER_OUTREACH: "Recruiter outreach",
    EventType.SCREENING_INVITATION: "Screening invitation",
    EventType.SCREENING_CONFIRMATION: "Screening confirmed",
    EventType.TECHNICAL_INTERVIEW: "Technical interview",
    EventType.HIRING_MANAGER_INTERVIEW: "Hiring manager interview",
    EventType.FINAL_INTERVIEW: "Final interview",
    EventType.SCHEDULING: "Scheduling update",
    EventType.ASSESSMENT_ASSIGNED: "Take-home / assessment assigned",
    EventType.ASSESSMENT_SUBMITTED: "Assessment submitted",
    EventType.RECRUITER_FOLLOWUP: "Recruiter follow-up",
    EventType.CANDIDATE_FOLLOWUP: "Candidate follow-up",
    EventType.OFFER: "Offer",
    EventType.REJECTION: "Rejection",
    EventType.WITHDRAWAL: "Withdrawal",
    EventType.OTHER_RECRUITMENT: "Recruitment correspondence",
    EventType.NOT_RECRUITMENT: "Non-recruitment",
}

STATUS_LABELS: dict[ApplicationStatus, str] = {
    ApplicationStatus.ACTION_REQUIRED: "Action required",
    ApplicationStatus.SCHEDULED: "Scheduled",
    ApplicationStatus.AWAITING_RESPONSE: "Awaiting response",
    ApplicationStatus.OFFER: "Offer",
    ApplicationStatus.REJECTED: "Rejected",
    ApplicationStatus.WITHDRAWN: "Withdrawn",
    ApplicationStatus.NEEDS_REVIEW: "Needs review",
}


def event_label(event_type: EventType) -> str:
    return _EVENT_LABELS.get(event_type, event_type.value.replace("_", " ").title())


def status_label(status: ApplicationStatus) -> str:
    return STATUS_LABELS.get(status, status.value.replace("_", " ").title())


def _activity_label(event: InterviewEvent) -> str:
    base = event_label(event.event_type)
    suffix = {
        Occurrence.CANCELED: " (canceled)",
        Occurrence.SCHEDULED: " (scheduled)",
        Occurrence.COMPLETED: " (completed)",
    }.get(event.occurrence, "")
    return f"{base}{suffix}"


def _milestone_label(event: InterviewEvent) -> str:
    """Milestone wording that never overstates what the evidence supports."""
    base = event_label(event.event_type)
    return {
        Occurrence.COMPLETED: f"{base} — completed (explicitly confirmed)",
        Occurrence.SCHEDULED: f"{base} — scheduled (completion not confirmed)",
        Occurrence.CANCELED: f"{base} — canceled",
    }.get(event.occurrence, f"{base} — mentioned")
