"""Reviewer journeys exercise the production service and stored CC evidence."""
from interviewtrack.fixtures.mailboxes import REVIEWER_EMAIL, STAFFING_CC
from interviewtrack.interview_policy import interview_bucket
from interviewtrack.enums import EventType, Stage


def test_expanded_reviewer_journeys(service):
    outcome = service.load_demo_data(profile_key="reviewer")
    assert not outcome.errors
    apps = service.interview_applications("demo:reviewer", REVIEWER_EMAIL)
    assert len(apps) == 6
    assert sum(interview_bucket(a) == "Rejected" for a in apps) == 2
    ml = next(a for a in apps if a.employer == "Lumen Robotics" and a.job_title == "Machine Learning Engineer")
    other = next(a for a in apps if a.employer == "Lumen Robotics" and a.job_title == "Backend Engineer")
    assert ml.current_stage == Stage.OFFER
    assert len(ml.events) == 9
    assert ml.key != other.key
    assert {e.event_type for e in ml.events} >= {
        EventType.SCREENING_INVITATION, EventType.TECHNICAL_INTERVIEW,
        EventType.ASSESSMENT_ASSIGNED, EventType.ASSESSMENT_SUBMITTED,
        EventType.FINAL_INTERVIEW, EventType.CANDIDATE_FOLLOWUP, EventType.OFFER}
    assert {"fictional-16", "fictional-17"} <= {e.gmail_id for e in ml.events}
    displayed = {e.gmail_id for a in apps for e in a.events}
    assert displayed.isdisjoint({"fictional-12", "fictional-13", "fictional-30", "fictional-31"})
    messages = {m.gmail_id: m for m in service.db.load_messages("demo:reviewer")}
    assert messages["fictional-14"].cc_addrs == STAFFING_CC
    assert messages["fictional-23"].cc_addrs == STAFFING_CC
    again = service.load_demo_data(profile_key="reviewer")
    assert again.new_messages == 0
