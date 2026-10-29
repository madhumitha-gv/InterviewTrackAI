"""One complete fictional interview journey through the production service."""
from datetime import datetime, timezone
from interviewtrack.models import RawMessage, Application, InterviewEvent
from interviewtrack.gmail.client import FixtureTransport
from interviewtrack.fixtures.synthetic import gmail_messages_as_api_payloads
from interviewtrack.enums import EventType, ApplicationStatus, Occurrence, Stage
from interviewtrack.interview_policy import interview_bucket
from interviewtrack.status import resolve


def test_screen_to_followups_to_offer(service):
    email='avery@example.com';uid='journey';service.register_user(uid,email)
    content=[
        ('Application received','Thank you for applying for the AI Engineer role at Cedar Systems. Requisition REQ-7788.'),
        ('Phone screen for AI Engineer at Cedar Systems','We invite you to a screening call for the AI Engineer role at Cedar Systems. Requisition REQ-7788.'),
        ('Technical interview for AI Engineer at Cedar Systems','We would like to schedule your technical interview for the AI Engineer role at Cedar Systems. Requisition REQ-7788.'),
        ('Take-home assignment','Please complete the take-home assignment for the AI Engineer role at Cedar Systems. Requisition REQ-7788.'),
        ('Assessment submission','I have submitted my assessment for the AI Engineer role at Cedar Systems. Requisition REQ-7788.'),
        ('Final interview for AI Engineer at Cedar Systems','We invite you to the final interview for the AI Engineer role at Cedar Systems. Requisition REQ-7788.'),
        ('Following up after our interview','I wanted to follow up after our final interview for the AI Engineer role at Cedar Systems. Requisition REQ-7788.'),
        ('Offer letter from Cedar Systems','We are pleased to extend you an offer for the AI Engineer role at Cedar Systems. Requisition REQ-7788.'),
    ]
    raws=[]
    for i,(subject,body) in enumerate(content):
        sent=i in (4,6)
        raws.append(RawMessage(gmail_id=f'm{i}',thread_id=f't{i}',
            from_addr=email if sent else 'hr@cedarsystems.example',
            to_addrs=['hr@cedarsystems.example'] if sent else [email],
            subject=subject,body_text=body,body_fetched=True,
            labels=['SENT'] if sent else ['INBOX'],date=datetime(2026,10,i+1,tzinfo=timezone.utc)))
    now=datetime(2026,10,10,tzinfo=timezone.utc)
    service.synchronize(uid,email,FixtureTransport(gmail_messages_as_api_payloads(raws[:-1])),now=now)
    before=service.interview_applications(uid,email)
    assert len(before)==1
    assert before[0].current_stage==Stage.FINAL
    assert before[0].timeline[-1].event_type==EventType.CANDIDATE_FOLLOWUP
    assert before[0].timeline[0].gmail_id=='m1' # receipt is not the process start
    service.synchronize(uid,email,FixtureTransport(gmail_messages_as_api_payloads(raws)),now=now)
    apps=service.interview_applications(uid,email)
    assert len(apps)==1
    assert apps[0].status==ApplicationStatus.OFFER
    assert apps[0].current_stage==Stage.OFFER
    assert apps[0].thread_count==7
    assert interview_bucket(apps[0])=='Active'
    assert {e.event_type for e in apps[0].events} >= {EventType.SCREENING_INVITATION,EventType.TECHNICAL_INTERVIEW,EventType.ASSESSMENT_ASSIGNED,EventType.ASSESSMENT_SUBMITTED,EventType.FINAL_INTERVIEW,EventType.CANDIDATE_FOLLOWUP,EventType.OFFER}


def test_review_after_rejection_does_not_reopen_dashboard():
    app=Application(key='closed',employer='Cedar',employer_canonical='cedar',events=[
        InterviewEvent(gmail_id='reject',thread_id='t',event_type=EventType.REJECTION,occurrence=Occurrence.MENTIONED,occurred_at=datetime(2026,10,2,tzinfo=timezone.utc)),
        InterviewEvent(gmail_id='followup',thread_id='t',event_type=EventType.CANDIDATE_FOLLOWUP,occurrence=Occurrence.MENTIONED,occurred_at=datetime(2026,10,3,tzinfo=timezone.utc))])
    resolve(app)
    assert app.needs_review
    assert interview_bucket(app)=='Rejected'
