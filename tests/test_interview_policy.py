from datetime import datetime, timezone
import pytest
from interviewtrack.interview_policy import interview_start_evidence, is_bulk_opportunity, interview_bucket
from interviewtrack.models import RawMessage
from interviewtrack.parsing import parse_message
from interviewtrack.classify import classify_message

@pytest.mark.parametrize('subject',['New job opportunities at Example','Fresh remote US tech jobs for you','Indeed job alert: Data Scientist','New jobs posted from example.jobs','Now Hiring: Data Scientist and other open positions!','Interview tips for your next role'])
def test_ads_excluded(subject):
    body='Our hiring process includes a phone screen and technical interview.'
    assert is_bulk_opportunity(subject,body)
    assert not interview_start_evidence(subject,body)
    raw=RawMessage(gmail_id='ad',thread_id='ad',subject=subject,body_text=body,from_addr='jobs@example.com')
    assert not classify_message(parse_message(raw)).is_recruitment

@pytest.mark.parametrize('subject,body',[
 ('Thank You for Your Example Application','If you are selected, we will schedule a phone screen.'),
 ('Your application is under review','Our process includes an initial interview and final round.'),
 ('Application received','We look forward to reviewing your application.'),
 ('Thanks for applying','If we decide to move forward with your application, we will reach out.'),
 ('Application decision','We have decided not to move forward with your application.'),
 ('Application update','Unfortunately we are not moving forward with your application.')])
def test_not_started(subject,body):
    assert not interview_start_evidence(subject,body)

@pytest.mark.parametrize('subject,body',[
 ('Next steps','We are excited to move forward with your application.'),
 ('Your application','We would like to invite you to a recruiter screening call.'),
 ('Phone screen — Data Scientist','Please select a time.'),
 ('Interview Availability Needed for JR123','Please reply with available times.'),
 ('Thank you for applying','We reviewed your profile. We would like to schedule a screening call.')])
def test_started(subject,body):
    assert interview_start_evidence(subject,body)
    assert not interview_start_evidence(subject,body,from_candidate=True)


def test_projection_preserves_later_steps_and_rejection(service):
    from interviewtrack.models import Application,InterviewEvent
    from interviewtrack.enums import EventType,Occurrence,ApplicationStatus
    uid='policy-user';email='candidate@example.com';service.register_user(uid,email)
    texts=[('receipt','Thank you for applying','We received your application.'),('first','Phone screen — Data Scientist at Example','Please pick a time for your interview.'),('next','Technical interview at Example','We invite you to the next interview.'),('reject','Application update','We are not moving forward with your application.')]
    raws=[RawMessage(gmail_id=i,thread_id=i,subject=sub,body_text=body,from_addr='recruiter@example.com',date=datetime(2026,10,n+1,tzinfo=timezone.utc)) for n,(i,sub,body) in enumerate(texts)]
    service.db.upsert_messages(uid,raws)
    events=[InterviewEvent(gmail_id=m.gmail_id,thread_id=m.thread_id,subject=m.subject,occurred_at=m.date,event_type=EventType.OTHER_RECRUITMENT,occurrence=Occurrence.MENTIONED) for m in raws]
    app=Application(key='tracked',employer='Example',employer_canonical='example',events=events,status=ApplicationStatus.REJECTED)
    unstarted=Application(key='receipt-only',employer='Other',employer_canonical='other',events=events[:1],status=ApplicationStatus.REJECTED)
    service.db.replace_applications(uid,[app,unstarted])
    visible=service.interview_applications(uid,email)
    assert len(visible)==1
    assert [e.gmail_id for e in visible[0].timeline]==['first','next','reject']
    assert interview_bucket(visible[0])=='Rejected'
    assert len(service.applications(uid))==2


def test_employer_corroborated_by_signature_not_ats_domain():
    from interviewtrack.classify import extract_employer
    raw=RawMessage(gmail_id='sig',thread_id='sig',subject='Interview availability',
        from_addr='tenant@myworkday.com',from_name='Cedar Systems Talent Acquisition',
        body_text='We invite you to an interview.\nRegards,\nTaylor\nCedar Systems\nTalent Acquisition Team')
    assert extract_employer(parse_message(raw))[0]=='Cedar Systems'
