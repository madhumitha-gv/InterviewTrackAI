"""Interview-only product policy: explicit invitations start a tracked process.

Rules describe language, never specific people, employers or account addresses.
Job-board delivery alone is not a reason to reject a genuine interview invitation.
"""
from __future__ import annotations
import re


def _text(value):
    return re.sub(r'\s+', ' ', (value or '').replace('’', "'")).strip()

BULK_SUBJECT = re.compile(
    r'new job opportunit|new jobs (?:posted|for you)|fresh .*jobs for you|'
    r'job alerts?|recommended jobs|jobs you may like|jobs matching|weekly digest|'
    r'now hiring:|other open positions|strong matches .*waiting|'
    r"isn.t matching jobs|let hiring companies know|interview (?:tips|advice|guide)|"
    r'how to (?:ace|prepare for).*interview|welcome to (?:our|the) talent community|'
    r'your verification code|verify your candidate account', re.I)
BULK_BODY = re.compile(r'jobs you may like|recommended jobs|view all jobs|manage (?:your )?job alerts|unsubscribe from (?:these |our )?job alerts', re.I)
RECEIPT = re.compile(r'thank you for (?:applying|your .*application)|thanks for applying|application (?:received|is under review|is complete)|we (?:have )?received your application', re.I)
NEGATIVE = re.compile(r'not (?:to |be )?mov(?:e|ing) forward|will not.*(?:interview|proceed)|regret to inform|if (?:you are|you.re) selected|should you be selected|if we (?:decide|choose|elect)|if (?:selected|successful)|may (?:invite|contact)|might (?:invite|contact)', re.I)
# Concrete candidate-directed action, not generic descriptions of a hiring process.
INVITATION = re.compile(
    r"(?:we(?:'re| are| would be)? (?:excited|pleased|delighted|happy) to |we(?:'d| would) like to |we want to )"
    r'(?:move forward|proceed|progress|advance).{0,80}(?:your application|you|interview|next (?:step|stage|round))|'
    r'(?:move|moving) forward with your application|'
    r'(?:invite|inviting|invited) you (?:to|for).{0,70}(?:interview|screen|call|next (?:step|stage|round))|'
    r"(?:we(?:'d| would) like|we want|would love) to interview you|"
    r'(?:schedule|arrange|set up|book).{0,50}(?:interview|screening|phone screen|recruiter call)|'
    r'(?:share|send|provide|confirm).{0,30}availability.{0,60}(?:interview|screen|call)|'
    r'(?:selected|shortlisted).{0,45}(?:interview|screen|next (?:step|round))|'
    r'your (?:interview|phone screen|screening call) (?:is |has been )?(?:confirmed|scheduled)', re.I)
INTERVIEW_SUBJECT = re.compile(
    r'phone screen|recruiter screen|screening (?:call|interview)|first[- ](?:stage|round) interview|'
    r'(?:technical|hiring manager|final|initial) interview|interview (?:availability|invitation|invite|confirmation|details)|'
    r'availability (?:requested|needed|required).{0,60}(?:\bat\b|interview)|'
    r'\binterview\b', re.I)


def is_bulk_opportunity(subject: str, body: str = '') -> bool:
    subject, body = _text(subject), _text(body)
    # Explicit advertising subjects cannot be rescued by sample interview language.
    if BULK_SUBJECT.search(subject):
        return True
    return bool(BULK_BODY.search(body) and not INVITATION.search(subject + ' ' + body))


def interview_start_evidence(subject: str, body: str = '', *, from_candidate=False) -> str | None:
    if from_candidate or is_bulk_opportunity(subject, body):
        return None
    subject, body = _text(subject), _text(body)
    text = subject + ' ' + body
    # Evaluate sentences independently so generic conditional boilerplate cannot
    # manufacture a screening invitation from an application acknowledgement.
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', text):
        if not NEGATIVE.search(sentence):
            match = INVITATION.search(sentence)
            if match:
                return match.group(0)
    if RECEIPT.search(subject) or NEGATIVE.search(text):
        return None
    match = INTERVIEW_SUBJECT.search(subject)
    if match:
        return match.group(0)
    return None


def interview_bucket(app) -> str | None:
    from .enums import ApplicationStatus, EventType
    if app.status == ApplicationStatus.WITHDRAWN:
        return None
    if app.status == ApplicationStatus.REJECTED:
        return 'Rejected'
    # Review is an uncertainty flag, not evidence that a closed process reopened.
    closures = [e for e in app.timeline if e.event_type in {EventType.REJECTION, EventType.WITHDRAWAL}]
    if closures:
        return 'Rejected' if closures[-1].event_type == EventType.REJECTION else None
    return 'Active'
