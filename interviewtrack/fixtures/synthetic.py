"""Synthetic labeled email fixtures.

All names, companies and addresses are fictional. The dataset is built to exercise
the eight behaviors required by the spec:

  T1 cross-thread matching       — external recruiter + company HR + scheduling tool
                                   about one role  -> ONE application
  T2 semantic matching           — different subjects/domains, same employer+role
  T3 similar language, different employers -> must NOT merge
  T4 same employer, different roles        -> two independent applications
  T5 follow-up after final interview       -> activity updates, stage does not advance
  T6 ambiguous staffing message            -> REVIEW_REQUIRED, no invented employer
  T7 malformed message                     -> graceful handling
  T8 duplicate sync                        -> no duplicate events/applications
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..models import RawMessage

USER_EMAIL = "avery.candidate@gmail.com"


def _dt(month: int, day: int, hour: int = 9, minute: int = 0) -> datetime:
    return datetime(2025, month, day, hour, minute, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ExpectedApplication:
    """Ground truth for one reconstructed application."""

    label: str
    employer_canonical: str
    role_family: str | None
    gmail_ids: tuple[str, ...]
    min_threads: int = 1
    expect_status: str | None = None
    expect_stage: str | None = None
    expect_needs_review: bool = False


def _msg(
    gid: str,
    tid: str,
    frm: str,
    name: str,
    subject: str,
    body: str,
    date: datetime,
    *,
    to: str = USER_EMAIL,
    cc: tuple[str, ...] = (),
    labels: tuple[str, ...] = ("INBOX",),
    rfc: str | None = None,
    in_reply_to: str | None = None,
    references: tuple[str, ...] = (),
) -> RawMessage:
    return RawMessage(
        gmail_id=gid,
        thread_id=tid,
        rfc_message_id=rfc or f"<{gid}@mail.example>",
        in_reply_to=in_reply_to,
        references=list(references),
        from_addr=frm,
        from_name=name,
        to_addrs=[to],
        cc_addrs=list(cc),
        subject=subject,
        date=date,
        snippet=body.strip().replace("\n", " ")[:120],
        labels=list(labels),
        body_text=body.strip(),
        body_fetched=True,
    )


# =============================================================================
# T1 + T2: Northwind Analytics — AI Engineer. Four unrelated threads, four senders.
# =============================================================================

M01 = _msg(
    "m001", "t101", "sarah.chen@harnham.com", "Sarah Chen",
    "AI Engineer opportunity with a client of ours",
    """Hi Avery,

I came across your profile and wanted to reach out about an AI Engineer role
I am recruiting for. I am a technical recruiter with Harnham, and my client is
Northwind Analytics. The requisition ID is REQ-4417.

They are building an applied LLM team and your background looks like a strong fit.
Would you be open to a quick chat this week?

Best,
Sarah Chen
Harnham
""",
    _dt(10, 3, 10, 15),
)

M02 = _msg(
    "m002", "t101", USER_EMAIL, "Avery",
    "Re: AI Engineer opportunity with a client of ours",
    """Hi Sarah,

Thanks for reaching out - yes, I would be interested in learning more about the
AI Engineer position at Northwind Analytics. I am generally free weekday afternoons.

Best,
Avery
""",
    _dt(10, 3, 16, 40),
    labels=("SENT",),
    in_reply_to="<m001@mail.example>",
    references=("<m001@mail.example>",),
)

# Separate thread, company's own HR domain. No shared Gmail thread with t101.
M03 = _msg(
    "m003", "t102", "priya.raman@northwind.com", "Priya Raman",
    "Introductory call - AI Engineer (REQ-4417)",
    """Hello Avery,

I am Priya Raman from the talent acquisition team at Northwind Analytics. Sarah Chen
passed along your details for the AI Engineer opening, requisition REQ-4417.

I would like to schedule an initial screening call. Are you available on
Wednesday, October 8 at 11:00 AM Eastern?

Warm regards,
Priya Raman
Talent Acquisition, Northwind Analytics
""",
    _dt(10, 6, 8, 5),
)

M04 = _msg(
    "m004", "t102", "priya.raman@northwind.com", "Priya Raman",
    "Re: Introductory call - AI Engineer (REQ-4417)",
    """Great - you are all set. Confirming our screening call for
Wednesday, October 8 at 11:00 AM Eastern. Looking forward to speaking on Wednesday.

Priya
""",
    _dt(10, 7, 9, 30),
    in_reply_to="<m003@mail.example>",
    references=("<m003@mail.example>",),
)

# Third thread: a scheduling platform. Employer appears only in the body.
M05 = _msg(
    "m005", "t103", "no-reply@goodtime.io", "GoodTime Scheduling",
    "Invitation: Technical Interview - Northwind Analytics",
    """You have been invited to a technical interview.

Position: AI Engineer
Company: Northwind Analytics
Interviewers: Daniel Okafor, Staff Engineer
When: Saturday, October 18 at 2:00 PM Eastern

This is a 60 minute technical interview covering ML system design and coding.
Please confirm your attendance using the scheduling link below.
""",
    _dt(10, 13, 12, 0),
)

# T2: semantic case. New thread, personal freemail sender, no requisition ID,
# subject shares no keywords with the other threads. Employer+role are in the text.
M06 = _msg(
    "m006", "t104", "d.okafor.eng@gmail.com", "Daniel Okafor",
    "Looking forward to our session on Saturday",
    """Hi Avery,

I will be one of your interviewers for the applied AI engineering position at
Northwind Analytics this Saturday. I lead the LLM platform group here at Northwind.

We will spend about half the time on a system design discussion and half on
practical coding. Nothing to prepare in advance.

Daniel
""",
    _dt(10, 15, 17, 20),
)

M07 = _msg(
    "m007", "t103", "no-reply@goodtime.io", "GoodTime Scheduling",
    "Rescheduled: Technical Interview - Northwind Analytics",
    """Your technical interview has been moved at the interviewer's request.

Position: AI Engineer
Company: Northwind Analytics
New time: Tuesday, October 21 at 3:00 PM Eastern

Please pick a time using the scheduling link if this does not work for you.
""",
    _dt(10, 17, 11, 10),
    in_reply_to="<m005@mail.example>",
    references=("<m005@mail.example>",),
)

# T5: candidate follow-up after the technical interview. Must update latest activity
# WITHOUT inventing a new interview round or marking a later stage.
M08 = _msg(
    "m008", "t102", USER_EMAIL, "Avery",
    "Thank you for your time",
    """Hi Priya,

Thank you so much for your time yesterday. It was great speaking with Daniel about
the LLM platform work. I wanted to follow up on our conversation and reiterate how
excited I am about the AI Engineer role.

Please let me know if there is anything else you need from me.

Best,
Avery
""",
    _dt(10, 22, 14, 0),
    labels=("SENT",),
    in_reply_to="<m004@mail.example>",
    references=("<m003@mail.example>", "<m004@mail.example>"),
)

# =============================================================================
# T4: Northwind Analytics — Data Analyst (DIFFERENT role, same employer).
# A rejection here must NOT close the AI Engineer application.
# =============================================================================

M09 = _msg(
    "m009", "t201", "careers@northwind.com", "Northwind Careers",
    "Your application for Data Analyst at Northwind Analytics",
    """Dear Avery,

Thank you for applying to the Data Analyst position at Northwind Analytics
(requisition REQ-5120). We have received your application and the hiring team
is reviewing it.

Northwind Careers Team
""",
    _dt(10, 5, 13, 0),
)

M10 = _msg(
    "m010", "t201", "careers@northwind.com", "Northwind Careers",
    "Update on your Data Analyst application",
    """Dear Avery,

Thank you for your interest in the Data Analyst role at Northwind Analytics
(requisition REQ-5120).

Unfortunately, we have decided not to move forward with your application at this
time. We moved forward with other candidates whose experience more closely matched
the requirements.

We wish you the best in your search.

Northwind Careers Team
""",
    _dt(10, 20, 10, 0),
    in_reply_to="<m009@mail.example>",
    references=("<m009@mail.example>",),
)

# =============================================================================
# T3: Different employer, near-identical language about a Data Scientist interview.
# Must NOT merge with each other or with Northwind.
# =============================================================================

M11 = _msg(
    "m011", "t301", "recruiting@globexlabs.com", "Amara Diallo",
    "Technical interview for Data Scientist at Globex Labs",
    """Hi Avery,

We would like to schedule your technical interview for the Data Scientist position
at Globex Labs. The session will cover statistics, machine learning fundamentals
and a live coding exercise.

Are you available on Thursday, October 16 at 1:00 PM?

Amara Diallo
Globex Labs Recruiting
""",
    _dt(10, 9, 9, 45),
)

M12 = _msg(
    "m012", "t302", "talent@initechdata.com", "Marco Rossi",
    "Technical interview for Data Scientist at Initech Data",
    """Hi Avery,

We would like to schedule your technical interview for the Data Scientist position
at Initech Data. The session will cover statistics, machine learning fundamentals
and a live coding exercise.

Are you available on Friday, October 17 at 1:00 PM?

Marco Rossi
Initech Data Talent
""",
    _dt(10, 9, 10, 5),
)

# =============================================================================
# Assessment flow — Globex Labs, Data Scientist (continues t301 semantically,
# arrives via an ATS domain on a brand new thread).
# =============================================================================

M13 = _msg(
    "m013", "t303", "no-reply@us.greenhouse-mail.io", "Globex Labs via Greenhouse",
    "Your take-home assignment for the Data Scientist role",
    """Hello Avery,

As the next step in your interview process for the Data Scientist role at
Globex Labs, please complete the take-home assignment linked below.

The coding challenge should take about three hours and is due within five days.

Globex Labs Recruiting
""",
    _dt(10, 18, 15, 30),
)

M14 = _msg(
    "m014", "t303", USER_EMAIL, "Avery",
    "Re: Your take-home assignment for the Data Scientist role",
    """Hi,

I have just submitted my completed take-home assignment for the Data Scientist
role at Globex Labs. Here is my write-up and the accompanying notebook.

Thanks,
Avery
""",
    _dt(10, 21, 20, 15),
    labels=("SENT",),
    in_reply_to="<m013@mail.example>",
    references=("<m013@mail.example>",),
)

# =============================================================================
# Offer flow — Initech Data, Data Scientist: final interview then offer.
# =============================================================================

M15 = _msg(
    "m015", "t302", "talent@initechdata.com", "Marco Rossi",
    "Final round onsite loop at Initech Data",
    """Hi Avery,

Congratulations on advancing! We would like to invite you to the final round
onsite interview loop for the Data Scientist position at Initech Data.

The loop is scheduled for Monday, October 27 and includes a meeting with the
hiring manager and two technical sessions.

Marco
""",
    _dt(10, 20, 11, 0),
    in_reply_to="<m012@mail.example>",
    references=("<m012@mail.example>",),
)

M16 = _msg(
    "m016", "t304", "hr@initechdata.com", "Initech Data People Team",
    "Your offer from Initech Data",
    """Dear Avery,

We are thrilled to extend an offer for the Data Scientist position at Initech Data.
Your offer letter is attached and includes the full compensation package.

Please let us know if you have any questions.

Initech Data People Team
""",
    _dt(10, 24, 16, 45),
)

# =============================================================================
# T6: Ambiguous staffing-agency message. Employer is deliberately unidentifiable.
# Expected: REVIEW_REQUIRED, no invented employer, no silent association.
# =============================================================================

M17 = _msg(
    "m017", "t401", "recruiter@talentsphere.io", "Jordan Blake",
    "Confidential opportunity - great fit for your background",
    """Hi Avery,

I am working on a confidential search for a well-funded client in the analytics
space. The role is a strong match for your background and the compensation is
competitive.

I am not able to share the company name until we have had a brief conversation.
Would you be open to a 15 minute call this week?

Jordan Blake
TalentSphere
""",
    _dt(10, 11, 14, 25),
)

# =============================================================================
# T7: Malformed messages. Must degrade gracefully, never crash the pipeline.
# =============================================================================

M18 = RawMessage(
    gmail_id="m018",
    thread_id="t501",
    rfc_message_id=None,
    from_addr="not-an-email-address",          # invalid sender
    from_name="",
    to_addrs=[],
    subject="",                                 # no subject
    date=None,                                  # no date
    snippet="",
    labels=["INBOX"],
    body_text="",                               # no body
    body_fetched=False,
)

M19 = _msg(
    "m019", "t502", "hr@acmerobotics.com", "Acme Robotics HR",
    "Interview details",
    # Broken HTML only, no plain-text alternative.
    "",
    _dt(10, 12, 9, 0),
)
M19 = M19.model_copy(
    update={
        "body_text": "",
        "body_html": (
            "<html><body><div><p>Hi Avery<br>Your <b>technical interview<i> for the "
            "Robotics Software Engineer role at Acme Robotics is scheduled for "
            "Thursday, October 23 at 10:00 AM.</p><div><span>Unclosed tags everywhere"
            "<script>var x = 'drop me';</script><style>.a{color:red}</style>"
        ),
        "body_fetched": True,
    }
)

# =============================================================================
# Noise: must be filtered out as non-recruitment.
# =============================================================================

M20 = _msg(
    "m020", "t601", "jobs-noreply@linkedin.com", "LinkedIn Job Alerts",
    "Jobs you may like: 12 new Data Scientist openings",
    """Jobs you may like this week

Based on your profile, here are recommended jobs near you:
- Data Scientist at Company A
- Machine Learning Engineer at Company B

This is your weekly job alert digest. Unsubscribe at any time.
""",
    _dt(10, 14, 6, 0),
    labels=("INBOX", "CATEGORY_PROMOTIONS"),
)

M21 = _msg(
    "m021", "t602", "billing@cloudhost.example", "CloudHost Billing",
    "Your October invoice is ready",
    """Your invoice for October is now available. Amount due: $42.00.
Thank you for being a CloudHost customer.
""",
    _dt(10, 2, 7, 0),
)

# A recruiter follow-up on the ambiguous staffing thread (stays ambiguous).
M22 = _msg(
    "m022", "t401", "recruiter@talentsphere.io", "Jordan Blake",
    "Re: Confidential opportunity - great fit for your background",
    """Hi Avery,

Just following up on my note below. Happy to share more details on a quick call.

Jordan
""",
    _dt(10, 16, 10, 30),
    in_reply_to="<m017@mail.example>",
    references=("<m017@mail.example>",),
)


ALL_MESSAGES: list[RawMessage] = [
    M01, M02, M03, M04, M05, M06, M07, M08, M09, M10, M11,
    M12, M13, M14, M15, M16, M17, M18, M19, M20, M21, M22,
]


#: Ground truth used by tests and scripts/evaluate.py.
EXPECTED: list[ExpectedApplication] = [
    ExpectedApplication(
        label="T1+T2: Northwind Analytics / AI Engineer (4 threads)",
        employer_canonical="northwind analytics",
        role_family="ai_engineer",
        gmail_ids=("m001", "m002", "m003", "m004", "m005", "m006", "m007", "m008"),
        min_threads=4,
        expect_status="scheduled",
        expect_stage="technical_interview",
    ),
    ExpectedApplication(
        label="T4: Northwind Analytics / Data Analyst (rejected, independent)",
        employer_canonical="northwind analytics",
        role_family="data_analyst",
        gmail_ids=("m009", "m010"),
        min_threads=1,
        expect_status="rejected",
    ),
    ExpectedApplication(
        label="T3: Globex Labs / Data Scientist (+ assessment)",
        employer_canonical="globex labs",
        role_family="data_scientist",
        gmail_ids=("m011", "m013", "m014"),
        min_threads=2,
        expect_status="awaiting_response",
    ),
    ExpectedApplication(
        label="T3: Initech Data / Data Scientist (offer)",
        employer_canonical="initech data",
        role_family="data_scientist",
        gmail_ids=("m012", "m015", "m016"),
        min_threads=2,
        expect_status="offer",
    ),
    ExpectedApplication(
        label="T6: Unidentified employer via staffing agency",
        employer_canonical="",
        role_family=None,
        gmail_ids=("m017", "m022"),
        min_threads=1,
        expect_needs_review=True,
    ),
]

#: Messages that must be filtered out as non-recruitment.
EXPECTED_NON_RECRUITMENT = {"m020", "m021"}

#: Messages that must be handled without crashing (malformed input).
EXPECTED_MALFORMED = {"m018", "m019"}

#: Pairs that must NEVER end up in the same application.
MUST_NOT_MERGE: list[tuple[str, str, str]] = [
    ("m011", "m012", "T3: same role wording, different employers (Globex vs Initech)"),
    ("m001", "m009", "T4: same employer, different roles (AI Engineer vs Data Analyst)"),
    ("m005", "m011", "Different employers entirely (Northwind vs Globex)"),
    ("m017", "m001", "T6: ambiguous staffing message must not attach to a real application"),
]

#: Pairs that MUST end up in the same application (cross-thread reconstruction).
MUST_MERGE: list[tuple[str, str, str]] = [
    ("m001", "m003", "T1: external recruiter thread + company HR thread"),
    ("m003", "m005", "T1: company HR thread + scheduling platform thread"),
    ("m005", "m006", "T2: scheduling thread + interviewer's personal-domain email"),
    ("m001", "m008", "T1/T5: recruiter outreach + candidate follow-up"),
    ("m011", "m013", "Globex: recruiter thread + ATS assessment thread"),
    ("m012", "m016", "Initech: recruiter thread + offer from HR domain"),
]


def gmail_messages_as_api_payloads(messages: list[RawMessage] | None = None) -> list[dict]:
    """Render fixtures as Gmail API `users.messages.get` payloads.

    Exercises the real parsing path (`raw_from_gmail_payload`) rather than bypassing
    it. Defaults to the full dataset (dev + holdout) when it is available, so new
    fixtures are picked up automatically.
    """
    import base64

    if messages is None:
        messages = globals().get("FULL_DATASET") or ALL_MESSAGES

    payloads = []
    for m in messages:
        headers = [
            {"name": "From", "value": f"{m.from_name} <{m.from_addr}>" if m.from_name else m.from_addr},
            {"name": "To", "value": ", ".join(m.to_addrs)},
            {"name": "Subject", "value": m.subject},
        ]
        if m.cc_addrs:
            headers.append({"name": "Cc", "value": ", ".join(m.cc_addrs)})
        if m.date:
            headers.append({"name": "Date", "value": m.date.strftime("%a, %d %b %Y %H:%M:%S %z")})
        if m.rfc_message_id:
            headers.append({"name": "Message-ID", "value": m.rfc_message_id})
        if m.in_reply_to:
            headers.append({"name": "In-Reply-To", "value": m.in_reply_to})
        if m.references:
            headers.append({"name": "References", "value": " ".join(m.references)})

        parts = []
        if m.body_text:
            parts.append({
                "mimeType": "text/plain",
                "body": {"data": base64.urlsafe_b64encode(m.body_text.encode()).decode()},
            })
        if m.body_html:
            parts.append({
                "mimeType": "text/html",
                "body": {"data": base64.urlsafe_b64encode(m.body_html.encode()).decode()},
            })

        payloads.append({
            "id": m.gmail_id,
            "threadId": m.thread_id,
            "snippet": m.snippet,
            "labelIds": m.labels,
            "internalDate": str(int(m.date.timestamp() * 1000)) if m.date else None,
            "payload": {"mimeType": "multipart/alternative", "headers": headers, "parts": parts},
        })
    return payloads


# =============================================================================
# HARD PROBE SET — paraphrase-heavy messages used by scripts/evaluate.py.
#
# These deliberately defeat deterministic extraction: the employer is referred to
# obliquely or not at all, and the role is paraphrased rather than named. They are
# the cases where semantic similarity is supposed to earn its place. They are NOT
# part of ALL_MESSAGES, so they do not affect the main reconstruction tests.
# =============================================================================

H01 = _msg(
    "h001", "t701", "ops@northwind.com", "Northwind People Ops",
    "Logistics for your upcoming session",
    """Hi Avery,

A few practical notes ahead of your conversation with the applied LLM platform group
next week. You will meet two engineers from the team you would be joining.

Parking is available on level 2. Let us know if you need anything.

People Ops
""",
    _dt(10, 16, 9, 0),
)

H02 = _msg(
    "h002", "t702", "k.mensah@contractor-mail.example", "Kofi Mensah",
    "Quick note before Tuesday",
    """Avery,

Looking forward to meeting you Tuesday about the generative AI engineering position
we discussed. I work on the same platform team at Northwind that Daniel mentioned.

Kofi
""",
    _dt(10, 19, 13, 0),
)

H03 = _msg(
    "h003", "t703", "noreply@calendly.com", "Calendly",
    "Confirmed: 45 minutes with the analytics reporting group",
    """Your meeting is confirmed.

Invitee: Avery
Host: Globex Labs
Topic: follow-up discussion about the statistics and modelling role

Tuesday, October 28 at 11:00 AM
""",
    _dt(10, 25, 8, 0),
)

#: Hard probe messages plus the gold application each one belongs to, expressed as a
#: representative message id already present in ALL_MESSAGES.
HARD_PROBES: list[tuple[RawMessage, str, str]] = [
    (H01, "m005", "employer by domain only; role paraphrased as 'applied LLM platform group'"),
    (H02, "m005", "personal domain; role paraphrased as 'generative AI engineering position'"),
    (H03, "m011", "scheduling tool; role paraphrased as 'statistics and modelling role'"),
]


# =============================================================================
# v3 SCENARIO FIXTURES
# Added for the spec-v3 test list. All fictional.
# =============================================================================

# --- v3 #2: recruiter handoff, agency -> internal, mid-process -----------------
# The internal recruiter is a brand-new sender on a brand-new thread. What links
# them is the shared participant set (the agency recruiter is CC'd) plus employer
# and role. Without contact learning this looks like a new application.

H_HANDOFF = _msg(
    "v201", "t801", "marcus.webb@sterlingsearch.com", "Marcus Webb",
    "Vantage Systems - Backend Engineer opportunity",
    """Hi Avery,

I am a recruiter with Sterling Search. My client is Vantage Systems and they are
hiring a Backend Engineer. Requisition VS-7781.

Would you be open to a short call?

Marcus Webb
Sterling Search
""",
    _dt(10, 6, 9, 30),
)

H_HANDOFF_2 = _msg(
    "v202", "t802", "nina.patel@vantagesystems.com", "Nina Patel",
    "Taking over your Backend Engineer process",
    """Hi Avery,

Marcus has handed your candidacy over to me - I am on the internal talent
acquisition team at Vantage Systems and will run the rest of your Backend Engineer
process from here (requisition VS-7781).

Nina Patel
Talent Acquisition, Vantage Systems
""",
    _dt(10, 13, 11, 0),
    cc=("marcus.webb@sterlingsearch.com",),
)

# --- v3 #3: a coordinator, not the recruiter, sends the interview invite -------
H_COORD = _msg(
    "v203", "t803", "jess.liu@vantagesystems.com", "Jess Liu",
    "Scheduling your next conversation",
    """Hello Avery,

I am the recruiting coordinator at Vantage Systems. Nina asked me to set up your
technical interview for the Backend Engineer role.

Are you free on Thursday, October 23 at 1:00 PM?

Jess Liu
""",
    _dt(10, 17, 15, 20),
)

# --- v3 #4: same recruiter, two roles at one company ---------------------------
# Nina also runs a Platform Engineer opening. Different role, different requisition,
# so this must NOT merge into the Backend Engineer application.
H_SAME_REC_ROLE2 = _msg(
    "v204", "t804", "nina.patel@vantagesystems.com", "Nina Patel",
    "Also - Platform Engineer opening at Vantage Systems",
    """Hi Avery,

Separately from the Backend Engineer process, we have just opened a Platform
Engineer position at Vantage Systems (requisition VS-9042). Different team and a
different hiring manager.

Would you like to be considered for that one as well?

Nina
""",
    _dt(10, 20, 10, 0),
)

# A role-less email from that same recruiter: ambiguous between her two roles.
H_SAME_REC_ROLELESS = _msg(
    "v205", "t805", "nina.patel@vantagesystems.com", "Nina Patel",
    "Quick question",
    """Hi Avery,

Do you have an updated resume you could send over? The team asked for the most
recent version.

Nina
""",
    _dt(10, 22, 9, 15),
)

# --- v3 #10: ONE Gmail thread that genuinely spans TWO roles -------------------
# The thread is shared, but the second message is explicitly about a different
# role with a conflicting requisition. A native thread must NOT force a merge.
H_THREAD_ROLE_A = _msg(
    "v206", "t806", "dana.okoro@meridianlabs.com", "Dana Okoro",
    "Meridian Labs - Research Engineer",
    """Hi Avery,

We would like to move forward with your application for the Research Engineer role
at Meridian Labs, requisition ML-3310. I will set up a technical interview.

Dana Okoro
Talent Acquisition, Meridian Labs
""",
    _dt(10, 9, 10, 0),
)

H_THREAD_ROLE_B = _msg(
    "v207", "t806", "dana.okoro@meridianlabs.com", "Dana Okoro",
    "Re: Meridian Labs - Research Engineer",
    """Hi Avery,

Changing topic on this thread - the Research Engineer headcount was frozen.

Instead I would like to put you forward for our Data Engineer position at Meridian
Labs, which is a different team under requisition ML-8822. It is a separate process
with its own interview loop.

Dana
""",
    _dt(10, 15, 14, 30),
    in_reply_to="<v206@mail.example>",
    references=("<v206@mail.example>",),
)

# --- v3 #11-adjacent: marketing blast from an EMPLOYER's own domain ------------
H_MARKETING = _msg(
    "v208", "t807", "news@vantagesystems.com", "Vantage Systems",
    "Vantage Systems Q4 product newsletter",
    """Read about our latest platform release, customer stories and upcoming webinars.

We are always hiring - browse open roles on our careers page.

You are receiving this because you subscribed. Unsubscribe at any time.
""",
    _dt(10, 11, 7, 0),
    labels=("INBOX", "CATEGORY_PROMOTIONS"),
)

#: Held-out evaluation set. Thresholds are NEVER tuned against these.
HOLDOUT_MESSAGES: list[RawMessage] = [
    H_HANDOFF, H_HANDOFF_2, H_COORD, H_SAME_REC_ROLE2,
    H_SAME_REC_ROLELESS, H_THREAD_ROLE_A, H_THREAD_ROLE_B, H_MARKETING,
]

#: Development set — the original labelled fixtures. Tuning is allowed here only.
DEV_MESSAGES: list[RawMessage] = list(ALL_MESSAGES)

#: Everything, for end-to-end runs.
FULL_DATASET: list[RawMessage] = DEV_MESSAGES + HOLDOUT_MESSAGES

#: Holdout ground truth: (gmail_id, expected grouping label or None if review/excluded)
HOLDOUT_EXPECTED: dict[str, str | None] = {
    "v201": "vantage_backend",
    "v202": "vantage_backend",      # handoff: same application as the agency email
    "v203": "vantage_backend",      # coordinator joins the right application
    "v204": "vantage_platform",     # different role -> separate application
    "v205": None,                   # role-less, ambiguous between two -> review
    "v206": "meridian_research",
    "v207": "meridian_data",        # same thread, different role -> must NOT merge
    "v208": None,                   # marketing -> excluded entirely
}

#: Pairs within the holdout that must never share an application.
HOLDOUT_MUST_NOT_MERGE: list[tuple[str, str, str]] = [
    ("v201", "v204", "v3 #4: same recruiter, two different roles at one company"),
    ("v206", "v207", "v3 #10: one Gmail thread spanning two roles must not merge"),
]

#: Pairs within the holdout that must share an application.
HOLDOUT_MUST_MERGE: list[tuple[str, str, str]] = [
    ("v201", "v202", "v3 #2: agency -> internal recruiter handoff keeps one process"),
    ("v202", "v203", "v3 #3: coordinator's invite joins the right application"),
]


# =============================================================================
# The DISCRIMINATING case for v3 #10.
#
# H_THREAD_ROLE_A/B carry conflicting requisition IDs, so even the older
# narrower check caught them — that pair does not prove the v3 change works.
# This pair shares a thread, names two different roles, and has NO requisition
# IDs anywhere, so ONLY a role-family conflict can prevent the merge.
# =============================================================================

H_THREAD_NOREQ_A = _msg(
    "v209", "t808", "owen.fitz@summitpoint.com", "Owen Fitzgerald",
    "Summit Point - Research Engineer",
    """Hi Avery,

We would like to interview you for the Research Engineer role at Summit Point.
I will set up a technical conversation with the team this week.

Owen Fitzgerald
Talent Acquisition, Summit Point
""",
    _dt(10, 10, 9, 0),
)

H_THREAD_NOREQ_B = _msg(
    "v210", "t808", "owen.fitz@summitpoint.com", "Owen Fitzgerald",
    "Re: Summit Point - Research Engineer",
    """Hi Avery,

Changing topic on this thread - the Research Engineer headcount was frozen.

Instead I would like to put you forward for our Data Analyst position at Summit
Point. It is a different team with its own interview loop.

Owen
""",
    _dt(10, 18, 14, 0),
    in_reply_to="<v209@mail.example>",
    references=("<v209@mail.example>",),
)

HOLDOUT_MESSAGES.extend([H_THREAD_NOREQ_A, H_THREAD_NOREQ_B])
FULL_DATASET.extend([H_THREAD_NOREQ_A, H_THREAD_NOREQ_B])
HOLDOUT_EXPECTED.update({"v209": "summit_research", "v210": "summit_analyst"})
HOLDOUT_MUST_NOT_MERGE.append(
    ("v209", "v210", "v3 #10 (discriminating): one thread, two roles, NO requisition "
                     "IDs — only a role conflict can prevent the merge")
)
