"""Fixtures modelled on real recruiting mail patterns.

**Entirely fictional.** These are structural analogues of real inbox patterns, not
real messages: every name, company, address, requisition and date is invented. No real
email content is stored in this repository.

The patterns they reproduce, which the original synthetic set did not cover:

  P1  A vendor sends from a SUBDOMAIN of the employer (an external-workforce host
      like ``xwv.employer.com``). The sender is agency staff; the employer is still
      the employer.
  P2  A recruiter hands off to a second recruiter by CC-ing them on the introduction.
  P3  A THIRD person (a scheduling coordinator) joins the same process later.
  P4  The same human writes first through an ATS relay, then from their corporate
      address — one person, two addresses.
  P5  Role title drifts inside one process ("software engineering roles" ->
      "Software Engineer" -> "Software Engineer III").
  P6  Enormous legal/marketing footers and inline quoted reply headers.
  P7  A reschedule conversation spanning several threads.
  P8  A candidate reply that names the role when no recruiter message did.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..models import RawMessage

USER_EMAIL = "candidate.example@gmail.com"

#: Boilerplate that must be stripped before extraction or embedding. Modelled on the
#: accessibility/confidentiality blocks real recruiting mail carries.
_FOOTER = """

--
Pinnacle Sourceright, Supporting Northvale Systems
careers.northvale.example

If you require interview accommodations due to a disability, medical condition or
neurodivergence, please email accommodations@northvale.example for a confidential
consultation. Information shared will only be viewed by members of the Accommodations
team, and will not be used or referenced in any hiring decisions.

If you received this communication by mistake, please don't forward it to anyone else
(it may contain confidential or privileged information), please erase all copies of it,
including all attachments, and please let the sender know it went to the wrong person.
"""


def _dt(month: int, day: int, hour: int = 9, minute: int = 0) -> datetime:
    return datetime(2025, month, day, hour, minute, tzinfo=timezone.utc)


def _msg(gid, tid, frm, name, subject, body, date, *, cc=(), labels=("INBOX",),
         in_reply_to=None, references=(), to=None):
    # Mail the candidate SENDS is addressed to the recruiter, not to themselves —
    # so the counterparty is in To:, which is how a reply links to its process.
    recipients = [to] if to else ([USER_EMAIL] if frm != USER_EMAIL else [])
    return RawMessage(
        gmail_id=gid, thread_id=tid, rfc_message_id=f"<{gid}@mail.example>",
        in_reply_to=in_reply_to, references=list(references),
        from_addr=frm, from_name=name, to_addrs=recipients, cc_addrs=list(cc),
        subject=subject, date=date,
        snippet=body.strip().replace("\n", " ")[:120],
        labels=list(labels), body_text=body.strip(), body_fetched=True,
    )


# =============================================================================
# PROCESS 1 — Northvale Systems / Software Engineer III
# Vendor subdomain senders, a two-step recruiter handoff, a coordinator, a
# reschedule across threads, and role-title drift.
# =============================================================================

R01 = _msg(
    "r001", "tr01", "reyesa@xwv.northvale.com", "Alina Reyes (xWV)",
    "Northvale Systems - engineering opportunities",
    """Hi,

I'm Alina from a specialized recruiting team supporting Northvale Systems
Engineering. I've reviewed your application for our software engineering roles and
I'm excited to move forward.

To continue in the hiring process, please complete the short questionnaire linked
below.

Best,
Alina
""" + _FOOTER,
    _dt(8, 20, 10, 23),
)

R02 = _msg(
    "r002", "tr01", USER_EMAIL, "Candidate Example",
    "Re: Northvale Systems - engineering opportunities",
    """Hi Alina,

Thank you for reaching out and for considering my application. I have completed the
questionnaire as requested and I'm excited to move forward.

Best,
Candidate
""",
    _dt(8, 20, 22, 38), labels=("SENT",), to="reyesa@xwv.northvale.com",
    in_reply_to="<r001@mail.example>", references=("<r001@mail.example>",),
)

# P5: the role is named precisely for the first time here.
R03 = _msg(
    "r003", "tr02", "reyesa@xwv.northvale.com", "Alina Reyes (xWV)",
    "Moving forward - Software Engineer III at Northvale Systems",
    """Hi,

I'm thrilled to tell you that we'd like to move forward and interview you for our
Software Engineer III role at Northvale Systems.

Round 1 consists of two 45-minute interviews: one programming and algorithms
interview, and one non-technical behavioural interview.

Please reply to confirm you are ready to begin and let me know which coding language
you'd like to use. Our scheduling team will then reach out to collect your
availability.

Thanks,
Alina
""" + _FOOTER,
    _dt(8, 21, 9, 47),
)

# P1 + P3: a coordinator from the same vendor subdomain schedules the rounds.
R04 = _msg(
    "r004", "tr03", "pnandan@xwv.northvale.com", "Priya Nandan (xWV)",
    "Your upcoming Northvale Systems interviews",
    """Hi,

I'm Priya, a Recruiting Coordinator working with Alina to schedule your interviews
for the Software Engineer III role.

1st Interview: Friday, September 5, 2025 at 2:00pm - 2:45pm
2nd Interview: Friday, September 5, 2025 at 4:00pm - 4:45pm

Please reply to confirm you can make it.

Regards,
Priya
""" + _FOOTER,
    _dt(8, 26, 15, 0),
)

# P7: reschedule request, on a NEW thread, naming the role only in passing.
R05 = _msg(
    "r005", "tr04", USER_EMAIL, "Candidate Example",
    "Request to reschedule",
    """Hi Priya,

Thank you again for coordinating my interviews for the Software Engineer III
opportunity.

I wanted to ask whether it would be possible to reschedule my upcoming interviews to
a later date. I remain very excited about the opportunity and appreciate your
flexibility.

Best,
Candidate
""",
    _dt(9, 2, 13, 33), labels=("SENT",), to="pnandan@xwv.northvale.com",
)

R06 = _msg(
    "r006", "tr04", "pnandan@xwv.northvale.com", "Priya Nandan (xWV)",
    "Re: Request to reschedule",
    """Hi,

Thank you for letting us know. We can certainly assist with rescheduling your
interviews. Please provide a few alternative dates that work for you over the next
two weeks and we will update your schedule accordingly.

Regards,
Priya
""" + _FOOTER,
    _dt(9, 2, 20, 49),
    in_reply_to="<r005@mail.example>", references=("<r005@mail.example>",),
)

R07 = _msg(
    "r007", "tr05", "pnandan@xwv.northvale.com", "Priya Nandan (xWV)",
    "Rescheduled interview details - Software Engineer III",
    """Hi,

As discussed, please find the rescheduled interview details below and kindly confirm
by replying to this email.

1st Interview: Wednesday, September 17, 2025 at 2:30pm - 3:15pm
2nd Interview: Wednesday, September 17, 2025 at 4:00pm - 4:45pm

All timings are listed in Eastern time.

Regards,
Priya
""" + _FOOTER,
    _dt(9, 3, 21, 56),
)

# Explicit completion + progression.
R08 = _msg(
    "r008", "tr02", "reyesa@xwv.northvale.com", "Alina Reyes (xWV)",
    "Re: Moving forward - Software Engineer III at Northvale Systems",
    """Hi,

We're thrilled to share that you've successfully passed your first round of
interviews. Congratulations!

The team was very impressed. We're now connecting you with a dedicated recruiting
point of contact who will guide you through the next stages.

Best wishes,
Alina
""" + _FOOTER,
    _dt(9, 21, 9, 51),
    in_reply_to="<r003@mail.example>", references=("<r003@mail.example>",),
)

# P2: the handoff. A NEW recruiter is introduced, CC'd on the message.
R09 = _msg(
    "r009", "tr06", "reyesa@xwv.northvale.com", "Alina Reyes (xWV)",
    "Introduction to your new point of contact",
    """Hi,

I would like to introduce you to Kiran (cc'd here) who will be your main point of
contact from here on out for the Software Engineer III process.

Kiran specialises in understanding Northvale's hiring needs and will speak with you
about your background and aspirations.

Sincerely,
Alina
""" + _FOOTER,
    _dt(9, 21, 9, 52), cc=("kmehta@xwv.northvale.com",),
)

# The handed-off recruiter writes from a cold thread of their own.
R10 = _msg(
    "r010", "tr07", "kmehta@xwv.northvale.com", "Kiran Mehta (xWV)",
    "Next steps at Northvale Systems",
    """Hello,

I am Staffing Specialist Kiran. I will be your point of contact moving forward for
the Software Engineer III role. Our next step is to discuss the interview process.

Please select a 30-minute window through my scheduling link so we can connect. If
none of the available slots work for you, send over your availability and I will
schedule a time manually.

Best,
Kiran
""" + _FOOTER,
    _dt(10, 14, 7, 0), cc=("reyesa@xwv.northvale.com",),
)


# =============================================================================
# PROCESS 2 — Brightline Energy / Associate Data Scientist
# P4: the same recruiter writes first through an ATS relay, then from their own
# corporate address. One person, two addresses, one process.
# =============================================================================

R11 = _msg(
    "r011", "tr08", "brightline@myworkday.com", "Brightline Energy Talent Acquisition",
    "Interview Availability Needed for JR229183 - Associate Data Scientist",
    """Dear Candidate,

We are excited to let you know we will be moving forward with your application for
the Associate Data Scientist role. We'd like to schedule a one-hour virtual
interview with you this week. Our current availability is:

Thursday: 2:30-3:30 PM MT
Friday: Any time after 10:15 AM MT

Interview times are scheduled on a first-come, first-served basis, so please let me
know what time works best for you.

Thanks,
Dale

Regards,
Dale Renner
Brightline Energy Talent Acquisition Team
""",
    _dt(10, 26, 12, 57),
)

R12 = _msg(
    "r012", "tr08", USER_EMAIL, "Candidate Example",
    "Re: Interview Availability Needed for JR229183 - Associate Data Scientist",
    """Hi Dale,

Friday afternoon works well for me. Thank you for coordinating the interview.

I'm very excited about the opportunity to interview for the Associate Data Scientist
role at Brightline Energy and look forward to speaking with the team.

Best,
Candidate
""",
    _dt(10, 26, 18, 43), labels=("SENT",), to="brightline@myworkday.com",
    in_reply_to="<r011@mail.example>", references=("<r011@mail.example>",),
)

# P4 + P6: same human, corporate address now, with an inline quoted header block.
R13 = _msg(
    "r013", "tr09", "dale.renner@brightlineenergy.com", "Renner, Dale",
    "RE: Interview Availability Needed for JR229183 - Associate Data Scientist",
    """Thank you, I just sent the invite!

Dale Renner
Brightline Energy
Recruiter, Talent Acquisition
dale.renner@brightlineenergy.com

From: Candidate Example <candidate.example@gmail.com>
Sent: Monday, October 26, 2025 4:43 PM
To: Renner, Dale <Dale.Renner@brightlineenergy.com>
Subject: Re: Interview Availability Needed for JR229183 - Associate Data Scientist

Hi Dale, Friday afternoon works well for me...
""",
    _dt(10, 27, 12, 0),
)


# =============================================================================
# PROCESS 3 — Calloway AI / AI Engineer (Tax)
# A take-home case study, and a candidate follow-up that asks a question rather
# than thanking anyone — it must not advance the stage.
# =============================================================================

R14 = _msg(
    "r014", "tr10", "talent@callowayai.com", "Camille Serrano",
    "Calloway AI | AI Engineer (Tax) Case Study",
    """Hi,

Thank you for your interest in the AI Engineer (Tax) role at Calloway AI.

We've reviewed your application and would like to invite you to move forward in our
interview process by completing the attached case study.

Once completed, please send us your written submission, a working prototype, the
source code, and a README with setup and usage instructions.

We are not looking for perfection. We're most interested in understanding your
thought process and how you approach solving real-world problems.

Best regards,
Talent Acquisition
Calloway AI
""",
    _dt(10, 27, 9, 50),
)

R15 = _msg(
    "r015", "tr10", USER_EMAIL, "Candidate Example",
    "Re: Calloway AI | AI Engineer (Tax) Case Study",
    """Hi Camille,

Thank you for the opportunity to move forward in the interview process for the AI
Engineer (Tax) role at Calloway AI. I'm looking forward to working on the case study.

Could you please let me know the expected deadline for submitting the completed case
study, prototype, source code, and README?

Best regards,
Candidate
""",
    _dt(10, 27, 12, 42), labels=("SENT",), to="talent@callowayai.com",
    in_reply_to="<r014@mail.example>", references=("<r014@mail.example>",),
)


REALISTIC_MESSAGES: list[RawMessage] = [
    R01, R02, R03, R04, R05, R06, R07, R08, R09, R10,
    R11, R12, R13, R14, R15,
]

#: Ground truth: gmail_id -> the process it belongs to.
REALISTIC_EXPECTED: dict[str, str] = {
    **{g: "northvale_swe3" for g in
       ("r001", "r002", "r003", "r004", "r005", "r006", "r007", "r008", "r009", "r010")},
    **{g: "brightline_ads" for g in ("r011", "r012", "r013")},
    **{g: "calloway_ai_tax" for g in ("r014", "r015")},
}

REALISTIC_MUST_MERGE: list[tuple[str, str, str]] = [
    ("r001", "r003", "P5: role drifts from 'engineering roles' to 'Software Engineer III'"),
    ("r003", "r004", "P1/P3: a coordinator on the vendor subdomain joins the process"),
    ("r003", "r007", "P7: reschedule thread belongs to the same process"),
    ("r003", "r010", "P2: handoff to a second recruiter keeps one process"),
    ("r011", "r013", "P4: same recruiter via ATS relay, then corporate address"),
]

REALISTIC_MUST_NOT_MERGE: list[tuple[str, str, str]] = [
    ("r003", "r011", "different employers (Northvale vs Brightline)"),
    ("r003", "r014", "different employers (Northvale vs Calloway)"),
    ("r011", "r014", "different employers (Brightline vs Calloway)"),
]
