"""A second mock inbox, modelled on real recruiting-mail structure.

Entirely fictional, and deliberately sharing **no companies, people or addresses**
with `synthetic.py` or `realistic.py`. Four concurrent hiring processes plus noise,
reproducing structures observed in real inboxes:

  * an ATS relay and the same recruiter's corporate address  (Process A)
  * a vendor/contractor subdomain of the employer            (Process B)
  * a recruiter handoff by CC, plus a scheduling coordinator (Process B)
  * a reschedule conversation spanning separate threads      (Process B)
  * a take-home case study and an explicit rejection         (Process C)
  * an agency that hides its client, then names it           (Process D)
  * marketing mail from an employer's own domain             (noise)
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..models import RawMessage

USER_EMAIL = "jordan.quinn.demo@gmail.com"

_LEGAL = """

--
This e-mail and any attachments may contain information which is confidential and
intended for the sole use of the recipient(s) named above. Any use of the information
or the taking of any action in reliance on the contents, by persons other than the
designated recipient(s), is strictly prohibited.
"""


def _dt(month: int, day: int, hour: int = 9, minute: int = 0) -> datetime:
    return datetime(2025, month, day, hour, minute, tzinfo=timezone.utc)


def _m(gid, tid, frm, name, subject, body, date, *, to=None, cc=(),
       labels=("INBOX",), irt=None, refs=()):
    recipients = [to] if to else ([USER_EMAIL] if frm != USER_EMAIL else [])
    return RawMessage(
        gmail_id=gid, thread_id=tid, rfc_message_id=f"<{gid}@mock.example>",
        in_reply_to=irt, references=list(refs), from_addr=frm, from_name=name,
        to_addrs=recipients, cc_addrs=list(cc), subject=subject, date=date,
        snippet=body.strip().replace("\n", " ")[:120], labels=list(labels),
        body_text=body.strip(), body_fetched=True,
    )


# =============================================================================
# PROCESS A — Helix Dynamics / Machine Learning Engineer
# ATS relay then corporate address; screening questions; a candidate chasing
# next steps; a recruiter waiting on a shortlist; first stage then onsite.
# =============================================================================

A1 = _m("k001", "tk01", "helixdynamics@myworkday.com", "Helix Dynamics Talent Acquisition",
        "Interview Availability Needed for REQ-88214 - Machine Learning Engineer",
        """Hi Jordan,

Thank you for taking the time to apply for the Machine Learning Engineer role. This is
a key hire for us as we continue to grow the team.

We were impressed with your application and would like to invite you to a first-stage
interview via video call. The interview will last approximately 30-40 minutes and will
be with Priyanka Raval (Director of Applied ML).

If successful, the next stage would be a one-hour, in-person interview at our Austin
office.

In the meantime, could you please confirm your salary expectations, your right to work
status, and your notice period.

Available slots (first come, first served):
Tuesday 14 October, 19:30 / 20:00 / 20:30
Thursday 16 October, 19:30 / 20:00 / 20:30

Thanks,
Nadia

Nadia Brennan
Talent Acquisition Partner, Helix Dynamics
""", _dt(10, 13, 7, 31))

A2 = _m("k002", "tk01", USER_EMAIL, "Jordan Quinn",
        "Re: Interview Availability Needed for REQ-88214 - Machine Learning Engineer",
        """Hi Nadia,

Thank you for reaching out about the Machine Learning Engineer role at Helix Dynamics.
Thursday at 19:30 would work well for me.

On your questions: I am targeting a base in the 135-155k range depending on the overall
package, I am authorised to work without sponsorship, and I could start within about
three weeks.

Best regards,
Jordan
""", _dt(10, 13, 17, 55), to="helixdynamics@myworkday.com", labels=("SENT",),
        irt="<k001@mock.example>", refs=("<k001@mock.example>",))

# Same human, now from her corporate address, with an inline quoted header block.
A3 = _m("k003", "tk02", "nadia.brennan@helixdynamics.com", "Brennan, Nadia",
        "RE: Interview Availability Needed for REQ-88214 - Machine Learning Engineer",
        """Thank you, I've just sent the invite across with all the details.

Nadia Brennan
Helix Dynamics
Talent Acquisition Partner
nadia.brennan@helixdynamics.com

From: Jordan Quinn <jordan.quinn.demo@gmail.com>
Sent: Monday, October 13, 2025 5:55 PM
To: Helix Dynamics Talent Acquisition
Subject: Re: Interview Availability Needed for REQ-88214 - Machine Learning Engineer

Hi Nadia, Thursday at 19:30 would work well for me...
""", _dt(10, 14, 11, 34))

A4 = _m("k004", "tk02", USER_EMAIL, "Jordan Quinn",
        "Following up on next steps",
        """Hi Nadia,

I hope you're doing well. I wanted to follow up regarding the Machine Learning Engineer
process. When we last spoke I understood second-stage interviews were expected to begin
the week of 27 October, so I wanted to check whether there are any updates.

I'd appreciate any information on timing, as I'll need to plan travel accordingly.

Best regards,
Jordan
""", _dt(10, 20, 12, 58), to="nadia.brennan@helixdynamics.com", labels=("SENT",))

A5 = _m("k005", "tk02", "nadia.brennan@helixdynamics.com", "Brennan, Nadia",
        "RE: Following up on next steps",
        """Hi Jordan,

I'm well thank you, I hope you are too.

I'm currently waiting for Priyanka to provide her shortlist for second-stage interviews
next week. I will update you as soon as possible.

Nadia
""" + _LEGAL, _dt(10, 21, 11, 25),
        irt="<k004@mock.example>", refs=("<k004@mock.example>",))

A6 = _m("k006", "tk03", "nadia.brennan@helixdynamics.com", "Brennan, Nadia",
        "Second-stage interview - Machine Learning Engineer",
        """Hi Jordan,

We'd like to invite you to a second-stage interview next week. This will be an
in-person interview at our Austin office and will last approximately one hour.

Would you be available on Friday 31 October at 11:30?

Helix Dynamics, 1200 Loyola Street, Austin, TX
""", _dt(10, 24, 10, 42))

A7 = _m("k007", "tk03", USER_EMAIL, "Jordan Quinn",
        "Re: Second-stage interview - Machine Learning Engineer",
        """Hi Nadia,

Thank you for the invitation. Friday 31 October at 11:30 works well for me and I'm
looking forward to meeting the team in person.

I also wanted to ask whether the company offers any assistance with travel to Austin
for the interview.

Best,
Jordan
""", _dt(10, 24, 11, 16), to="nadia.brennan@helixdynamics.com", labels=("SENT",),
        irt="<k006@mock.example>", refs=("<k006@mock.example>",))


# =============================================================================
# PROCESS B — Ardent Financial / Quantitative Analyst
# Contractor subdomain, vague-then-specific role, coordinator, reschedule
# across threads, and a handoff to a second recruiter by CC.
# =============================================================================

B1 = _m("k008", "tk04", "t.okonkwo@ext.ardentfinancial.com", "Tobi Okonkwo (ext)",
        "Ardent Financial - analytics opportunities",
        """Hi Jordan,

I'm Tobi, part of a sourcing team supporting Ardent Financial. I reviewed your
application for our analytics roles and would like to move forward.

To continue, please complete the short questionnaire linked below.

Best,
Tobi
Sourcing Specialist, supporting Ardent Financial
""", _dt(9, 8, 10, 15))

B2 = _m("k009", "tk04", USER_EMAIL, "Jordan Quinn",
        "Re: Ardent Financial - analytics opportunities",
        """Hi Tobi,

Thank you for reaching out. I've completed the questionnaire and I'm glad to move
forward.

Best,
Jordan
""", _dt(9, 8, 21, 2), to="t.okonkwo@ext.ardentfinancial.com", labels=("SENT",),
        irt="<k008@mock.example>", refs=("<k008@mock.example>",))

B3 = _m("k010", "tk05", "t.okonkwo@ext.ardentfinancial.com", "Tobi Okonkwo (ext)",
        "Moving forward - Quantitative Analyst at Ardent Financial",
        """Hi Jordan,

Good news - we'd like to interview you for the Quantitative Analyst role at Ardent
Financial, requisition AF-5530.

Round one is two 45-minute sessions: one technical and one behavioural. Please reply to
confirm you're ready to begin and our scheduling team will collect your availability.

Thanks,
Tobi
""", _dt(9, 9, 9, 47))

B4 = _m("k011", "tk06", "s.varga@ext.ardentfinancial.com", "Sofia Varga (ext)",
        "Your upcoming Ardent Financial interviews",
        """Hi Jordan,

I'm Sofia, a Recruiting Coordinator working with Tobi to schedule your interviews for
the Quantitative Analyst role.

1st Interview: Tuesday, 23 September, 14:00 - 14:45
2nd Interview: Tuesday, 23 September, 16:00 - 16:45

Please reply to confirm you can make it.

Regards,
Sofia
""" + _LEGAL, _dt(9, 15, 14, 0))

B5 = _m("k012", "tk07", USER_EMAIL, "Jordan Quinn",
        "Request to move my interviews",
        """Hi Sofia,

Thank you again for coordinating my interviews for the Quantitative Analyst
opportunity.

Would it be possible to reschedule to a later date? I remain very interested and
appreciate your flexibility.

Best,
Jordan
""", _dt(9, 17, 13, 33), to="s.varga@ext.ardentfinancial.com", labels=("SENT",))

B6 = _m("k013", "tk07", "s.varga@ext.ardentfinancial.com", "Sofia Varga (ext)",
        "Re: Request to move my interviews",
        """Hi Jordan,

Thank you for letting us know - we can certainly help. Please send a few alternative
dates over the next two weeks and I'll update your schedule.

Regards,
Sofia
""", _dt(9, 17, 20, 49), irt="<k012@mock.example>", refs=("<k012@mock.example>",))

B7 = _m("k014", "tk08", "s.varga@ext.ardentfinancial.com", "Sofia Varga (ext)",
        "Rescheduled interview details - Quantitative Analyst",
        """Hi Jordan,

As discussed, here are your rescheduled interview details. Please confirm by replying
to this email.

1st Interview: Wednesday, 1 October, 14:30 - 15:15
2nd Interview: Wednesday, 1 October, 16:00 - 16:45

Regards,
Sofia
""" + _LEGAL, _dt(9, 18, 21, 56))

B8 = _m("k015", "tk05", "t.okonkwo@ext.ardentfinancial.com", "Tobi Okonkwo (ext)",
        "Re: Moving forward - Quantitative Analyst at Ardent Financial",
        """Hi Jordan,

You've successfully passed your first round of interviews - congratulations. The team
was impressed.

I'm connecting you with a dedicated point of contact who will guide you through the
next stages.

Best wishes,
Tobi
""", _dt(10, 6, 9, 51), irt="<k010@mock.example>", refs=("<k010@mock.example>",))

B9 = _m("k016", "tk09", "t.okonkwo@ext.ardentfinancial.com", "Tobi Okonkwo (ext)",
        "Introducing your new point of contact",
        """Hi Jordan,

I'd like to introduce you to Mateo (cc'd) who will be your main point of contact from
here for the Quantitative Analyst process.

Sincerely,
Tobi
""", _dt(10, 6, 9, 52), cc=("m.delgado@ext.ardentfinancial.com",))

B10 = _m("k017", "tk10", "m.delgado@ext.ardentfinancial.com", "Mateo Delgado (ext)",
         "Next steps at Ardent Financial",
         """Hello Jordan,

I'm Mateo, Staffing Specialist. I'll be your point of contact going forward for the
Quantitative Analyst role. Our next step is a short call to discuss the remaining
process.

Please pick a 30-minute slot using my scheduling link, or send your availability.

Best,
Mateo
""" + _LEGAL, _dt(10, 15, 7, 0), cc=("t.okonkwo@ext.ardentfinancial.com",))


# =============================================================================
# PROCESS C — Lumenwave Health / Clinical Data Scientist
# Case study, candidate asking about the deadline, submission, rejection.
# =============================================================================

C1 = _m("k018", "tk11", "talent@lumenwavehealth.com", "Rosa Villanueva",
        "Lumenwave Health | Clinical Data Scientist Case Study",
        """Hi Jordan,

Thank you for your interest in the Clinical Data Scientist role at Lumenwave Health.

We've reviewed your application and would like to invite you to complete the attached
case study as the next step.

Once complete, please send your written submission, a working prototype, the source
code, and a README with setup instructions.

We're most interested in your thought process rather than perfection.

Best regards,
Talent Acquisition
Lumenwave Health
""", _dt(10, 2, 9, 50))

C2 = _m("k019", "tk11", USER_EMAIL, "Jordan Quinn",
        "Re: Lumenwave Health | Clinical Data Scientist Case Study",
        """Hi Rosa,

Thank you for the opportunity to move forward for the Clinical Data Scientist role at
Lumenwave Health.

Could you let me know the expected deadline for the case study, prototype and README?

Best regards,
Jordan
""", _dt(10, 2, 12, 42), to="talent@lumenwavehealth.com", labels=("SENT",),
        irt="<k018@mock.example>", refs=("<k018@mock.example>",))

C3 = _m("k020", "tk11", USER_EMAIL, "Jordan Quinn",
        "Re: Lumenwave Health | Clinical Data Scientist Case Study",
        """Hi Rosa,

I've just submitted my completed case study for the Clinical Data Scientist role, along
with the prototype, source code and README.

Thanks,
Jordan
""", _dt(10, 9, 20, 15), to="talent@lumenwavehealth.com", labels=("SENT",),
        irt="<k019@mock.example>", refs=("<k018@mock.example>", "<k019@mock.example>"))

C4 = _m("k021", "tk12", "talent@lumenwavehealth.com", "Rosa Villanueva",
        "Update on your Lumenwave Health application",
        """Dear Jordan,

Thank you for your interest in the Clinical Data Scientist role at Lumenwave Health and
for the time you put into the case study.

Unfortunately, we have decided not to move forward with your application at this time.
We moved forward with other candidates whose experience more closely matched the
requirements.

We wish you the best in your search.

Lumenwave Health Talent Team
""", _dt(10, 22, 10, 0))


# =============================================================================
# PROCESS D — an agency that hides its client, then names it
# =============================================================================

D1 = _m("k022", "tk13", "r.sandoval@pikearrowsearch.com", "Renata Sandoval",
        "Confidential search - strong match for your background",
        """Hi Jordan,

I'm running a confidential search for a well-funded client in the robotics space. The
role is a strong match for your background and the compensation is competitive.

I'm not able to share the company name until we've had a brief conversation. Would you
be open to a 15-minute call?

Renata Sandoval
Pike & Arrow Search
""", _dt(10, 6, 14, 25))

D2 = _m("k023", "tk13", "r.sandoval@pikearrowsearch.com", "Renata Sandoval",
        "Re: Confidential search - strong match for your background",
        """Hi Jordan,

Now that we've spoken I can share the details: my client is Caldera Robotics and the
role is Perception Engineer, requisition CR-2041.

Shall I put you forward?

Renata
""", _dt(10, 10, 9, 30), irt="<k022@mock.example>", refs=("<k022@mock.example>",))


# =============================================================================
# NOISE — must be excluded
# =============================================================================

N1 = _m("k024", "tk14", "news@helixdynamics.com", "Helix Dynamics",
        "Helix Dynamics Q4 product newsletter",
        """Read about our latest platform release, customer stories and upcoming webinars.

We're always hiring - browse open roles on our careers page.

You are receiving this because you subscribed. Unsubscribe at any time.
""", _dt(10, 11, 7, 0), labels=("INBOX", "CATEGORY_PROMOTIONS"))

N2 = _m("k025", "tk15", "alerts@jobboard.example", "JobBoard Alerts",
        "Jobs you may like: 18 new Machine Learning roles",
        """Jobs you may like this week, based on your profile.

This is your weekly job alert digest. Unsubscribe at any time.
""", _dt(10, 12, 6, 0), labels=("INBOX", "CATEGORY_PROMOTIONS"))


MOCK_MESSAGES: list[RawMessage] = [
    A1, A2, A3, A4, A5, A6, A7,
    B1, B2, B3, B4, B5, B6, B7, B8, B9, B10,
    C1, C2, C3, C4,
    D1, D2,
    N1, N2,
]

#: gmail_id -> process label, or None when it must be excluded entirely.
MOCK_EXPECTED: dict[str, str | None] = {
    **{g: "helix_mle" for g in ("k001", "k002", "k003", "k004", "k005", "k006", "k007")},
    **{g: "ardent_quant" for g in ("k008", "k009", "k010", "k011", "k012", "k013",
                                   "k014", "k015", "k016", "k017")},
    **{g: "lumenwave_cds" for g in ("k018", "k019", "k020", "k021")},
    **{g: "caldera_perception" for g in ("k022", "k023")},
    "k024": None, "k025": None,
}

MOCK_MUST_MERGE: list[tuple[str, str, str]] = [
    ("k001", "k003", "ATS relay and the recruiter's corporate address are one process"),
    ("k001", "k006", "first-stage and second-stage invitations are one process"),
    ("k008", "k010", "vague 'analytics roles' then the specific Quantitative Analyst"),
    ("k010", "k011", "the coordinator's scheduling mail joins the process"),
    ("k010", "k014", "the reschedule thread belongs to the same process"),
    ("k010", "k017", "handoff to a second point of contact keeps one process"),
    ("k018", "k021", "the rejection closes the process it belongs to"),
    ("k022", "k023", "the agency's follow-up naming the client joins its own thread"),
]

MOCK_MUST_NOT_MERGE: list[tuple[str, str, str]] = [
    ("k001", "k010", "Helix Dynamics vs Ardent Financial"),
    ("k001", "k018", "Helix Dynamics vs Lumenwave Health"),
    ("k010", "k018", "Ardent Financial vs Lumenwave Health"),
    ("k018", "k022", "Lumenwave Health vs the agency search"),
]

#: Messages that must be filtered out as non-recruitment.
MOCK_EXCLUDED = {"k024", "k025"}
