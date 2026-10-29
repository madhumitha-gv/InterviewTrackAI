"""Three structurally-distinct mock mailboxes, for cross-inbox generalization.

Each is a separate account with its own `user_email` and message list. They use
companies, domains, labels and senders that appear in **no hardcoded list** anywhere
in the engine — that is the whole point: the acceptance criterion is that the same
code processes all three with no edits to recruiter/domain/company/role lists.

  PROFILE_CONVENTIONAL  — a tidy recruiting inbox: standard phrasing, a known ATS and
                          scheduler, Gmail CATEGORY_* labels.
  PROFILE_UNFAMILIAR    — unusual TLDs, an UNKNOWN staffing agency, a self-hosted
                          scheduler, and custom user labels (Label_12, Jobs2026)
                          instead of CATEGORY_*. Nothing may key off known domains or
                          label names.
  PROFILE_SPARSE        — a high-volume inbox: ~4 recruitment messages buried in
                          newsletters, receipts, notifications and social mail. Tests
                          precision — the detector must find the few and reject the many.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..models import RawMessage


@dataclass
class MailboxProfile:
    key: str
    label: str
    user_email: str
    messages: list[RawMessage]
    expected: dict[str, str | None]          # gmail_id -> process, or None = excluded
    must_merge: list[tuple[str, str, str]] = field(default_factory=list)
    must_not_merge: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def expected_applications(self) -> int:
        return len({v for v in self.expected.values() if v})

    @property
    def recruitment_ids(self) -> set[str]:
        return {g for g, v in self.expected.items() if v}

    @property
    def excluded_ids(self) -> set[str]:
        return {g for g, v in self.expected.items() if v is None}


def _dt(m, d, h=9, mi=0):
    return datetime(2025, m, d, h, mi, tzinfo=timezone.utc)


def _msg(gid, tid, frm, name, subj, body, date, user, *, to=None, cc=(),
         labels=("INBOX",), irt=None, refs=()):
    recipients = [to] if to else ([user] if frm != user else [])
    return RawMessage(
        gmail_id=gid, thread_id=tid, rfc_message_id=f"<{gid}@mbx.example>",
        in_reply_to=irt, references=list(refs), from_addr=frm, from_name=name,
        to_addrs=recipients, cc_addrs=list(cc), subject=subj, date=date,
        snippet=body.strip().replace("\n", " ")[:120], labels=list(labels),
        body_text=body.strip(), body_fetched=True,
    )


# =============================================================================
# PROFILE 1 — CONVENTIONAL
# =============================================================================
U1 = "taylor.rivers.demo@gmail.com"
_P1 = [
    _msg("c01", "c-t1", "recruiting@novagridenergy.com", "Dana Pierce",
         "Nova Grid Energy - Data Engineer",
         "Hi Taylor, I'm a recruiter at Nova Grid Energy. We'd like to interview you "
         "for the Data Engineer role, requisition NG-4420. Are you free this week?",
         _dt(10, 3), U1),
    _msg("c02", "c-t2", "no-reply@goodtime.io", "GoodTime",
         "Invitation: Technical Interview - Nova Grid Energy",
         "You have been invited to a technical interview.\nPosition: Data Engineer\n"
         "Company: Nova Grid Energy\nWhen: Friday, October 17 at 2:00 PM.", _dt(10, 10), U1),
    _msg("c03", "c-t3", "no-reply@us.greenhouse-mail.io", "Nova Grid via Greenhouse",
         "Your take-home assignment for the Data Engineer role",
         "As the next step for the Data Engineer role at Nova Grid Energy, please "
         "complete the attached take-home assignment within five days.", _dt(10, 11), U1),
    _msg("c04", "c-t3", U1, "Taylor Rivers",
         "Re: Your take-home assignment for the Data Engineer role",
         "Hi, I've just submitted my completed take-home assignment for the Data "
         "Engineer role at Nova Grid Energy. Thanks!", _dt(10, 14, 20),
         U1, to="no-reply@us.greenhouse-mail.io", labels=("SENT",),
         irt="<c03@mbx.example>", refs=("<c03@mbx.example>",)),
    _msg("c05", "c-t4", "news@novagridenergy.com", "Nova Grid Energy",
         "Nova Grid Energy monthly newsletter",
         "Our latest grid-tech updates and upcoming webinars. We're always hiring - "
         "see our careers page. Unsubscribe anytime.", _dt(10, 8),
         U1, labels=("INBOX", "CATEGORY_PROMOTIONS")),
]
PROFILE_CONVENTIONAL = MailboxProfile(
    "conventional", "Conventional recruiting inbox", U1, _P1,
    expected={"c01": "nova_de", "c02": "nova_de", "c03": "nova_de", "c04": "nova_de",
              "c05": None},
    must_merge=[("c01", "c03", "recruiter + ATS assessment, one process")],
    must_not_merge=[],
)


# =============================================================================
# PROFILE 2 — UNFAMILIAR DOMAINS & CUSTOM LABELS
# Unknown agency, unusual TLDs, self-hosted scheduler, user labels.
# =============================================================================
U2 = "sam.okafor.demo@gmail.com"
_P2 = [
    # Unknown staffing agency (not in STAFFING_DOMAINS), names its client in text.
    _msg("u01", "u-t1", "recruiter@bridgepointhunt.io", "Lena Hart",
         "A role that fits your background",
         "Hi Sam, I'm a recruiter with Bridgepoint Hunt. My client is Vireo Analytics "
         "and they're hiring a Platform Engineer, requisition VA-2207. Interested?",
         _dt(9, 29), U2, labels=("INBOX", "Label_12")),
    # Employer on an unusual TLD, custom label.
    _msg("u02", "u-t2", "people@vireo.analytics", "Marcus Bell",
         "Vireo Analytics - next steps for Platform Engineer",
         "Hi Sam, I'm on the talent team at Vireo Analytics. We'd like to move forward "
         "and interview you for the Platform Engineer role (req VA-2207). Round one is "
         "a technical screen.", _dt(10, 6), U2, labels=("INBOX", "Jobs2026")),
    # Self-hosted scheduler on the employer's own subdomain (unknown to lists).
    _msg("u03", "u-t3", "scheduler@meet.vireo.analytics", "Vireo Scheduling",
         "Confirmed: your Platform Engineer technical screen",
         "Your interview is confirmed.\nRole: Platform Engineer\nCompany: Vireo "
         "Analytics\nWhen: Thursday, October 16 at 11:00.", _dt(10, 12), U2,
         labels=("INBOX", "Jobs2026")),
    _msg("u04", "u-t3", U2, "Sam Okafor",
         "Re: Confirmed: your Platform Engineer technical screen",
         "Thanks for confirming - looking forward to the Platform Engineer screen with "
         "Vireo Analytics on Thursday.", _dt(10, 12, 15), U2,
         to="scheduler@meet.vireo.analytics", labels=("SENT",),
         irt="<u03@mbx.example>", refs=("<u03@mbx.example>",)),
    # A DELIBERATELY MALFORMED message: broken HTML, no plain text, odd TLD sender.
    RawMessage(
        gmail_id="u05", thread_id="u-t4", rfc_message_id="<u05@mbx.example>",
        from_addr="hr@quanta.engineering", from_name="Quanta HR",
        to_addrs=[U2], subject="Interview details",
        date=_dt(10, 13), snippet="", labels=["INBOX", "Label_12"],
        body_text="",
        body_html="<html><body><div><p>Hi Sam<br>Your <b>technical interview<i> for "
                  "the Reliability Engineer role at Quanta Engineering is on Tuesday "
                  "October 21 at 10:00.<script>junk()</script><style>.x{}</style>",
        body_fetched=True,
    ),
]
PROFILE_UNFAMILIAR = MailboxProfile(
    "unfamiliar", "Unfamiliar domains & custom labels", U2, _P2,
    expected={"u01": "vireo_pe", "u02": "vireo_pe", "u03": "vireo_pe", "u04": "vireo_pe",
              "u05": "quanta_re"},
    must_merge=[("u01", "u03", "unknown agency + employer + self-hosted scheduler, one process")],
    must_not_merge=[("u01", "u05", "Vireo Analytics vs Quanta Engineering")],
)


# =============================================================================
# PROFILE 3 — HIGH-VOLUME, SPARSE RECRUITMENT
# ~4 recruitment messages among noise.
# =============================================================================
U3 = "jordan.lee.demo@gmail.com"
_noise = [
    ("s90", "shipping@parceltrack.example", "ParcelTrack", "Your package is out for delivery",
     "Your order #88213 is arriving today. Track it here."),
    ("s91", "digest@devweekly.example", "Dev Weekly", "Dev Weekly: 10 articles you missed",
     "This week in software: newsletters, releases and tips. Unsubscribe anytime."),
    ("s92", "noreply@socialapp.example", "SocialApp", "You have 3 new connection requests",
     "People you may know want to connect. See who."),
    ("s93", "billing@cloudhost.example", "CloudHost", "Your October invoice is ready",
     "Invoice for October: $42.00. Thank you for being a customer."),
    ("s94", "alerts@jobboard.example", "JobBoard", "Jobs you may like: 15 new Data roles",
     "Jobs you may like this week based on your profile. Weekly digest."),
    ("s95", "team@projecttool.example", "ProjectTool", "Weekly summary for your workspace",
     "Here's what happened in your projects this week."),
    ("s96", "news@retailstore.example", "RetailStore", "Flash sale - 30% off this weekend",
     "Don't miss our biggest sale. Shop now."),
    ("s97", "receipts@rideshare.example", "RideShare", "Your trip receipt",
     "Thanks for riding. Total: $18.40."),
    ("s98", "noreply@calendarapp.example", "CalendarApp", "Reminder: Dentist at 3pm",
     "You have an appointment today at 3:00 PM."),
    ("s99", "updates@streamingsvc.example", "StreamingSvc", "New shows added this week",
     "Fresh titles just landed. Start watching."),
]
_P3 = [_msg(gid, f"s-{gid}", frm, name, subj, body, _dt(10, 2 + i), U3,
            labels=("INBOX", "CATEGORY_PROMOTIONS" if i % 2 else "CATEGORY_UPDATES"))
       for i, (gid, frm, name, subj, body) in enumerate(_noise)]
_P3 += [
    _msg("s01", "s-r1", "talent@meridianpathhealth.com", "Olivia Crane",
         "Meridian Path Health - Research Scientist",
         "Hi Jordan, I'm a recruiter at Meridian Path Health. We'd like to interview you "
         "for the Research Scientist role, requisition MP-7781. Are you available next "
         "week?", _dt(10, 9), U3),
    _msg("s02", "s-r1", U3, "Jordan Lee",
         "Re: Meridian Path Health - Research Scientist",
         "Hi Olivia, yes I'd be glad to interview for the Research Scientist role at "
         "Meridian Path Health. Tuesday afternoon works well.", _dt(10, 9, 18), U3,
         to="talent@meridianpathhealth.com", labels=("SENT",),
         irt="<s01@mbx.example>", refs=("<s01@mbx.example>",)),
    _msg("s03", "s-r2", "no-reply@goodtime.io", "GoodTime",
         "Invitation: Technical Interview - Meridian Path Health",
         "You have been invited to a technical interview.\nPosition: Research Scientist\n"
         "Company: Meridian Path Health\nWhen: Tuesday, October 21 at 1:00 PM.",
         _dt(10, 15), U3),
    _msg("s04", "s-r2", "talent@meridianpathhealth.com", "Olivia Crane",
         "Your offer from Meridian Path Health",
         "Dear Jordan, we're delighted to extend an offer for the Research Scientist "
         "role at Meridian Path Health. Your offer letter is attached.", _dt(10, 24), U3),
]
PROFILE_SPARSE = MailboxProfile(
    "sparse", "High-volume inbox, sparse recruitment", U3, _P3,
    expected={**{gid: None for gid, *_ in _noise},
              "s01": "meridianpath_rs", "s02": "meridianpath_rs",
              "s03": "meridianpath_rs", "s04": "meridianpath_rs"},
    must_merge=[("s01", "s03", "recruiter + scheduler for the one real process")],
    must_not_merge=[],
)


ALL_PROFILES: list[MailboxProfile] = [
    PROFILE_CONVENTIONAL, PROFILE_UNFAMILIAR, PROFILE_SPARSE,
]


# Fictional rewrites of observed inbox patterns, not copied private messages.
REVIEWER_EMAIL = "avery.quinn.interviewtrack.demo@gmail.com"
_reviewer = []
def _review_msg(i, company, domain, role, subject, text, thread=None, sent=False):
    recruiter = "Morgan Vale" if company == "Juniper Labs" else "Riley Brooks"
    body = f"Hi Avery,\n{text}\nRole: {role}\nCompany: {company}\nRegards, {recruiter}"
    _reviewer.append(_msg(f"fictional-{i}", thread or f"fictional-thread-{i}",
        REVIEWER_EMAIL if sent else f"talent@{domain}",
        "Avery Quinn" if sent else recruiter, subject, body,
        _dt(10, i), REVIEWER_EMAIL,
        to=f"talent@{domain}" if sent else None,
        labels=("SENT",) if sent else ("INBOX",)))

_review_msg(1, "Juniper Labs", "juniperlabs.example", "AI Engineer",
    "AI Engineer screening interview at Juniper Labs",
    "We would like to invite you to a screening call for the AI Engineer role at Juniper Labs. Requisition JL-8101.", "fictional-juniper")
_review_msg(2, "Juniper Labs", "juniperlabs.example", "AI Engineer",
    "Re: AI Engineer screening interview at Juniper Labs",
    "Thank you for the invitation. Could you share the interview format and the next steps? Requisition JL-8101.", "fictional-juniper", True)
for i, subject, body, sent in [
    (3, "Phone screen", "We invite you to a screening call", False),
    (4, "Technical interview", "We would like to schedule your technical interview", False),
    (5, "Take-home assignment", "Please complete the take-home assignment", False),
    (6, "Assessment submission", "I have submitted my assessment", True),
    (7, "Final interview", "We invite you to the final interview", False),
    (8, "Following up after our interview", "I wanted to follow up after our final interview", True),
    (9, "Offer letter", "We are pleased to extend you an offer", False),
]:
    _review_msg(i, "Harbor Energy", "harborenergy.example", "Data Engineer",
        subject + " for Data Engineer at Harbor Energy",
        body + " for the Data Engineer role at Harbor Energy. Requisition HE-8202.", sent=sent)
_review_msg(10, "Harbor Energy", "harborenergy.example", "Platform Engineer",
    "Screening interview for Platform Engineer at Harbor Energy",
    "We invite you to a screening call for the Platform Engineer role at Harbor Energy. Requisition HE-8303.")
_review_msg(11, "Harbor Energy", "harborenergy.example", "Platform Engineer",
    "Update on Platform Engineer at Harbor Energy",
    "Thank you for interviewing. We have decided not to move forward with your application for the Platform Engineer role at Harbor Energy. Requisition HE-8303.", "fictional-thread-10")
_review_msg(12, "Career Digest", "careerdigest.example", "Software Engineer",
    "Jobs you may like: new opportunities", "Job alert: recommended jobs matching your profile. Unsubscribe.")
_review_msg(13, "Career Digest", "careerdigest.example", "Data Scientist",
    "Your application was received", "Thank you for applying. We have received your application and will review your qualifications.")
PROFILE_REVIEWER = MailboxProfile("reviewer", "Avery Quinn — fictional interview journeys", REVIEWER_EMAIL,
    _reviewer, expected={**{f"fictional-{i}": "juniper" for i in (1,2)},
        **{f"fictional-{i}": "harbor_data" for i in range(3,10)},
        "fictional-10": "harbor_platform", "fictional-11": "harbor_platform",
        "fictional-12": None, "fictional-13": "receipt_context"},
    must_merge=[("fictional-3", "fictional-9", "screening through offer")],
    must_not_merge=[("fictional-3", "fictional-10", "different roles at same employer")])
ALL_PROFILES.append(PROFILE_REVIEWER)


# Additional entirely invented cases. Shared CC contacts do not identify a job.
STAFFING_CC = ["sasha.reed@clearpathstaffing.example", "devon.park@clearpathstaffing.example"]
def _expanded(i, company, role, req, subject, body, *, sender=None, sent=False, thread=None):
    sender = sender or "elena.ward@clearpathstaffing.example"
    text = (f"Hi Avery,\n{body} for the {role} role at {company}. "
            f"Requisition {req}.\nCompany: {company}\nPosition: {role}\n"
            "Sasha Reed and Devon Park are copied for interview coordination.\n"
            "Regards, Elena Ward\nClearpath Staffing")
    message = _msg(f"fictional-{i}", thread or f"fictional-thread-{i}",
        REVIEWER_EMAIL if sent else sender, "Avery Quinn" if sent else "Elena Ward",
        f"{subject} — {role} at {company}", text, _dt(10, i - 10), REVIEWER_EMAIL,
        to=sender if sent else None, cc=STAFFING_CC,
        labels=("SENT",) if sent else ("INBOX",))
    _reviewer.append(message)

# A complete, separate journey with changing specialists and multiple CCs.
for i, subject, text, sent, sender in [
    (14, "Screening invitation", "We invite you to a screening call", False, None),
    (15, "Screening completed", "Thank you for completing your screening interview. We would like to schedule a technical interview", False, "sasha.reed@clearpathstaffing.example"),
    (16, "Technical interview round 1", "We invite you to a technical interview, round one", False, None),
    (17, "Technical interview round 2", "We invite you to a technical interview, round two focused on system design", False, "devon.park@clearpathstaffing.example"),
    (18, "Take-home assignment", "Please complete the take-home assignment", False, None),
    (19, "Assessment submission", "I have submitted my completed assessment", True, None),
    (20, "Final interview", "We invite you to the final interview with the hiring manager", False, "people@lumenrobotics.example"),
    (21, "Following up after our interview", "I wanted to follow up after our final interview", True, None),
    (22, "Offer letter", "We are pleased to extend you an offer", False, "people@lumenrobotics.example"),
]:
    _expanded(i, "Lumen Robotics", "Machine Learning Engineer", "LR-9401",
              subject, text, sent=sent, sender=sender)

# Same company and same staffing CCs, but a different role/requisition.
_expanded(23, "Lumen Robotics", "Backend Engineer", "LR-9402", "Screening invitation",
          "We invite you to a screening call")
_expanded(24, "Lumen Robotics", "Backend Engineer", "LR-9402", "Technical interview",
          "We invite you to a technical interview", sender="devon.park@clearpathstaffing.example")
_expanded(25, "Lumen Robotics", "Backend Engineer", "LR-9402", "Application update",
          "After your interview, we have decided not to move forward with your application",
          thread="fictional-thread-24")

# An ongoing multi-round process at another employer, sharing the staffing team.
_expanded(26, "Summit Analytics", "Data Engineer", "SA-9501", "Screening invitation",
          "We invite you to a screening call")
_expanded(27, "Summit Analytics", "Data Engineer", "SA-9501", "Technical interview round 1",
          "We invite you to a technical interview, round one")
_expanded(28, "Summit Analytics", "Data Engineer", "SA-9501", "Technical interview round 2",
          "We invite you to a technical interview, round two", sender="sasha.reed@clearpathstaffing.example")
_expanded(29, "Summit Analytics", "Data Engineer", "SA-9501", "Following up after our interview",
          "I wanted to follow up after our technical interview", sent=True)
for i, subject, body in [
    (30, "Jobs you may like: interview opportunities", "Job alert: recommended jobs matching your profile. Unsubscribe from this weekly digest."),
    (31, "Save 30 percent on interview coaching", "Special offer: interview coaching sale. Shop now and unsubscribe anytime."),
]:
    _reviewer.append(_msg(f"fictional-{i}", f"fictional-thread-{i}",
        "ads@careerbulletin.example", "Career Bulletin", subject, body,
        _dt(10, i - 10), REVIEWER_EMAIL, labels=("INBOX", "CATEGORY_PROMOTIONS")))
PROFILE_REVIEWER.expected.update({
    **{f"fictional-{i}": "lumen_ml" for i in range(14,23)},
    **{f"fictional-{i}": "lumen_backend" for i in range(23,26)},
    **{f"fictional-{i}": "summit_data" for i in range(26,30)},
    "fictional-30": None, "fictional-31": None,
})
PROFILE_REVIEWER.must_merge.extend([
    ("fictional-14", "fictional-22", "staffing handoffs through employer offer"),
    ("fictional-16", "fictional-17", "multiple technical rounds"),
    ("fictional-26", "fictional-29", "ongoing process with candidate followup"),
])
PROFILE_REVIEWER.must_not_merge.extend([
    ("fictional-14", "fictional-23", "same company and CCs but different roles"),
    ("fictional-14", "fictional-26", "same staffing team but different employers"),
])
