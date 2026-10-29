"""Entity normalization (Stage B) and deterministic classification."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from interviewtrack.classify import classify_message, extract_employer, extract_job_title
from interviewtrack.enums import EventType, Occurrence
from interviewtrack.models import RawMessage
from interviewtrack.parsing import parse_message
from interviewtrack.normalize import (
    canonical_company,
    canonical_role,
    company_from_domain,
    extract_requisition_id,
    is_ats_domain,
    is_non_employer_domain,
    is_staffing_domain,
    requisitions_conflict,
    role_family,
    role_similarity,
)

USER = "avery.candidate@gmail.com"


def mk(subject: str, body: str, frm: str = "r@acme.com", name: str = "R",
       labels=("INBOX",), tid: str = "t1"):
    return parse_message(RawMessage(
        gmail_id="m1", thread_id=tid, from_addr=frm, from_name=name,
        to_addrs=[USER], subject=subject, date=datetime(2025, 10, 3, tzinfo=timezone.utc),
        snippet=body[:120], body_text=body, body_fetched=True, labels=list(labels),
    ))


class TestCompanyNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("Northwind Analytics, Inc.", "northwind analytics"),
        ("NORTHWIND ANALYTICS", "northwind analytics"),
        ("Globex Labs LLC", "globex labs"),
        ("Initech Data Ltd", "initech data"),
    ])
    def test_canonicalization(self, raw, expected):
        assert canonical_company(raw) == expected

    def test_domain_derivation(self):
        assert company_from_domain("careers.northwind.com") == "northwind"

    def test_intermediary_domains_yield_no_employer(self):
        for domain in ("roberthalf.com", "hire.lever.co", "us.greenhouse-mail.io",
                       "calendly.com", "gmail.com"):
            assert company_from_domain(domain) == "", domain
            assert is_non_employer_domain(domain), domain

    def test_domain_classification(self):
        assert is_staffing_domain("harnham.com")
        assert is_ats_domain("us.greenhouse-mail.io")
        assert not is_non_employer_domain("northwind.com")


class TestRoleNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("Senior Data Scientist II (Remote)", "data scientist"),
        ("ML Engineer", "machine learning engineer"),
        ("AI Engineer", "artificial intelligence engineer"),
        ("Staff Software Engineer", "software engineer"),
    ])
    def test_canonicalization(self, raw, expected):
        assert canonical_role(raw) == expected

    def test_families(self):
        assert role_family("Senior Data Analyst") == "data_analyst"
        assert role_family("AI Engineer") == "ai_engineer"
        assert role_family("Machine Learning Engineer") == "ml_engineer"
        assert role_family("Zookeeper") is None

    def test_similarity_high_for_equivalents(self):
        assert role_similarity("AI Engineer", "Artificial Intelligence Engineer") >= 0.95
        assert role_similarity("Senior Data Scientist", "Data Scientist II") >= 0.95

    def test_similarity_low_for_different_roles(self):
        assert role_similarity("Data Analyst", "AI Engineer") < 0.35


class TestRequisitions:
    @pytest.mark.parametrize("text,expected", [
        ("The requisition ID is REQ-4417.", "REQ4417"),
        ("requisition REQ-4417", "REQ4417"),
        ("(requisition REQ-5120)", "REQ5120"),
        ("ref JR-1029384", "JR1029384"),
        ("no identifier here", None),
    ])
    def test_extraction(self, text, expected):
        assert extract_requisition_id(text) == expected

    def test_prefix_variants_do_not_conflict(self):
        assert requisitions_conflict("REQ-4417", "req_4417") is False
        assert requisitions_conflict("REQ4417", "4417") is False

    def test_genuinely_different_ids_conflict(self):
        assert requisitions_conflict("REQ-4417", "REQ-5120") is True

    def test_unknown_never_conflicts(self):
        assert requisitions_conflict(None, "REQ-1") is False


class TestEmployerExtraction:
    def test_staffing_agency_is_not_the_employer(self):
        """The single most important extraction rule."""
        msg = mk(
            "AI Engineer opportunity",
            "I am a technical recruiter with Harnham, and my client is "
            "Northwind Analytics. Requisition REQ-4417.",
            frm="sarah@harnham.com",
        )
        employer, conf, source = extract_employer(msg)
        assert canonical_company(employer) == "northwind analytics"
        assert conf >= 0.85

    def test_ats_sender_is_not_the_employer(self):
        msg = mk(
            "Your take-home assignment",
            "Next step in your interview process for the Data Scientist role at "
            "Globex Labs.",
            frm="no-reply@us.greenhouse-mail.io", name="Globex Labs via Greenhouse",
        )
        assert canonical_company(extract_employer(msg)[0]) == "globex labs"

    def test_labeled_company_field(self):
        msg = mk("Invitation: Technical Interview",
                 "Position: AI Engineer\nCompany: Northwind Analytics\nWhen: Oct 18",
                 frm="no-reply@goodtime.io")
        assert canonical_company(extract_employer(msg)[0]) == "northwind analytics"

    def test_line_wrapped_company_name(self):
        msg = mk("Assignment", "for the Data Scientist role at\nGlobex Labs, please complete")
        assert canonical_company(extract_employer(msg)[0]) == "globex labs"

    def test_does_not_bleed_past_sentence_end(self):
        msg = mk("Update", "the role at Northwind Analytics. We moved forward with others.",
                 frm="hr@x-corp.com")
        assert extract_employer(msg)[0] == "Northwind Analytics"

    def test_job_title_is_not_an_employer(self):
        msg = mk("Technical interview for Data Scientist at Initech Data",
                 "We would like to schedule your technical interview for the Data "
                 "Scientist position at Initech Data.", frm="talent@initechdata.com")
        assert canonical_company(extract_employer(msg)[0]) == "initech data"

    def test_unidentifiable_employer_returns_none(self):
        """Must never invent a company name."""
        msg = mk("Confidential opportunity",
                 "I am working on a confidential search for a well-funded client. "
                 "I cannot share the company name yet.",
                 frm="recruiter@talentsphere.io")
        assert extract_employer(msg)[0] is None


class TestJobTitleExtraction:
    def test_extracts_known_titles(self):
        assert extract_job_title(mk("AI Engineer role", "the AI Engineer role")) == "AI Engineer"

    def test_rejects_non_titles(self):
        msg = mk("Confidential opportunity - great fit for your background",
                 "A strong match for your background.")
        assert extract_job_title(msg) is None


class TestEventClassification:
    @pytest.mark.parametrize("subject,body,expected", [
        ("Opportunity", "I came across your profile and wanted to reach out about a role",
         EventType.RECRUITER_OUTREACH),
        ("Intro call", "I would like to schedule an initial screening call",
         EventType.SCREENING_INVITATION),
        ("Tech interview", "We'd like to schedule your technical interview",
         EventType.TECHNICAL_INTERVIEW),
        ("Final round", "We would like to invite you to the final round onsite loop",
         EventType.FINAL_INTERVIEW),
        ("HM chat", "Next is a hiring manager interview with the director",
         EventType.HIRING_MANAGER_INTERVIEW),
        ("Assignment", "Please complete the take-home assignment linked below",
         EventType.ASSESSMENT_ASSIGNED),
        ("Offer", "We are thrilled to extend an offer. Your offer letter is attached.",
         EventType.OFFER),
        ("Update", "Unfortunately, we have decided not to move forward with your application",
         EventType.REJECTION),
        ("Reschedule", "Your technical interview has been moved. Please pick a time.",
         EventType.SCHEDULING),
    ])
    def test_event_types(self, subject, body, expected):
        assert classify_message(mk(subject, body), user_email=USER).event_type == expected

    def test_candidate_authored_is_a_candidate_followup(self):
        cls = classify_message(
            mk("Thanks", "Thank you so much for your time yesterday. Following up on our "
               "conversation.", frm=USER, name="Avery", labels=("SENT",)),
            user_email=USER,
        )
        assert cls.event_type == EventType.CANDIDATE_FOLLOWUP

    def test_candidate_submission_beats_assignment_wording(self):
        cls = classify_message(
            mk("Re: Your take-home assignment",
               "I have just submitted my completed take-home assignment.",
               frm=USER, name="Avery", labels=("SENT",)),
            user_email=USER,
        )
        assert cls.event_type == EventType.ASSESSMENT_SUBMITTED

    def test_newsletter_is_not_recruitment(self):
        cls = classify_message(
            mk("Jobs you may like", "Jobs you may like this week: 12 new Data Scientist "
               "openings near you. Job alert digest.", frm="jobs@linkedin.com",
               labels=("INBOX", "CATEGORY_PROMOTIONS")),
            user_email=USER,
        )
        assert cls.event_type == EventType.NOT_RECRUITMENT


class TestOccurrence:
    def test_scheduled_not_completed(self):
        cls = classify_message(
            mk("Interview", "Your interview is scheduled for Monday, October 20 at 2:00 PM"),
            user_email=USER,
        )
        assert cls.occurrence == Occurrence.SCHEDULED

    def test_completed_requires_explicit_confirmation(self):
        cls = classify_message(
            mk("Thanks", "Thank you for your time yesterday. It was great speaking with you.",
               frm=USER, labels=("SENT",)),
            user_email=USER,
        )
        assert cls.occurrence == Occurrence.COMPLETED

    def test_canceled(self):
        cls = classify_message(
            mk("Cancelled", "Your technical interview has been cancelled."), user_email=USER
        )
        assert cls.occurrence == Occurrence.CANCELED

    def test_event_datetime_extracted(self):
        cls = classify_message(
            mk("Interview", "Your technical interview is on Thursday, October 18 at 2:00 PM"),
            user_email=USER,
        )
        assert cls.event_datetime is not None
        assert (cls.event_datetime.month, cls.event_datetime.day) == (10, 18)

    def test_no_invented_datetime(self):
        cls = classify_message(mk("Hello", "We will be in touch soon."), user_email=USER)
        assert cls.event_datetime is None
