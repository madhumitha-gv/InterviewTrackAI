"""Patterns taken from real recruiting mail (reproduced as fictional analogues).

Each test corresponds to a structure observed in a real inbox. Several of these
failed when first written — see docs/email-templates.md for what each one exposed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from interviewtrack.contacts import classify_contact
from interviewtrack.enums import ContactType
from interviewtrack.fixtures.realistic import (
    REALISTIC_EXPECTED,
    REALISTIC_MESSAGES,
    REALISTIC_MUST_MERGE,
    REALISTIC_MUST_NOT_MERGE,
    USER_EMAIL,
)
from interviewtrack.normalize import company_from_domain, registrable_domain
from interviewtrack.reconstruct import reconstruct

NOW = datetime(2025, 11, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def built(backend, settings):
    return reconstruct(REALISTIC_MESSAGES, user_email=USER_EMAIL, backend=backend,
                       settings=settings, now=NOW)


@pytest.fixture(scope="module")
def placed(built):
    return {e.gmail_id: a.key for a in built.applications for e in a.events}


def app_of(built, gmail_id):
    return next((a for a in built.applications
                 if any(e.gmail_id == gmail_id for e in a.events)), None)


class TestVendorSubdomain:
    """A vendor sending from a subdomain of the employer is not a new company."""

    @pytest.mark.parametrize("domain,expected", [
        ("xwv.northvale.com", "northvale"),
        ("xwf.example.com", "example"),
        ("careers.northwind.com", "northwind"),
        ("talent.sub.globex.co.uk", "globex"),
        ("northwind.com", "northwind"),
    ])
    def test_registrable_domain_ignores_subdomains(self, domain, expected):
        assert registrable_domain(domain) == expected
        assert company_from_domain(domain) == expected

    def test_vendor_staff_do_not_create_a_separate_employer(self, built):
        employers = {a.employer_canonical for a in built.applications}
        assert not any("xwv" in e for e in employers), (
            "a vendor subdomain prefix must never be read as the employer"
        )


class TestRecruiterHandoff:
    def test_handoff_by_cc_keeps_one_process(self, placed):
        assert placed["r003"] == placed["r010"]

    def test_all_three_staff_are_linked_to_one_application(self, built):
        app = app_of(built, "r003")
        emails = {c.email for c in app.contacts}
        assert {"reyesa@xwv.northvale.com", "pnandan@xwv.northvale.com",
                "kmehta@xwv.northvale.com"} <= emails

    def test_coordinator_is_typed_distinctly(self, built):
        app = app_of(built, "r004")
        contact = app.contact_for("pnandan@xwv.northvale.com")
        assert contact.contact_type == ContactType.COORDINATOR


class TestAtsRelayThenCorporateAddress:
    """One recruiter writes via an ATS relay, then from their own address."""

    def test_both_addresses_land_in_one_application(self, placed):
        assert placed["r011"] == placed["r013"]

    def test_relay_is_typed_as_automated(self):
        assert classify_contact("brightline@myworkday.com", "Talent Acquisition",
                                "") == ContactType.ATS_NOREPLY

    def test_corporate_address_is_typed_as_human(self):
        assert classify_contact(
            "dale.renner@brightlineenergy.com", "Renner, Dale",
            "Dale Renner, Recruiter, Talent Acquisition",
        ) == ContactType.INTERNAL_RECRUITER


class TestCandidateReplyLinksThroughRecipient:
    """A candidate's own reply names the role but not the employer."""

    def test_reschedule_thread_joins_the_right_process(self, placed):
        assert placed["r005"] == placed["r003"], (
            "a reply addressed to a known contact must not start a new application"
        )

    def test_the_whole_reschedule_conversation_stays_together(self, placed):
        for gid in ("r005", "r006", "r007"):
            assert placed[gid] == placed["r003"]


class TestRoleDrift:
    def test_vague_then_specific_role_is_one_process(self, placed):
        assert placed["r001"] == placed["r003"]


class TestBoilerplateStripping:
    def test_legal_footers_are_not_stored_as_evidence(self, built):
        for a in built.applications:
            for e in a.events:
                assert "received this communication by mistake" not in e.preview
                assert "accommodations@" not in e.preview

    def test_inline_quoted_headers_are_stripped(self, built):
        event = next(e for a in built.applications for e in a.events
                     if e.gmail_id == "r013")
        assert "Sent: Monday, October 26" not in event.preview


class TestProcessSeparation:
    @pytest.mark.parametrize("a,b,why", REALISTIC_MUST_NOT_MERGE,
                             ids=[m[2][:40] for m in REALISTIC_MUST_NOT_MERGE])
    def test_distinct_employers_stay_separate(self, placed, a, b, why):
        assert placed.get(a) != placed.get(b) or placed.get(a) is None, why

    @pytest.mark.parametrize("a,b,why", REALISTIC_MUST_MERGE,
                             ids=[m[2][:40] for m in REALISTIC_MUST_MERGE])
    def test_one_process_stays_together(self, placed, a, b, why):
        assert placed.get(a) is not None and placed[a] == placed.get(b), why

    def test_every_message_is_placed(self, placed):
        missing = [g for g in REALISTIC_EXPECTED if g not in placed]
        assert not missing, f"unplaced: {missing}"

    def test_three_distinct_processes(self, built):
        assert len(built.applications) == 3
