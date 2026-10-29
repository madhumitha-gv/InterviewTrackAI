"""Regression tests for the second mock inbox (fixtures/mock_inbox.py).

A fresh set of fictional companies, people and addresses, modelling real recruiting
structures with no overlap with the other fixture sets.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from interviewtrack.enums import ApplicationStatus, ContactType
from interviewtrack.fixtures.mock_inbox import (
    MOCK_EXCLUDED,
    MOCK_EXPECTED,
    MOCK_MESSAGES,
    MOCK_MUST_MERGE,
    MOCK_MUST_NOT_MERGE,
    USER_EMAIL,
)
from interviewtrack.reconstruct import reconstruct

NOW = datetime(2025, 11, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def built(backend, settings):
    return reconstruct(MOCK_MESSAGES, user_email=USER_EMAIL, backend=backend,
                       settings=settings, now=NOW)


@pytest.fixture(scope="module")
def placed(built):
    return {e.gmail_id: a.key for a in built.applications for e in a.events}


def app_of(built, gmail_id):
    return next((a for a in built.applications
                 if any(e.gmail_id == gmail_id for e in a.events)), None)


class TestStructure:
    def test_four_distinct_processes(self, built):
        assert len(built.applications) == 4

    def test_noise_is_excluded(self, built):
        for gid in MOCK_EXCLUDED:
            assert gid in built.skipped_non_recruitment

    def test_every_recruitment_message_is_placed(self, placed):
        missing = [g for g, v in MOCK_EXPECTED.items() if v and g not in placed]
        assert not missing, f"unplaced: {missing}"

    @pytest.mark.parametrize("a,b,why", MOCK_MUST_MERGE, ids=[m[2][:38] for m in MOCK_MUST_MERGE])
    def test_must_merge(self, placed, a, b, why):
        assert placed.get(a) is not None and placed[a] == placed.get(b), why

    @pytest.mark.parametrize("a,b,why", MOCK_MUST_NOT_MERGE, ids=[m[2][:38] for m in MOCK_MUST_NOT_MERGE])
    def test_must_not_merge(self, placed, a, b, why):
        assert placed.get(a) != placed.get(b) or placed.get(a) is None, why


class TestExtraction:
    def test_roles_are_extracted_correctly(self, built):
        roles = {a.employer_canonical: a.job_title for a in built.applications}
        assert roles.get("helix dynamics") == "Machine Learning Engineer"
        assert roles.get("ardent financial") == "Quantitative Analyst"
        assert roles.get("caldera robotics") == "Perception Engineer"

    def test_vendor_subdomain_is_not_the_employer(self, built):
        assert not any("ext" in a.employer_canonical.split()
                       for a in built.applications)
        assert any(a.employer_canonical == "ardent financial" for a in built.applications)

    def test_confidential_agency_does_not_become_the_employer(self, built):
        """An agency hiding its client names no employer — the agency is not it."""
        employers = {a.employer_canonical for a in built.applications}
        assert "pikearrowsearch" not in employers
        # Once the agency names the real client, that becomes the employer.
        assert "caldera robotics" in employers


class TestStatus:
    def test_rejection_closes_only_its_process(self, built):
        lumenwave = next(a for a in built.applications
                         if a.employer_canonical == "lumenwave health")
        assert lumenwave.status == ApplicationStatus.REJECTED
        for a in built.applications:
            if a.employer_canonical != "lumenwave health":
                assert a.status != ApplicationStatus.REJECTED

    def test_identified_employer_clears_a_stale_review_flag(self, built):
        """Helix's first message (ATS relay) had no employer; a later one supplied it."""
        helix = next(a for a in built.applications
                     if a.employer_canonical == "helix dynamics")
        assert helix.needs_review is False, (
            "once the employer is identified, the 'employer unknown' review must clear"
        )


class TestContactTyping:
    def test_handoff_and_coordinator_are_typed(self, built):
        ardent = next(a for a in built.applications
                      if a.employer_canonical == "ardent financial")
        by_email = {c.email: c.contact_type for c in ardent.contacts}
        assert by_email["s.varga@ext.ardentfinancial.com"] == ContactType.COORDINATOR
        assert "m.delgado@ext.ardentfinancial.com" in by_email

    def test_ats_relay_is_automated(self, built):
        helix = next(a for a in built.applications
                     if a.employer_canonical == "helix dynamics")
        relay = helix.contact_for("helixdynamics@myworkday.com")
        assert relay is not None and relay.contact_type == ContactType.ATS_NOREPLY
