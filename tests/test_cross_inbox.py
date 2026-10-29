"""Cross-inbox generalization: the same code on three distinct mailbox profiles.

Acceptance criterion (from the task): the same application code processes multiple
mailbox profiles using only account-specific data, with NO changes to hardcoded
recruiter / domain / company / role lists. The profiles in fixtures/mailboxes.py use
companies, domains and labels absent from every list, so passing proves generality
rather than memorization.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from interviewtrack.config import get_settings
from interviewtrack.fixtures.mailboxes import (
    ALL_PROFILES,
    PROFILE_SPARSE,
    PROFILE_UNFAMILIAR,
)
from interviewtrack.fixtures.synthetic import gmail_messages_as_api_payloads
from interviewtrack.gmail.client import FixtureTransport, fetch_messages
from interviewtrack.gmail.desktop_oauth import (
    stored_accounts,
    token_path_for,
    tokens_dir,
)
from interviewtrack.reconstruct import reconstruct

NOW = datetime(2025, 11, 1, tzinfo=timezone.utc)


def _run(profile, backend, settings):
    """The full pipeline for one profile: retrieval → parse → classify → reconstruct."""
    payloads = gmail_messages_as_api_payloads(profile.messages)
    messages, report = fetch_messages(
        FixtureTransport(payloads, page_size=4), settings
    )
    result = reconstruct(messages, user_email=profile.user_email, backend=backend,
                         settings=settings, now=NOW)
    placed = {e.gmail_id: a.key for a in result.applications for e in a.events}
    return result, placed, report


@pytest.mark.parametrize("profile", ALL_PROFILES, ids=[p.key for p in ALL_PROFILES])
class TestEachProfileEndToEnd:
    def test_retrieval_is_not_domain_filtered(self, profile, settings):
        """Every eligible message is listed regardless of sender domain."""
        payloads = gmail_messages_as_api_payloads(profile.messages)
        _, report = fetch_messages(FixtureTransport(payloads, page_size=4), settings)
        assert report.listed == len(profile.messages)

    def test_detection_recall(self, profile, backend, settings):
        _, placed, _ = _run(profile, backend, settings)
        missed = [g for g in profile.recruitment_ids if g not in placed]
        assert not missed, f"{profile.key}: missed recruitment {missed}"

    def test_noise_is_excluded(self, profile, backend, settings):
        _, placed, _ = _run(profile, backend, settings)
        leaked = [g for g in profile.excluded_ids if g in placed]
        assert not leaked, f"{profile.key}: noise leaked into applications {leaked}"

    def test_expected_application_count(self, profile, backend, settings):
        result, _, _ = _run(profile, backend, settings)
        assert len(result.applications) == profile.expected_applications

    def test_required_merges(self, profile, backend, settings):
        _, placed, _ = _run(profile, backend, settings)
        for a, b, why in profile.must_merge:
            assert placed.get(a) is not None and placed[a] == placed.get(b), why

    def test_required_separations(self, profile, backend, settings):
        _, placed, _ = _run(profile, backend, settings)
        for a, b, why in profile.must_not_merge:
            assert placed.get(a) != placed.get(b) or placed.get(a) is None, why

    def test_every_event_carries_evidence(self, profile, backend, settings):
        result, _, _ = _run(profile, backend, settings)
        for app in result.applications:
            for ev in app.events:
                assert ev.evidence_ref is not None
                assert ev.evidence_ref.gmail_url


class TestUnfamiliarInbox:
    """Unknown agency, unusual TLDs, custom labels, a malformed message."""

    def _run(self, backend, settings):
        return _run(PROFILE_UNFAMILIAR, backend, settings)

    def test_unknown_agency_is_not_the_employer(self, backend, settings):
        result, _, _ = self._run(backend, settings)
        employers = {a.employer_canonical for a in result.applications}
        assert "bridgepoint hunt" not in employers
        assert "vireo analytics" in employers   # the named client is the employer

    def test_unusual_tld_employer_is_extracted(self, backend, settings):
        result, _, _ = self._run(backend, settings)
        assert any(a.employer_canonical == "vireo analytics"
                   for a in result.applications)

    def test_custom_labels_do_not_break_triage(self, backend, settings):
        """Nothing keys off CATEGORY_* label names."""
        _, placed, _ = self._run(backend, settings)
        assert placed.get("u01") is not None   # arrived under Label_12

    def test_malformed_message_handled_not_dropped(self, backend, settings):
        result, placed, _ = self._run(backend, settings)
        # u05 has broken HTML, no plain text — it must still be parsed and placed,
        # and its script contents stripped.
        assert placed.get("u05") is not None
        event = next(e for a in result.applications for e in a.events
                     if e.gmail_id == "u05")
        assert "junk()" not in event.preview


class TestSparseInbox:
    """~4 recruitment messages among 10 noise messages: precision + recall."""

    def test_high_precision_no_noise(self, backend, settings):
        _, placed, _ = _run(PROFILE_SPARSE, backend, settings)
        leaked = [g for g in PROFILE_SPARSE.excluded_ids if g in placed]
        assert not leaked

    def test_sparse_process_fully_recovered(self, backend, settings):
        result, placed, _ = _run(PROFILE_SPARSE, backend, settings)
        assert len(result.applications) == 1
        app = result.applications[0]
        assert {e.gmail_id for e in app.events} == {"s01", "s02", "s03", "s04"}

    def test_thin_reply_on_known_thread_is_kept(self, backend, settings):
        """s02 is a bare 'yes, Tuesday works' — rescued by thread continuity."""
        _, placed, _ = _run(PROFILE_SPARSE, backend, settings)
        assert placed.get("s02") is not None


class TestAccountIsolation:
    """Multiple accounts: independent tokens and data, no shared mutable file."""

    def test_tokens_are_per_account(self, monkeypatch, tmp_path):
        monkeypatch.setenv("INTERVIEWTRACK_DATA_DIR", str(tmp_path))
        settings = get_settings(refresh=True)
        d = tokens_dir(settings)
        d.mkdir(parents=True, exist_ok=True)
        for acct in ("alice@gmail.com", "bob@company.io"):
            token_path_for(acct, settings).write_text(
                json.dumps({"token": "x", "account_email": acct, "refresh_token": "r"})
            )
        assert token_path_for("alice@gmail.com", settings) != token_path_for(
            "bob@company.io", settings
        )
        assert set(stored_accounts(settings)) == {"alice@gmail.com", "bob@company.io"}
        get_settings(refresh=True)

    def test_database_rows_are_per_user(self, db):
        from interviewtrack.fixtures.mailboxes import PROFILE_CONVENTIONAL as P
        db.upsert_user("gmail:alice@gmail.com", "alice@gmail.com")
        db.upsert_user("gmail:bob@company.io", "bob@company.io")
        db.upsert_messages("gmail:alice@gmail.com", P.messages)
        assert len(db.load_messages("gmail:alice@gmail.com")) == len(P.messages)
        assert db.load_messages("gmail:bob@company.io") == []

    def test_sanitized_account_filenames(self):
        from interviewtrack.gmail.desktop_oauth import _sanitize_account
        assert _sanitize_account("Foo.Bar+tag@Gmail.com") == "foo.bar_tag@gmail.com"
        assert _sanitize_account("") == "default"


class TestNoHardcodedListsWereEdited:
    """Guard: the profiles' companies/domains must be ABSENT from the engine lists."""

    def test_profile_domains_are_not_in_known_lists(self):
        from interviewtrack.normalize import (
            ATS_DOMAINS,
            SCHEDULING_DOMAINS,
            STAFFING_DOMAINS,
        )
        # The deliberately-unknown agency and employers must not be enumerated.
        for d in ("bridgepointhunt.io", "vireo.analytics", "meet.vireo.analytics",
                  "novagridenergy.com", "meridianpathhealth.com", "quanta.engineering"):
            assert d not in STAFFING_DOMAINS
            assert d not in ATS_DOMAINS
            assert d not in SCHEDULING_DOMAINS
