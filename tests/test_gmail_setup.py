"""Offline tests for local Gmail Mode setup: expected-account, verification, smoke.

Live OAuth is NOT tested here — it requires a browser and real credentials, and is the
user's own step. These cover everything that can be verified without a live account.
"""

from __future__ import annotations

import pytest

from interviewtrack.config import get_settings
from interviewtrack.gmail.desktop_oauth import (
    SCOPES,
    _sanitize_account,
    assert_readonly,
    token_path_for,
    verify_account,
)


class TestExpectedAccountConfig:
    def test_default_is_empty(self, monkeypatch):
        monkeypatch.delenv("GMAIL_EXPECTED_ACCOUNT", raising=False)
        assert get_settings(refresh=True).gmail_expected_account == ""
        get_settings(refresh=True)

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("GMAIL_EXPECTED_ACCOUNT", "person@gmail.com")
        assert get_settings(refresh=True).gmail_expected_account == "person@gmail.com"
        get_settings(refresh=True)

    def test_env_example_has_no_real_address(self):
        text = open(".env.example").read().lower()
        assert "gannavaram" not in text, "no real email may appear in .env.example"
        assert "gmail_expected_account=you@gmail.com" in text.replace(" ", "") \
            or "gmail_expected_account=you@" in text, "generic placeholder expected"

    def test_no_real_email_hardcoded_in_source(self):
        """The real address must live only in local .env, never in the code."""
        import pathlib

        for py in pathlib.Path("interviewtrack").rglob("*.py"):
            assert "gannavaram" not in py.read_text().lower(), f"{py} hardcodes the email"


class TestAccountVerification:
    def test_match(self):
        ok, msg = verify_account("a@x.com", "a@x.com")
        assert ok and "matches" in msg

    def test_case_insensitive(self):
        ok, _ = verify_account("A@X.com", "a@x.com")
        assert ok

    def test_mismatch_warns_does_not_swap(self):
        ok, msg = verify_account("a@x.com", "b@y.com")
        assert not ok
        assert "MISMATCH" in msg
        assert "No account was swapped" in msg

    def test_no_expected_account_is_permissive(self):
        ok, msg = verify_account("a@x.com", "")
        assert ok and "no expected account" in msg.lower()

    def test_unreadable_connected_with_expected(self):
        ok, msg = verify_account(None, "a@x.com")
        assert not ok


class TestReadonlyAndPaths:
    def test_only_readonly_scope(self):
        joined = " ".join(SCOPES)
        assert "gmail.readonly" in joined
        for bad in ("gmail.modify", "gmail.send", "gmail.compose", "mail.google.com"):
            assert bad not in joined

    def test_write_scope_refused(self):
        from interviewtrack.gmail.desktop_oauth import AuthError

        with pytest.raises(AuthError):
            assert_readonly("openid https://www.googleapis.com/auth/gmail.modify")

    def test_tokens_are_per_account(self):
        assert token_path_for("a@x.com") != token_path_for("b@y.com")
        assert token_path_for("a@x.com").name == "a@x.com.json"

    def test_sanitize(self):
        assert _sanitize_account("A@X.com") == "a@x.com"
        assert "/" not in _sanitize_account("a/b@x.com")


class TestSmokeIsPrivacySafe:
    """The smoke test must report counts only — never email content."""

    def test_smoke_output_contains_no_content(self, capsys, monkeypatch, tmp_path):
        from interviewtrack.fixtures.mock_inbox import MOCK_MESSAGES, USER_EMAIL
        from interviewtrack.fixtures.synthetic import gmail_messages_as_api_payloads
        from interviewtrack.gmail.client import FixtureTransport, fetch_messages
        import interviewtrack.gmail_smoke as smoke

        # Stub the live-only pieces so run_smoke uses a fixture transport offline.
        monkeypatch.setattr(smoke, "__name__", smoke.__name__)
        transport = FixtureTransport(
            gmail_messages_as_api_payloads(MOCK_MESSAGES), page_size=4
        )

        import interviewtrack.gmail.desktop_oauth as oauth

        monkeypatch.setattr(oauth, "load_credentials", lambda *a, **k: object())
        monkeypatch.setattr(oauth, "account_email", lambda *a, **k: USER_EMAIL)
        monkeypatch.setattr(
            "interviewtrack.gmail.client.GoogleApiTransport",
            lambda *a, **k: transport,
        )

        rc = smoke.run_smoke(settings=get_settings(refresh=True))
        out = capsys.readouterr().out
        assert rc == 0
        # counts present
        assert "Listed" in out and "Applications" in out
        # NO content: company names, candidate address, roles, subjects
        for leak in ("Helix", "Ardent", "Lumenwave", "Caldera", "Machine Learning",
                     "Perception", "jordan.lee"):
            assert leak not in out, f"content leaked into smoke output: {leak!r}"
        assert "Safe to share" in out
        get_settings(refresh=True)
