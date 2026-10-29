"""Gmail retrieval, OAuth security properties and deep links."""

from __future__ import annotations

import time

import pytest

from interviewtrack.config import get_settings
from interviewtrack.fixtures.synthetic import gmail_messages_as_api_payloads
from interviewtrack.gmail import oauth
from interviewtrack.gmail.client import FixtureTransport, build_query, fetch_messages
from interviewtrack.gmail.links import best_message_link, message_search_link, thread_link


@pytest.fixture
def payloads():
    return gmail_messages_as_api_payloads()


class TestQuery:
    def test_excludes_spam_and_trash_and_covers_archive(self):
        q = build_query(92)
        assert "-in:spam" in q and "-in:trash" in q
        assert "in:anywhere" in q      # inbox + sent + archived
        assert "after:" in q


class TestRetrieval:
    def test_pagination(self, payloads):
        page_size = 5
        t = FixtureTransport(payloads, page_size=page_size)
        msgs, report = fetch_messages(t, get_settings())
        expected_pages = -(-len(payloads) // page_size)   # ceiling division
        assert report.pages == expected_pages
        assert report.listed == len(payloads)
        assert msgs

    def test_metadata_first_then_selective_bodies(self, payloads):
        t = FixtureTransport(payloads, page_size=100)
        _, report = fetch_messages(t, get_settings())
        assert report.metadata_fetched == len(payloads)
        assert report.bodies_fetched < report.metadata_fetched, (
            "bodies must be fetched selectively, not for every message"
        )
        metadata_calls = [c for c in t.calls if c[1] == "metadata"]
        assert len(metadata_calls) == len(payloads)

    def test_recruitment_mail_is_retained(self, payloads):
        msgs, _ = fetch_messages(FixtureTransport(payloads, page_size=8), get_settings())
        kept = {m.gmail_id for m in msgs}
        # Short confirmations and personal-domain mail must survive triage.
        assert {"m003", "m004", "m006"} <= kept

    def test_obvious_noise_is_dropped(self, payloads):
        msgs, _ = fetch_messages(FixtureTransport(payloads, page_size=8), get_settings())
        kept = {m.gmail_id for m in msgs}
        assert "m020" not in kept and "m021" not in kept

    def test_incremental_sync_skips_known_ids(self, payloads):
        """Already-stored messages are never re-fetched.

        Messages previously rejected by triage were never stored, so they are
        re-examined cheaply on the metadata pass; crucially, no redundant *body*
        fetch happens for them.
        """
        t = FixtureTransport(payloads, page_size=100)
        msgs, report1 = fetch_messages(t, get_settings())
        known = {m.gmail_id for m in msgs}
        _, report2 = fetch_messages(t, get_settings(), known_ids=known)

        assert report2.skipped_known == len(known)
        assert report2.metadata_fetched == len(payloads) - len(known)
        assert report2.bodies_fetched == 0

    def test_body_failure_keeps_the_metadata_version(self, payloads):
        t = FixtureTransport(payloads, page_size=100, fail_bodies={"m001"})
        msgs, report = fetch_messages(t, get_settings())
        assert "m001" in {m.gmail_id for m in msgs}
        assert any("m001" in e for e in report.errors)

    def test_results_are_deduplicated_and_sorted(self, payloads):
        msgs, _ = fetch_messages(FixtureTransport(payloads * 2, page_size=9), get_settings())
        ids = [m.gmail_id for m in msgs]
        assert len(ids) == len(set(ids))
        dates = [m.sort_key for m in msgs]
        assert dates == sorted(dates)

    def test_listing_failure_is_contained(self):
        class Broken:
            def list_messages(self, *a, **k):
                raise RuntimeError("network down")

            def get_message(self, *a, **k):
                raise AssertionError("should not be reached")

        msgs, report = fetch_messages(Broken(), get_settings())
        assert msgs == []
        assert report.errors


class TestOAuthSecurity:
    def test_state_is_random_and_long(self):
        a, b = oauth.new_pending_auth(), oauth.new_pending_auth()
        assert a.state != b.state
        assert len(a.state) >= 32

    def test_state_verification_accepts_matching_state(self):
        p = oauth.new_pending_auth()
        oauth.verify_state(p.state, p)  # must not raise

    def test_state_mismatch_is_rejected(self):
        p = oauth.new_pending_auth()
        with pytest.raises(oauth.OAuthError, match="mismatch"):
            oauth.verify_state("attacker-value", p)

    def test_missing_state_is_rejected(self):
        with pytest.raises(oauth.OAuthError):
            oauth.verify_state(None, oauth.new_pending_auth())

    def test_absent_pending_auth_is_rejected(self):
        with pytest.raises(oauth.OAuthError):
            oauth.verify_state("anything", None)

    def test_expired_pending_auth_is_rejected(self):
        p = oauth.new_pending_auth()
        p.created_at = time.time() - 10_000
        with pytest.raises(oauth.OAuthError, match="expired"):
            oauth.verify_state(p.state, p)

    def test_pkce_challenge_is_s256_of_the_verifier(self):
        import base64
        import hashlib

        p = oauth.new_pending_auth()
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(p.code_verifier.encode()).digest()
        ).decode().rstrip("=")
        assert oauth.code_challenge_for(p.code_verifier) == expected

    def test_only_readonly_gmail_scope_is_requested(self):
        scopes = get_settings().google_scopes
        assert "https://www.googleapis.com/auth/gmail.readonly" in scopes
        joined = " ".join(scopes)
        for forbidden in ("gmail.modify", "gmail.send", "gmail.compose", "mail.google.com"):
            assert forbidden not in joined

    def test_write_capable_token_is_refused(self):
        with pytest.raises(oauth.OAuthError):
            oauth._assert_readonly_scope(
                "openid https://www.googleapis.com/auth/gmail.modify"
            )

    def test_authorization_url_carries_security_params(self, monkeypatch):
        import urllib.parse as up

        monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid.apps.googleusercontent.com")
        monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
        settings = get_settings(refresh=True)
        p = oauth.new_pending_auth()
        params = up.parse_qs(up.urlparse(oauth.build_authorization_url(p, settings)).query)

        assert params["state"] == [p.state]
        assert params["code_challenge_method"] == ["S256"]
        assert params["access_type"] == ["offline"]
        assert "gmail.readonly" in params["scope"][0]
        get_settings(refresh=True)

    def test_unconfigured_oauth_raises_a_clear_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "")
        monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "")
        settings = get_settings(refresh=True)
        with pytest.raises(oauth.OAuthError, match="not configured"):
            oauth.build_authorization_url(oauth.new_pending_auth(), settings)
        get_settings(refresh=True)

    def test_expiry_detection(self):
        assert oauth.is_expired({}) is True
        assert oauth.is_expired({"obtained_at": time.time(), "expires_in": 3600}) is False
        assert oauth.is_expired({"obtained_at": time.time() - 4000, "expires_in": 3600}) is True

    def test_refresh_without_refresh_token_raises(self):
        with pytest.raises(oauth.OAuthError, match="refresh token"):
            oauth.refresh_token({"access_token": "a"})


class TestGmailLinks:
    def test_thread_link(self):
        assert thread_link("t103").endswith("#all/t103")

    def test_message_search_link_uses_rfc822msgid(self):
        link = message_search_link("<abc@mail.example>")
        assert "rfc822msgid" in link
        assert "%3C" not in link  # angle brackets stripped before quoting

    def test_best_link_prefers_message_id(self):
        assert "rfc822msgid" in best_message_link("<a@b>", "t1")

    def test_best_link_falls_back_to_thread(self):
        assert best_message_link(None, "t1").endswith("#all/t1")

    def test_best_link_handles_nothing(self):
        assert best_message_link(None, None).startswith("https://mail.google.com")
