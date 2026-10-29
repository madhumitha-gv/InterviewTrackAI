"""Google OAuth 2.0 (Web application) — the only sign-in method.

Security properties:
  * `openid email profile` + `gmail.readonly` requested in ONE consent screen, so
    login and Gmail authorization happen in a single onboarding flow.
  * CSRF-resistant `state`, generated with `secrets` and verified on callback.
  * PKCE (S256) code challenge/verifier.
  * Read-only Gmail scope. No write/modify/delete scope is ever requested.
  * Tokens are stored per-user in SQLite, never in the URL or page state.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"


class OAuthError(RuntimeError):
    """Raised for any OAuth failure (misconfiguration, state mismatch, HTTP error)."""


@dataclass
class PendingAuth:
    """Server-side state for one in-flight authorization attempt."""

    state: str
    code_verifier: str
    created_at: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: int = 600) -> bool:
        return (time.time() - self.created_at) > ttl_seconds


def _b64url_nopad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def new_pending_auth() -> PendingAuth:
    """Generate fresh CSRF state and a PKCE verifier."""
    return PendingAuth(
        state=_b64url_nopad(secrets.token_bytes(32)),
        code_verifier=_b64url_nopad(secrets.token_bytes(64)),
    )


def code_challenge_for(verifier: str) -> str:
    """PKCE S256 challenge."""
    return _b64url_nopad(hashlib.sha256(verifier.encode("ascii")).digest())


def build_authorization_url(pending: PendingAuth, settings: Settings | None = None) -> str:
    """Build the Google consent URL for the single sign-in + Gmail authorization."""
    settings = settings or get_settings()
    if not settings.google_configured:
        raise OAuthError(
            "Google OAuth is not configured. Set GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET (see README -> Google Cloud setup)."
        )
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.oauth_redirect_uri,
        "response_type": "code",
        "scope": " ".join(settings.google_scopes),
        "state": pending.state,
        "code_challenge": code_challenge_for(pending.code_verifier),
        "code_challenge_method": "S256",
        "access_type": "offline",     # we need a refresh token
        "prompt": "consent",          # ensure a refresh token is issued
        "include_granted_scopes": "true",
    }
    return f"{AUTH_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _post_form(url: str, data: dict[str, str], timeout: int = 30) -> dict[str, Any]:
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise OAuthError(f"Google returned HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise OAuthError(f"Token request failed: {exc}") from exc


def verify_state(returned_state: str | None, pending: PendingAuth | None) -> None:
    """Validate the OAuth `state` parameter. Raises on any mismatch."""
    if pending is None:
        raise OAuthError("No pending authorization for this session. Start sign-in again.")
    if pending.is_expired():
        raise OAuthError("Authorization request expired. Start sign-in again.")
    if not returned_state:
        raise OAuthError("Missing OAuth state parameter.")
    # Constant-time comparison.
    if not secrets.compare_digest(str(returned_state), pending.state):
        raise OAuthError("OAuth state mismatch — possible CSRF. Authorization rejected.")


def exchange_code(code: str, pending: PendingAuth,
                  settings: Settings | None = None) -> dict[str, Any]:
    """Exchange an authorization code for tokens, sending the PKCE verifier."""
    settings = settings or get_settings()
    token = _post_form(
        TOKEN_ENDPOINT,
        {
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": settings.oauth_redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": pending.code_verifier,
        },
    )
    if "access_token" not in token:
        raise OAuthError(f"Token response missing access_token: {sorted(token)}")
    token["obtained_at"] = time.time()
    _assert_readonly_scope(token.get("scope", ""))
    return token


def refresh_token(token: dict[str, Any],
                  settings: Settings | None = None) -> dict[str, Any]:
    """Refresh an expired access token, preserving the refresh token."""
    settings = settings or get_settings()
    refresh = token.get("refresh_token")
    if not refresh:
        raise OAuthError("No refresh token stored; the user must sign in again.")
    fresh = _post_form(
        TOKEN_ENDPOINT,
        {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "refresh_token": refresh,
            "grant_type": "refresh_token",
        },
    )
    merged = {**token, **fresh, "obtained_at": time.time()}
    merged["refresh_token"] = refresh  # Google omits it on refresh responses
    return merged


def _assert_readonly_scope(granted: str) -> None:
    """Fail loudly if a write-capable Gmail scope was somehow granted."""
    forbidden = ("gmail.modify", "gmail.send", "gmail.compose", "mail.google.com")
    for bad in forbidden:
        if bad in granted:
            raise OAuthError(
                f"Refusing a token with write-capable Gmail scope ({bad}). "
                "InterviewTrack AI requires read-only access only."
            )


def is_expired(token: dict[str, Any], skew_seconds: int = 120) -> bool:
    obtained = float(token.get("obtained_at") or 0)
    expires_in = float(token.get("expires_in") or 0)
    if not obtained or not expires_in:
        return True
    return (time.time() - obtained) > (expires_in - skew_seconds)


def ensure_fresh(token: dict[str, Any],
                 settings: Settings | None = None) -> tuple[dict[str, Any], bool]:
    """Return (token, was_refreshed), refreshing only when needed."""
    if is_expired(token):
        return refresh_token(token, settings), True
    return token, False


def fetch_userinfo(token: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    """Fetch the OpenID Connect profile (sub, email, name, picture)."""
    req = urllib.request.Request(
        USERINFO_ENDPOINT,
        headers={"Authorization": f"Bearer {token.get('access_token', '')}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise OAuthError(f"userinfo failed: HTTP {exc.code}") from exc
    except Exception as exc:
        raise OAuthError(f"userinfo failed: {exc}") from exc


def revoke(token: dict[str, Any]) -> bool:
    """Best-effort token revocation on logout."""
    target = token.get("refresh_token") or token.get("access_token")
    if not target:
        return False
    try:
        _post_form(REVOKE_ENDPOINT, {"token": target}, timeout=15)
        return True
    except OAuthError:
        logger.info("Token revocation failed (already revoked or network issue).")
        return False


def to_google_credentials(token: dict[str, Any], settings: Settings | None = None):
    """Build a `google.oauth2.credentials.Credentials` for the Gmail API client."""
    settings = settings or get_settings()
    try:
        from google.oauth2.credentials import Credentials  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise OAuthError("google-auth is not installed; run pip install -r requirements.txt") from exc
    return Credentials(
        token=token.get("access_token"),
        refresh_token=token.get("refresh_token"),
        token_uri=TOKEN_ENDPOINT,
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        scopes=settings.google_scopes,
    )
