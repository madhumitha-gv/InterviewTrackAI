"""Desktop OAuth for single-user Gmail Mode.

Why a Desktop (installed-app) client rather than a Web client:

  * Authorization runs **once, from the terminal**, as its own command. Streamlit
    reruns its whole script on every interaction, so a web redirect flow driven from
    the UI would re-trigger OAuth constantly.
  * No redirect URI to register or keep in sync — the installed-app flow spins up a
    temporary local server.

Scope is `gmail.readonly` and nothing else. A token carrying any write-capable Gmail
scope is refused outright.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Read-only. No modify, send, compose or full-mailbox scope is ever requested.
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
]

FORBIDDEN_SCOPE_FRAGMENTS = (
    "gmail.modify", "gmail.send", "gmail.compose", "gmail.settings",
    "mail.google.com",
)


class AuthError(RuntimeError):
    """Raised for any authorization failure, with a message safe to show a user."""


def client_secret_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return Path(settings.client_secret_file)


def _sanitize_account(account_email: str) -> str:
    """Filesystem-safe filename for an account email."""
    return re.sub(r"[^a-z0-9._@-]+", "_", (account_email or "").strip().lower()) or "default"


def tokens_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return Path(settings.tokens_dir)


def token_path_for(account_email: str | None, settings: Settings | None = None) -> Path:
    """The token file for one account.

    With an account email, returns a per-account path under tokens_dir, so multiple
    accounts never share a token. Without one, falls back to the legacy single file
    (one-account setup).
    """
    settings = settings or get_settings()
    if account_email:
        return tokens_dir(settings) / f"{_sanitize_account(account_email)}.json"
    legacy = Path(settings.token_file)
    if legacy.exists():
        return legacy
    candidates = sorted(tokens_dir(settings).glob("*.json"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise AuthError("Multiple accounts are authorized. Select one with --account EMAIL.")
    return legacy


def token_path(settings: Settings | None = None) -> Path:
    """Legacy single-account token path (one-account setup)."""
    settings = settings or get_settings()
    return Path(settings.token_file)


def stored_accounts(settings: Settings | None = None) -> list[str]:
    """Every account that currently has a stored token."""
    settings = settings or get_settings()
    out: list[str] = []
    d = tokens_dir(settings)
    if d.exists():
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text())
                out.append(data.get("account_email") or f.stem)
            except (json.JSONDecodeError, OSError):
                out.append(f.stem)
    if Path(settings.token_file).exists():
        out.append("(legacy single-account token)")
    return out


def verify_account(connected: str | None, expected: str | None) -> tuple[bool, str]:
    """Compare the connected account to the expected one (when configured).

    Returns (ok, message). A mismatch is a *warning*, not a hard failure: the user may
    legitimately authorize a different account, but they must be told rather than have
    one silently swapped for another.
    """
    conn = (connected or "").strip().lower()
    exp = (expected or "").strip().lower()
    if not exp:
        return True, f"Connected account: {connected or '(unknown)'} (no expected account set)."
    if not conn:
        return False, f"Expected {expected}, but the connected account could not be read."
    if conn == exp:
        return True, f"Connected account matches the expected account ({connected})."
    return False, (
        f"ACCOUNT MISMATCH: connected as {connected}, but GMAIL_EXPECTED_ACCOUNT is "
        f"{expected}. No account was swapped — re-authorize with the right account, or "
        f"update GMAIL_EXPECTED_ACCOUNT in .env."
    )


def assert_readonly(scopes: list[str] | str) -> None:
    """Refuse any token that can modify the mailbox."""
    granted = " ".join(scopes) if isinstance(scopes, list) else str(scopes)
    for bad in FORBIDDEN_SCOPE_FRAGMENTS:
        if bad in granted:
            raise AuthError(
                f"Refusing a token with write-capable Gmail scope ({bad}). "
                "InterviewTrack AI requires read-only access only."
            )


def has_credentials(settings: Settings | None = None) -> bool:
    return client_secret_path(settings).exists()


def has_token(account_email: str | None = None, settings: Settings | None = None) -> bool:
    return token_path_for(account_email, settings).exists()


def authorize(settings: Settings | None = None, port: int = 0, account: str | None = None) -> dict[str, Any]:
    """Run the installed-app flow and persist the token. Interactive."""
    settings = settings or get_settings()
    secret = client_secret_path(settings)
    if not secret.exists():
        raise AuthError(
            f"No OAuth client file at {secret}.\n"
            "Create a **Desktop app** OAuth client in Google Cloud Console "
            "(APIs & Services -> Credentials), download the JSON, and save it there. "
            "See README -> Gmail Mode setup."
        )
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: PLC0415
    except ImportError as exc:
        raise AuthError(
            "google-auth-oauthlib is not installed. Run "
            "`pip install -r requirements.txt`."
        ) from exc

    flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES, autogenerate_code_verifier=True)
    # port=0 picks a free port; the flow opens a browser and serves the redirect.
    creds = flow.run_local_server(port=port, prompt="consent", **({"login_hint": account} if account else {}))
    assert_readonly(list(creds.scopes or []))

    # Learn which account this token is for, so it is stored under its own name.
    email = _account_email_from_creds(creds)
    if not email:
        raise AuthError("Google account identity could not be verified. No token was saved; retry authorization.")
    if account and email.lower() != account.strip().lower():
        raise AuthError("The authorized account differs from --account. No token was replaced.")
    token_file = token_path_for(email, settings)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(creds.to_json())
    payload["account_email"] = email
    token_file.write_text(json.dumps(payload))
    try:
        token_file.chmod(0o600)   # owner-only; the token grants mailbox read access
    except OSError:
        pass
    logger.info("Authorization for %s stored at %s", email, token_file)
    return payload


def load_credentials(account_email: str | None = None,
                     settings: Settings | None = None):
    """Load an account's stored credentials, refreshing when expired.

    Raises AuthError with actionable text when authorization is missing — it never
    falls back to synthetic data.
    """
    settings = settings or get_settings()
    token_file = token_path_for(account_email, settings)
    if not token_file.exists():
        hint = f" --account {account_email}" if account_email else ""
        raise AuthError(
            f"Not authorized yet (no token at {token_file}).\n"
            f"Run:  python -m interviewtrack.auth{hint}"
        )
    try:
        from google.auth.transport.requests import Request  # noqa: PLC0415
        from google.oauth2.credentials import Credentials  # noqa: PLC0415
    except ImportError as exc:
        raise AuthError("google-auth is not installed.") from exc

    try:
        payload = json.loads(token_file.read_text())
        assert_readonly(payload.get("scopes") or [])
        creds = Credentials.from_authorized_user_info(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthError(
            f"Stored token at {token_file} is unreadable ({type(exc).__name__}). "
            "Delete it and run `python -m interviewtrack.auth` again."
        ) from exc

    assert_readonly(list(creds.scopes or SCOPES))

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                fresh = json.loads(creds.to_json())
                if payload.get("account_email"):
                    fresh["account_email"] = payload["account_email"]
                token_file.write_text(json.dumps(fresh))
                token_file.chmod(0o600)
            except Exception as exc:
                raise AuthError(
                    f"Token refresh failed ({type(exc).__name__}). Re-authorize with "
                    "`python -m interviewtrack.auth`."
                ) from exc
        else:
            raise AuthError(
                "Stored credentials are invalid and cannot be refreshed. "
                "Run `python -m interviewtrack.auth` again."
            )
    return creds


def revoke(account_email: str | None = None, settings: Settings | None = None) -> bool:
    """Revoke and delete one account's stored token."""
    settings = settings or get_settings()
    token_file = token_path_for(account_email, settings)
    if not token_file.exists():
        return False
    try:
        import urllib.parse
        import urllib.request

        data = json.loads(token_file.read_text())
        target = data.get("refresh_token") or data.get("token")
        if target:
            req = urllib.request.Request(
                "https://oauth2.googleapis.com/revoke",
                data=urllib.parse.urlencode({"token": target}).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            urllib.request.urlopen(req, timeout=15)
    except Exception:
        logger.info("Revocation call failed; deleting the local token anyway.")
    token_file.unlink(missing_ok=True)
    return True


def _account_email_from_creds(creds) -> str:
    """Read the account address from a fresh credential via the Gmail profile."""
    try:
        from googleapiclient.discovery import build  # noqa: PLC0415

        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return str(service.users().getProfile(userId="me").execute()
                   .get("emailAddress") or "")
    except Exception:
        try:
            from google.auth.transport.requests import AuthorizedSession
            with AuthorizedSession(creds) as session:
                response = session.get("https://openidconnect.googleapis.com/v1/userinfo", timeout=15)
                response.raise_for_status()
                return str(response.json().get("email") or "")
        except Exception:
            return ""


def account_email(account_email: str | None = None,
                  settings: Settings | None = None) -> str | None:
    """The authorized account's address, used for account-aware Gmail links."""
    # If a stored token already records the email, trust it (no network).
    token_file = token_path_for(account_email, settings)
    if token_file.exists():
        try:
            stored = json.loads(token_file.read_text()).get("account_email")
            if stored:
                return stored
        except (json.JSONDecodeError, OSError):
            pass
    try:
        creds = load_credentials(account_email, settings)
    except AuthError:
        return account_email or None
    return _account_email_from_creds(creds) or account_email or None
