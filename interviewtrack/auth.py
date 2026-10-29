"""Gmail Mode authorization — run this once, from the terminal.

    python -m interviewtrack.auth            # authorize
    python -m interviewtrack.auth --status   # show current state
    python -m interviewtrack.auth --revoke   # revoke and delete the local token

Kept out of the Streamlit app deliberately: Streamlit re-runs its whole script on
every interaction, so driving OAuth from the UI would re-trigger the flow repeatedly.
"""

from __future__ import annotations

import argparse
import sys

from .config import get_settings
from .gmail.desktop_oauth import (
    AuthError,
    account_email,
    authorize,
    client_secret_path,
    has_credentials,
    has_token,
    revoke,
    stored_accounts,
    token_path_for,
    verify_account,
)


def _status(account: str | None = None) -> int:
    settings = get_settings()
    secret = client_secret_path(settings)
    token = token_path_for(account, settings)
    accounts = stored_accounts(settings)
    print("InterviewTrack AI — Gmail Mode status")
    print(f"  OAuth client file : {secret}  [{'found' if secret.exists() else 'MISSING'}]")
    print(f"  Authorized accounts: {', '.join(accounts) if accounts else 'none'}")
    if account:
        print(f"  This account token: {token}  [{'found' if token.exists() else 'none'}]")
    if not secret.exists():
        print("\n  Not configured. Create a DESKTOP APP OAuth client in Google Cloud")
        print("  Console (APIs & Services -> Credentials), download the JSON, and save")
        print(f"  it to {secret}. See README -> Gmail Mode setup.")
        return 1
    if not token.exists():
        print("\n  Not authorized. Run:  python -m interviewtrack.auth")
        return 1
    email = account_email(account, settings)
    print(f"  Authorized account: {email or '(could not read profile)'}")
    print("  Scope: gmail.readonly (read-only)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m interviewtrack.auth",
        description="Authorize read-only Gmail access for InterviewTrack AI.",
    )
    parser.add_argument("--account", default=None,
                        help="which Google account to authorize/show/revoke "
                             "(omit for a single-account setup)")
    parser.add_argument("--status", action="store_true", help="show current state")
    parser.add_argument("--revoke", action="store_true",
                        help="revoke and delete this account's stored token")
    parser.add_argument("--smoke", action="store_true",
                        help="privacy-safe retrieval check (counts only, no content)")
    parser.add_argument("--port", type=int, default=0,
                        help="local redirect port (default: pick a free one)")
    args = parser.parse_args(argv)

    settings = get_settings()

    if args.status:
        return _status(args.account)

    if args.smoke:
        from .gmail_smoke import run_smoke
        return run_smoke(args.account, settings)

    if args.revoke:
        if revoke(args.account, settings):
            print("Token revoked and deleted.")
            return 0
        print("No stored token to revoke.")
        return 0

    if not has_credentials(settings):
        print(f"ERROR: no OAuth client file at {client_secret_path(settings)}",
              file=sys.stderr)
        print("\nCreate a DESKTOP APP OAuth client in Google Cloud Console:", file=sys.stderr)
        print("  1. https://console.cloud.google.com/apis/credentials", file=sys.stderr)
        print("  2. Create credentials -> OAuth client ID -> Desktop app", file=sys.stderr)
        print("  3. Enable the Gmail API for the project", file=sys.stderr)
        print("  4. Add your Google account under OAuth consent screen -> Test users",
              file=sys.stderr)
        print(f"  5. Download the JSON and save it to {client_secret_path(settings)}",
              file=sys.stderr)
        return 1

    if has_token(args.account, settings):
        print(f"Already authorized ({token_path_for(args.account, settings)}).")
        print("Re-authorizing will replace the stored token.")

    print("Opening a browser for Google authorization…")
    print("Requesting READ-ONLY Gmail access (gmail.readonly) and basic profile.")
    try:
        result = authorize(settings, port=args.port, account=args.account)
    except AuthError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # the browser flow can fail in many ways
        print(f"\nERROR: authorization failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    email = result.get("account_email")
    print(f"\nAuthorized{f' as {email}' if email else ''}.")
    print(f"Token stored at {token_path_for(email, settings)} (git-ignored, owner-only).")
    print("\nNext:  streamlit run app.py   then choose Gmail Mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
