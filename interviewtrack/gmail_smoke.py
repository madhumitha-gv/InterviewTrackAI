"""Privacy-safe Gmail smoke test.

Runs the real retrieval and reconstruction for an authorized account and reports
**counts only** — never a subject, sender, snippet or body. This is deliberately safe
to paste into a bug report: there is nothing in its output that identifies a person,
company or message.

    python -m interviewtrack.auth --smoke
    python -m interviewtrack.gmail_smoke            # same thing
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Settings, get_settings


def run_smoke(account: str | None = None, settings: Settings | None = None,
              persist: bool = False) -> int:
    """Fetch + reconstruct for the authorized account; print counts only."""
    settings = settings or get_settings()

    # Local imports keep `--help` fast and avoid importing googleapiclient unless used.
    from .gmail.client import GoogleApiTransport, build_query, fetch_messages
    from .gmail.desktop_oauth import (
        AuthError,
        account_email,
        load_credentials,
        verify_account,
    )
    from .reconstruct import reconstruct

    print("=" * 70)
    print("InterviewTrack AI — Gmail smoke test (counts only, no content)")
    print("=" * 70)

    try:
        creds = load_credentials(account, settings)
    except AuthError as exc:
        print(f"Not authorized: {exc}", file=sys.stderr)
        return 1

    connected = account_email(account, settings)
    print(f"Account identity  : {'verified' if connected else 'unavailable'}")
    if settings.gmail_expected_account:
        ok, msg = verify_account(connected, settings.gmail_expected_account)
        print(f"Account check     : {'OK' if ok else 'MISMATCH OR UNAVAILABLE'}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.lookback_days)
    print(f"Window            : last {settings.lookback_days} days "
          f"(since {cutoff.date().isoformat()})")
    print(f"Query             : {build_query(settings.lookback_days)}")
    print("Scope             : gmail.readonly (read-only)")
    print("-" * 70)

    try:
        transport = GoogleApiTransport(creds)
        messages, report = fetch_messages(transport, settings)
    except Exception as exc:
        print(f"Retrieval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # Apply the exact window cutoff (Gmail's after: is day-granular).
    in_window = [m for m in messages if m.date is None or m.date >= cutoff]

    result = reconstruct(
        in_window,
        user_email=connected or settings.gmail_expected_account or None,
        settings=settings,
    )

    review = sum(1 for a in result.applications if a.needs_review)
    print(f"Listed            : {report.listed}")
    print(f"Metadata fetched  : {report.metadata_fetched}")
    print(f"Bodies fetched    : {report.bodies_fetched}")
    print(f"In-window messages: {len(in_window)}")
    print(f"Classified recruit: {result.processed_count}")
    print(f"Skipped (non-recr): {len(result.skipped_non_recruitment)}")
    print(f"Parse issues      : {len(result.parse_error_ids)}")
    print(f"Retrieval errors  : {len(report.errors)}")
    if report.errors:
        print("Retrieval incomplete. Check Google Cloud Gmail API enablement and authorization; zero counts do not prove an empty inbox.")
    print("-" * 70)
    print(f"Applications      : {len(result.applications)}")
    print(f"  needs review    : {review}")
    print(f"Total events      : {sum(len(a.events) for a in result.applications)}")
    print(f"Threads involved  : {len({t for a in result.applications for t in a.thread_ids})}")
    print(f"Embedding backend : {result.embedding_backend}")
    print("=" * 70)
    print("No subjects, senders, snippets or bodies were printed. Safe to share.")

    if persist and connected:
        from .service import InterviewTrackService
        from .sources import GmailEmailSource

        svc = InterviewTrackService(settings=settings)
        uid = f"gmail:{connected}"
        svc.register_user(uid, connected)
        svc.sync_from_source(uid, connected, GmailEmailSource(account=account))
        print(f"\nPersisted to the local database under user_id={uid}.")
    return 1 if report.errors else 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m interviewtrack.gmail_smoke",
        description="Privacy-safe Gmail retrieval check (counts only).",
    )
    parser.add_argument("--account", default=None, help="which authorized account")
    parser.add_argument("--persist", action="store_true",
                        help="also store results in the local database")
    args = parser.parse_args(argv)
    return run_smoke(args.account, get_settings(), persist=args.persist)


if __name__ == "__main__":
    raise SystemExit(main())
