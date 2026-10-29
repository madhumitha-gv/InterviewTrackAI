"""Email source adapters.

Two interchangeable sources feed the same reconstruction engine:

  * ``SyntheticEmailSource`` — Demo Mode. Fixtures rendered as Gmail API payloads and
    parsed by the production parser, so Demo Mode exercises the real code path.
  * ``GmailEmailSource``     — Gmail Mode. Read-only, single user, Desktop OAuth.

**Gmail Mode never falls back to synthetic data.** If authorization is missing the
source raises, and the caller must surface that — silently substituting fixtures for a
real mailbox would make the UI lie about where its data came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Protocol, runtime_checkable

from .config import Settings, get_settings
from .gmail.client import FixtureTransport, GoogleApiTransport, SyncReport, fetch_messages
from .gmail.desktop_oauth import (
    AuthError,
    account_email,
    load_credentials,
    verify_account,
)
from .models import RawMessage

logger = logging.getLogger(__name__)


@runtime_checkable
class EmailSource(Protocol):
    """What the engine needs from any mail source."""

    name: str

    def fetch(
        self,
        known_ids: Iterable[str] = (),
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[RawMessage], SyncReport]: ...


@dataclass
class SyntheticEmailSource:
    """Demo Mode: the labelled fixtures, with no network and no credentials."""

    name: str = "Synthetic (Demo Mode)"
    include_holdout: bool = True
    settings: Settings | None = None

    @property
    def account_email(self) -> str:
        from .fixtures.synthetic import USER_EMAIL

        return USER_EMAIL

    def fetch(
        self,
        known_ids: Iterable[str] = (),
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[RawMessage], SyncReport]:
        from .fixtures.synthetic import (
            DEV_MESSAGES,
            FULL_DATASET,
            gmail_messages_as_api_payloads,
        )

        wanted = {m.gmail_id for m in (FULL_DATASET if self.include_holdout
                                       else DEV_MESSAGES)}
        payloads = [p for p in gmail_messages_as_api_payloads() if p["id"] in wanted]
        transport = FixtureTransport(payloads, page_size=8)
        return fetch_messages(
            transport, self.settings or get_settings(),
            known_ids=known_ids, progress=progress,
        )


@dataclass
class GmailEmailSource:
    """Gmail Mode: one authorized mailbox, read-only, rolling window."""

    name: str = "Gmail (read-only)"
    settings: Settings | None = None
    account: str | None = None       # which authorized account to read
    _resolved: str | None = None

    @property
    def account_email(self) -> str | None:
        if self._resolved is None:
            self._resolved = account_email(self.account, self.settings or get_settings())
        return self._resolved

    def fetch(
        self,
        known_ids: Iterable[str] = (),
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[RawMessage], SyncReport]:
        settings = self.settings or get_settings()
        # Raises AuthError when not authorized. The caller must show that, not
        # quietly substitute fixtures.
        credentials = load_credentials(self.account, settings)
        transport = GoogleApiTransport(credentials)
        messages, report = fetch_messages(
            transport, settings, known_ids=known_ids, progress=progress
        )
        # Gmail's `after:` is day-granular, so trim to the exact cutoff here.
        # Warn (never swap) if the connected account is not the expected one.
        if settings.gmail_expected_account:
            ok, msg = verify_account(self.account_email, settings.gmail_expected_account)
            if not ok:
                report.errors.append(msg)

        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.lookback_days)
        kept = [m for m in messages if m.date is None or m.date >= cutoff]
        dropped = len(messages) - len(kept)
        if dropped:
            report.errors.append(
                f"{dropped} message(s) outside the exact "
                f"{settings.lookback_days}-day window were discarded"
            )
        return kept, report


def build_source(mode: str, settings: Settings | None = None) -> EmailSource:
    """Pick a source by mode name. Unknown modes raise rather than defaulting."""
    mode = (mode or "").strip().lower()
    if mode in {"demo", "synthetic"}:
        return SyntheticEmailSource(settings=settings)
    if mode in {"gmail", "real"}:
        return GmailEmailSource(settings=settings)
    raise ValueError(f"Unknown email source mode: {mode!r}")


def gmail_available(account_email: str | None = None,
                    settings: Settings | None = None) -> tuple[bool, str]:
    """Whether Gmail Mode can run for an account, and why not when it cannot."""
    try:
        load_credentials(account_email, settings or get_settings())
        return True, "authorized"
    except AuthError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{type(exc).__name__}: {exc}"
