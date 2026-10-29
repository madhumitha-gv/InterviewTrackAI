"""Metadata-first Gmail retrieval.

Two-pass strategy (section 4 of the spec):
  1. List message IDs in the last three months (inbox + sent + archived, excluding
     Spam and Trash), then fetch cheap `format=metadata` headers for each.
  2. Triage those on headers alone, and fetch `format=full` bodies ONLY for the
     messages that look recruitment-related.

This keeps quota use proportional to the hiring mail, not the whole mailbox, while
still giving the classifier real bodies where it matters.
"""

from __future__ import annotations

import logging
import json
import random
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Protocol, Sequence

from ..classify import recruitment_signal_score
from ..config import Settings, get_settings
from ..models import RawMessage
from ..parsing import parse_message, raw_from_gmail_payload

logger = logging.getLogger(__name__)

#: Headers worth requesting in the metadata pass.
METADATA_HEADERS = [
    "From", "To", "Cc", "Subject", "Date", "Message-ID", "In-Reply-To", "References",
]

#: Gmail search query. `-in:spam -in:trash` excludes those by default; `in:anywhere`
#: covers inbox, sent and archived mail.
def build_query(lookback_days: int) -> str:
    """Gmail search for the rolling window.

    `after:` takes unix seconds, which Gmail honours more precisely than the
    `YYYY/MM/DD` form. It is still only approximate on Gmail's side, so
    `GmailEmailSource.fetch` applies an exact cutoff on the parsed date afterwards.
    `in:anywhere` covers inbox, sent and archived mail; Spam and Trash are excluded.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    return f"in:anywhere -in:spam -in:trash after:{int(cutoff.timestamp())}"


class GmailTransport(Protocol):
    """The slice of the Gmail API this module needs.

    Declared as a Protocol so tests can supply a fake without touching the network.
    """

    def list_messages(self, query: str, page_token: str | None,
                      max_results: int) -> dict[str, Any]: ...

    def get_message(self, message_id: str, fmt: str,
                    headers: Sequence[str] | None = None) -> dict[str, Any]: ...


_request_lock = threading.Lock()
_last_request = 0.0


def _paced_execute(request):
    """Conservative process-wide pacing, including simultaneous app sessions."""
    global _last_request
    with _request_lock:
        delay = 0.5 - (time.monotonic() - _last_request)
        if delay > 0:
            time.sleep(delay)
        _last_request = time.monotonic()
        return request.execute()


def _rate_limited(exc):
    status = getattr(getattr(exc, "resp", None), "status", None)
    if status == 429:
        return True
    try:
        reasons = {e.get("reason") for e in json.loads(exc.content).get("error", {}).get("errors", [])}
    except (AttributeError, ValueError, TypeError):
        reasons = set()
    return status == 403 and bool(reasons & {"rateLimitExceeded", "userRateLimitExceeded"})


class GoogleApiTransport:
    """Real transport backed by `googleapiclient`."""

    def __init__(self, credentials: Any, user_id: str = "me"):
        from googleapiclient.discovery import build  # noqa: PLC0415

        self._service = build("gmail", "v1", credentials=credentials,
                              cache_discovery=False)
        self._user = user_id

    def list_messages(self, query: str, page_token: str | None,
                      max_results: int) -> dict[str, Any]:
        return _paced_execute(
            self._service.users().messages()
            .list(userId=self._user, q=query, pageToken=page_token,
                  maxResults=max_results, includeSpamTrash=False)
        )

    def get_message(self, message_id: str, fmt: str,
                    headers: Sequence[str] | None = None) -> dict[str, Any]:
        req = self._service.users().messages().get(
            userId=self._user, id=message_id, format=fmt,
            metadataHeaders=list(headers) if headers else None,
        )
        return _paced_execute(req)


@dataclass
class SyncReport:
    """What a synchronization actually did — surfaced in the UI, never guessed."""

    listed: int = 0
    metadata_fetched: int = 0
    bodies_fetched: int = 0
    skipped_known: int = 0
    errors: list[str] = field(default_factory=list)
    rate_limited: bool = False
    query: str = ""
    pages: int = 0

    def summary(self) -> str:
        return (
            f"listed {self.listed}, metadata {self.metadata_fetched}, "
            f"bodies {self.bodies_fetched}, already-known {self.skipped_known}, "
            f"errors {len(self.errors)}"
        )


def _with_retries(fn: Callable[[], Any], attempts: int = 4,
                  base_delay: float = 2.0, report: SyncReport | None = None) -> Any:
    """Retry transient Gmail errors with exponential backoff + jitter."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # googleapiclient raises HttpError subclasses
            last = exc
            code = getattr(getattr(exc, "resp", None), "status", None)
            transient = _rate_limited(exc) or code in {500, 502, 503, 504} or code is None
            if not transient or i == attempts - 1:
                break
            delay = base_delay * (2**i) + random.uniform(0, 0.4)
            try:
                retry_after = float(exc.resp.get("retry-after", 0))
            except (AttributeError, TypeError, ValueError):
                retry_after = 0
            # Long server cooldowns are surfaced instead of blocking the UI.
            if retry_after > 60:
                break
            delay = max(delay, retry_after)
            logger.info("Gmail call failed (%s); retrying in %.1fs", code, delay)
            time.sleep(delay)
    if report is not None and last is not None:
        status = getattr(getattr(last, "resp", None), "status", None)
        reason = ""
        try:
            reasons = json.loads(last.content).get("error", {}).get("errors", [])
            reason = next((r.get("reason", "") for r in reasons), "")
        except (AttributeError, ValueError, TypeError):
            pass
        if _rate_limited(last):
            report.errors.append("Google rate limit reached. Synchronization paused; retrieved data is retained. Wait a few minutes before refreshing, and keep only one app instance running.")
        elif reason == "accessNotConfigured":
            report.errors.append("Gmail API is disabled for this OAuth project's Google Cloud configuration. Enable Gmail API in APIs & Services, then retry.")
        else:
            report.errors.append(f"Gmail request failed: {type(last).__name__}" + (f" (HTTP {status})" if status else ""))
    raise last if last else RuntimeError("retry loop exhausted")


def list_message_ids(transport: GmailTransport, query: str, max_messages: int,
                     page_size: int, report: SyncReport) -> list[str]:
    """Page through message IDs, de-duplicating and respecting `max_messages`."""
    ids: list[str] = []
    seen: set[str] = set()
    page_token: str | None = None

    while len(ids) < max_messages:
        try:
            page = _with_retries(
                lambda: transport.list_messages(query, page_token, page_size),
                report=report,
            )
        except Exception as exc:
            report.rate_limited = _rate_limited(exc)
            break
        report.pages += 1
        for item in page.get("messages") or []:
            mid = str(item.get("id") or "")
            if mid and mid not in seen:
                seen.add(mid)
                ids.append(mid)
                if len(ids) >= max_messages:
                    break
        page_token = page.get("nextPageToken")
        if not page_token:
            break

    report.listed = len(ids)
    return ids


def fetch_messages(
    transport: GmailTransport,
    settings: Settings | None = None,
    known_ids: Iterable[str] = (),
    body_threshold: float = 0.12,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[RawMessage], SyncReport]:
    """Run the two-pass retrieval and return parsed-ready RawMessages.

    `known_ids` are skipped entirely, making repeated syncs cheap and idempotent.
    `body_threshold` is deliberately low: the spec warns against discarding valid
    recruiting conversations, so borderline messages still get their body fetched.
    """
    settings = settings or get_settings()
    report = SyncReport(query=build_query(settings.lookback_days))
    known = set(known_ids)

    ids = list_message_ids(
        transport, report.query, settings.max_messages, settings.gmail_page_size, report
    )
    if report.rate_limited:
        return [], report
    to_fetch = [i for i in ids if i not in known]
    report.skipped_known = len(ids) - len(to_fetch)

    # --- pass 1: cheap metadata for every candidate message ---
    staged: list[tuple[RawMessage, float]] = []
    for n, mid in enumerate(to_fetch, start=1):
        if progress is not None:
            progress(n, len(to_fetch))
        try:
            meta = _with_retries(
                lambda: transport.get_message(mid, "metadata", METADATA_HEADERS),
                report=report,
            )
        except Exception as exc:
            if _rate_limited(exc):
                report.rate_limited = True
                break
            continue
        report.metadata_fetched += 1
        try:
            raw = raw_from_gmail_payload(meta)
        except Exception as exc:
            report.errors.append(f"parse {mid}: {type(exc).__name__}")
            continue
        score, _ = recruitment_signal_score(parse_message(raw))
        staged.append((raw, score))

    # --- thread continuity: if ANY message in a Gmail thread looks like recruitment,
    # the whole thread is in scope. Short replies ("Confirming our call for
    # Wednesday") score low on keywords alone but are part of the conversation, and
    # dropping them would silently break cross-thread reconstruction.
    recruitment_threads = {
        raw.thread_id for raw, score in staged if score >= body_threshold
    }

    # --- pass 2: full bodies only for in-scope messages ---
    messages: list[RawMessage] = []
    for raw, score in staged:
        if score < body_threshold and raw.thread_id not in recruitment_threads:
            continue
        if report.rate_limited:
            messages.append(raw)
            continue
        try:
            full = _with_retries(
                lambda: transport.get_message(raw.gmail_id, "full"), report=report
            )
            raw = raw_from_gmail_payload(full)
            report.bodies_fetched += 1
        except Exception as exc:
            # Keep the metadata-only version rather than losing the message.
            if _rate_limited(exc):
                report.rate_limited = True
            else:
                report.errors.append(f"body {raw.gmail_id}: {type(exc).__name__}")
        messages.append(raw)

    # Stable chronological order; dedupe defensively.
    unique: dict[str, RawMessage] = {}
    for m in messages:
        unique.setdefault(m.gmail_id, m)
    return sorted(unique.values(), key=lambda m: m.sort_key), report


class FixtureTransport:
    """In-memory transport over Gmail API payloads.

    Used by the demo/tests to exercise the *real* retrieval and parsing code paths
    (pagination, metadata-then-body, retries) with no network and no credentials.
    """

    def __init__(self, payloads: Sequence[dict[str, Any]], page_size: int = 5,
                 fail_bodies: set[str] | None = None):
        self._payloads = list(payloads)
        self._by_id = {str(p["id"]): p for p in self._payloads}
        self.page_size = page_size
        self.fail_bodies = fail_bodies or set()
        self.calls: list[tuple[str, str]] = []

    def list_messages(self, query: str, page_token: str | None,
                      max_results: int) -> dict[str, Any]:
        start = int(page_token) if page_token else 0
        size = min(self.page_size, max_results)
        chunk = self._payloads[start : start + size]
        out: dict[str, Any] = {
            "messages": [{"id": p["id"], "threadId": p["threadId"]} for p in chunk]
        }
        nxt = start + size
        if nxt < len(self._payloads):
            out["nextPageToken"] = str(nxt)
        return out

    def get_message(self, message_id: str, fmt: str,
                    headers: Sequence[str] | None = None) -> dict[str, Any]:
        self.calls.append((message_id, fmt))
        payload = self._by_id.get(message_id)
        if payload is None:
            raise KeyError(f"unknown message {message_id}")
        if fmt == "full" and message_id in self.fail_bodies:
            raise RuntimeError("simulated body fetch failure")
        if fmt == "metadata":
            # Strip bodies, mirroring what Gmail returns for format=metadata.
            inner = dict(payload.get("payload") or {})
            inner.pop("parts", None)
            return {**payload, "payload": inner}
        return payload
