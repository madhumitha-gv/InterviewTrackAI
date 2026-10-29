"""Gmail deep links built from original RFC Message-IDs.

Two deliberate choices:

  * **Search on `rfc822msgid:` rather than linking a message ID.** Gmail API message
    and thread IDs are API identifiers; they are not reliable web-UI URLs. An
    `rfc822msgid:` search resolves to the right message in the web client.
  * **Account-aware.** `authuser=<email>` targets the correct account when the user is
    signed into several, instead of opening whichever happens to be account 0.
"""

from __future__ import annotations

import urllib.parse

GMAIL_BASE = "https://mail.google.com/mail"


def _prefix(account_email: str | None = None) -> str:
    """Build the account-aware base URL."""
    if account_email and "@" in account_email:
        qs = urllib.parse.urlencode({"authuser": account_email.strip().lower()})
        return f"{GMAIL_BASE}/u/0/?{qs}"
    return f"{GMAIL_BASE}/u/0/"


def thread_link(thread_id: str, account_email: str | None = None) -> str:
    """Open a Gmail thread by its Gmail thread ID.

    Less reliable than an rfc822msgid search; used only as a fallback when no RFC
    Message-ID is available.
    """
    return f"{_prefix(account_email)}#all/{urllib.parse.quote(thread_id)}"


def message_search_link(rfc_message_id: str, account_email: str | None = None) -> str:
    """Account-aware Gmail search for an exact RFC Message-ID."""
    mid = rfc_message_id.strip().strip("<>")
    query = f"rfc822msgid:{mid}"
    return f"{_prefix(account_email)}#search/{urllib.parse.quote(query)}"


def best_message_link(rfc_message_id: str | None, thread_id: str | None,
                      account_email: str | None = None) -> str:
    """Prefer an exact RFC Message-ID search; fall back to the thread view."""
    if rfc_message_id:
        return message_search_link(rfc_message_id, account_email)
    if thread_id:
        return thread_link(thread_id, account_email)
    return f"{_prefix(account_email)}#all"
