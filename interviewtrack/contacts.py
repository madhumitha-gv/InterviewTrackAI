"""Contact registry: who sent this, and what role do they play in the process?

Contact type is *supporting evidence* for matching, never application identity. One
recruiter can run several roles, and several people work one role — so a known contact
narrows candidates, but employer and role decide the match.

The graph is built from From, To, Cc, Reply-To, References, signatures and explicit
handoffs, so a recruiter handoff or a coordinator's first email has something to anchor
on.
"""

from __future__ import annotations

import re
from datetime import datetime

from .enums import ContactType
from .models import Contact, ParsedMessage
from .normalize import (
    canonical_person,
    domain_of,
    is_ats_domain,
    is_freemail_domain,
    is_scheduling_domain,
    is_staffing_domain,
    name_from_email,
)

#: Local-parts that mark a machine sender regardless of domain.
_NOREPLY_LOCALS = (
    "no-reply", "noreply", "donotreply", "do-not-reply", "notifications",
    "notification", "mailer-daemon", "automated", "auto-reply", "bounce",
)

#: Signature / body cues for each human contact type, most specific first.
_ROLE_CUES: list[tuple[ContactType, re.Pattern[str]]] = [
    (ContactType.COORDINATOR, re.compile(
        r"\b(recruit(?:ing|ment)\s+coordinator|interview\s+coordinator|"
        r"scheduling\s+coordinator|talent\s+coordinator|candidate\s+experience|"
        r"coordinator,\s*(?:talent|recruiting|people))\b", re.I)),
    # Must be SELF-identification. A bare mention of "hiring manager" is usually
    # third-person ("a different hiring manager"), which describes someone else.
    (ContactType.HIRING_MANAGER, re.compile(
        r"\b(i\s+am\s+(?:the\s+|your\s+)?(?:hiring|engineering)\s+manager|"
        r"your\s+(?:hiring|engineering)\s+manager\s+(?:here|at|for)|"
        r"i(?:'ll| will)\s+be\s+(?:one\s+of\s+)?your\s+interviewer|"
        r"i\s+will\s+be\s+interviewing\s+you|"
        r"i\s+lead\s+the\b|i\s+manage\s+the\b|"
        r"i\s+am\s+(?:a\s+|the\s+)?(?:staff|principal)\s+engineer)", re.I)),
    (ContactType.AGENCY_RECRUITER, re.compile(
        r"\b(agency\s+recruiter|(?:technical\s+|senior\s+)?recruiter\s+(?:at|with)\s+"
        r"\w+|my\s+client|on\s+behalf\s+of|executive\s+search|staffing|"
        r"recruitment\s+consultant)\b", re.I)),
    (ContactType.INTERNAL_RECRUITER, re.compile(
        r"\b(talent\s+acquisition|internal\s+recruiter|people\s+team|"
        r"recruiting\s+team|talent\s+team|hr\s+(?:team|partner)|"
        r"(?:technical\s+)?recruiter\b)", re.I)),
]

#: Sender local-parts that imply a recruiting function.
_RECRUITING_LOCALS = (
    "recruit", "talent", "careers", "jobs", "hiring", "hr", "people", "candidates",
)


def classify_contact(
    from_addr: str,
    from_name: str = "",
    text: str = "",
    user_email: str | None = None,
) -> ContactType:
    """Determine the sender's role in the process.

    Order matters: self and machine senders are decided on the address alone, because
    their body text often *describes* a recruiter without being one.
    """
    addr = (from_addr or "").strip().lower()
    if not addr or "@" not in addr:
        return ContactType.UNKNOWN

    # 1. The candidate themselves.
    if user_email and addr == user_email.strip().lower():
        return ContactType.SELF

    domain = domain_of(addr)
    local = addr.split("@", 1)[0]

    # 2. Machine senders — decided by address, never by body text.
    if is_scheduling_domain(domain):
        return ContactType.SCHEDULING_TOOL
    if is_ats_domain(domain):
        return ContactType.ATS_NOREPLY
    if any(tok in local for tok in _NOREPLY_LOCALS):
        return ContactType.ATS_NOREPLY

    # 3. A staffing domain makes the sender an agency recruiter outright.
    if is_staffing_domain(domain):
        return ContactType.AGENCY_RECRUITER

    # 4. Human role from signature / body cues.
    haystack = f"{from_name}\n{text}"
    for contact_type, pattern in _ROLE_CUES:
        if pattern.search(haystack):
            # "recruiter at X" on a corporate domain is internal, not agency.
            if contact_type == ContactType.AGENCY_RECRUITER and not is_freemail_domain(domain):
                if re.search(r"\b(my client|on behalf of|staffing|executive search)\b",
                             haystack, re.I):
                    return ContactType.AGENCY_RECRUITER
                return ContactType.INTERNAL_RECRUITER
            return contact_type

    # 5. Recruiting-shaped mailbox on a corporate domain.
    if any(tok in local for tok in _RECRUITING_LOCALS):
        return ContactType.INTERNAL_RECRUITER

    return ContactType.UNKNOWN


def contact_from_message(msg: ParsedMessage, user_email: str | None = None,
                         resolved: dict[str, ContactType] | None = None) -> Contact:
    """Build the Contact record for a message's sender.

    `resolved` supplies a type decided once across all of that sender's messages.
    """
    raw = msg.raw
    addr = (raw.from_addr or "").lower()
    contact_type = (resolved or {}).get(addr) or classify_contact(
        raw.from_addr, raw.from_name, msg.clean_text, user_email
    )
    display = (raw.from_name or "").strip() or name_from_email(raw.from_addr).title()
    return Contact(
        email=(raw.from_addr or "").lower(),
        display_name=display,
        org_domain=domain_of(raw.from_addr),
        contact_type=contact_type,
        first_seen=raw.date,
    )


def participants(msg: ParsedMessage, user_email: str | None = None) -> set[str]:
    """Every address involved in a message, excluding the candidate.

    Used to carry a handoff: when a new sender appears on a thread that already has
    known contacts, the overlap is what links them.
    """
    raw = msg.raw
    addrs = {raw.from_addr or ""} | set(raw.to_addrs) | set(raw.cc_addrs)
    me = (user_email or "").strip().lower()
    return {a.lower() for a in addrs if a and "@" in a and a.lower() != me}


#: How specific each type is. When one sender's messages classify differently
#: (a recruiter who mentions "hiring manager" in one email), the most specific
#: observation wins, so the same person is not typed three ways.
_TYPE_SPECIFICITY: dict[ContactType, int] = {
    ContactType.SELF: 100,              # unambiguous: it is the candidate
    ContactType.ATS_NOREPLY: 90,        # decided by domain, never by prose
    ContactType.SCHEDULING_TOOL: 90,
    ContactType.COORDINATOR: 60,
    ContactType.HIRING_MANAGER: 50,
    ContactType.AGENCY_RECRUITER: 40,
    ContactType.INTERNAL_RECRUITER: 30,
    ContactType.UNKNOWN: 0,
}


def resolve_contact_types(
    messages, user_email: str | None = None
) -> dict[str, ContactType]:
    """Decide one type per sender address across every message they sent.

    Typing per-message makes the same person appear as three different roles,
    because the cue that identifies them may appear in only one email.
    """
    best: dict[str, ContactType] = {}
    for msg in messages:
        addr = (msg.raw.from_addr or "").lower()
        if not addr or "@" not in addr:
            continue
        observed = classify_contact(
            addr, msg.raw.from_name, msg.clean_text, user_email
        )
        current = best.get(addr, ContactType.UNKNOWN)
        if _TYPE_SPECIFICITY[observed] > _TYPE_SPECIFICITY[current]:
            best[addr] = observed
    return best


class ContactRegistry:
    """Which contacts are known to belong to which applications.

    A sender is *learned* for an application once one of their messages is matched on
    identity evidence (thread, requisition, employer+role) or confirmed by the user.
    """

    def __init__(self) -> None:
        self._by_app: dict[str, set[str]] = {}
        self._by_email: dict[str, set[str]] = {}
        self._contacts: dict[str, Contact] = {}

    def register(self, contact: Contact) -> None:
        """Record a contact we have seen, without linking it to an application."""
        email = contact.email.lower()
        if not email:
            return
        existing = self._contacts.get(email)
        if existing is None or (
            existing.contact_type == ContactType.UNKNOWN
            and contact.contact_type != ContactType.UNKNOWN
        ):
            self._contacts[email] = contact

    def learn(self, app_key: str, email: str) -> None:
        """Link a contact to an application (contact learning)."""
        email = (email or "").lower()
        if not email or "@" not in email:
            return
        self._by_app.setdefault(app_key, set()).add(email)
        self._by_email.setdefault(email, set()).add(app_key)

    def learn_participants(self, app_key: str, emails: set[str]) -> None:
        for email in emails:
            self.learn(app_key, email)

    def applications_for(self, email: str) -> set[str]:
        """Applications this sender is already known to."""
        return set(self._by_email.get((email or "").lower(), set()))

    def contacts_for(self, app_key: str) -> list[Contact]:
        emails = self._by_app.get(app_key, set())
        return [self._contacts[e] for e in sorted(emails) if e in self._contacts]

    def get(self, email: str) -> Contact | None:
        return self._contacts.get((email or "").lower())

    def is_known(self, email: str) -> bool:
        return bool(self.applications_for(email))

    def clear(self) -> None:
        self._by_app.clear()
        self._by_email.clear()
        self._contacts.clear()
