"""Email parsing: MIME decode, HTML->text, and removal of quoted history/boilerplate.

Everything here is defensive: malformed metadata or bodies must degrade to a partial
result with recorded `parse_errors`, never raise.
"""

from __future__ import annotations

import base64
import binascii
import re
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any

from bs4 import BeautifulSoup

from .models import ParsedMessage, RawMessage

# --- quoted-history / boilerplate detection ------------------------------------

#: Lines that mark the start of quoted history. Everything after is dropped.
_QUOTE_MARKERS = [
    re.compile(r"^\s*On .{3,80}\bwrote:\s*$", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*Forwarded message\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*_{5,}\s*$"),
    re.compile(r"^\s*From:\s*.+@.+$", re.IGNORECASE),
    re.compile(r"^\s*Sent from my (iPhone|iPad|Android|mobile).*$", re.IGNORECASE),
]

#: Signature separator per RFC 3676.
_SIG_SEP = re.compile(r"^--\s?$")

#: Legal disclaimers and footer noise — high-frequency, zero semantic value.
_DISCLAIMER_MARKERS = [
    re.compile(r"confidentiality notice", re.IGNORECASE),
    re.compile(r"this (e-?mail|message) (and any attachments )?is (intended|confidential)", re.I),
    re.compile(r"if you are not the intended recipient", re.IGNORECASE),
    re.compile(r"please consider the environment", re.IGNORECASE),
    re.compile(r"unsubscribe", re.IGNORECASE),
    re.compile(r"view this (e-?mail|message) in your browser", re.IGNORECASE),
    re.compile(r"©\s*\d{4}", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"^\s*sent via\b", re.IGNORECASE),
]

_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


def decode_b64url(data: str) -> bytes:
    """Decode Gmail's base64url payload, tolerating missing padding."""
    if not data:
        return b""
    try:
        padded = data + "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii", errors="ignore"))
    except (binascii.Error, ValueError, UnicodeEncodeError):
        return b""


def html_to_text(html: str) -> str:
    """Convert HTML to readable plain text, preserving block boundaries."""
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return re.sub(r"<[^>]+>", " ", html)

    for tag in soup(["script", "style", "head", "title", "meta", "noscript"]):
        tag.decompose()
    # Keep link targets out of the text but keep anchor labels.
    for br in soup.find_all(["br"]):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"]):
        block.append("\n")
    text = soup.get_text(separator=" ")
    return normalize_whitespace(text)


#: A line that ends without sentence punctuation is a soft wrap, not a paragraph
#: break. Email bodies are hard-wrapped around 72 characters, which splits names
#: like "Meridian\nLabs" across lines.
_SOFT_WRAP = re.compile(r"(?<![.!?:;])\n(?!\n)(?=[^\s])")


def unwrap_soft_breaks(text: str) -> str:
    """Join lines broken mid-sentence by hard wrapping.

    Paragraph breaks (blank lines) and lines ending in sentence punctuation are
    preserved, so "Northwind Analytics.\nHi Avery" keeps its break.
    """
    return _SOFT_WRAP.sub(" ", text or "")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def strip_quoted_and_boilerplate(text: str) -> tuple[str, str]:
    """Split body into (clean, quoted/boilerplate).

    Returns the author's own prose first. This matters a great deal for embedding
    quality: an unstripped reply chain makes every message in a thread look
    semantically identical.
    """
    if not text:
        return "", ""

    lines = text.split("\n")
    cut = len(lines)

    for i, line in enumerate(lines):
        if any(m.match(line) for m in _QUOTE_MARKERS):
            cut = i
            break
        if _SIG_SEP.match(line):
            cut = i
            break
        # A run of '>' quoted lines also ends the original content.
        if line.startswith(">") and i > 0:
            cut = i
            break
        if any(m.search(line) for m in _DISCLAIMER_MARKERS):
            cut = i
            break

    clean_lines = [ln for ln in lines[:cut] if not ln.startswith(">")]
    clean = normalize_whitespace("\n".join(clean_lines))
    quoted = normalize_whitespace("\n".join(lines[cut:]))

    # If stripping removed everything, the whole body was probably boilerplate-ish;
    # fall back to the unstripped text rather than embedding an empty string.
    if not clean and text.strip():
        return normalize_whitespace(text), ""
    return clean, quoted


# --- header parsing -------------------------------------------------------------


def parse_date_header(value: str | None) -> datetime | None:
    """Parse a Date: header into tz-aware UTC, tolerating junk."""
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def split_addresses(value: str | None) -> list[str]:
    """Split an address header into lowercase email addresses."""
    if not value:
        return []
    out: list[str] = []
    for chunk in re.split(r"[,;]", value):
        _, addr = parseaddr(chunk.strip())
        if addr and "@" in addr:
            out.append(addr.lower())
    return out


def split_references(value: str | None) -> list[str]:
    """Extract <message-id> tokens from References/In-Reply-To."""
    if not value:
        return []
    return [f"<{m}>" for m in re.findall(r"<([^<>]+)>", value)]


def headers_to_dict(headers: list[dict[str, Any]] | None) -> dict[str, str]:
    """Flatten the Gmail header list into a lowercase-keyed dict."""
    out: dict[str, str] = {}
    for h in headers or []:
        if not isinstance(h, dict):
            continue
        name = str(h.get("name", "")).strip().lower()
        if name and name not in out:
            out[name] = str(h.get("value", ""))
    return out


def extract_body_parts(payload: dict[str, Any] | None) -> tuple[str, str]:
    """Walk a MIME tree and return (text/plain, text/html).

    Prefers the first text/plain and first text/html found via depth-first traversal,
    skipping attachments.
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part: Any, depth: int = 0) -> None:
        if not isinstance(part, dict) or depth > 12:
            return
        mime = str(part.get("mimeType", "")).lower()
        body = part.get("body") or {}
        filename = part.get("filename") or ""
        data = body.get("data") if isinstance(body, dict) else None

        if data and not filename:
            decoded = decode_b64url(data).decode("utf-8", errors="replace")
            if mime == "text/plain":
                plain_parts.append(decoded)
            elif mime == "text/html":
                html_parts.append(decoded)
        for child in part.get("parts") or []:
            walk(child, depth + 1)

    walk(payload or {})
    return ("\n".join(plain_parts).strip(), "\n".join(html_parts).strip())


def raw_from_gmail_payload(message: dict[str, Any]) -> RawMessage:
    """Build a RawMessage from a Gmail API `users.messages.get` response.

    Works for both `format=metadata` and `format=full` responses.
    """
    if not isinstance(message, dict):
        raise ValueError("Gmail message payload must be a dict")

    gmail_id = str(message.get("id") or "").strip()
    if not gmail_id:
        raise ValueError("Gmail message is missing an id")
    thread_id = str(message.get("threadId") or gmail_id).strip()

    payload = message.get("payload") or {}
    headers = headers_to_dict(payload.get("headers") if isinstance(payload, dict) else None)

    from_name, from_addr = parseaddr(headers.get("from", ""))
    body_text, body_html = extract_body_parts(payload if isinstance(payload, dict) else None)

    date = parse_date_header(headers.get("date"))
    if date is None:
        # internalDate is epoch millis and is always present on real messages.
        internal = message.get("internalDate")
        if internal:
            try:
                date = datetime.fromtimestamp(int(internal) / 1000, tz=timezone.utc)
            except (TypeError, ValueError, OSError, OverflowError):
                date = None

    return RawMessage(
        gmail_id=gmail_id,
        thread_id=thread_id,
        rfc_message_id=(headers.get("message-id") or None),
        in_reply_to=(headers.get("in-reply-to") or None),
        references=split_references(headers.get("references")),
        from_addr=(from_addr or "").lower(),
        from_name=from_name or "",
        to_addrs=split_addresses(headers.get("to")),
        cc_addrs=split_addresses(headers.get("cc")),
        subject=headers.get("subject", "") or "",
        date=date,
        snippet=str(message.get("snippet") or ""),
        labels=[str(x) for x in (message.get("labelIds") or [])],
        body_text=body_text,
        body_html=body_html,
        body_fetched=bool(body_text or body_html),
    )


def parse_message(raw: RawMessage) -> ParsedMessage:
    """Produce embedding-ready text. Never raises; records issues in parse_errors."""
    errors: list[str] = []
    text = ""

    try:
        if raw.body_text:
            text = normalize_whitespace(raw.body_text)
        elif raw.body_html:
            text = html_to_text(raw.body_html)
    except Exception as exc:  # defensive: malformed markup must not kill a sync
        errors.append(f"body_extraction_failed: {type(exc).__name__}")
        text = ""

    if not text:
        # Metadata-only messages still carry a usable snippet.
        text = normalize_whitespace(raw.snippet)
        if not text:
            errors.append("empty_body")

    try:
        clean, quoted = strip_quoted_and_boilerplate(text)
    except Exception as exc:
        errors.append(f"quote_strip_failed: {type(exc).__name__}")
        clean, quoted = text, ""

    if not raw.date:
        errors.append("missing_date")
    if not raw.from_addr or "@" not in raw.from_addr:
        errors.append("missing_or_invalid_sender")
    if not raw.subject.strip():
        errors.append("missing_subject")

    return ParsedMessage(raw=raw, clean_text=clean, quoted_text=quoted, parse_errors=errors)
