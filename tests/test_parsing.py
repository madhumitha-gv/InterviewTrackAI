"""Parsing, MIME handling, HTML conversion and malformed-input tolerance."""

from __future__ import annotations

import base64
from datetime import datetime, timezone

import pytest

from interviewtrack.models import RawMessage
from interviewtrack.parsing import (
    decode_b64url,
    extract_body_parts,
    html_to_text,
    parse_date_header,
    parse_message,
    raw_from_gmail_payload,
    split_addresses,
    split_references,
    strip_quoted_and_boilerplate,
)


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


class TestHtmlToText:
    def test_strips_scripts_and_styles(self):
        out = html_to_text("<div>Hello<script>evil()</script><style>a{}</style></div>")
        assert "evil" not in out and "a{}" not in out
        assert "Hello" in out

    def test_preserves_block_boundaries(self):
        out = html_to_text("<p>First line</p><p>Second line</p>")
        assert "First line" in out and "Second line" in out

    def test_unclosed_tags_do_not_raise(self):
        out = html_to_text("<div><p><b>Interview <i>Thursday<span>unclosed")
        assert "Interview" in out and "Thursday" in out

    def test_empty(self):
        assert html_to_text("") == ""


class TestQuoteStripping:
    def test_removes_reply_chain(self):
        body = (
            "Here is my answer.\n\n"
            "On Mon, Oct 10, 2025 at 9:00 AM Sarah <s@x.com> wrote:\n"
            "> original question\n"
        )
        clean, quoted = strip_quoted_and_boilerplate(body)
        assert clean == "Here is my answer."
        assert "original question" in quoted

    def test_removes_disclaimer(self):
        clean, _ = strip_quoted_and_boilerplate(
            "Real content here.\n\nConfidentiality Notice: this email is confidential."
        )
        assert "Confidentiality" not in clean
        assert "Real content here." in clean

    def test_removes_signature_separator(self):
        clean, _ = strip_quoted_and_boilerplate("Body text.\n--\nSarah Chen\nRecruiter")
        assert clean == "Body text."

    def test_all_boilerplate_falls_back_to_full_text(self):
        """Stripping must never return an empty document for a non-empty body."""
        clean, _ = strip_quoted_and_boilerplate("> only quoted content")
        assert clean.strip()


class TestHeaders:
    def test_parse_date_header(self):
        dt = parse_date_header("Fri, 3 Oct 2025 10:00:00 -0400")
        assert dt == datetime(2025, 10, 3, 14, 0, tzinfo=timezone.utc)

    def test_parse_garbage_date_returns_none(self):
        assert parse_date_header("not a date") is None
        assert parse_date_header(None) is None

    def test_split_addresses(self):
        assert split_addresses("A <a@x.com>, B <b@y.com>") == ["a@x.com", "b@y.com"]
        assert split_addresses("garbage") == []

    def test_split_references(self):
        assert split_references("<a@x> <b@y>") == ["<a@x>", "<b@y>"]


class TestGmailPayload:
    def test_full_payload(self):
        raw = raw_from_gmail_payload({
            "id": "m1", "threadId": "t1", "snippet": "snip", "labelIds": ["INBOX"],
            "payload": {
                "headers": [
                    {"name": "From", "value": "Sarah Chen <sarah@acme.com>"},
                    {"name": "Subject", "value": "Interview"},
                    {"name": "Date", "value": "Fri, 3 Oct 2025 10:00:00 +0000"},
                    {"name": "Message-ID", "value": "<abc@acme.com>"},
                ],
                "parts": [{"mimeType": "text/plain", "body": {"data": _b64("Body")}}],
            },
        })
        assert raw.from_addr == "sarah@acme.com"
        assert raw.from_name == "Sarah Chen"
        assert raw.rfc_message_id == "<abc@acme.com>"
        assert raw.body_text == "Body"
        assert raw.body_fetched is True

    def test_nested_multipart(self):
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [{
                "mimeType": "multipart/alternative",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": _b64("plain text")}},
                    {"mimeType": "text/html", "body": {"data": _b64("<p>html</p>")}},
                ],
            }],
        }
        plain, html = extract_body_parts(payload)
        assert plain == "plain text" and "html" in html

    def test_attachments_are_ignored(self):
        payload = {"parts": [
            {"mimeType": "text/plain", "filename": "cv.txt",
             "body": {"data": _b64("ATTACHMENT")}},
            {"mimeType": "text/plain", "body": {"data": _b64("REAL")}},
        ]}
        plain, _ = extract_body_parts(payload)
        assert plain == "REAL"

    def test_internal_date_fallback(self):
        raw = raw_from_gmail_payload({
            "id": "m1", "threadId": "t1", "internalDate": "1759485600000",
            "payload": {"headers": []},
        })
        assert raw.date is not None and raw.date.year == 2025

    def test_missing_id_raises(self):
        with pytest.raises(ValueError):
            raw_from_gmail_payload({"threadId": "t1"})

    def test_non_dict_raises(self):
        with pytest.raises(ValueError):
            raw_from_gmail_payload("not a dict")  # type: ignore[arg-type]

    def test_malformed_headers_tolerated(self):
        raw = raw_from_gmail_payload({"id": "x", "payload": {"headers": "not-a-list"}})
        assert raw.gmail_id == "x"
        assert raw.thread_id == "x"  # falls back to the message id

    def test_bad_base64_tolerated(self):
        assert decode_b64url("!!!not base64!!!") in (b"", b"\x8e\xdb")


class TestParseMessage:
    def test_records_errors_without_raising(self):
        msg = parse_message(RawMessage(gmail_id="x", thread_id="t"))
        assert "empty_body" in msg.parse_errors
        assert "missing_date" in msg.parse_errors
        assert "missing_or_invalid_sender" in msg.parse_errors
        assert "missing_subject" in msg.parse_errors

    def test_falls_back_to_snippet(self):
        msg = parse_message(RawMessage(
            gmail_id="x", thread_id="t", snippet="Interview on Thursday",
            from_addr="a@b.com", subject="Hi", date=datetime.now(timezone.utc),
        ))
        assert "Interview on Thursday" in msg.clean_text
        assert msg.parse_errors == []

    def test_html_only_body(self):
        msg = parse_message(RawMessage(
            gmail_id="x", thread_id="t", from_addr="a@b.com", subject="S",
            date=datetime.now(timezone.utc), body_html="<p>Hello <b>world</b></p>",
        ))
        assert "Hello" in msg.clean_text and "<p>" not in msg.clean_text
