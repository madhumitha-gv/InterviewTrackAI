#!/usr/bin/env python3
"""Cross-inbox generalization report.

Runs the whole pipeline over three structurally-distinct mailbox profiles and prints
measured outcomes per profile: retrieval coverage, recruitment-detection recall,
correct cross-thread associations, false merges, unresolved (REVIEW) count and parse
failures. Numbers are measured, not asserted. Profiles are synthetic.

    python scripts/evaluate_cross_inbox.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interviewtrack.config import get_settings
from interviewtrack.embeddings import get_backend
from interviewtrack.fixtures.mailboxes import ALL_PROFILES
from interviewtrack.fixtures.synthetic import gmail_messages_as_api_payloads
from interviewtrack.gmail.client import FixtureTransport, fetch_messages
from interviewtrack.reconstruct import reconstruct

NOW = datetime(2025, 11, 1, tzinfo=timezone.utc)


def main() -> int:
    settings = get_settings()
    backend = get_backend(force_hashed=True, refresh=True)

    print("=" * 92)
    print("InterviewTrack AI — cross-inbox generalization")
    print("=" * 92)
    print(f"Embedding backend: {backend.describe()}")
    print("Same application code, three structurally-different synthetic mailboxes.")
    print("Companies, domains and labels below appear in NO hardcoded engine list.\n")

    header = (f"{'profile':<16} {'retrieved':>10} {'recall':>8} {'apps':>6} "
              f"{'merges':>8} {'seps':>6} {'review':>7} {'noise_leak':>11} {'parse_err':>10}")
    print(header)
    print("-" * len(header))

    all_pass = True
    for prof in ALL_PROFILES:
        payloads = gmail_messages_as_api_payloads(prof.messages)
        messages, report = fetch_messages(
            FixtureTransport(payloads, page_size=4), settings
        )
        result = reconstruct(messages, user_email=prof.user_email, backend=backend,
                             settings=settings, now=NOW)
        placed = {e.gmail_id: a.key for a in result.applications for e in a.events}

        found = sum(1 for g in prof.recruitment_ids if g in placed)
        recall = found / max(1, len(prof.recruitment_ids))
        merges = sum(1 for a, b, _ in prof.must_merge
                     if placed.get(a) and placed.get(a) == placed.get(b))
        seps = sum(1 for a, b, _ in prof.must_not_merge
                   if placed.get(a) != placed.get(b) or placed.get(a) is None)
        review = sum(1 for a in result.applications if a.needs_review)
        noise_leak = sum(1 for g in prof.excluded_ids if g in placed)

        ok = (found == len(prof.recruitment_ids) and noise_leak == 0
              and merges == len(prof.must_merge) and seps == len(prof.must_not_merge)
              and len(result.applications) == prof.expected_applications)
        all_pass &= ok

        print(f"{prof.key:<16} {len(messages):>4}/{len(prof.messages):<5} "
              f"{recall:>7.0%} {len(result.applications):>3}/{prof.expected_applications:<2} "
              f"{merges:>3}/{len(prof.must_merge):<4} {seps:>2}/{len(prof.must_not_merge):<3} "
              f"{review:>7} {noise_leak:>11} {len(result.parse_error_ids):>10}")

    print()
    print("Columns: retrieved = listed/total; recall = recruitment messages placed;")
    print("apps = reconstructed/expected; merges & seps = required associations held;")
    print("review = NEEDS_REVIEW apps; noise_leak = noise wrongly attached (0 is good);")
    print("parse_err = messages with recorded parse issues (handled, not dropped).")
    print()
    print("Profiles:")
    for prof in ALL_PROFILES:
        print(f"  {prof.key:<14} {prof.label}  ({prof.user_email})")
    print()
    print(f"RESULT: {'ALL PROFILES GENERALIZE' if all_pass else 'SOME PROFILE FAILED'}")
    print()
    print("Note: profiles are synthetic and small; these demonstrate that the same")
    print("code path handles different inbox structures, not real-world accuracy.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
