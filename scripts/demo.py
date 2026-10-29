#!/usr/bin/env python3
"""Run the reconstruction pipeline over the synthetic fixtures and print a report.

    python scripts/demo.py             # use best available embedding backend
    python scripts/demo.py --hashed    # force the deterministic fallback
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interviewtrack.embeddings import get_backend
from interviewtrack.enums import MatchDecisionKind
from interviewtrack.fixtures.synthetic import (
    ALL_MESSAGES,
    EXPECTED,
    MUST_MERGE,
    MUST_NOT_MERGE,
    USER_EMAIL,
)
from interviewtrack.parsing import raw_from_gmail_payload
from interviewtrack.fixtures.synthetic import gmail_messages_as_api_payloads
from interviewtrack.reconstruct import reconstruct
from interviewtrack.status import event_label

NOW = datetime(2025, 10, 28, tzinfo=timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hashed", action="store_true", help="force deterministic fallback")
    ap.add_argument("--via-api", action="store_true",
                    help="route fixtures through the Gmail API payload parser")
    args = ap.parse_args()

    backend = get_backend(force_hashed=args.hashed, refresh=True)

    if args.via_api:
        messages = [raw_from_gmail_payload(p) for p in gmail_messages_as_api_payloads()]
        source = "Gmail API payload parser"
    else:
        messages = ALL_MESSAGES
        source = "in-memory fixtures"

    print("=" * 78)
    print("InterviewTrack AI — synthetic reconstruction demo")
    print("=" * 78)
    print(f"Input          : {len(messages)} messages via {source}")
    print(f"Embeddings     : {backend.describe()}")
    print(f"Semantic tier  : {'NEURAL' if backend.is_semantic else 'deterministic fallback'}")
    print()

    result = reconstruct(messages, user_email=USER_EMAIL, backend=backend, now=NOW)

    print(f"Classified as recruitment : {result.processed_count}")
    print(f"Filtered as non-recruitment: {len(result.skipped_non_recruitment)} "
          f"{sorted(result.skipped_non_recruitment)}")
    print(f"Messages with parse issues : {sorted(result.parse_error_ids)}")
    print(f"Reconstructed applications : {len(result.applications)}")
    print()

    for app in result.applications:
        flag = "  [NEEDS REVIEW]" if app.needs_review else ""
        print("-" * 78)
        print(f"{app.display_title}{flag}")
        print(f"  key={app.key}  status={app.status.value}  stage={app.current_stage.value}")
        print(f"  requisition={app.requisition_id}  threads={app.thread_count} "
              f"{sorted(set(app.thread_ids))}")
        print(f"  match_confidence={app.match_confidence:.2f}")
        print(f"  latest activity: {app.latest_activity_label} "
              f"({app.latest_activity_at.date() if app.latest_activity_at else 'n/a'})")
        print(f"  last supported milestone: {app.last_supported_milestone}")
        if app.recruiters:
            print(f"  recruiters: " + ", ".join(
                f"{r.name or r.email} <{r.email}>" for r in app.recruiters))
        if app.history_possibly_incomplete:
            print("  ! history possibly incomplete")
        print("  timeline:")
        for ev in app.timeline:
            when = ev.occurred_at.strftime("%b %d") if ev.occurred_at else "  ?   "
            print(f"    {when}  {event_label(ev.event_type):32s} [{ev.occurrence.value:9s}] "
                  f"{ev.gmail_id}/{ev.thread_id}  {ev.subject[:40]}")
        for note in app.notes:
            print(f"  note: {note}")

    # ---- correctness checks against ground truth ----
    print()
    print("=" * 78)
    print("GROUND-TRUTH CHECKS")
    print("=" * 78)

    app_of: dict[str, str] = {}
    for app in result.applications:
        for ev in app.events:
            app_of[ev.gmail_id] = app.key

    failures = 0

    print("\nMUST MERGE (cross-thread reconstruction):")
    for a, b, why in MUST_MERGE:
        ok = app_of.get(a) is not None and app_of.get(a) == app_of.get(b)
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {a}+{b}: {why}")

    print("\nMUST NOT MERGE (separation):")
    for a, b, why in MUST_NOT_MERGE:
        ok = app_of.get(a) != app_of.get(b) or app_of.get(a) is None
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {a}|{b}: {why}")

    print("\nEXPECTED APPLICATIONS:")
    for exp in EXPECTED:
        keys = {app_of.get(g) for g in exp.gmail_ids if app_of.get(g)}
        one_app = len(keys) == 1
        app = result.by_key(next(iter(keys))) if one_app else None
        covered = sum(1 for g in exp.gmail_ids if app and any(
            e.gmail_id == g for e in app.events))
        ok = one_app and covered == len(exp.gmail_ids)
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {exp.label}")
        print(f"         -> {len(keys)} application(s), {covered}/{len(exp.gmail_ids)} messages")
        if app:
            if exp.expect_status and app.status.value != exp.expect_status:
                failures += 1
                print(f"         !! status {app.status.value} != expected {exp.expect_status}")
            if exp.expect_stage and app.current_stage.value != exp.expect_stage:
                failures += 1
                print(f"         !! stage {app.current_stage.value} != expected {exp.expect_stage}")
            if exp.expect_needs_review and not app.needs_review:
                failures += 1
                print("         !! expected needs_review=True")

    # ---- decision breakdown ----
    counts: dict[str, int] = {}
    for d in result.decisions:
        counts[d.kind.value] = counts.get(d.kind.value, 0) + 1
    print(f"\nDecision breakdown (all pairs scored): {counts}")

    print()
    print("=" * 78)
    print(f"RESULT: {'ALL GROUND-TRUTH CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    print("=" * 78)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
