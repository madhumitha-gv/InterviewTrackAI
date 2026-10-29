#!/usr/bin/env python3
"""Rules-only vs rules+embeddings, measured on the HELD-OUT fixtures.

The holdout was never used to tune a threshold. It is also small (8 messages), so
every number here is preliminary and is reported as such — no metric is invented.

    python scripts/evaluate_matching.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interviewtrack.config import get_settings
from interviewtrack.embeddings import get_backend
from interviewtrack.enums import MatchDecisionKind
from interviewtrack.fixtures.synthetic import (
    DEV_MESSAGES,
    FULL_DATASET,
    HOLDOUT_EXPECTED,
    HOLDOUT_MESSAGES,
    HOLDOUT_MUST_MERGE,
    HOLDOUT_MUST_NOT_MERGE,
    USER_EMAIL,
)
from interviewtrack.reconstruct import reconstruct

NOW = datetime(2025, 10, 28, tzinfo=timezone.utc)


@dataclass
class Result:
    label: str
    available: bool = True
    note: str = ""
    false_merges: int = 0
    missed_links: int = 0
    review_count: int = 0
    applications: int = 0
    excluded_correctly: int = 0
    detail: list[str] = field(default_factory=list)


def run(label: str, use_embeddings: bool, require_neural: bool = False) -> Result:
    settings = get_settings()
    if not use_embeddings:
        settings = settings.model_copy(
            update={"weights": settings.weights.model_copy(
                update={"semantic_similarity": 0.0})}
        )

    backend = get_backend(force_hashed=not require_neural, refresh=True)
    if require_neural and not backend.is_semantic:
        return Result(label, available=False,
                      note=f"no neural model could be loaded — {backend.describe()}")

    res = reconstruct(FULL_DATASET, user_email=USER_EMAIL, backend=backend,
                      settings=settings, now=NOW)
    placed = {e.gmail_id: a.key for a in res.applications for e in a.events}

    out = Result(label)
    out.applications = len(res.applications)
    out.review_count = sum(1 for a in res.applications if a.needs_review)

    # Pairwise over the labelled holdout only.
    labelled = {k: v for k, v in HOLDOUT_EXPECTED.items() if v is not None}
    for a, b in combinations(sorted(labelled), 2):
        same_gold = labelled[a] == labelled[b]
        pa, pb = placed.get(a), placed.get(b)
        same_pred = pa is not None and pa == pb
        if same_pred and not same_gold:
            out.false_merges += 1
            out.detail.append(f"FALSE MERGE: {a} + {b}")
        elif same_gold and not same_pred:
            out.missed_links += 1
            out.detail.append(f"MISSED LINK: {a} + {b}")

    # Messages expected to be excluded or sent to review.
    for gid, gold in HOLDOUT_EXPECTED.items():
        if gold is None:
            app = next((a for a in res.applications
                        if any(e.gmail_id == gid for e in a.events)), None)
            if app is None or app.needs_review:
                out.excluded_correctly += 1
            else:
                out.detail.append(
                    f"NOT FLAGGED: {gid} silently joined {app.employer}/{app.job_title}"
                )
    return out


def main() -> int:
    print("=" * 84)
    print("InterviewTrack AI — matching evaluation on the HELD-OUT fixture set")
    print("=" * 84)
    print(f"Dev set: {len(DEV_MESSAGES)} messages (thresholds tuned here)")
    print(f"Holdout: {len(HOLDOUT_MESSAGES)} messages (NEVER tuned against)")
    print()

    results = [
        run("rules only (semantic weight = 0)", use_embeddings=False),
        run("rules + deterministic fallback vectorizer", use_embeddings=True),
        run("rules + neural embeddings", use_embeddings=True, require_neural=True),
    ]

    header = (f"{'configuration':<44} {'false':>6} {'missed':>7} {'review':>7} "
              f"{'flagged':>8} {'apps':>5}")
    print(header)
    print(f"{'':<44} {'merges':>6} {'links':>7} {'items':>7} {'correctly':>8}")
    print("-" * len(header))
    for r in results:
        if not r.available:
            print(f"{r.label:<44} {'UNAVAILABLE':>36}")
            print(f"{'':<44} {r.note}")
            continue
        n_excl = sum(1 for v in HOLDOUT_EXPECTED.values() if v is None)
        print(f"{r.label:<44} {r.false_merges:>6} {r.missed_links:>7} "
              f"{r.review_count:>7} {r.excluded_correctly:>4}/{n_excl:<3} {r.applications:>5}")

    print()
    print("Required holdout invariants:")
    settings = get_settings()
    backend = get_backend(force_hashed=True, refresh=True)
    res = reconstruct(FULL_DATASET, user_email=USER_EMAIL, backend=backend,
                      settings=settings, now=NOW)
    placed = {e.gmail_id: a.key for a in res.applications for e in a.events}
    for a, b, why in HOLDOUT_MUST_MERGE:
        ok = placed.get(a) is not None and placed.get(a) == placed.get(b)
        print(f"  [{'PASS' if ok else 'FAIL'}] {why}")
    for a, b, why in HOLDOUT_MUST_NOT_MERGE:
        ok = placed.get(a) != placed.get(b) or placed.get(a) is None
        print(f"  [{'PASS' if ok else 'FAIL'}] {why}")

    for r in results:
        if r.available and r.detail:
            print(f"\nIssues — {r.label}:")
            for line in r.detail:
                print(f"  {line}")

    usable = [r for r in results if r.available]
    print()
    print("Observations (measured, not estimated):")
    if len(usable) >= 2:
        rules, hybrid = usable[0], usable[1]
        if (rules.false_merges, rules.missed_links) == (hybrid.false_merges, hybrid.missed_links):
            print("  • On this holdout, adding the deterministic vectorizer changes nothing:")
            print(f"    both configurations give {rules.false_merges} false merges and "
                  f"{rules.missed_links} missed links.")
            print("    The hard constraints and entity normalization are doing the work.")
        else:
            print(f"  • Embeddings change false merges {rules.false_merges}→"
                  f"{hybrid.false_merges}, missed links {rules.missed_links}→"
                  f"{hybrid.missed_links}.")
    if not any(r.available and "neural" in r.label for r in results):
        print("  • The neural configuration could NOT be measured in this environment.")
        print("    No claim is made about its effect. Run this where huggingface.co is")
        print("    reachable to obtain real numbers.")
    print()
    print(f"  ⚠ The holdout is {len(HOLDOUT_MESSAGES)} messages. These results are PRELIMINARY and cannot")
    print("    support a claim about real-world accuracy. The fixtures were authored")
    print("    for this project and therefore encode its own assumptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
