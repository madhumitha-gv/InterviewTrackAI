#!/usr/bin/env python3
"""Export the synthetic reconstruction to JSON for the static viewer.

Reuses the production pipeline end to end — the Gmail API payload parser, the
metadata-first retrieval path and the reconstruction engine — so the exported data is
exactly what the Streamlit app shows, not a hand-built sample.

    python scripts/export_json.py [output.json]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interviewtrack.config import get_settings
from interviewtrack.embeddings import get_backend
from interviewtrack.fixtures.synthetic import USER_EMAIL, gmail_messages_as_api_payloads
from interviewtrack.gmail.client import FixtureTransport, fetch_messages
from interviewtrack.gmail.links import best_message_link, thread_link
from interviewtrack.reconstruct import reconstruct
from interviewtrack.status import event_label

NOW = datetime(2025, 10, 28, tzinfo=timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def build() -> dict:
    settings = get_settings()
    backend = get_backend()

    transport = FixtureTransport(gmail_messages_as_api_payloads(), page_size=8)
    messages, report = fetch_messages(transport, settings)
    result = reconstruct(messages, user_email=USER_EMAIL, backend=backend,
                         settings=settings, now=NOW)

    apps = []
    for app in result.applications:
        apps.append({
            "key": app.key,
            "employer": app.employer,
            "job_title": app.job_title,
            "requisition_id": app.requisition_id,
            "status": app.status.value,
            "stage": app.current_stage.value,
            "latest_activity_label": app.latest_activity_label,
            "latest_activity_at": _iso(app.latest_activity_at),
            "last_milestone": app.last_supported_milestone,
            "match_confidence": round(app.match_confidence, 3),
            "needs_review": app.needs_review,
            "history_possibly_incomplete": app.history_possibly_incomplete,
            "notes": app.notes,
            "thread_ids": sorted(set(app.thread_ids)),
            "thread_count": app.thread_count,
            "threads": [
                {"thread_id": t, "url": thread_link(t),
                 "message_count": sum(1 for e in app.events if e.thread_id == t)}
                for t in sorted(set(app.thread_ids))
            ],
            "recruiters": [
                {"name": r.name, "email": r.email, "organization": r.organization}
                for r in app.recruiters
            ],
            "events": [
                {
                    "gmail_id": e.gmail_id,
                    "thread_id": e.thread_id,
                    "occurred_at": _iso(e.occurred_at),
                    "event_type": e.event_type.value,
                    "event_label": event_label(e.event_type),
                    "occurrence": e.occurrence.value,
                    "sender": e.sender,
                    "sender_name": e.sender_name,
                    "subject": e.subject,
                    "preview": e.preview,
                    "evidence": e.evidence,
                    "confidence": round(e.confidence, 3),
                    "event_datetime": _iso(e.event_datetime),
                    "gmail_url": best_message_link(e.rfc_message_id, e.thread_id),
                }
                for e in app.timeline
            ],
            "decisions": [
                {
                    "gmail_id": d.gmail_id,
                    "kind": d.kind.value,
                    "score": round(d.score, 3),
                    "source": d.source.value,
                    "reasons": d.reasons,
                    "blockers": d.blockers,
                    "features": {
                        "requisition_match": d.features.requisition_match,
                        "employer_match": d.features.employer_match,
                        "role_similarity": round(d.features.role_similarity, 3),
                        "semantic_similarity": round(d.features.semantic_similarity, 3),
                        "same_gmail_thread": d.features.same_gmail_thread,
                        "thread_reference_link": d.features.thread_reference_link,
                        "shared_recruiter": d.features.shared_recruiter,
                        "days_apart": round(d.features.days_apart, 1),
                    },
                }
                for d in app.decisions
                if d.kind.value != "NO_MATCH"
            ],
        })

    return {
        "generated_for": "2025-10-28",
        "embedding_backend": backend.describe(),
        "semantic_enabled": backend.is_semantic,
        "thresholds": {
            "auto_match": settings.auto_match_threshold,
            "review": settings.review_threshold,
            "semantic_only_cap": settings.weights.semantic_only_cap,
        },
        "retrieval": {
            "query": report.query,
            "listed": report.listed,
            "metadata_fetched": report.metadata_fetched,
            "bodies_fetched": report.bodies_fetched,
        },
        "totals": {
            "applications": len(result.applications),
            "events": sum(len(a.events) for a in result.applications),
            "threads": len({t for a in result.applications for t in a.thread_ids}),
            "messages_classified": result.processed_count,
            "filtered_non_recruitment": len(result.skipped_non_recruitment),
        },
        "applications": apps,
    }


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reconstruction.json")
    data = build()
    out.write_text(json.dumps(data, indent=2))
    t = data["totals"]
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    print(f"  {t['applications']} applications, {t['events']} events, "
          f"{t['threads']} threads, {t['filtered_non_recruitment']} filtered")
    print(f"  backend: {data['embedding_backend']}")
