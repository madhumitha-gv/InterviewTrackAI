"""Persistence, per-user isolation, sticky corrections and data deletion."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from interviewtrack.enums import ApplicationStatus
from interviewtrack.fixtures.synthetic import ALL_MESSAGES, USER_EMAIL
from interviewtrack.models import Correction
from interviewtrack.service import DEMO_USER_ID, FILTERS, search_applications, sort_applications

from .conftest import NOW


class TestMessagePersistence:
    def test_roundtrip(self, db):
        db.upsert_user("u1", USER_EMAIL)
        db.upsert_messages("u1", ALL_MESSAGES)
        loaded = db.load_messages("u1")
        assert len(loaded) == len(ALL_MESSAGES)
        original = {m.gmail_id: m for m in ALL_MESSAGES}
        for m in loaded:
            assert m.subject == original[m.gmail_id].subject
            assert m.thread_id == original[m.gmail_id].thread_id

    def test_upsert_is_idempotent(self, db):
        db.upsert_user("u1", USER_EMAIL)
        db.upsert_messages("u1", ALL_MESSAGES)
        db.upsert_messages("u1", ALL_MESSAGES)
        assert len(db.load_messages("u1")) == len(ALL_MESSAGES)

    def test_metadata_resync_does_not_erase_a_fetched_body(self, db):
        db.upsert_user("u1", USER_EMAIL)
        full = ALL_MESSAGES[0]
        db.upsert_messages("u1", [full])
        stub = full.model_copy(update={"body_text": "", "body_html": "", "body_fetched": False})
        db.upsert_messages("u1", [stub])
        assert db.load_messages("u1")[0].body_text == full.body_text

    def test_per_user_isolation(self, db):
        db.upsert_user("u1", "a@x.com")
        db.upsert_user("u2", "b@x.com")
        db.upsert_messages("u1", ALL_MESSAGES)
        assert len(db.load_messages("u2")) == 0
        assert db.known_gmail_ids("u2") == set()


class TestApplicationPersistence:
    def test_roundtrip_preserves_reconstruction(self, service):
        service.load_demo_data(now=NOW)
        apps = service.applications(DEMO_USER_ID)
        assert apps
        northwind = next(
            a for a in apps if a.employer_canonical == "northwind analytics"
            and a.role_family == "ai_engineer"
        )
        assert northwind.thread_count >= 4
        assert len(northwind.events) == 8
        assert northwind.timeline[0].occurred_at <= northwind.timeline[-1].occurred_at

    def test_decisions_are_stored(self, service):
        service.load_demo_data(now=NOW)
        rows = service.decisions_for(DEMO_USER_ID, "m006")
        assert rows
        assert any(r["kind"] == "AUTO_MATCH" for r in rows)


class TestStickyCorrections:
    def test_merge_survives_resync(self, service):
        service.load_demo_data(now=NOW)
        apps = service.applications(DEMO_USER_ID)
        # Pick the specific ambiguous record (employer never identified), not merely
        # "the first flagged one" — several fixtures now land in review.
        review = next(a for a in apps if a.needs_review and not a.employer_canonical)
        target = next(a for a in apps if a.employer_canonical == "initech data")

        service.merge_applications(DEMO_USER_ID, USER_EMAIL, review.key, target.key, now=NOW)
        after = service.applications(DEMO_USER_ID)
        assert all(a.key != review.key for a in after)
        merged = next(a for a in after if a.key == target.key)
        assert {"m017", "m022"} <= {e.gmail_id for e in merged.events}

        # Re-sync: the correction must be re-applied, not lost.
        service.load_demo_data(now=NOW)
        resynced = service.applications(DEMO_USER_ID)
        merged2 = next(a for a in resynced if a.key == target.key)
        assert {"m017", "m022"} <= {e.gmail_id for e in merged2.events}
        assert all(a.key != review.key for a in resynced)

    def test_split_survives_resync(self, service):
        service.load_demo_data(now=NOW)
        apps = service.applications(DEMO_USER_ID)
        northwind = next(
            a for a in apps if a.employer_canonical == "northwind analytics"
            and a.role_family == "ai_engineer"
        )
        service.split_message(DEMO_USER_ID, USER_EMAIL, northwind.key, "m006", now=NOW)

        after = service.applications(DEMO_USER_ID)
        host = next(a for a in after if a.key == northwind.key)
        assert "m006" not in {e.gmail_id for e in host.events}

        service.load_demo_data(now=NOW)
        resynced = service.applications(DEMO_USER_ID)
        host2 = next(a for a in resynced if a.key == northwind.key)
        assert "m006" not in {e.gmail_id for e in host2.events}, (
            "a manual split must not be undone by a later synchronization"
        )

    def test_field_correction_survives_resync(self, service):
        service.load_demo_data(now=NOW)
        app = next(a for a in service.applications(DEMO_USER_ID) if a.needs_review)
        service.correct_fields(DEMO_USER_ID, USER_EMAIL, app.key,
                               employer="Stealth Startup", job_title="Data Scientist",
                               now=NOW)
        service.load_demo_data(now=NOW)
        fixed = next(a for a in service.applications(DEMO_USER_ID) if a.key == app.key)
        assert fixed.employer == "Stealth Startup"
        assert fixed.job_title == "Data Scientist"
        assert fixed.needs_review is False

    def test_confirm_review_clears_the_flag(self, service):
        service.load_demo_data(now=NOW)
        app = next(a for a in service.applications(DEMO_USER_ID) if a.needs_review)
        service.confirm_review(DEMO_USER_ID, USER_EMAIL, app.key, now=NOW)
        after = next(a for a in service.applications(DEMO_USER_ID) if a.key == app.key)
        assert after.needs_review is False

    def test_invalid_correction_does_not_break_a_sync(self, service):
        service.load_demo_data(now=NOW)
        service.db.add_correction(Correction(
            user_id=DEMO_USER_ID, kind="merge",
            application_key="does_not_exist", target_key="also_missing",
        ))
        outcome = service.load_demo_data(now=NOW)
        assert outcome.applications > 0


class TestDeletion:
    def test_delete_removes_everything(self, service):
        service.load_demo_data(now=NOW)
        assert service.applications(DEMO_USER_ID)
        service.delete_everything(DEMO_USER_ID)
        assert service.applications(DEMO_USER_ID) == []
        assert service.db.load_messages(DEMO_USER_ID) == []
        assert service.db.load_corrections(DEMO_USER_ID) == []
        assert service.db.get_user(DEMO_USER_ID) is None


class TestSearchAndFilters:
    def test_search_by_company(self, service):
        service.load_demo_data(now=NOW)
        apps = service.applications(DEMO_USER_ID)
        assert len(search_applications(apps, "northwind")) == 2
        assert len(search_applications(apps, "globex")) == 1

    def test_search_by_position(self, service):
        service.load_demo_data(now=NOW)
        apps = service.applications(DEMO_USER_ID)
        assert len(search_applications(apps, "data scientist")) == 2

    def test_search_by_recruiter(self, service):
        service.load_demo_data(now=NOW)
        apps = service.applications(DEMO_USER_ID)
        assert len(search_applications(apps, "priya")) == 1

    def test_empty_query_returns_all(self, service):
        service.load_demo_data(now=NOW)
        apps = service.applications(DEMO_USER_ID)
        assert len(search_applications(apps, "")) == len(apps)

    def test_filters(self, service):
        service.load_demo_data(now=NOW)
        apps = service.applications(DEMO_USER_ID)
        assert len([a for a in apps if FILTERS["All"](a)]) == len(apps)
        assert len([a for a in apps if FILTERS["Rejected"](a)]) == 1
        assert len([a for a in apps if FILTERS["Offers"](a)]) == 1
        # Several fixtures are deliberately ambiguous; assert the filter selects
        # exactly the flagged ones rather than a fixed count.
        flagged = [a for a in apps if FILTERS["Needs review"](a)]
        assert flagged
        assert all(a.needs_review for a in flagged)
        assert len(flagged) == len([a for a in apps if a.needs_review])
        assert any(not a.employer_canonical for a in flagged), (
            "the unidentified-employer record must be among those needing review"
        )

    def test_rejected_applications_are_preserved_not_hidden(self, service):
        service.load_demo_data(now=NOW)
        apps = sort_applications(service.applications(DEMO_USER_ID))
        assert any(a.status == ApplicationStatus.REJECTED for a in apps)
        # Active work sorts ahead of closed work.
        statuses = [a.status for a in apps]
        assert statuses.index(ApplicationStatus.OFFER) < statuses.index(
            ApplicationStatus.REJECTED
        ), "active work must sort ahead of closed work"


class TestSyncState:
    def test_sync_state_recorded(self, service):
        service.load_demo_data(now=NOW)
        state = service.sync_state(DEMO_USER_ID)
        assert state and state["message_count"] > 0
        assert state["last_sync_at"]
