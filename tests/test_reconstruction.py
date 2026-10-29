"""The eight behaviors required by section 9 of the spec.

Each test maps to a numbered requirement and asserts on the reconstruction of the
labeled synthetic dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from interviewtrack.enums import ApplicationStatus, EventType, MatchDecisionKind, Occurrence, Stage
from interviewtrack.fixtures.synthetic import (
    ALL_MESSAGES,
    EXPECTED,
    EXPECTED_NON_RECRUITMENT,
    MUST_MERGE,
    MUST_NOT_MERGE,
    USER_EMAIL,
)
from interviewtrack.normalize import canonical_company
from interviewtrack.reconstruct import UNKNOWN_EMPLOYER, reconstruct

from .conftest import NOW


def app_for(result, gmail_id):
    return next((a for a in result.applications if any(e.gmail_id == gmail_id for e in a.events)), None)


# =============================================================================
# Test 1 — cross-thread matching (THE central success criterion)
# =============================================================================

class TestT1CrossThreadMatching:
    def test_four_unrelated_threads_become_one_application(self, result):
        """External recruiter + company HR + scheduling platform + interviewer."""
        app = app_for(result, "m001")
        assert app is not None
        ids = {e.gmail_id for e in app.events}
        assert {"m001", "m003", "m005"} <= ids, (
            "recruiter, HR and scheduling threads must reconstruct into one application"
        )
        assert app.thread_count >= 4, f"expected >= 4 Gmail threads, got {app.thread_count}"
        assert canonical_company(app.employer) == "northwind analytics"

    def test_all_eight_northwind_messages_present(self, result):
        app = app_for(result, "m001")
        expected = {"m001", "m002", "m003", "m004", "m005", "m006", "m007", "m008"}
        assert {e.gmail_id for e in app.events} == expected

    @pytest.mark.parametrize("a,b,why", MUST_MERGE, ids=[m[2][:40] for m in MUST_MERGE])
    def test_pairs_that_must_merge(self, app_of, a, b, why):
        assert app_of.get(a) is not None, f"{a} was not placed in any application"
        assert app_of[a] == app_of.get(b), why

    def test_timeline_is_chronological(self, result):
        app = app_for(result, "m001")
        dates = [e.occurred_at for e in app.timeline]
        assert dates == sorted(dates)

    def test_every_event_keeps_its_gmail_identifiers(self, result):
        for app in result.applications:
            for ev in app.events:
                assert ev.gmail_id and ev.thread_id


# =============================================================================
# Test 2 — semantic matching across different subjects and sender domains
# =============================================================================

class TestT2SemanticMatching:
    def test_personal_domain_interviewer_joins_the_application(self, app_of):
        """m006 shares no thread, no requisition and no corporate domain with m005."""
        assert app_of.get("m006") == app_of.get("m005")

    def test_association_is_backed_by_supported_context(self, result):
        app = app_for(result, "m006")
        decision = next(d for d in app.decisions if d.gmail_id == "m006")
        assert decision.kind == MatchDecisionKind.AUTO_MATCH
        # The merge must rest on real identity evidence, not similarity alone.
        assert decision.features.employer_match is True
        assert not decision.blockers


# =============================================================================
# Test 3 — similar language, different employers: must NOT merge
# =============================================================================

class TestT3DifferentEmployers:
    def test_near_identical_emails_stay_separate(self, app_of):
        assert app_of["m011"] != app_of["m012"]

    def test_employer_conflict_is_a_hard_blocker(self, result):
        decisions = [
            d for d in result.decisions
            if d.gmail_id == "m012" and d.features.employer_conflict
        ]
        assert decisions, "expected at least one employer-conflict decision for m012"
        assert all(d.kind == MatchDecisionKind.NO_MATCH for d in decisions)

    @pytest.mark.parametrize("a,b,why", MUST_NOT_MERGE, ids=[m[2][:40] for m in MUST_NOT_MERGE])
    def test_pairs_that_must_not_merge(self, app_of, a, b, why):
        assert app_of.get(a) != app_of.get(b) or app_of.get(a) is None, why

    def test_high_semantic_similarity_alone_never_merges(self, backend, settings):
        """The central safety rule, asserted directly on the scorer."""
        from interviewtrack.matching import ApplicationContext, decide
        from interviewtrack.classify import classify_message
        from interviewtrack.parsing import parse_message
        from interviewtrack.models import Application, RawMessage

        # A message with NO identifiable employer, deliberately worded to look like
        # the target application.
        raw = RawMessage(
            gmail_id="z1", thread_id="tz", from_addr="x@talentsphere.io",
            subject="Technical interview for a data role",
            date=datetime(2025, 10, 14, tzinfo=timezone.utc),
            body_text=("We would like to schedule your technical interview covering "
                       "statistics, machine learning fundamentals and live coding."),
            body_fetched=True, labels=["INBOX"],
        )
        msg = parse_message(raw)
        cls = classify_message(msg, user_email=USER_EMAIL)
        assert cls.entities.employer is None

        ctx = ApplicationContext(application=Application(
            key="app_target", employer="Globex Labs", employer_canonical="globex labs",
            job_title="Data Scientist",
        ))
        ctx.last_activity = datetime(2025, 10, 13, tzinfo=timezone.utc)

        # Force maximum possible semantic agreement.
        vec = ctx.ensure_embedding(backend)
        decision = decide(msg, cls, ctx, settings, msg_embedding=vec, backend=backend)

        assert decision.features.semantic_similarity > 0.99
        assert decision.kind != MatchDecisionKind.AUTO_MATCH, (
            "semantic similarity alone must never produce an automatic merge"
        )
        assert decision.score <= settings.weights.semantic_only_cap


# =============================================================================
# Test 4 — same employer, different roles: independent applications
# =============================================================================

class TestT4SameEmployerDifferentRoles:
    def test_two_separate_applications(self, app_of):
        assert app_of["m001"] != app_of["m009"]

    def test_rejection_closes_only_the_matching_application(self, result):
        analyst = app_for(result, "m009")
        engineer = app_for(result, "m001")

        assert analyst.status == ApplicationStatus.REJECTED

        assert engineer.status != ApplicationStatus.REJECTED, (
            "a rejection for Data Analyst must not close the AI Engineer application"
        )
        assert engineer.status in {
            ApplicationStatus.ACTION_REQUIRED, ApplicationStatus.SCHEDULED,
            ApplicationStatus.AWAITING_RESPONSE,
        }

    def test_requisition_conflict_prevents_merge(self, result):
        analyst = app_for(result, "m009")
        engineer = app_for(result, "m001")
        assert analyst.requisition_id == "REQ5120"
        assert engineer.requisition_id == "REQ4417"


# =============================================================================
# Test 5 — follow-up after an interview must not invent a new stage
# =============================================================================

class TestT5FollowUpDoesNotAdvanceStage:
    def test_latest_activity_reflects_the_followup(self, result):
        app = app_for(result, "m008")
        assert "Candidate follow-up" in app.latest_activity_label
        assert app.latest_activity_at.date().isoformat() == "2025-10-22"

    def test_stage_stays_at_the_last_supported_round(self, result):
        app = app_for(result, "m008")
        assert app.current_stage == Stage.TECHNICAL, (
            "a thank-you note must not advance the funnel beyond the technical round"
        )

    def test_scheduled_interview_is_not_marked_completed_by_clock_time(self, result):
        """The technical interview (Oct 21) is in the past relative to NOW (Oct 28)."""
        app = app_for(result, "m005")
        tech = [e for e in app.events if e.event_type == EventType.TECHNICAL_INTERVIEW]
        assert tech
        assert all(e.occurrence != Occurrence.COMPLETED for e in tech)
        assert "completion not confirmed" in app.last_supported_milestone


# =============================================================================
# Test 6 — ambiguous staffing message: review required, nothing invented
# =============================================================================

class TestT6AmbiguousStaffingMessage:
    def test_flagged_for_review(self, result):
        app = app_for(result, "m017")
        assert app is not None
        assert app.needs_review is True
        assert app.status == ApplicationStatus.NEEDS_REVIEW

    def test_no_employer_is_invented(self, result):
        app = app_for(result, "m017")
        assert app.employer == UNKNOWN_EMPLOYER
        assert not app.employer_canonical
        assert app.job_title is None

    def test_not_attached_to_a_real_application(self, app_of):
        for other in ("m001", "m009", "m011", "m012"):
            assert app_of["m017"] != app_of.get(other)

    def test_explains_why(self, result):
        app = app_for(result, "m017")
        assert any("could not be identified" in n for n in app.notes)

    def test_follow_up_stays_on_the_same_ambiguous_record(self, app_of):
        assert app_of["m017"] == app_of["m022"]


# =============================================================================
# Test 7 — malformed input is handled gracefully
# =============================================================================

class TestT7MalformedInput:
    def test_pipeline_does_not_raise(self, backend, settings):
        res = reconstruct(ALL_MESSAGES, user_email=USER_EMAIL, backend=backend,
                          settings=settings, now=NOW)
        assert res.applications

    def test_message_with_no_sender_date_or_body_is_skipped_not_fatal(self, result):
        assert "m018" in result.skipped_non_recruitment
        assert "m018" in result.parse_error_ids

    def test_broken_html_is_still_parsed_and_classified(self, result, app_of):
        """m019 has unclosed tags, embedded script/style and no plain-text part."""
        assert app_of.get("m019") is not None
        app = app_for(result, "m019")
        assert canonical_company(app.employer) == "acme robotics"
        ev = next(e for e in app.events if e.gmail_id == "m019")
        assert ev.event_type == EventType.TECHNICAL_INTERVIEW
        assert "drop me" not in ev.preview  # script contents stripped

    def test_empty_input(self, backend, settings):
        res = reconstruct([], user_email=USER_EMAIL, backend=backend, settings=settings)
        assert res.applications == []

    def test_none_dates_do_not_break_ordering(self, backend, settings):
        from interviewtrack.models import RawMessage
        msgs = list(ALL_MESSAGES) + [RawMessage(
            gmail_id="zz", thread_id="tzz", from_addr="a@b.com",
            subject="Technical interview", body_text="We'd like to schedule your technical interview.",
            body_fetched=True, date=None, labels=["INBOX"],
        )]
        res = reconstruct(msgs, user_email=USER_EMAIL, backend=backend, settings=settings, now=NOW)
        assert res.applications


# =============================================================================
# Test 8 — duplicate synchronization produces no duplicates
# =============================================================================

class TestT8DuplicateSync:
    def test_duplicate_messages_in_one_batch(self, backend, settings, result):
        doubled = list(ALL_MESSAGES) + list(ALL_MESSAGES)
        res = reconstruct(doubled, user_email=USER_EMAIL, backend=backend,
                          settings=settings, now=NOW)
        assert len(res.applications) == len(result.applications)
        total = sum(len(a.events) for a in res.applications)
        assert total == sum(len(a.events) for a in result.applications)

    def test_no_duplicate_event_ids_within_an_application(self, result):
        for app in result.applications:
            ids = [e.gmail_id for e in app.events]
            assert len(ids) == len(set(ids)), f"duplicate events in {app.key}"

    def test_a_message_belongs_to_exactly_one_application(self, result):
        seen: dict[str, str] = {}
        for app in result.applications:
            for ev in app.events:
                assert ev.gmail_id not in seen, (
                    f"{ev.gmail_id} appears in {seen.get(ev.gmail_id)} and {app.key}"
                )
                seen[ev.gmail_id] = app.key

    def test_application_keys_are_stable_across_runs(self, backend, settings, result):
        again = reconstruct(ALL_MESSAGES, user_email=USER_EMAIL, backend=backend,
                            settings=settings, now=NOW)
        assert {a.key for a in again.applications} == {a.key for a in result.applications}

    def test_repeated_sync_through_the_service_is_idempotent(self, service):
        first = service.load_demo_data(now=NOW)
        second = service.load_demo_data(now=NOW)
        assert second.new_messages == 0
        assert second.total_messages == first.total_messages
        assert second.applications == first.applications


# =============================================================================
# Dataset-level expectations
# =============================================================================

class TestDatasetExpectations:
    @pytest.mark.parametrize("expected", EXPECTED, ids=[e.label[:46] for e in EXPECTED])
    def test_expected_applications(self, result, app_of, expected):
        keys = {app_of.get(g) for g in expected.gmail_ids}
        assert len(keys) == 1 and None not in keys, (
            f"{expected.label}: messages landed in {len(keys)} application(s)"
        )
        app = result.by_key(next(iter(keys)))
        assert {e.gmail_id for e in app.events} >= set(expected.gmail_ids)
        if expected.employer_canonical:
            assert canonical_company(app.employer) == expected.employer_canonical
        if expected.expect_status:
            assert app.status.value == expected.expect_status
        if expected.expect_stage:
            assert app.current_stage.value == expected.expect_stage
        if expected.expect_needs_review:
            assert app.needs_review

    def test_noise_is_filtered(self, result):
        for gid in EXPECTED_NON_RECRUITMENT:
            assert gid in result.skipped_non_recruitment

    def test_short_confirmations_are_not_discarded(self, app_of):
        """m004 scores low on keywords but is unambiguous recruitment mail."""
        assert app_of.get("m004") is not None

    def test_decisions_are_recorded_with_evidence(self, result):
        auto = [d for d in result.decisions if d.kind == MatchDecisionKind.AUTO_MATCH]
        assert auto
        assert all(d.reasons for d in auto)
