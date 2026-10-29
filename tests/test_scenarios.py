"""The 14 scenarios required by spec v3 (Phase 4 — TEST).

Each class maps to one numbered requirement. These run offline against the full
synthetic dataset (dev + holdout) using the deterministic embedding tier, so the
default `pytest -q` downloads no model.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from interviewtrack.contacts import classify_contact
from interviewtrack.enums import (
    ApplicationStatus,
    ContactType,
    EventType,
    MatchDecisionKind,
    Occurrence,
    Stage,
    StageState,
)
from interviewtrack.fixtures.synthetic import (
    FULL_DATASET,
    HOLDOUT_MUST_MERGE,
    HOLDOUT_MUST_NOT_MERGE,
    USER_EMAIL,
    gmail_messages_as_api_payloads,
)
from interviewtrack.gmail.links import best_message_link, message_search_link
from interviewtrack.normalize import canonical_company
from interviewtrack.parsing import raw_from_gmail_payload
from interviewtrack.reconstruct import reconstruct

NOW = datetime(2025, 10, 28, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def full(backend, settings):
    """Reconstruction over dev + holdout fixtures."""
    return reconstruct(FULL_DATASET, user_email=USER_EMAIL, backend=backend,
                       settings=settings, now=NOW)


@pytest.fixture(scope="module")
def placed(full):
    return {e.gmail_id: a.key for a in full.applications for e in a.events}


def app_of(full, gmail_id):
    return next(
        (a for a in full.applications if any(e.gmail_id == gmail_id for e in a.events)),
        None,
    )


# --- 1 ------------------------------------------------------------------------
class TestS01MultipleThreadsMerge:
    def test_agency_internal_and_scheduler_become_one_application(self, full, placed):
        app = app_of(full, "m001")
        assert app is not None
        assert {"m001", "m003", "m005"} <= {e.gmail_id for e in app.events}
        assert app.thread_count >= 4
        assert canonical_company(app.employer) == "northwind analytics"

    def test_every_event_carries_evidence(self, full):
        for a in full.applications:
            for e in a.events:
                assert e.evidence_ref is not None
                assert e.evidence_ref.message_id == e.gmail_id
                assert e.evidence_ref.gmail_url


# --- 2 ------------------------------------------------------------------------
class TestS02RecruiterHandoff:
    def test_agency_to_internal_handoff_keeps_one_process(self, placed):
        assert placed.get("v201") is not None
        assert placed["v201"] == placed.get("v202")

    def test_both_contacts_are_linked(self, full):
        app = app_of(full, "v201")
        emails = {c.email for c in app.contacts}
        assert "marcus.webb@sterlingsearch.com" in emails
        assert "nina.patel@vantagesystems.com" in emails

    def test_contact_types_are_distinguished(self, full):
        app = app_of(full, "v201")
        by_email = {c.email: c.contact_type for c in app.contacts}
        assert by_email["marcus.webb@sterlingsearch.com"] == ContactType.AGENCY_RECRUITER
        assert by_email["nina.patel@vantagesystems.com"] == ContactType.INTERNAL_RECRUITER


# --- 3 ------------------------------------------------------------------------
class TestS03Coordinator:
    def test_coordinator_invite_joins_the_right_application(self, placed):
        assert placed.get("v203") == placed.get("v202")

    def test_coordinator_is_typed_as_such(self, full):
        app = app_of(full, "v203")
        contact = app.contact_for("jess.liu@vantagesystems.com")
        assert contact is not None
        assert contact.contact_type == ContactType.COORDINATOR


# --- 4 ------------------------------------------------------------------------
class TestS04SameRecruiterDifferentRoles:
    def test_two_roles_stay_separate(self, placed):
        assert placed["v201"] != placed["v204"]

    def test_each_keeps_its_own_requisition(self, full):
        backend_app = app_of(full, "v201")
        platform_app = app_of(full, "v204")
        assert backend_app.requisition_id == "VS7781"
        assert platform_app.requisition_id == "VS9042"

    def test_roleless_email_from_that_recruiter_goes_to_review(self, full):
        """The decisive case: one recruiter, two roles, an email naming neither."""
        app = app_of(full, "v205")
        assert app is not None, "a role-less email must not be silently dropped"
        assert app.needs_review is True
        assert app.job_title is None, "no role may be invented"

    def test_review_lists_ranked_candidates(self, full):
        decisions = [d for d in full.decisions if d.gmail_id == "v205"]
        review = [d for d in decisions if d.kind == MatchDecisionKind.REVIEW_REQUIRED]
        assert review
        assert len(review[-1].candidate_keys) >= 2


# --- 5 ------------------------------------------------------------------------
class TestS05DifferentEmployersSimilarTitles:
    def test_similar_data_scientist_emails_do_not_merge(self, placed):
        assert placed["m011"] != placed["m012"]

    def test_employer_conflict_is_a_hard_blocker(self, full):
        conflicted = [
            d for d in full.decisions
            if d.gmail_id == "m012" and d.features.employer_conflict
        ]
        assert conflicted
        assert all(d.kind == MatchDecisionKind.NO_MATCH for d in conflicted)


# --- 6 ------------------------------------------------------------------------
class TestS06RejectionScope:
    def test_rejection_closes_only_its_own_application(self, full, placed):
        analyst = app_of(full, "m009")
        engineer = app_of(full, "m001")
        assert placed["m009"] != placed["m001"]
        assert analyst.status == ApplicationStatus.REJECTED
        assert engineer.status != ApplicationStatus.REJECTED

    def test_rejection_leaves_the_stage_unchanged(self, full):
        """v3 table: a rejection changes status, not stage."""
        analyst = app_of(full, "m009")
        assert analyst.current_stage != Stage.CLOSED


# --- 7 ------------------------------------------------------------------------
class TestS07FollowUpInventsNothing:
    def test_follow_up_updates_activity_only(self, full):
        app = app_of(full, "m008")
        assert "Candidate follow-up" in app.latest_activity_label
        assert app.current_stage == Stage.TECHNICAL

    def test_follow_up_is_not_a_status_changing_event(self, full):
        app = app_of(full, "m008")
        latest = app.timeline[-1]
        assert latest.event_type == EventType.CANDIDATE_FOLLOWUP
        # The status must come from an earlier, status-changing message.
        assert app.evidence_message_id != latest.gmail_id


# --- 8 ------------------------------------------------------------------------
class TestS08AmbiguousGoesToReview:
    def test_unidentifiable_employer_is_flagged(self, full):
        app = app_of(full, "m017")
        assert app.needs_review is True
        assert app.employer_canonical == ""

    def test_no_employer_is_invented(self, full):
        app = app_of(full, "m017")
        assert app.job_title is None
        assert any("could not be identified" in n for n in app.notes)

    def test_ambiguous_message_does_not_attach_to_a_real_application(self, placed):
        for other in ("m001", "m009", "m011", "m012"):
            assert placed["m017"] != placed.get(other)


# --- 9 ------------------------------------------------------------------------
class TestS09RequisitionConflict:
    def test_conflicting_requisitions_prevent_merging(self, placed):
        assert placed["m001"] != placed["m009"]

    def test_conflict_is_recorded_as_a_blocker(self, full):
        blocked = [
            d for d in full.decisions
            if d.features.requisition_conflict
            and d.kind in {MatchDecisionKind.NO_MATCH, MatchDecisionKind.REVIEW_REQUIRED}
        ]
        assert blocked
        assert any("requisition" in b for d in blocked for b in d.blockers)


# --- 10 -----------------------------------------------------------------------
class TestS10ThreadSpanningTwoRoles:
    """A native Gmail thread is strong evidence, NOT unconditional proof."""

    def test_one_thread_two_roles_does_not_force_a_merge(self, placed):
        assert placed.get("v206") is not None
        assert placed["v206"] != placed.get("v207"), (
            "a shared Gmail thread must not override a conflicting requisition"
        )

    def test_the_split_message_is_flagged_rather_than_guessed(self, full):
        app = app_of(full, "v207")
        assert app is not None
        assert app.needs_review is True

    def test_discriminating_case_no_requisition_ids(self, placed):
        """The case that proves the v3 change matters.

        v206/v207 carry conflicting requisition IDs, so the older, narrower check
        caught them too — that pair does not discriminate. v209/v210 share a thread,
        name two different roles and have NO requisition IDs, so ONLY the role-family
        hard constraint can stop the merge. Before this change, both merged.
        """
        assert placed.get("v209") is not None
        assert placed["v209"] != placed.get("v210"), (
            "a thread naming two different roles must split even with no requisition IDs"
        )

    def test_role_switch_is_extracted_not_inherited_from_reply_subject(self, full):
        """The reply subject says Research Engineer; the body says Data Analyst."""
        app = app_of(full, "v210")
        assert app is not None
        assert app.job_title is not None
        assert "analyst" in app.job_title.lower(), (
            f"expected the replacement role, got {app.job_title!r} "
            "(stale role inherited from the reply subject)"
        )

    def test_decision_records_why_the_thread_was_not_followed(self, full):
        decisions = [
            d for d in full.decisions
            if d.gmail_id == "v207" and d.features.same_gmail_thread
        ]
        assert decisions
        assert all(d.kind != MatchDecisionKind.AUTO_MATCH for d in decisions)
        assert any(d.blockers for d in decisions)


# --- 11 -----------------------------------------------------------------------
class TestS11DuplicateSync:
    def test_duplicate_input_produces_no_duplicates(self, backend, settings, full):
        again = reconstruct(list(FULL_DATASET) + list(FULL_DATASET),
                            user_email=USER_EMAIL, backend=backend,
                            settings=settings, now=NOW)
        assert len(again.applications) == len(full.applications)
        assert sum(len(a.events) for a in again.applications) == sum(
            len(a.events) for a in full.applications
        )

    def test_each_message_belongs_to_exactly_one_application(self, full):
        seen: dict[str, str] = {}
        for a in full.applications:
            for e in a.events:
                assert e.gmail_id not in seen
                seen[e.gmail_id] = a.key

    def test_keys_are_stable_across_runs(self, backend, settings, full):
        again = reconstruct(FULL_DATASET, user_email=USER_EMAIL, backend=backend,
                            settings=settings, now=NOW)
        assert {a.key for a in again.applications} == {a.key for a in full.applications}


# --- 12 -----------------------------------------------------------------------
class TestS12MalformedInput:
    def test_no_sender_no_date_no_body_is_skipped_not_fatal(self, full):
        assert "m018" in full.skipped_non_recruitment

    def test_broken_html_is_still_parsed(self, full, placed):
        assert placed.get("m019") is not None
        app = app_of(full, "m019")
        event = next(e for e in app.events if e.gmail_id == "m019")
        assert "drop me" not in event.preview   # <script> contents stripped

    def test_empty_input(self, backend, settings):
        assert reconstruct([], user_email=USER_EMAIL, backend=backend,
                           settings=settings).applications == []


# --- 13 -----------------------------------------------------------------------
class TestS13PassedDateIsNotCompletion:
    def test_passed_scheduled_interview_stays_outcome_unknown(self, full):
        """The technical interview (Oct 21) is in the past relative to NOW (Oct 28)."""
        app = app_of(full, "m005")
        assert app.current_stage == Stage.TECHNICAL
        assert app.stage_state == StageState.UNKNOWN
        assert app.stage_note == "date passed, outcome unknown"

    def test_no_event_is_marked_completed_without_evidence(self, full):
        app = app_of(full, "m005")
        for e in app.events:
            if e.occurrence == Occurrence.COMPLETED:
                assert e.evidence, "a completed event must carry its evidence quote"

    def test_explicit_completion_is_honoured(self, backend, settings):
        from interviewtrack.models import Application, InterviewEvent
        from interviewtrack.status import resolve

        app = Application(key="k", employer="Acme", employer_canonical="acme", events=[
            InterviewEvent(gmail_id="g1", thread_id="t", event_type=EventType.FINAL_INTERVIEW,
                           occurrence=Occurrence.COMPLETED,
                           occurred_at=datetime(2025, 10, 10, tzinfo=timezone.utc)),
        ])
        resolve(app, now=NOW)
        assert app.stage_state == StageState.COMPLETED
        assert app.status == ApplicationStatus.AWAITING_RESPONSE


# --- 14 -----------------------------------------------------------------------
class TestS14GmailParsingAndLinks:
    """Gmail payload parsing and account-aware links.

    The parsing and link construction here are the real production code paths. A
    *live authorized* Gmail round-trip is NOT covered — that requires credentials and
    is reported as untested in docs/test-report.md.
    """

    def test_real_gmail_payload_shape_parses(self):
        payloads = gmail_messages_as_api_payloads()
        parsed = [raw_from_gmail_payload(p) for p in payloads]
        assert len(parsed) == len(payloads)
        first = parsed[0]
        assert first.gmail_id and first.thread_id and first.rfc_message_id
        assert first.from_addr and "@" in first.from_addr

    def test_links_are_account_aware(self):
        link = message_search_link("<abc@mail.example>", "avery@gmail.com")
        assert "authuser=avery%40gmail.com" in link
        assert "rfc822msgid" in link

    def test_links_use_rfc_message_id_not_api_thread_id(self):
        """Gmail API thread IDs are not valid web-UI URLs."""
        link = best_message_link("<abc@mail.example>", "thread-api-id-123")
        assert "rfc822msgid" in link
        assert "thread-api-id-123" not in link

    def test_link_falls_back_to_thread_without_message_id(self):
        assert best_message_link(None, "t1").endswith("#all/t1")

    @pytest.mark.skip(reason="requires real Google OAuth credentials; see docs/test-report.md")
    def test_live_gmail_round_trip(self):
        pass


# --- cross-cutting -------------------------------------------------------------
class TestHoldoutInvariants:
    @pytest.mark.parametrize("a,b,why", HOLDOUT_MUST_MERGE, ids=[m[2][:40] for m in HOLDOUT_MUST_MERGE])
    def test_holdout_must_merge(self, placed, a, b, why):
        assert placed.get(a) is not None and placed[a] == placed.get(b), why

    @pytest.mark.parametrize("a,b,why", HOLDOUT_MUST_NOT_MERGE, ids=[m[2][:40] for m in HOLDOUT_MUST_NOT_MERGE])
    def test_holdout_must_not_merge(self, placed, a, b, why):
        assert placed.get(a) != placed.get(b) or placed.get(a) is None, why

    def test_contact_typing_covers_every_sender_class(self):
        assert classify_contact("x@harnham.com", "", "my client is Acme") == ContactType.AGENCY_RECRUITER
        assert classify_contact("no-reply@goodtime.io", "", "") == ContactType.SCHEDULING_TOOL
        assert classify_contact("no-reply@lever.co", "", "") == ContactType.ATS_NOREPLY
        assert classify_contact(USER_EMAIL, "", "", USER_EMAIL) == ContactType.SELF
