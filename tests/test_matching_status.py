"""Hybrid scoring internals (Stage D/E) and status resolution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from interviewtrack.classify import classify_message
from interviewtrack.config import MatchWeights
from interviewtrack.enums import ApplicationStatus, EventType, MatchDecisionKind, Occurrence, Stage
from interviewtrack.matching import ApplicationContext, compute_features, decide, find_blockers, score_features
from interviewtrack.models import Application, InterviewEvent, MatchFeatures, RawMessage
from interviewtrack.parsing import parse_message
from interviewtrack.status import resolve

USER = "avery.candidate@gmail.com"
T0 = datetime(2025, 10, 15, tzinfo=timezone.utc)


def ctx(employer="Northwind Analytics", title="AI Engineer", req="REQ4417",
        threads=("t1",), last=T0):
    c = ApplicationContext(application=Application(
        key="app_1", employer=employer,
        employer_canonical=employer.lower() if employer else "",
        job_title=title, requisition_id=req,
        role_family="ai_engineer" if title == "AI Engineer" else None,
    ))
    c.thread_ids = set(threads)
    c.last_activity = last
    return c


def msg(subject="Interview", body="We'd like to schedule your technical interview.",
        frm="hr@northwind.com", tid="t9", gid="mX", refs=(), in_reply_to=None):
    return parse_message(RawMessage(
        gmail_id=gid, thread_id=tid, from_addr=frm, from_name="HR",
        to_addrs=[USER], subject=subject, date=T0 + timedelta(days=1),
        body_text=body, body_fetched=True, labels=["INBOX"],
        references=list(refs), in_reply_to=in_reply_to,
    ))


class TestFeatures:
    def test_same_thread_detected(self, backend):
        m = msg(tid="t1")
        f = compute_features(m, classify_message(m, user_email=USER), ctx(threads=("t1",)))
        assert f.same_gmail_thread is True

    def test_reply_header_link_detected(self, backend):
        c = ctx(threads=("t1",))
        c.rfc_message_ids = {"<parent@mail>"}
        m = msg(tid="tOther", in_reply_to="<parent@mail>")
        f = compute_features(m, classify_message(m, user_email=USER), c)
        assert f.thread_reference_link is True

    def test_requisition_conflict(self, backend):
        m = msg(body="Role at Northwind Analytics, requisition REQ-9999.")
        f = compute_features(m, classify_message(m, user_email=USER), ctx(req="REQ4417"))
        assert f.requisition_conflict is True

    def test_employer_conflict(self, backend):
        m = msg(body="Technical interview for the Data Scientist role at Globex Labs.")
        f = compute_features(m, classify_message(m, user_email=USER), ctx())
        assert f.employer_conflict is True

    def test_temporal_proximity_decays(self):
        near = MatchFeatures(days_apart=1.0, temporal_proximity=0.95)
        far = MatchFeatures(days_apart=90.0, temporal_proximity=0.01)
        w = MatchWeights()
        assert score_features(near, w)[0] > score_features(far, w)[0]


class TestBlockers:
    def test_requisition_conflict_blocks(self):
        assert "conflicting requisition IDs" in find_blockers(
            MatchFeatures(requisition_conflict=True))

    def test_employer_conflict_blocks(self):
        assert "conflicting employer identity" in find_blockers(
            MatchFeatures(employer_conflict=True))

    def test_role_conflict_blocks_unless_requisition_matches(self):
        assert find_blockers(MatchFeatures(role_conflict=True))
        assert not find_blockers(
            MatchFeatures(role_conflict=True, requisition_match=True))


class TestDecisions:
    def test_same_thread_is_authoritative(self, backend, settings):
        m = msg(tid="t1")
        d = decide(m, classify_message(m, user_email=USER), ctx(threads=("t1",)),
                   settings, backend=backend)
        assert d.kind == MatchDecisionKind.AUTO_MATCH
        assert d.score == 1.0

    def test_conflicting_requisition_vetoes_even_with_strong_overlap(self, backend, settings):
        m = msg(body="The AI Engineer role at Northwind Analytics, requisition REQ-9999.")
        d = decide(m, classify_message(m, user_email=USER), ctx(), settings, backend=backend)
        assert d.kind == MatchDecisionKind.NO_MATCH
        assert "conflicting requisition IDs" in d.blockers

    def test_no_identity_evidence_caps_the_score(self, backend, settings):
        m = msg(subject="Interview", body="We'd like to schedule your technical interview.",
                frm="x@talentsphere.io")
        d = decide(m, classify_message(m, user_email=USER), ctx(), settings, backend=backend)
        assert d.kind != MatchDecisionKind.AUTO_MATCH
        assert d.score <= settings.weights.semantic_only_cap
        assert any("no explicit" in b for b in d.blockers)

    def test_strong_deterministic_evidence_auto_matches(self, backend, settings):
        m = msg(body="Your technical interview for the AI Engineer role at "
                     "Northwind Analytics, requisition REQ-4417.")
        d = decide(m, classify_message(m, user_email=USER), ctx(), settings, backend=backend)
        assert d.kind == MatchDecisionKind.AUTO_MATCH
        assert d.features.requisition_match and d.features.employer_match

    def test_decision_carries_reasons(self, backend, settings):
        m = msg(body="The AI Engineer role at Northwind Analytics, requisition REQ-4417.")
        d = decide(m, classify_message(m, user_email=USER), ctx(), settings, backend=backend)
        assert d.reasons and all(isinstance(r, str) for r in d.reasons)


class TestStatusResolution:
    def _app(self, *events):
        return Application(key="k", employer="Acme", employer_canonical="acme",
                           events=list(events))

    def _ev(self, day, etype, occ=Occurrence.MENTIONED, gid=None):
        return InterviewEvent(
            gmail_id=gid or f"g{day}", thread_id="t", event_type=etype, occurrence=occ,
            occurred_at=datetime(2025, 10, day, tzinfo=timezone.utc),
        )

    def test_stage_is_monotonic(self):
        app = self._app(
            self._ev(1, EventType.RECRUITER_OUTREACH),
            self._ev(10, EventType.TECHNICAL_INTERVIEW, Occurrence.SCHEDULED),
            self._ev(12, EventType.SCHEDULING, Occurrence.SCHEDULED),
        )
        resolve(app, now=datetime(2025, 10, 28, tzinfo=timezone.utc))
        assert app.current_stage == Stage.TECHNICAL

    def test_canceled_round_does_not_advance_the_stage(self):
        app = self._app(
            self._ev(1, EventType.SCREENING_INVITATION, Occurrence.SCHEDULED),
            self._ev(10, EventType.FINAL_INTERVIEW, Occurrence.CANCELED),
        )
        resolve(app, now=datetime(2025, 10, 28, tzinfo=timezone.utc))
        assert app.current_stage == Stage.SCREENING

    def test_rejection_closes_the_application(self):
        app = self._app(
            self._ev(1, EventType.TECHNICAL_INTERVIEW, Occurrence.SCHEDULED),
            self._ev(10, EventType.REJECTION, Occurrence.COMPLETED),
        )
        resolve(app, now=datetime(2025, 10, 28, tzinfo=timezone.utc))
        assert app.status == ApplicationStatus.REJECTED
        # v3 transition table: a rejection leaves the stage UNCHANGED. The funnel
        # still records how far the process actually got.
        assert app.current_stage == Stage.TECHNICAL

    def test_offer_status(self):
        app = self._app(self._ev(10, EventType.OFFER, Occurrence.COMPLETED))
        resolve(app, now=datetime(2025, 10, 11, tzinfo=timezone.utc))
        assert app.status == ApplicationStatus.OFFER

    def test_open_assessment_is_action_required(self):
        """v3: take-home sent -> stage Assessment, status Action required."""
        app = self._app(self._ev(26, EventType.ASSESSMENT_ASSIGNED, Occurrence.SCHEDULED))
        resolve(app, now=datetime(2025, 10, 28, tzinfo=timezone.utc))
        assert app.current_stage == Stage.ASSESSMENT
        assert app.status == ApplicationStatus.ACTION_REQUIRED

    def test_submitted_assessment_awaits_response(self):
        """v3: assessment submitted -> stage Assessment, status Awaiting response."""
        app = self._app(
            self._ev(20, EventType.ASSESSMENT_ASSIGNED, Occurrence.SCHEDULED),
            self._ev(26, EventType.ASSESSMENT_SUBMITTED, Occurrence.COMPLETED),
        )
        resolve(app, now=datetime(2025, 10, 28, tzinfo=timezone.utc))
        assert app.current_stage == Stage.ASSESSMENT
        assert app.status == ApplicationStatus.AWAITING_RESPONSE

    def test_milestone_wording_never_overstates(self):
        app = self._app(self._ev(26, EventType.TECHNICAL_INTERVIEW, Occurrence.SCHEDULED))
        resolve(app, now=datetime(2025, 11, 30, tzinfo=timezone.utc))
        assert "completion not confirmed" in app.last_supported_milestone

    def test_latest_activity_independent_of_stage(self):
        app = self._app(
            self._ev(10, EventType.FINAL_INTERVIEW, Occurrence.SCHEDULED),
            self._ev(20, EventType.CANDIDATE_FOLLOWUP, Occurrence.COMPLETED),
        )
        resolve(app, now=datetime(2025, 10, 28, tzinfo=timezone.utc))
        assert app.current_stage == Stage.FINAL
        assert "Candidate follow-up" in app.latest_activity_label

    def test_incomplete_history_flag(self):
        app = self._app(self._ev(2, EventType.TECHNICAL_INTERVIEW, Occurrence.SCHEDULED))
        resolve(app, now=datetime(2025, 10, 28, tzinfo=timezone.utc),
                window_start=datetime(2025, 10, 1, tzinfo=timezone.utc))
        assert app.history_possibly_incomplete

    def test_empty_timeline_is_safe(self):
        app = Application(key="k", employer="X", employer_canonical="x")
        resolve(app)
        assert app.current_stage == Stage.OUTREACH

    def test_resolve_is_idempotent(self):
        app = self._app(
            self._ev(1, EventType.RECRUITER_OUTREACH),
            self._ev(10, EventType.TECHNICAL_INTERVIEW, Occurrence.SCHEDULED),
        )
        now = datetime(2025, 10, 28, tzinfo=timezone.utc)
        resolve(app, now=now)
        first = (app.current_stage, app.status, app.last_supported_milestone)
        resolve(app, now=now)
        assert (app.current_stage, app.status, app.last_supported_milestone) == first
