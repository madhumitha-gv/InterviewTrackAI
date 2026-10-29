"""Embedding backends (including the real neural path) and the optional LLM layer."""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np
import pytest

from interviewtrack.config import get_settings
from interviewtrack.embeddings import (
    EmbeddingBackend,
    build_application_document,
    build_message_document,
    cosine_matrix,
    cosine_similarity,
    get_backend,
)
from interviewtrack.enums import MatchDecisionKind
from interviewtrack.llm_review import (
    apply_verdict,
    build_payload,
    parse_verdict,
    redact,
    review_ambiguous,
)
from interviewtrack.matching import ApplicationContext
from interviewtrack.models import Application, LLMAssociationVerdict, MatchDecision


class TestDeterministicBackend:
    def test_shape_and_normalization(self, backend):
        v = backend.encode(["alpha beta", "gamma delta"])
        assert v.shape == (2, backend.dimension)
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)

    def test_deterministic(self):
        a = get_backend(force_hashed=True, refresh=True).encode_one("same text")
        b = get_backend(force_hashed=True, refresh=True).encode_one("same text")
        assert np.allclose(a, b)

    def test_empty_inputs(self, backend):
        assert backend.encode([]).shape == (0, backend.dimension)
        assert float(np.linalg.norm(backend.encode_one(""))) == 0.0

    def test_similar_text_scores_higher(self, backend):
        base = backend.encode_one("AI Engineer role at Northwind Analytics")
        near = backend.encode_one("Artificial Intelligence Engineer at Northwind")
        far = backend.encode_one("Your October invoice is ready, amount due $42")
        assert cosine_similarity(base, near) > cosine_similarity(base, far)

    def test_reports_its_tier_honestly(self, backend):
        assert backend.is_semantic is False
        assert "deterministic" in backend.describe()

    def test_cosine_bounds(self, backend):
        v = backend.encode(["a b c", "d e f"])
        assert -1.0 <= cosine_similarity(v[0], v[1]) <= 1.0
        m = cosine_matrix(v, v)
        assert m.shape == (2, 2)
        assert np.allclose(np.diag(m), 1.0, atol=1e-5)

    def test_zero_vectors_are_safe(self):
        assert cosine_similarity(np.zeros(4), np.ones(4)) == 0.0


class TestSemanticDocuments:
    def test_message_document_is_structured(self):
        doc = build_message_document(
            subject="Interview", clean_text="Body text here",
            employer="Northwind Analytics", job_title="AI Engineer",
            event_type="technical_interview", recruiter="priya raman",
        )
        assert "employer: Northwind Analytics" in doc
        assert "role: AI Engineer" in doc
        assert "event: technical interview" in doc

    def test_body_excerpt_is_bounded(self):
        doc = build_message_document("s", "x" * 5000, None, None, None)
        assert len(doc) < 1200

    def test_application_document(self):
        doc = build_application_document(
            "Globex Labs", "Data Scientist", "REQ1", ["amara diallo"], ["Subject A"]
        )
        assert "employer: Globex Labs" in doc and "requisition: REQ1" in doc


class TestNeuralBackendCodePath:
    """Exercises the SentenceTransformer path with a locally built model.

    The published BAAI/bge-small-en-v1.5 weights cannot be downloaded in every
    environment, so this test constructs a tiny local SentenceTransformer instead.
    It verifies the real `_encode_neural` branch, dimension detection and
    normalization — everything except the specific pretrained weights.
    """

    @pytest.fixture(scope="class")
    def tiny_model_dir(self):
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("sentence_transformers")
        from transformers import BertConfig, BertModel, BertTokenizerFast

        d = pathlib.Path(tempfile.mkdtemp()) / "tiny-st"
        d.mkdir(parents=True)
        BertModel(BertConfig(
            vocab_size=1000, hidden_size=32, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=64, max_position_embeddings=64,
        )).save_pretrained(d)
        (d / "vocab.txt").write_text(
            "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
                      + [f"w{i}" for i in range(995)])
        )
        BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
        return d

    def test_neural_encode_path(self, tiny_model_dir):
        from sentence_transformers import SentenceTransformer, models

        we = models.Transformer(str(tiny_model_dir), max_seq_length=64)
        pooling = models.Pooling(we.get_word_embedding_dimension(), pooling_mode="mean")
        model = SentenceTransformer(modules=[we, pooling], device="cpu")

        backend = EmbeddingBackend(
            kind="sentence-transformer", model_name="tiny-local",
            dimension=int(model.get_sentence_embedding_dimension()), _model=model,
        )
        assert backend.is_semantic is True
        assert "SentenceTransformer" in backend.describe()

        vecs = backend.encode([
            build_message_document("Interview", "technical interview",
                                   "Northwind Analytics", "AI Engineer",
                                   "technical_interview"),
            build_application_document("Northwind Analytics", "AI Engineer", "REQ4417"),
        ])
        assert vecs.shape == (2, backend.dimension)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
        assert -1.0 <= cosine_similarity(vecs[0], vecs[1]) <= 1.0

    def test_runtime_failure_falls_back_to_hashed(self):
        class Exploding:
            def encode(self, *a, **k):
                raise RuntimeError("model died mid-run")

        backend = EmbeddingBackend(kind="sentence-transformer", model_name="broken",
                                   dimension=384, _model=Exploding())
        vecs = backend.encode(["still works"])
        assert vecs.shape == (1, 384)
        assert backend.is_semantic is False
        assert "runtime_encode_failed" in backend.load_error

    def test_unloadable_model_name_degrades_gracefully(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_MODEL", "definitely/not-a-real-model-xyz")
        monkeypatch.setenv("EMBEDDING_FALLBACK_MODEL", "also/not-real-xyz")
        get_settings(refresh=True)
        backend = get_backend(refresh=True)
        assert backend.encode(["text"]).shape[0] == 1   # never raises
        get_settings(refresh=True)
        get_backend(force_hashed=True, refresh=True)


class TestLLMGuardRails:
    def _decision(self, **kw):
        base = dict(gmail_id="m1", application_key="app_x",
                    kind=MatchDecisionKind.REVIEW_REQUIRED, score=0.34)
        base.update(kw)
        return MatchDecision(**base)

    def _verdict(self, **kw):
        base = dict(associate=True, application_key="app_x", shared_evidence=["same employer"],
                    contradictions=[], rationale="ok", uncertainty=0.1,
                    recommend_human_review=False)
        base.update(kw)
        return LLMAssociationVerdict(**base)

    def test_disabled_by_default(self):
        settings = get_settings(refresh=True)
        assert settings.llm_review_enabled is False
        assert settings.llm_configured is False

    def test_disabled_returns_an_explanatory_error(self, monkeypatch):
        from interviewtrack.classify import classify_message
        from interviewtrack.fixtures.synthetic import M17
        from interviewtrack.parsing import parse_message

        msg = parse_message(M17)
        out = review_ambiguous(msg, classify_message(msg), [], get_settings(refresh=True))
        assert out.verdict is None
        assert "disabled" in out.error.lower()

    def test_enabled_without_api_key_is_refused(self, monkeypatch):
        monkeypatch.setenv("LLM_REVIEW_ENABLED", "true")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        from interviewtrack.classify import classify_message
        from interviewtrack.fixtures.synthetic import M17
        from interviewtrack.parsing import parse_message

        msg = parse_message(M17)
        out = review_ambiguous(msg, classify_message(msg), [], get_settings(refresh=True))
        assert out.verdict is None
        assert "ANTHROPIC_API_KEY" in out.error
        get_settings(refresh=True)

    def test_payload_is_minimized_and_redacted(self):
        from interviewtrack.classify import classify_message
        from interviewtrack.fixtures.synthetic import M17
        from interviewtrack.parsing import parse_message

        msg = parse_message(M17)
        contexts = [
            ApplicationContext(application=Application(
                key=f"app_{i}", employer=f"Co{i}", employer_canonical=f"co{i}"))
            for i in range(10)
        ]
        payload = build_payload(msg, classify_message(msg), contexts, max_candidates=3)
        assert len(payload["candidate_applications"]) == 3
        blob = json.dumps(payload)
        assert "@talentsphere.io" not in blob
        assert len(payload["email"]["excerpt"]) <= 420

    def test_redaction(self):
        out = redact("write to a.b@c.com or call +1 (555) 123-4567 at https://x.io/p")
        assert "@c.com" not in out and "555" not in out and "https://" not in out

    def test_parses_fenced_json(self):
        v = parse_verdict('```json\n{"associate": true, "application_key": "a", '
                          '"shared_evidence": [], "contradictions": [], "rationale": "r", '
                          '"uncertainty": 0.2, "recommend_human_review": false}\n```')
        assert v.associate is True

    def test_rejects_invalid_response(self):
        with pytest.raises(Exception):
            parse_verdict("absolutely not json")

    def test_invalid_response_is_handled(self, monkeypatch):
        monkeypatch.setenv("LLM_REVIEW_ENABLED", "true")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from interviewtrack.classify import classify_message
        from interviewtrack.fixtures.synthetic import M17
        from interviewtrack.parsing import parse_message

        msg = parse_message(M17)
        out = review_ambiguous(msg, classify_message(msg), [], get_settings(refresh=True),
                               caller=lambda p, s: "not json at all")
        assert out.verdict is None and "Invalid LLM response" in out.error
        get_settings(refresh=True)

    # --- the guard rails that matter most ---

    def test_cannot_override_a_hard_blocker(self):
        blocked = self._decision(blockers=["conflicting requisition IDs"])
        out = apply_verdict(blocked, self._verdict())
        assert out.kind == MatchDecisionKind.REVIEW_REQUIRED
        assert any("hard blockers remain" in r for r in out.reasons)

    def test_cannot_merge_contradictory_records(self):
        out = apply_verdict(self._decision(),
                            self._verdict(contradictions=["different requisition IDs"]))
        assert out.kind == MatchDecisionKind.REVIEW_REQUIRED

    def test_high_uncertainty_requires_human_review(self):
        out = apply_verdict(self._decision(), self._verdict(uncertainty=0.9))
        assert out.kind == MatchDecisionKind.REVIEW_REQUIRED

    def test_explicit_review_request_is_honored(self):
        out = apply_verdict(self._decision(), self._verdict(recommend_human_review=True))
        assert out.kind == MatchDecisionKind.REVIEW_REQUIRED

    def test_pointing_at_a_different_application_requires_review(self):
        out = apply_verdict(self._decision(), self._verdict(application_key="app_other"))
        assert out.kind == MatchDecisionKind.REVIEW_REQUIRED

    def test_may_upgrade_a_clean_confident_case(self):
        out = apply_verdict(self._decision(), self._verdict())
        assert out.kind == MatchDecisionKind.AUTO_MATCH

    def test_may_always_be_more_conservative(self):
        out = apply_verdict(self._decision(), self._verdict(associate=False))
        assert out.kind == MatchDecisionKind.NO_MATCH

    def test_verdict_schema_is_validated(self):
        with pytest.raises(Exception):
            LLMAssociationVerdict.model_validate({"associate": "yes please"})
        with pytest.raises(Exception):
            LLMAssociationVerdict.model_validate({"associate": True, "uncertainty": 5.0})
