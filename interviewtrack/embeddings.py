"""Stage C — semantic representations.

Three tiers, in order of preference:
  1. ``BAAI/bge-small-en-v1.5`` via SentenceTransformer on CPU (preferred).
  2. ``sentence-transformers/all-MiniLM-L6-v2`` (fallback if the preferred model
     cannot be loaded).
  3. A deterministic hashed character-n-gram vectorizer (always available, no
     downloads, no torch). Lexical only — but it keeps the whole pipeline runnable
     offline and makes tests hermetic.

The active tier is always reported via :meth:`EmbeddingBackend.describe`, so the UI
and the implementation report never overstate what ran.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import get_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class EmbeddingBackend:
    """Wraps whichever embedding tier is actually available."""

    kind: str = "hashed"  # "sentence-transformer" | "hashed"
    model_name: str = "deterministic-hashed-ngrams"
    dimension: int = 384
    _model: object | None = field(default=None, repr=False)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    load_error: str = ""

    @property
    def is_semantic(self) -> bool:
        """True only when a real neural embedding model is loaded."""
        return self.kind == "sentence-transformer"

    def describe(self) -> str:
        if self.is_semantic:
            return f"SentenceTransformer({self.model_name}, dim={self.dimension}, cpu)"
        reason = f" [reason: {self.load_error}]" if self.load_error else ""
        return f"deterministic hashed n-grams (dim={self.dimension}, no model){reason}"

    # --- encoding -------------------------------------------------------------

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Encode texts into L2-normalized row vectors of shape (n, dimension)."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        todo_idx: list[int] = []
        todo_txt: list[str] = []

        for i, text in enumerate(texts):
            key = text or ""
            cached = self._cache.get(key)
            if cached is not None:
                out[i] = cached
            else:
                todo_idx.append(i)
                todo_txt.append(key)

        if todo_txt:
            if self.is_semantic and self._model is not None:
                vectors = self._encode_neural(todo_txt, batch_size)
                if not self.is_semantic:
                    return self.encode(texts, batch_size)
            else:
                vectors = np.vstack([self._encode_hashed(t) for t in todo_txt])
            for pos, i in enumerate(todo_idx):
                vec = vectors[pos].astype(np.float32)
                out[i] = vec
                self._cache[todo_txt[pos]] = vec

        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _encode_neural(self, texts: list[str], batch_size: int) -> np.ndarray:
        try:
            vecs = self._model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vecs, dtype=np.float32)
        except Exception as exc:  # a mid-run model failure must not abort the sync
            logger.warning("Neural encode failed (%s); falling back to hashed.", exc)
            self.kind = "hashed"
            self.load_error = f"runtime_encode_failed: {type(exc).__name__}"
            self.dimension = 384
            self._cache.clear()
            return np.vstack([self._encode_hashed(t) for t in texts])

    def _encode_hashed(self, text: str) -> np.ndarray:
        """Deterministic fallback: hashed unigrams + bigrams + char 4-grams.

        Not a semantic model. It captures lexical overlap only, which is enough for
        the deterministic-only ablation and for offline/CI runs.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        lowered = (text or "").lower()
        tokens = _TOKEN_RE.findall(lowered)
        if not tokens:
            return vec

        features: list[tuple[str, float]] = [(t, 1.0) for t in tokens]
        features += [(f"{a}_{b}", 1.4) for a, b in zip(tokens, tokens[1:])]
        compact = " ".join(tokens)
        features += [(compact[i : i + 4], 0.5) for i in range(0, max(0, len(compact) - 3), 2)]

        for feat, weight in features:
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dimension
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign * weight

        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


_backend: EmbeddingBackend | None = None
_lock = threading.Lock()


def get_backend(force_hashed: bool = False, refresh: bool = False) -> EmbeddingBackend:
    """Load (once) and return the best available embedding backend.

    `force_hashed=True` skips model loading entirely — used by the deterministic
    ablation in scripts/evaluate.py and by fast unit tests.
    """
    global _backend
    with _lock:
        if _backend is not None and not refresh and not (force_hashed and _backend.is_semantic):
            return _backend

        settings = get_settings()
        if force_hashed or not settings.embeddings_enabled:
            _backend = EmbeddingBackend(
                load_error="disabled_by_config" if not settings.embeddings_enabled else "forced"
            )
            return _backend

        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except Exception as exc:
            logger.info("sentence-transformers unavailable (%s); using hashed fallback.", exc)
            _backend = EmbeddingBackend(load_error=f"import_failed: {type(exc).__name__}")
            return _backend

        last_error = ""
        for name in (settings.embedding_model, settings.embedding_fallback_model):
            if not name:
                continue
            try:
                model = SentenceTransformer(name, device="cpu")
                # sentence-transformers 6.x renamed this method; support both.
                getter = getattr(model, "get_embedding_dimension", None) or getattr(
                    model, "get_sentence_embedding_dimension", None
                )
                dim = int((getter() if getter else None) or 384)
                _backend = EmbeddingBackend(
                    kind="sentence-transformer", model_name=name, dimension=dim, _model=model
                )
                logger.info("Loaded embedding model %s (dim=%d)", name, dim)
                return _backend
            except Exception as exc:
                last_error = f"{name}: {type(exc).__name__}"
                logger.warning("Could not load %s (%s); trying next.", name, exc)

        _backend = EmbeddingBackend(load_error=f"load_failed: {last_error}")
        return _backend


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, clamped to [-1, 1]."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0))


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between row sets `a` (n,d) and `b` (m,d)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return np.clip(an @ bn.T, -1.0, 1.0).astype(np.float32)


# --- semantic document construction --------------------------------------------

#: Hard cap on body excerpt length. Keeps the embedding focused on the informative
#: opening of the message and bounds CPU cost.
_BODY_CHARS = 700


def build_message_document(subject: str, clean_text: str, employer: str | None,
                           job_title: str | None, event_type: str | None,
                           recruiter: str | None = None) -> str:
    """Build the concise semantic representation for a single message.

    Deliberately *structured*: employer and role are stated as labeled fields so the
    model attends to identity, not to shared recruiting boilerplate. The body excerpt
    is already quote-stripped by `parsing.strip_quoted_and_boilerplate`.
    """
    parts: list[str] = []
    if employer:
        parts.append(f"employer: {employer}")
    if job_title:
        parts.append(f"role: {job_title}")
    if event_type:
        parts.append(f"event: {event_type.replace('_', ' ')}")
    if recruiter:
        parts.append(f"contact: {recruiter}")
    if subject:
        parts.append(f"subject: {subject.strip()}")
    excerpt = (clean_text or "").strip()
    if excerpt:
        parts.append(f"body: {excerpt[:_BODY_CHARS]}")
    return " | ".join(parts)


def build_application_document(employer: str | None, job_title: str | None,
                               requisition_id: str | None,
                               recruiters: Sequence[str] = (),
                               recent_subjects: Sequence[str] = ()) -> str:
    """Build the semantic representation of an existing application *context*.

    Candidate messages are compared against these aggregate contexts rather than
    against every other email, which keeps comparisons O(messages x applications).
    """
    parts: list[str] = []
    if employer:
        parts.append(f"employer: {employer}")
    if job_title:
        parts.append(f"role: {job_title}")
    if requisition_id:
        parts.append(f"requisition: {requisition_id}")
    if recruiters:
        parts.append(f"contacts: {', '.join(sorted({r for r in recruiters if r})[:4])}")
    if recent_subjects:
        parts.append(f"threads: {' ; '.join(list(recent_subjects)[:4])}")
    return " | ".join(parts)
