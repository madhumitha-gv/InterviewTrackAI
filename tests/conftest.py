"""Shared pytest fixtures.

Tests run on the deterministic hashed embedding backend by default so they are fast,
hermetic and require no model download. `test_embeddings.py` separately verifies the
neural SentenceTransformer code path.
"""

from __future__ import annotations

import sys
import os
import tempfile

# Never let tests touch a real mailbox token or database.
_TEST_DATA = tempfile.TemporaryDirectory(prefix="interviewtrack-tests-")
os.environ["INTERVIEWTRACK_DATA_DIR"] = _TEST_DATA.name
for _name in ("GOOGLE_TOKEN_FILE", "GOOGLE_TOKENS_DIR", "GOOGLE_CLIENT_SECRET_FILE"):
    os.environ.pop(_name, None)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interviewtrack.config import get_settings
from interviewtrack.db import Database
from interviewtrack.embeddings import get_backend
from interviewtrack.fixtures.synthetic import ALL_MESSAGES, USER_EMAIL
from interviewtrack.reconstruct import reconstruct
from interviewtrack.service import InterviewTrackService

#: Fixed "now" so status/stage assertions are stable regardless of the real date.
NOW = datetime(2025, 10, 28, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def backend():
    return get_backend(force_hashed=True, refresh=True)


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def result(backend, settings):
    """The reconstruction of the full synthetic dataset (computed once)."""
    return reconstruct(ALL_MESSAGES, user_email=USER_EMAIL, backend=backend,
                       settings=settings, now=NOW)


@pytest.fixture(scope="session")
def app_of(result):
    """Map gmail_id -> application key."""
    return {e.gmail_id: a.key for a in result.applications for e in a.events}


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def service(db, backend, settings):
    return InterviewTrackService(db=db, settings=settings, backend=backend)
