"""Streamlit interaction checks against isolated synthetic storage."""
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest


def click(app, label):
    next(b for b in app.button if b.label == label).click().run(timeout=30)
    assert not app.exception


def test_demo_refresh_preserves_selected_profile(monkeypatch, tmp_path):
    from interviewtrack.config import get_settings
    from interviewtrack import db
    monkeypatch.setenv('INTERVIEWTRACK_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('EMBEDDINGS_ENABLED', 'false')
    get_settings(refresh=True)
    monkeypatch.setattr(db, '_db', None)
    app = AppTest.from_file(str(Path('app.py').resolve())).run(timeout=30)
    assert not app.exception
    app.selectbox[0].select('Profile 2 — Unfamiliar domains & custom labels').run()
    click(app, 'Start Demo Mode')
    assert next(r for r in app.radio if r.label == 'Filter').options == ['Active', 'Rejected']
    uid = app.session_state['user_id']
    database = db.get_db()
    before = database.known_gmail_ids(uid)
    assert before
    # A fresh test runner mirrors the next browser render after st.rerun.
    state = app.session_state.filtered_state
    app = AppTest.from_file(str(Path('app.py').resolve()))
    for key, value in state.items():
        app.session_state[key] = value
    app.run(timeout=30)
    click(app, '🔄 Refresh demo')
    assert database.known_gmail_ids(uid) == before
    click(app, '🚪 Log out')
    assert any(b.label == 'Start Demo Mode' for b in app.button)


def test_empty_gmail_sync_runs_once_and_keeps_logout(monkeypatch):
    from interviewtrack.service import InterviewTrackService, SyncOutcome
    calls = Mock(return_value=SyncOutcome())
    monkeypatch.setattr(InterviewTrackService, 'sync_from_source', calls)
    app = AppTest.from_file(str(Path('app.py').resolve()))
    for key, value in dict(user_id='gmail:empty@example.com', user_email='empty@example.com', mode='gmail').items():
        app.session_state[key] = value
    app.run(timeout=30)
    assert not app.exception
    assert calls.call_count == 1
    assert any(b.label == '🚪 Log out' for b in app.button)
    app.run(timeout=30)
    assert calls.call_count == 1
