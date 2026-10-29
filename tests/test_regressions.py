"""Regression checks for the recovered ZIP; no personal credentials or network."""
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from interviewtrack.config import Settings, get_settings
from interviewtrack.gmail import desktop_oauth as oauth
from interviewtrack.embeddings import EmbeddingBackend


def test_data_directory_resolved_when_settings_created(monkeypatch, tmp_path):
    monkeypatch.setenv('INTERVIEWTRACK_DATA_DIR', str(tmp_path))
    settings = get_settings(refresh=True)
    assert settings.db_path.parent == tmp_path
    assert Path(settings.token_file).parent == tmp_path
    assert Path(settings.tokens_dir).parent == tmp_path


def test_single_named_token_is_usable_without_account(tmp_path):
    settings = Settings(token_file=str(tmp_path/'legacy.json'), tokens_dir=str(tmp_path/'tokens'))
    token = oauth.token_path_for('one@example.com', settings)
    token.parent.mkdir()
    token.write_text('{}')
    assert oauth.token_path_for(None, settings) == token
    oauth.token_path_for('two@example.com', settings).write_text('{}')
    with pytest.raises(oauth.AuthError, match='Multiple accounts'):
        oauth.token_path_for(None, settings)


def test_status_uses_requested_account(monkeypatch, tmp_path):
    from interviewtrack import auth
    settings = Settings(client_secret_file=str(tmp_path/'client.json'), token_file=str(tmp_path/'token.json'))
    Path(settings.client_secret_file).write_text('{}')
    Path(settings.token_file).write_text('{}')
    monkeypatch.setattr(auth, 'get_settings', lambda: settings)
    lookup = Mock(return_value='person@example.com')
    monkeypatch.setattr(auth, 'account_email', lookup)
    assert auth.main(['--status']) == 0
    lookup.assert_called_once_with(None, settings)


def test_reauthorization_uses_returned_identity(monkeypatch, capsys):
    from interviewtrack import auth
    monkeypatch.setattr(auth, 'has_credentials', lambda *a: True)
    monkeypatch.setattr(auth, 'has_token', lambda *a: True)
    authorize = Mock(return_value={'account_email':'person@example.com'})
    monkeypatch.setattr(auth, 'authorize', authorize)
    assert auth.main(['--account','person@example.com']) == 0
    assert 'person@example.com.json' in capsys.readouterr().out
    assert authorize.call_args.kwargs['account'] == 'person@example.com'


def test_stored_write_scopes_rejected_before_loading(monkeypatch, tmp_path):
    settings = Settings(token_file=str(tmp_path/'token.json'))
    Path(settings.token_file).write_text(json.dumps({'scopes':['https://www.googleapis.com/auth/gmail.modify']}))
    with pytest.raises(oauth.AuthError, match='write-capable'):
        oauth.load_credentials(settings=settings)


def test_profile_lookup_has_identity_fallback(monkeypatch):
    import googleapiclient.discovery
    from google.auth.transport import requests
    monkeypatch.setattr(googleapiclient.discovery, 'build', Mock(side_effect=RuntimeError('unavailable')))
    response = Mock()
    response.json.return_value = {'email':'person@example.com'}
    session = Mock()
    session.get.return_value = response
    context = Mock()
    context.__enter__ = Mock(return_value=session)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(requests, 'AuthorizedSession', lambda creds: context)
    assert oauth._account_email_from_creds(object()) == 'person@example.com'


def test_runtime_failure_reencodes_cached_vectors_and_dimension():
    class Broken:
        def encode(self, *a, **k):
            raise RuntimeError('failed')
    backend = EmbeddingBackend(kind='sentence-transformer', dimension=32, _model=Broken())
    backend._cache['cached'] = np.ones(32)
    actual = backend.encode(['cached', 'new'])
    expected = EmbeddingBackend().encode(['cached', 'new'])
    assert actual.shape == (2,384)
    np.testing.assert_allclose(actual, expected)
