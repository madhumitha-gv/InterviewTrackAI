import json
from unittest.mock import Mock
import pytest
from interviewtrack.gmail import client
from interviewtrack.config import Settings

class Response(dict):
    status=403

class Failure(Exception):
    def __init__(self,reason, retry_after='0'):
        self.resp=Response({'retry-after':retry_after})
        self.content=json.dumps({'error':{'errors':[{'reason':reason}]}}).encode()

@pytest.mark.parametrize('reason',['rateLimitExceeded','userRateLimitExceeded'])
def test_403_rate_limit_retries(monkeypatch,reason):
    sleep=Mock();monkeypatch.setattr(client.time,'sleep',sleep)
    monkeypatch.setattr(client.random,'uniform',lambda *args:0)
    fn=Mock(side_effect=[Failure(reason),Failure(reason),'success'])
    assert client._with_retries(fn)=='success'
    assert [c.args[0] for c in sleep.call_args_list]==[2,4]


def test_permission_403_not_retried(monkeypatch):
    sleep=Mock();monkeypatch.setattr(client.time,'sleep',sleep)
    fn=Mock(side_effect=Failure('insufficientPermissions'))
    with pytest.raises(Failure):client._with_retries(fn)
    assert fn.call_count==1
    sleep.assert_not_called()


def test_exhausted_rate_limit_stops_remaining_metadata(monkeypatch):
    monkeypatch.setattr(client.time,'sleep',lambda _:None)
    transport=Mock()
    transport.list_messages.return_value={'messages':[{'id':'one'},{'id':'two'}]}
    transport.get_message.side_effect=Failure('rateLimitExceeded')
    messages,report=client.fetch_messages(transport,Settings())
    assert report.rate_limited
    assert transport.get_message.call_count==4
    assert not messages
    assert 'paused' in report.errors[0]


def test_retry_after_respected(monkeypatch):
    sleep=Mock();monkeypatch.setattr(client.time,'sleep',sleep)
    fn=Mock(side_effect=[Failure('rateLimitExceeded','20'),'ok'])
    assert client._with_retries(fn)=='ok'
    sleep.assert_called_once_with(20)


def test_long_cooldown_stops_without_hammering(monkeypatch):
    sleep=Mock();monkeypatch.setattr(client.time,'sleep',sleep)
    fn=Mock(side_effect=Failure('rateLimitExceeded','120'))
    with pytest.raises(Failure):client._with_retries(fn)
    assert fn.call_count==1
    sleep.assert_not_called()


def test_pacing(monkeypatch):
    monkeypatch.setattr(client,'_last_request',10.0)
    monkeypatch.setattr(client.time,'monotonic',lambda:10.1)
    sleep=Mock();monkeypatch.setattr(client.time,'sleep',sleep)
    request=Mock();client._paced_execute(request)
    assert sleep.call_args.args[0]==pytest.approx(.4)
    request.execute.assert_called_once()
