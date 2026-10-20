"""Exercise bounded endpoint fallback without real provider calls."""
import asyncio
from types import SimpleNamespace
import pytest
from apps.group_agent_api.app import assistant_usage as usage
from apps.group_agent_api.app import assistant_model_fallback as fallback

class Client:
    def __init__(self, name, error=None):
        self.model_name, self.error, self.calls = name, error, 0
    async def ainvoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.model_name, response_metadata={}, usage_metadata={})

@pytest.mark.parametrize('error', [None, TimeoutError(), ValueError('invalid'), asyncio.CancelledError()])
async def test_routing(monkeypatch, error):
    primary, standby = Client('primary', error), Client('standby')
    monkeypatch.setattr(fallback, 'primary_client', lambda _: primary)
    if isinstance(error, (ValueError, asyncio.CancelledError)):
        with pytest.raises(type(error)):
            await usage.invoke(standby, [], operation='conversation', timeout=1)
        assert standby.calls == 0
    else:
        result = await usage.invoke(standby, [], operation='conversation', timeout=1)
        assert result.content == ('standby' if error else 'primary')
        assert standby.calls == (1 if error else 0)

def test_disabled(monkeypatch):
    monkeypatch.delenv('AI_ASSISTANT_PRIMARY_ENABLED', raising=False)
    assert fallback.primary_client(object()) is None

@pytest.mark.parametrize('status', [400, 401, 403, 404, 422, 429, 500, 503])
async def test_http_failures_fallback_and_preserve_accounting(monkeypatch, status):
    import httpx
    from openai import APIStatusError
    error = APIStatusError('request failed', response=httpx.Response(status, request=httpx.Request('POST', 'https://example.invalid')), body=None)
    primary, standby = Client('auto', error), Client('qwen-plus')
    monkeypatch.setattr(fallback, 'primary_client', lambda _: primary)
    events = []
    async def emit(event, payload):
        events.append(payload['supplier_usage'])
    token = usage.sink.set(emit)
    try:
        result = await usage.invoke(standby, [], operation='conversation', timeout=1)
    finally:
        usage.sink.reset(token)
    assert result.content == 'qwen-plus'
    assert events[0]['model'] == 'auto'
    assert events[1]['usage_status'] == ('not_applicable' if status in {400, 401, 403, 404, 422, 429} else 'unknown')
    assert events[2]['model'] == 'qwen-plus'
    assert events[0]['call_id'] != events[2]['call_id']


def test_auto_omits_temperature(monkeypatch):
    from langchain_openai import ChatOpenAI
    monkeypatch.setenv('GROUP_AGENT_PROVIDER', 'qwen')
    monkeypatch.setenv('AI_ASSISTANT_PRIMARY_ENABLED', '1')
    monkeypatch.setenv('AI_ASSISTANT_PRIMARY_API_KEY', 'test')
    monkeypatch.setenv('AI_ASSISTANT_PRIMARY_MODEL', 'auto')
    monkeypatch.setenv('AI_ASSISTANT_PRIMARY_BASE_URL', 'https://example.invalid/v1')
    standby = ChatOpenAI(model='qwen-plus', api_key='test', temperature=0.3)
    primary = fallback.primary_client(standby)
    assert 'temperature' not in primary._default_params
    assert primary.max_retries == 0
    assert primary.extra_body['enable_thinking'] is False


def test_wrapped_provider_rejection_is_recognized():
    import httpx
    from openai import APIStatusError
    original = APIStatusError('invalid parameter', response=httpx.Response(400, request=httpx.Request('POST', 'https://example.invalid')), body=None)
    wrapped = ValueError('provider wrapper')
    wrapped.__cause__ = original
    assert fallback.may_fallback(wrapped)
