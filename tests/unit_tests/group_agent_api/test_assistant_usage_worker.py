"""Exercise worker _emit routing and real signed HTTP serialization."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import json
import httpx
import pytest
from apps.group_agent_worker import tasks
from apps.group_agent_api.app.assistant_usage import UsageReceipt, invoke, sink


@pytest.mark.parametrize('failure', [None, 'reject', 'network'])
async def test_worker_posts_usage_without_poisoning_final(monkeypatch, failure):
    posts = []
    real_client = httpx.AsyncClient
    def receive(request):
        envelope = json.loads(request.content)
        posts.append(envelope)
        if failure == 'network':
            raise httpx.ConnectError('offline', request=request)
        return httpx.Response(422 if failure == 'reject' else 200, json={'status': 'ok'})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(receive), **kw))
    monkeypatch.setenv('GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS', 'http://micro:3001/group_agent_callbacks')
    monkeypatch.setenv('GROUP_AGENT_CALLBACK_HMAC_SECRET', 'synthetic-test-secret')
    # Business fence is deliberately separate; usage must never enter its slot.
    business = []
    class Fence:
        aborted = False
        terminal_delivered = False
        final_callback_ok = None
        def __init__(self, **kw): pass
        def on_abort(self, callback): pass
        def assert_owner(self): return True
        async def recover_pending(self, **kw): return True
        async def emit_durable(self, event, payload, **kw):
            business.append(event)
            assert 'supplier_usage' not in payload
            if event == 'final':
                assert payload['usage_manifest'][0]['total_tokens'] == 15
                self.final_callback_ok = True
            return True
    monkeypatch.setattr(tasks, 'SideEffectFence', Fence)
    record = SimpleNamespace(attempt_count=1, pending_callback=None, routing_snapshot_hash=None)
    store = MagicMock()
    store.get.return_value = record
    monkeypatch.setattr(tasks, 'get_worker_runtime', lambda: {'store': store, 'state': None,
        'config': SimpleNamespace(heartbeat_fail_threshold=3, heartbeat_interval_s=100, max_attempts=3),
        'backpressure': MagicMock()})
    monkeypatch.setattr(tasks, 'set_active_fence', lambda *a, **kw: None)
    monkeypatch.setattr(tasks, 'clear_active_fence', lambda *a: None)
    class Model:
        async def ainvoke(self, messages):
            return SimpleNamespace(content='reply', usage_metadata={'input_tokens': 10, 'output_tokens': 5})
    async def core(**kw):
        receipt = UsageReceipt(kw['emit_callback'])
        token = sink.set(receipt.emit)
        try:
            result = await invoke(Model(), [], operation='conversation', timeout=1)
            assert result.content == 'reply'
            await kw['emit_callback']('final', {'assistant_result': {'reply': result.content}, 'usage_manifest': receipt.manifest()})
        finally:
            sink.reset(token)
    monkeypatch.setattr(tasks, 'execute_async_run_core', core)
    req = SimpleNamespace(request_schema_version=5, run_id='run-test', idempotency_key='test-key',
        conversation_id='ai_assistant_1', callback_url='http://micro:3001/group_agent_callbacks/run-test', input_kind='user_message')
    outcome = await tasks._run_attempt(claim=SimpleNamespace(attempt_id='attempt-test'), req=req,
        session=SimpleNamespace(principal=SimpleNamespace(user_id='1'), group_id='global'), tid='test')
    assert outcome == 'succeeded'
    assert business == ['progress', 'final']
    assert [p['payload']['supplier_usage']['status'] for p in posts] == ['started', 'completed']
    assert all(p['payload']['usage_channel_version'] == 1 for p in posts)
    store.create_pending_callback.assert_not_called()
