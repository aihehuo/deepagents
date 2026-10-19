"""Meter supplier calls without exposing dialogue or changing model results."""
from types import SimpleNamespace
import pytest
from apps.group_agent_api.app.assistant_usage import counters, invoke, sink


class Model:
    async def ainvoke(self, messages):
        return SimpleNamespace(content='private text', usage_metadata={
            'input_tokens': 100, 'output_tokens': 20,
            'input_token_details': {'cache_read': 30}, 'output_token_details': {'reasoning': 5}},
            response_metadata={'model_name': 'qwen-plus', 'id': 'supplier-1'})


async def test_records_each_invocation_and_preserves_actual_counts():
    events = []
    async def emit(event, payload):
        events.append(payload['supplier_usage'])
    token = sink.set(emit)
    try:
        await invoke(Model(), [], operation='conversation', timeout=1)
        await invoke(Model(), [], operation='conversation', timeout=1)
    finally:
        sink.reset(token)
    assert len(events) == 4
    assert events[0]['call_id'] == events[1]['call_id']
    assert events[0]['call_id'] != events[2]['call_id']
    assert events[1]['total_tokens'] == 120
    assert events[1]['cached_tokens'] == 30
    assert events[1]['reasoning_tokens'] == 5
    assert 'private text' not in str(events)


async def test_timeout_records_unknown_instead_of_zero():
    import asyncio
    class Slow:
        async def ainvoke(self, messages):
            await asyncio.sleep(1)
    events = []
    async def emit(event, payload):
        events.append(payload['supplier_usage'])
    token = sink.set(emit)
    try:
        with pytest.raises(TimeoutError):
            await invoke(Slow(), [], operation='match', timeout=.001)
    finally:
        sink.reset(token)
    assert events[-1]['status'] == 'failed'
    assert events[-1]['usage_status'] == 'unknown'
    assert 'input_tokens' not in events[-1]


def test_missing_or_invalid_usage_is_not_zero():
    assert counters(SimpleNamespace())['input_tokens'] is None
    assert counters(SimpleNamespace(usage_metadata={'input_tokens': True}))['usage_status'] == 'unknown'

async def test_metering_exception_never_consumes_generated_reply():
    async def broken(event, payload):
        raise RuntimeError('telemetry unavailable')
    token = sink.set(broken)
    try:
        output = await invoke(Model(), [], operation='conversation', timeout=1)
    finally:
        sink.reset(token)
    assert output.content == 'private text'

@pytest.mark.parametrize('metadata', [
    {'usage_metadata': ['invalid']},
    {'response_metadata': ['invalid']},
    {'usage_metadata': {'input_tokens': 10, 'output_tokens': 5, 'input_token_details': ['invalid']}},
    {'usage_metadata': {'input_tokens': 10, 'output_tokens': 5, 'output_token_details': ['invalid']}},
])
async def test_malformed_usage_does_not_discard_successful_reply(metadata):
    class Malformed:
        async def ainvoke(self, messages):
            return SimpleNamespace(content='successful reply', **metadata)
    events = []
    async def emit(event, payload):
        events.append(payload['supplier_usage'])
    token = sink.set(emit)
    try:
        output = await invoke(Malformed(), [], operation='conversation', timeout=1)
    finally:
        sink.reset(token)
    assert output.content == 'successful reply'
    assert events[-1]['status'] == 'completed'
    assert events[-1]['usage_status'] == 'unknown'
    assert 'total_tokens' not in events[-1]

async def test_durable_receipt_survives_progress_failure_and_contains_no_text():
    from apps.group_agent_api.app.assistant_usage import UsageReceipt
    async def unavailable(event, payload):
        raise RuntimeError('progress lost')
    receipt = UsageReceipt(unavailable)
    token = sink.set(receipt.emit)
    try:
        result = await invoke(Model(), [], operation='conversation', timeout=1)
    finally:
        sink.reset(token)
    assert result.content == 'private text'
    assert len(receipt.manifest()) == 1
    assert receipt.manifest()[0]['total_tokens'] == 120
    assert receipt.manifest()[0]['status'] == 'completed'
    assert 'private text' not in str(receipt.manifest())

async def test_overflow_receipt_is_incomplete_instead_of_partially_chargeable():
    from apps.group_agent_api.app.assistant_usage import UsageReceipt
    async def forward(event, payload):
        return True
    receipt = UsageReceipt(forward)
    for n in range(17):
        await receipt.emit('progress', {'supplier_usage': {'call_id': str(n), 'status': 'completed'}})
    assert receipt.manifest() == []
