import pytest
import asyncio
from apps.group_agent_api.agent_factory.stream_buffer import StreamChunkBuffer

@pytest.mark.asyncio
async def test_stream_chunk_buffer_normal_text():
    chunks = []
    async def fake_emit(event, payload):
        chunks.append((event, payload))
        return True

    buffer = StreamChunkBuffer(emit_callback=fake_emit, min_interval_s=0.05, max_buffer_chars=20)

    # Feed sentence with punctuation
    await buffer.feed_token("您好，")
    await buffer.feed_token("很高兴认识您。")
    await buffer.flush()

    assert len(chunks) >= 2
    full_text = "".join(c[1]["text"] for c in chunks)
    assert full_text == "您好，很高兴认识您。"

@pytest.mark.asyncio
async def test_stream_chunk_buffer_handles_normal_less_than_without_deadlock():
    chunks = []
    async def fake_emit(event, payload):
        chunks.append((event, payload))
        return True

    buffer = StreamChunkBuffer(emit_callback=fake_emit, min_interval_s=0.05, max_buffer_chars=20)

    # Normal comparison: "< 5000"
    await buffer.feed_token("预算 < ")
    await buffer.feed_token("5000 元，团队规模 < 10 人。")
    await buffer.flush()

    full_text = "".join(c[1]["text"] for c in chunks)
    assert "预算 < 5000 元，团队规模 < 10 人。" == full_text

@pytest.mark.asyncio
async def test_stream_chunk_buffer_strips_suggested_replies_tag():
    chunks = []
    async def fake_emit(event, payload):
        chunks.append((event, payload))
        return True

    buffer = StreamChunkBuffer(emit_callback=fake_emit, min_interval_s=0.05, max_buffer_chars=20)

    # Text containing private tag
    await buffer.feed_token("这是推荐给您的合伙人。")
    await buffer.feed_token("<suggested_replies>[{\"label\":\"了解更多\",\"reply\":\"请发我更多介绍\"}]</suggested_replies>")
    await buffer.flush()

    full_text = "".join(c[1]["text"] for c in chunks)
    assert full_text == "这是推荐给您的合伙人。"
    assert "<suggested_replies" in buffer.get_held_marker_content()
