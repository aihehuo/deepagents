from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from apps.wu_tanchang_api.app.stream_history import (
    extract_main_agent_ai_delta,
    should_emit_streamed_ai_text,
    snapshot_pre_turn_ai,
)

WELCOME = (
    "你好，创业者！我是吴探长智能体，你的\"实体店商业外脑\"。"
    "15年商业洞察与策略思维已就位。\n\n"
    "成都茶饮、投入40万、开店快一年，客流波动遇到瓶颈——这几个信息很有价值。"
)
REGION = (
    "高新区、茶小馆、15元客单——明白了。周末有量、平日难熬，"
    "这个节奏很多茶饮店都卡在这。"
)
NEXT_TURN = (
    "写字楼底商、50平、外流不错但进店少——这个画面很清楚了。"
    "你其实是在跟\"路过心智\"做博弈。"
)


def _snapshot() -> object:
    return snapshot_pre_turn_ai(
        [
            HumanMessage(content="哎，吴探长的助手你好啊", id="h1"),
            AIMessage(content=WELCOME, id="ai-hist-1"),
            HumanMessage(content="店在成都高新区", id="h2"),
            AIMessage(content=REGION, id="ai-hist-2"),
        ]
    )


def test_snapshot_collects_ai_ids_and_full_texts_only() -> None:
    snap = _snapshot()
    assert snap.ids == frozenset({"ai-hist-1", "ai-hist-2"})
    assert WELCOME in snap.texts
    assert REGION in snap.texts
    assert "哎，吴探长的助手你好啊" not in snap.texts


def test_skips_replayed_chunk_with_checkpoint_id() -> None:
    snap = _snapshot()
    replay = AIMessageChunk(content=WELCOME, id="ai-hist-1")
    assert should_emit_streamed_ai_text(replay, snap) is False
    assert (
        extract_main_agent_ai_delta((), "messages", (replay, {}), snap) is None
    )


def test_skips_replayed_full_text_even_with_new_id() -> None:
    snap = _snapshot()
    replay = AIMessageChunk(content=REGION, id="brand-new-id")
    assert should_emit_streamed_ai_text(replay, snap) is False


def test_emits_new_turn_tokens_and_full_reply() -> None:
    snap = _snapshot()
    token = AIMessageChunk(content="写", id="ai-new")
    assert should_emit_streamed_ai_text(token, snap) is True
    full = AIMessageChunk(content=NEXT_TURN, id="ai-new")
    assert should_emit_streamed_ai_text(full, snap) is True
    extracted = extract_main_agent_ai_delta((), "messages", (full, {}), snap)
    assert extracted == (NEXT_TURN, "ai-new")


def test_ignores_subagent_and_empty_chunks() -> None:
    snap = _snapshot()
    chunk = AIMessageChunk(content=NEXT_TURN, id="ai-new")
    assert (
        extract_main_agent_ai_delta(("task:1",), "messages", (chunk, {}), snap)
        is None
    )
    assert extract_main_agent_ai_delta((), "updates", {"agent": {}}, snap) is None
    empty = AIMessageChunk(content="", id="ai-new")
    assert should_emit_streamed_ai_text(empty, snap) is False


def test_short_repeatable_tokens_are_not_fingerprinted() -> None:
    snap = snapshot_pre_turn_ai([AIMessage(content="好的。", id="ai-short")])
    assert snap.texts == frozenset()
    assert should_emit_streamed_ai_text(
        AIMessageChunk(content="好的。", id="ai-new"), snap
    )
