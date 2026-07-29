"""Unit tests for opt-in group_agent debug turn traces."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from apps.group_agent_api.agent_factory.debug_trace import (
    serialize_messages_delta,
    write_turn_trace,
)


def test_serialize_messages_delta_includes_tools() -> None:
    messages = [
        HumanMessage(content="换方向做社区生鲜配送"),
        AIMessage(
            content="收到",
            tool_calls=[
                {
                    "name": "save_group_profile",
                    "args": {"doing": "社区生鲜配送", "token": "secret"},
                    "id": "tc1",
                }
            ],
        ),
        ToolMessage(
            content="ok: saved",
            name="save_group_profile",
            tool_call_id="tc1",
        ),
    ]
    delta = serialize_messages_delta(messages, 0)
    assert delta[0]["role"] == "human"
    assert delta[1]["tool_calls"][0]["name"] == "save_group_profile"
    assert delta[1]["tool_calls"][0]["args"]["token"] == "[redacted]"
    assert delta[2]["role"] == "tool"


def test_write_turn_trace_respects_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GROUP_AGENT_DEBUG_TRACE", raising=False)
    assert (
        write_turn_trace(
            base_dir=tmp_path,
            run_id="r1",
            thread_id="t1",
            user_id="u1",
            group_id="g1",
            conversation_id="c1",
            episode_id="e1",
            user_message="hi",
            messages=[HumanMessage(content="hi")],
            msg_count_before=0,
            reply="ok",
        )
        is None
    )

    monkeypatch.setenv("GROUP_AGENT_DEBUG_TRACE", "1")
    path = write_turn_trace(
        base_dir=tmp_path,
        run_id="r1",
        thread_id="t1",
        user_id="u1",
        group_id="g1",
        conversation_id="c1",
        episode_id="e1",
        user_message="换方向",
        messages=[
            HumanMessage(content="换方向"),
            AIMessage(content="明白了"),
        ],
        msg_count_before=0,
        reply="明白了",
        profile_status="stale_episode",
    )
    assert path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["run_id"] == "r1"
    assert data["turn_messages"][0]["content"] == "换方向"
    assert data["profile_status"] == "stale_episode"
