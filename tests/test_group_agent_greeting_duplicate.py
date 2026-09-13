"""Bounded duplicate greeting repairs without providers or persisted writes."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from apps.group_agent_api.agent_factory.host_context import HostContext
from apps.group_agent_api.agent_factory.venue_greeting import VenueGreetingError, execute_venue_greeting
from apps.group_agent_api.execution.models import RetryClass
from apps.group_agent_api.execution.retry import classify_error
from tests.test_group_agent_host_da001 import _host
from tests.test_group_agent_venue_da002 import BareGreetingModel

OLD = "回来了，想先聊聊咖啡的风味吗？"
NEW = "欢迎回来，今天想聊点什么？"


def _returning(tmp_path: Path, version: int) -> HostContext:
    host = _host(tmp_path, version=version)
    snapshot = copy.deepcopy(host.snapshot)
    snapshot["previous_greeting"] = {"message_id": "old", "text": OLD, "created_at": "2026-09-12T02:00:00Z"}
    snapshot["recent_messages"] = [{"message_id": "old", "role": "assistant", "text": OLD, "created_at": "2026-09-12T02:00:00Z", "result_kind": "greeting"}]
    snapshot["previous_visit_at"] = "2026-09-12T02:00:00Z"
    return replace(host, snapshot=snapshot)


def _output(reply: str) -> AIMessage:
    return AIMessage(content=json.dumps({"result_kind": "greeting", "reply": reply, "suggested_replies": []}, ensure_ascii=False))


@pytest.mark.parametrize("version", [1, 2])
async def test_duplicate_gets_precise_feedback_then_only_new_final(tmp_path: Path, version: int) -> None:
    host = _returning(tmp_path, version)
    before = copy.deepcopy(host.snapshot)
    model = BareGreetingModel([_output(OLD), _output(NEW)])
    emit = AsyncMock(return_value=True)
    await execute_venue_greeting(host=host, model=model, emit_callback=emit)
    assert len(model.calls) == 2
    assert "与上次完全相同" in model.calls[1]["messages"][-1].content
    assert "不要编造" in model.calls[1]["messages"][-1].content
    emit.assert_awaited_once()
    assert emit.await_args.args[1]["reply"] == NEW
    assert host.snapshot == before


@pytest.mark.parametrize("version", [1, 2])
async def test_second_duplicate_uses_reduced_input_but_original_validation(tmp_path: Path, version: int) -> None:
    host = _returning(tmp_path, version)
    before = copy.deepcopy(host.snapshot)
    model = BareGreetingModel([_output(OLD), _output(OLD), _output(NEW)])
    emit = AsyncMock(return_value=True)
    gate = MagicMock()
    await execute_venue_greeting(host=host, model=model, emit_callback=emit, side_effect_gate=gate)
    assert len(model.calls) == 3
    assert OLD not in "".join(item.content for item in model.calls[2]["messages"])
    assert "本次仍是重访" in model.calls[2]["messages"][0].content
    assert host.snapshot == before
    assert all(call["kwargs"]["tools"] == [] for call in model.calls)
    assert gate.call_count == 4
    emit.assert_awaited_once()


@pytest.mark.parametrize("last,code", [(OLD, "greeting_duplicate"), ("你的项目上线了，恭喜！", "security_guard_failed")])
async def test_last_attempt_never_bypasses_duplicate_or_safety(tmp_path: Path, last: str, code: str) -> None:
    model = BareGreetingModel([_output(OLD), _output(OLD), _output(last)])
    emit = AsyncMock()
    with pytest.raises(VenueGreetingError, match=code):
        await execute_venue_greeting(host=_returning(tmp_path, 2), model=model, emit_callback=emit)
    assert len(model.calls) == 3
    emit.assert_not_called()


@pytest.mark.parametrize("reply,code", [("   ", "greeting_reply_empty"), ("not json", "greeting_schema_invalid"), ("已经帮你发送邀请", "security_guard_failed")])
async def test_other_errors_stay_at_two_attempts(tmp_path: Path, reply: str, code: str) -> None:
    model = BareGreetingModel([AIMessage(content=reply) if reply == "not json" else _output(reply)])
    emit = AsyncMock()
    with pytest.raises(VenueGreetingError, match=code):
        await execute_venue_greeting(host=_returning(tmp_path, 2), model=model, emit_callback=emit)
    assert len(model.calls) == 2
    emit.assert_not_called()
    if code == "greeting_reply_empty":
        assert "正文为空" in model.calls[1]["messages"][-1].content


@pytest.mark.parametrize("code", ["greeting_duplicate", "greeting_reply_empty"])
def test_exhausted_local_repairs_are_not_requeued(code: str) -> None:
    assert classify_error(code) == RetryClass.PERMANENT
