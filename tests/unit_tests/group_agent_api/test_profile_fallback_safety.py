"""Bounded profile parsing and durable retry-budget regression tests."""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from apps.group_agent_api.app import async_manager as manager
from apps.group_agent_api.execution.models import ExecutionStatus, LeaseClaim
from apps.group_agent_worker import tasks


def test_repeated_missing_fields_are_bounded() -> None:
    # Previously this triggers repeated whole-document regex backtracking.
    for text in ("doing:项目 " * 280, "项目:创业 need:技术 " * 120):
        assert manager.extract_explicit_profile_dimensions(text) is None


def test_only_explicitly_requested_recent_human_history_is_used() -> None:
    profile = "doing:咖啡馆; need:运营合伙人; offer:供应链"
    history = [AIMessage(content=profile), ToolMessage(content=profile, tool_call_id="public")]
    assert manager.extract_explicit_profile_dimensions("保存我的画像", history) is None
    history.append(HumanMessage(content=profile))
    assert manager.extract_explicit_profile_dimensions("查询 FAQ", history) is None
    assert manager.extract_explicit_profile_dimensions("保存我的画像", history) == {"doing": "咖啡馆", "need": "运营合伙人", "offer": "供应链"}
    history.extend(HumanMessage(content="你好") for _ in range(6))
    assert manager.extract_explicit_profile_dimensions("保存我的画像", history) is None


@pytest.mark.parametrize("optout", ["只查询", "不要更新画像", "read-only", "do not save my profile"])
def test_read_only_request_never_extracts_profile(optout: str) -> None:
    profile = "doing:咖啡馆; need:运营; offer:供应链"
    assert manager.extract_explicit_profile_dimensions(f"{profile}; {optout}", [HumanMessage(content=profile)]) is None


async def test_extraction_failure_does_not_call_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = MagicMock()
    monkeypatch.setattr(manager, "extract_explicit_profile_dimensions", MagicMock(side_effect=ValueError("bad input")))
    from apps.group_agent_api.agent_factory import write_skills

    monkeypatch.setattr(write_skills, "member_write_configuration", lambda **_kwargs: ([writer], [], None))
    assert (
        await manager._attempt_deterministic_profile_save(message="保存我的画像", config={"metadata": {"can_write_profile": True}}, messages=[])
        is False
    )
    writer.assert_not_called()


@pytest.mark.parametrize("attempts", [3, 4])
async def test_timeout_and_lease_recovery_budget_emit_terminal_error(monkeypatch: pytest.MonkeyPatch, attempts: int) -> None:
    from apps.group_agent_api.app.models import AsyncCallRequest
    from tests.test_group_agent_mode_da001 import _body
    from tests.test_group_agent_venue_da001 import _session

    req = AsyncCallRequest.model_validate(_body(2))
    store = MagicMock()
    store.get.return_value = SimpleNamespace(attempt_count=attempts)
    cfg = SimpleNamespace(max_attempts=3, heartbeat_interval_s=60, heartbeat_fail_threshold=2, retry_base_s=1, retry_max_s=10)
    bp = MagicMock()
    monkeypatch.setattr(tasks, "get_worker_runtime", lambda: {"store": store, "state": object(), "config": cfg, "backpressure": bp})
    fence = MagicMock(unsafe=True)
    fence.assert_owner.return_value = True
    fence.emit = AsyncMock(return_value=True)
    monkeypatch.setattr(tasks, "SideEffectFence", lambda **_kwargs: fence)
    active = MagicMock(cancel_event=threading.Event())
    monkeypatch.setattr(tasks, "ActiveAttemptFence", lambda **_kwargs: active)
    monkeypatch.setattr(tasks, "set_active_fence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tasks, "clear_active_fence", lambda *_args: None)
    monkeypatch.setattr(tasks, "push_dlq_index", MagicMock())
    core = AsyncMock(side_effect=tasks.SoftTimeLimitExceeded())
    monkeypatch.setattr(tasks, "execute_async_run_core", core)
    claim = LeaseClaim(req.run_id, "attempt", "token", "worker", 9999999999)
    result = await tasks._run_attempt(claim=claim, req=req, session=_session(req), tid="test")
    assert result == ExecutionStatus.DEAD_LETTERED.value
    if attempts > cfg.max_attempts:
        core.assert_not_awaited()
    else:
        core.assert_awaited_once()
    assert fence.emit.await_args.args[0] == "error"
    store.finish.assert_called_once()
    assert store.finish.call_args.kwargs["error_code"] == "max_attempts_exceeded"
    bp.on_finish.assert_called_once()
