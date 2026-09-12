"""Policy-2 encrypted admission/recovery tests with isolated localhost Redis."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest

from apps.group_agent_api.app.models import AsyncCallRequest, CallbackEnvelope
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.execution.admission import admit_durable_async_call
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.models import ExecutionStatus
from apps.group_agent_api.execution.redis_store import ExecutionStore
from apps.group_agent_worker import tasks
from tests.test_group_agent_host_da001 import _request
from tests.test_group_agent_host_da002 import ChatModel, _profile, _state
from tests.test_group_agent_venue_da001 import _session

pytest_plugins = ["tests.test_group_agent_req032_integration"]
pytestmark = pytest.mark.skipif(urlparse(os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "")).hostname not in {"127.0.0.1", "localhost"}, reason="requires explicitly selected isolated localhost Redis")


@pytest.mark.parametrize("streaming", [False, True])
def test_encrypted_request_restoration_executes_host_without_profile_commit_or_tools(env_cfg: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool) -> None:
    config, store = env_cfg
    bp = BackpressureController(store._r, config)
    req = AsyncCallRequest.model_validate(_request(index=2))
    frozen = copy.deepcopy(req.model_dump())
    tid = thread_id(user_id=req.user_id, group_id=req.group_id, conversation_id=req.conversation_id, episode_id=req.episode_id)
    enqueue = MagicMock()
    admit_durable_async_call(req=req, session=_session(req), thread_id=tid, store=store, config=config, backpressure=bp, enqueue=enqueue)
    # Recreate the store as a new consumer would, never use request-local state.
    restored_store = ExecutionStore(store._r, config)
    from apps.group_agent_api.execution.crypto import digest_id

    # profile_epoch is also the attempt lease counter and legitimately advances
    # on claim; profile_fence/meta are the actual profile commit markers.
    commit_keys = [config.key(kind, digest_id(str(req.user_id)), digest_id(str(req.group_id))) for kind in ("profile_fence", "profile_fence_meta")]
    commit_before = store._r.mget(commit_keys)
    profile_cas = MagicMock(side_effect=AssertionError("profile CAS must not execute"))
    monkeypatch.setattr(restored_store, "cas_profile_write_fence", profile_cas)
    state = _state(tmp_path, ChatModel(captured=[], bindings=[]))
    _profile(tmp_path)
    profile_files = {str(path): path.read_bytes() for path in (tmp_path / "users").rglob("*") if path.is_file()}
    monkeypatch.setenv("GROUP_AGENT_STREAMING_ENABLED", "1" if streaming else "0")
    monkeypatch.setattr(tasks, "get_worker_runtime", lambda: {"store": restored_store, "config": config, "backpressure": bp, "state": state})
    send = AsyncMock(return_value=True)
    monkeypatch.setattr(tasks, "send_callback_event", send)
    result = tasks.process_run.run(enqueue.call_args.args[0].to_dict())
    assert result["status"] == "succeeded"
    assert restored_store.get(req.run_id).status == ExecutionStatus.SUCCEEDED
    assert store._r.mget(commit_keys) == commit_before
    profile_cas.assert_not_called()
    assert req.model_dump() == frozen
    assert profile_files == {str(path): path.read_bytes() for path in (tmp_path / "users").rglob("*") if path.is_file()}
    assert not any(state.greeting_model.bindings)
    assert "tools" not in state.cafe_agent.get_graph().nodes
    final = None
    for call in send.await_args_list:
        envelope = call.kwargs["envelope_dict"]
        CallbackEnvelope.model_validate(envelope)
        assert envelope["episode_id"] == req.episode_id
        assert envelope["space_id"] == "community_cafe"
        if envelope["event"] == "final":
            final = envelope["payload"]
    assert final is not None
    assert final["profile_status"] == "host_read_only" and final["profile_persisted"] is False
    assert final["candidates"] == [] and final["invite_text"] is None


def test_worker_rogue_tool_response_does_not_fallback_to_legacy(env_cfg: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_core.messages import AIMessage

    config, store = env_cfg
    bp = BackpressureController(store._r, config)
    req = AsyncCallRequest.model_validate(_request(index=2))
    tid = thread_id(user_id=req.user_id, group_id=req.group_id, conversation_id=req.conversation_id, episode_id=req.episode_id)
    enqueue = MagicMock()
    admit_durable_async_call(req=req, session=_session(req), thread_id=tid, store=store, config=config, backpressure=bp, enqueue=enqueue)
    state = _state(tmp_path, ChatModel(output=AIMessage(content="", tool_calls=[{"id":"rogue", "name":"save_group_profile", "args":{}}])))
    monkeypatch.setattr(tasks, "get_worker_runtime", lambda: {"store":store,"config":config,"backpressure":bp,"state":state})
    send = AsyncMock(return_value=True)
    monkeypatch.setattr(tasks, "send_callback_event", send)
    result = tasks.process_run.run(enqueue.call_args.args[0].to_dict())
    assert result["status"] != "succeeded"
    assert not any(path.is_file() for path in (tmp_path / "users").rglob("*"))
    assert not any(call.kwargs["envelope_dict"]["event"] == "final" for call in send.await_args_list)
