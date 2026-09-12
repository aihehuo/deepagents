"""Host public output after encrypted worker admission/restoration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest
from langchain_core.messages import AIMessage

from apps.group_agent_api.app.models import CallbackEnvelope
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.execution.admission import admit_durable_async_call
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.redis_store import ExecutionStore
from apps.group_agent_worker import tasks
from tests.test_group_agent_host_da002 import ChatModel, _profile, _state
from tests.test_group_agent_host_da003 import Reviewer, _req
from tests.test_group_agent_venue_da001 import _session

pytest_plugins = ["tests.test_group_agent_req032_integration"]
pytestmark = pytest.mark.skipif(
    urlparse(os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "")).hostname
    not in {"127.0.0.1", "localhost"},
    reason="requires explicitly selected isolated localhost Redis",
)


@pytest.mark.parametrize("streaming", [False, True])
def test_restored_worker_commits_only_validated_dialogue(
    env_cfg: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    config, store = env_cfg
    req = _req("这只是案例，给二手包三个诚实标题。")
    tid = thread_id(
        user_id=req.user_id,
        group_id=req.group_id,
        conversation_id=req.conversation_id,
        episode_id=req.episode_id,
    )
    bp = BackpressureController(store._r, config)
    enqueue = MagicMock()
    admit_durable_async_call(
        req=req,
        session=_session(req),
        thread_id=tid,
        store=store,
        config=config,
        backpressure=bp,
        enqueue=enqueue,
    )
    restored = ExecutionStore(store._r, config)
    titles = "1. 看锁扣内侧的细节\n2. 不同光线下看成色\n3. 先看五金再判断"
    state = _state(
        tmp_path,
        ChatModel(output=AIMessage(content="90%假货过不了这一关。\n" + titles)),
    )
    state.quality_model = Reviewer(captured=[])
    _profile(tmp_path)
    before = {
        str(path): path.read_bytes()
        for path in (tmp_path / "users").rglob("*")
        if path.is_file()
    }
    profile_cas = MagicMock(side_effect=AssertionError("café must not commit profile"))
    monkeypatch.setattr(restored, "cas_profile_write_fence", profile_cas)
    monkeypatch.setenv("GROUP_AGENT_STREAMING_ENABLED", "1" if streaming else "0")
    monkeypatch.setenv("GROUP_AGENT_GROUNDED_FINAL_ENABLED", "1")
    monkeypatch.setattr(
        tasks,
        "get_worker_runtime",
        lambda: {
            "store": restored,
            "config": config,
            "backpressure": bp,
            "state": state,
        },
    )
    send = AsyncMock(return_value=True)
    monkeypatch.setattr(tasks, "send_callback_event", send)
    result = tasks.process_run.run(enqueue.call_args.args[0].to_dict())
    assert result["status"] == "succeeded"
    chunks = []
    final = None
    for call in send.await_args_list:
        envelope = call.kwargs["envelope_dict"]
        CallbackEnvelope.model_validate(envelope)
        assert (
            envelope["episode_id"] == req.episode_id
            and envelope["space_id"] == "community_cafe"
        )
        if envelope["event"] == "chunk":
            chunks.append(envelope["payload"]["text"])
        if envelope["event"] == "final":
            final = envelope["payload"]
    assert final is not None
    assert (
        final["reply"]
        == final["dialogue_text"]
        == final["grounded_final"]["reply"]
        == titles
    )
    assert (
        final["reply_mode"] == "dialogue"
        and final["candidates"] == []
        and final["invite_text"] is None
    )
    assert "".join(chunks) == (titles if streaming else "")
    snapshot = state.cafe_agent.get_state({"configurable": {"thread_id": tid}})
    assert snapshot.values["messages"][-1].content == titles
    assert before == {
        str(path): path.read_bytes()
        for path in (tmp_path / "users").rglob("*")
        if path.is_file()
    }
    profile_cas.assert_not_called()
