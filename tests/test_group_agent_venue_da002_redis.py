"""DA-002 actual worker and run-fencing tests with isolated localhost Redis."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest

from apps.group_agent_api.app.models import AsyncCallRequest, CallbackEnvelope
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.execution.admission import admit_durable_async_call
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.models import ExecutionStatus
from apps.group_agent_worker import tasks
from tests.test_group_agent_req032_integration import env_cfg  # noqa: F401
from tests.test_group_agent_venue_da001 import REQUESTS, _session
from tests.test_group_agent_venue_da002 import BareGreetingModel

pytestmark = pytest.mark.skipif(urlparse(os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "")).hostname not in {"127.0.0.1", "localhost"}, reason="requires explicitly selected isolated localhost Redis")


def test_greeting_success_leaves_normal_checkpoint_lease_and_profile_epoch_untouched(env_cfg: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, store = env_cfg
    bp = BackpressureController(store._r, config)
    ordinary = AsyncCallRequest.model_validate(REQUESTS[2]["request"])
    greeting = AsyncCallRequest.model_validate(REQUESTS[0]["request"])
    enqueue = MagicMock()
    for req in (ordinary, greeting):
        tid = thread_id(user_id=req.user_id, group_id=req.group_id, conversation_id=req.conversation_id, episode_id=req.episode_id)
        admit_durable_async_call(req=req, session=_session(req), thread_id=tid, store=store, config=config, backpressure=bp, enqueue=enqueue)
    claimed = store.claim_lease(run_id=ordinary.run_id, conversation_id=ordinary.conversation_id, owner="normal-test-worker")
    assert claimed.kind == "claimed"
    epoch_key = store.epoch_key_for_record(store.get(ordinary.run_id), conversation_id=ordinary.conversation_id)
    original_epoch = store._r.get(epoch_key)
    model = BareGreetingModel()
    state = SimpleNamespace(base_dir=tmp_path, greeting_model=model, agent=MagicMock(), admin_agent=MagicMock(), thread_locks={})
    monkeypatch.setattr(tasks, "get_worker_runtime", lambda: {"store": store, "config": config, "state": state, "backpressure": bp})
    callback = AsyncMock(return_value=True)
    monkeypatch.setattr(tasks, "send_callback_event", callback)
    result = tasks.process_run.run(enqueue.call_args.args[0].to_dict())
    assert result["status"] == "succeeded"
    assert store.get(greeting.run_id).status == ExecutionStatus.SUCCEEDED
    assert store.get(ordinary.run_id).status == ExecutionStatus.RUNNING
    assert store._r.get(epoch_key) == original_epoch
    assert not state.agent.mock_calls and not state.admin_agent.mock_calls
    assert state.thread_locks == {}
    assert list(tmp_path.rglob("*")) == []
    assert len(model.calls) == 1
    events = [call.kwargs["envelope_dict"] for call in callback.await_args_list]
    assert [event["event"] for event in events] == ["progress", "final"]
    assert [event["seq"] for event in events] == [1, 2]
    for event in events:
        CallbackEnvelope.model_validate(event)
        assert event["conversation_id"] == greeting.conversation_id
        assert event["visit_id"] == greeting.event["visit_id"]
    assert events[1]["payload"]["result_kind"] == "greeting"
    assert set(events[1]["payload"]) == {"result_kind", "reply", "suggested_replies"}
