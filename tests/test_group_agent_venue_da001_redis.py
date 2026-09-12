"""DA-001 fault tests against an explicitly selected isolated localhost Redis."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest

from apps.group_agent_api.app.models import AsyncCallRequest, CallbackEnvelope
from apps.group_agent_api.app.utils import thread_id
from apps.group_agent_api.app.venue_protocol import lease_scope
from apps.group_agent_api.execution.admission import admit_durable_async_call
from apps.group_agent_api.execution.backpressure import BackpressureController
from apps.group_agent_api.execution.dlq import DlqAdmin
from apps.group_agent_api.execution.models import ExecutionStatus
from apps.group_agent_api.execution.recovery import run_recovery_once
from apps.group_agent_worker import tasks
from tests.test_group_agent_req032_integration import env_cfg  # noqa: F401
from tests.test_group_agent_venue_da001 import REQUESTS, _session

pytestmark = pytest.mark.skipif(urlparse(os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "")).hostname not in {"127.0.0.1", "localhost"}, reason="requires explicitly selected isolated localhost Redis")


def _tid(req: AsyncCallRequest) -> str:
    return thread_id(user_id=req.user_id, group_id=req.group_id, conversation_id=req.conversation_id, episode_id=req.episode_id)


def test_v2_worker_hold_does_not_lock_or_fence_normal_chat(env_cfg: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    config, store = env_cfg
    bp = BackpressureController(store._r, config)
    ordinary = AsyncCallRequest.model_validate(REQUESTS[2]["request"])
    event = AsyncCallRequest.model_validate(REQUESTS[0]["request"])
    enqueue = MagicMock()
    for req in (ordinary, event):
        result = admit_durable_async_call(req=req, session=_session(req), thread_id=_tid(req), store=store, config=config, backpressure=bp, enqueue=enqueue)
        assert result.response.queue_schema_version == 2
    normal_record = store.get(ordinary.run_id)
    event_record = store.get(event.run_id)
    assert event_record.input_kind == "venue_event"
    assert normal_record.input_kind == "user_message"
    normal_claim = store.claim_lease(run_id=ordinary.run_id, conversation_id=ordinary.conversation_id, owner="ordinary-test-worker")
    assert normal_claim.kind == "claimed"
    epoch_key = store.epoch_key_for_record(normal_record, conversation_id=ordinary.conversation_id)
    normal_epoch = store._r.get(epoch_key)
    assert store.epoch_key_for_record(event_record, conversation_id=lease_scope(event)) != epoch_key
    state = MagicMock()
    state.greeting_model = None
    runtime = {"store": store, "config": config, "state": state, "backpressure": bp}
    monkeypatch.setattr(tasks, "get_worker_runtime", lambda: runtime)
    callback = AsyncMock(return_value=True)
    monkeypatch.setattr(tasks, "send_callback_event", callback)
    event_ref = enqueue.call_args.args[0]
    result = tasks.process_run.run(event_ref.to_dict())
    assert result["status"] == "dead_lettered"
    assert store.get(event.run_id).last_error_code == "venue_event_executor_unavailable"
    assert store._r.get(epoch_key) == normal_epoch
    assert store.get(ordinary.run_id).status == ExecutionStatus.RUNNING
    assert not state.mock_calls
    assert callback.await_count == 2
    envelopes = [call.kwargs["envelope_dict"] for call in callback.await_args_list]
    for envelope in envelopes:
        CallbackEnvelope.model_validate(envelope)
        assert envelope["conversation_id"] == event.conversation_id
        assert envelope["episode_id"] == event.episode_id
        assert envelope["visit_id"] == event.event["visit_id"]
        assert "idempotency_key" not in envelope
    assert envelopes[0]["payload"] == {"phase": "preparing_greeting"}
    assert envelopes[1]["payload"]["error_code"] == "venue_event_executor_unavailable"
    replay_enqueue = MagicMock()
    admin = DlqAdmin(store, config, replay_enqueue)
    admin.replay(event.run_id, operator_id="synthetic-operator", reason="test version restoration")
    assert replay_enqueue.call_args.args[0].queue_schema_version == 2


def test_v2_enqueue_failure_recovery_keeps_full_request(env_cfg: Any) -> None:
    config, store = env_cfg
    bp = BackpressureController(store._r, config)
    req = AsyncCallRequest.model_validate(REQUESTS[0]["request"])
    with pytest.raises(Exception, match="enqueue_failed"):
        admit_durable_async_call(req=req, session=_session(req), thread_id=_tid(req), store=store, config=config, backpressure=bp, enqueue=MagicMock(side_effect=RuntimeError("synthetic broker failure")))
    assert store.get(req.run_id).status == ExecutionStatus.ENQUEUE_FAILED
    enqueue = MagicMock()
    report = run_recovery_once(store, config, enqueue, backpressure=bp)
    assert report.enqueue_failed_requeued == 1
    delivery = enqueue.call_args.args[0]
    assert delivery.queue_schema_version == 2
    assert delivery.request_fingerprint == req.request_fingerprint
    assert store.get(req.run_id).status == ExecutionStatus.QUEUED
