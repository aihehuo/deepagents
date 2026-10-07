"""Durable Micro v1 routing retains its encrypted request binding."""

from types import SimpleNamespace

import pytest

from apps.group_agent_api.app import async_manager
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.membership_client import MembershipResult
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.agent_factory.intent_routing.execution import QueryPlanError, prepare_intent_execution
from apps.group_agent_api.app.cafe_host_model_tools import conversation_resolution
from apps.group_agent_api.execution import active_fence


@pytest.fixture(autouse=True)
def callback_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS", "http://127.0.0.1:9")


def _request() -> AsyncCallRequest:
    return AsyncCallRequest(
        run_id="ga_test", idempotency_key="test", user_id="u1", group_id="global",
        conversation_id="ga_wecom_u1_1", message="你好",
        callback_url="http://127.0.0.1:9/group_agent_callbacks",
        request_schema_version=1, queue_schema_version=1,
        request_fingerprint="a" * 64,
    )


def _session() -> TrustedSession:
    return TrustedSession(
        principal=SessionPrincipal(user_id="u1", unionid=None, user_token=None, source="test"),
        membership=MembershipResult(tier=CapabilityTier.in_group, source="test"),
        group_id="global", group_token=None,
    )


@pytest.mark.parametrize("matches", [True, False])
def test_durable_v1_snapshot_retains_ledger_binding(monkeypatch: pytest.MonkeyPatch, matches: bool) -> None:
    req, session = _request(), _session()
    assert async_manager.calculate_request_fingerprint(req, session) != req.request_fingerprint
    record = SimpleNamespace(
        run_id=req.run_id, request_fingerprint=req.request_fingerprint if matches else "b" * 64,
        routing_snapshot_hash=None, attempt_count=1,
    )

    class Store:
        def get(self, run_id: str) -> SimpleNamespace:
            assert run_id == req.run_id
            return record

        def routing_crypto_material(self) -> tuple[dict[str, bytes], str]:
            return {"v1": bytes(32)}, "v1"

        def create_routing_snapshot(self, claim: object, **fields: object) -> None:
            record.routing_snapshot_hash = fields["snapshot_hash"]
            record.routing_snapshot_ciphertext = fields["envelope"]
            record.routing_snapshot_version = 1
            record.routing_snapshot_key_version = "v1"

    fence = SimpleNamespace(
        store=Store(), claim=SimpleNamespace(run_id=req.run_id, attempt_id="attempt", attempt_number=1),
    )
    monkeypatch.setattr(active_fence, "get_active_fence", lambda: fence)
    monkeypatch.setattr("apps.group_agent_api.agent_factory.intent_routing.execution.get_active_fence", lambda: fence)
    fingerprint = async_manager._routing_request_fingerprint(req, session)
    assert fingerprint == req.request_fingerprint
    if not matches:
        with pytest.raises(QueryPlanError, match="routing_snapshot_invalid"):
            prepare_intent_execution(conversation_resolution(), original_message=req.message,
                run_id=req.run_id, request_fingerprint=fingerprint, execution_mode="group_agent", can_search=False)
        assert record.routing_snapshot_hash is None
    else:
        execution = prepare_intent_execution(conversation_resolution(), original_message=req.message,
            run_id=req.run_id, request_fingerprint=fingerprint, execution_mode="group_agent", can_search=False)
        assert execution.snapshot.run_id == req.run_id
        assert record.routing_snapshot_hash == execution.snapshot.canonical_content_hash


def test_legacy_without_fence_keeps_existing_fingerprint() -> None:
    req, session = _request(), _session()
    assert async_manager._routing_request_fingerprint(req, session) == async_manager.calculate_request_fingerprint(req, session)


def test_durable_missing_fingerprint_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    req = _request().model_copy(update={"request_fingerprint": None})
    monkeypatch.setattr(active_fence, "get_active_fence", lambda: object())
    with pytest.raises(QueryPlanError, match="routing_snapshot_invalid"):
        async_manager._routing_request_fingerprint(req, _session())
