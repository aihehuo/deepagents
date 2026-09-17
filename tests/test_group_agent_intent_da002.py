"""INTENT-DA-002 durable snapshot, budget, receipt, and adoption integration."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid

import pytest
from redis import Redis

from apps.group_agent_api.agent_factory.intent_routing.execution import (
    QueryPlanError,
    prepare_intent_execution,
)
from apps.group_agent_api.agent_factory.intent_routing.resolver import resolve_intent
from apps.group_agent_api.execution.active_fence import (
    ActiveAttemptFence,
    clear_active_fence,
    set_active_fence,
)
from apps.group_agent_api.execution.config import load_durable_queue_config
from apps.group_agent_api.execution.crypto import digest_id
from apps.group_agent_api.execution.models import ExecutionRecord, ExecutionStatus
from apps.group_agent_api.execution.redis_store import ExecutionStore

REDIS_URL = os.environ.get("GROUP_AGENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


@pytest.fixture
def durable(monkeypatch: pytest.MonkeyPatch):
    prefix = f"ga:test:intent002:{uuid.uuid4().hex[:10]}"
    key = base64.b64encode(bytes(range(32))).decode()
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    monkeypatch.setenv("GROUP_AGENT_REDIS_URL", REDIS_URL)
    monkeypatch.setenv("GROUP_AGENT_REDIS_PREFIX", prefix)
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_KEYS", f"intent-k1:{key}")
    monkeypatch.setenv("GROUP_AGENT_QUEUE_PAYLOAD_CURRENT_VERSION", "intent-k1")
    monkeypatch.setenv("GROUP_AGENT_LEASE_TTL_S", "30")
    monkeypatch.setenv("GROUP_AGENT_HEARTBEAT_INTERVAL_S", "5")
    monkeypatch.setenv("GROUP_AGENT_VISIBILITY_TIMEOUT_S", "240")
    monkeypatch.setenv("GROUP_AGENT_TASK_SOFT_LIMIT_S", "150")
    monkeypatch.setenv("GROUP_AGENT_TASK_HARD_LIMIT_S", "180")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    config = load_durable_queue_config(require_enabled=True)
    assert config is not None
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - local integration dependency.
        pytest.skip(f"isolated Redis unavailable: {exc}")
    store = ExecutionStore(client, config)
    yield store
    for redis_key in client.scan_iter(match=f"{prefix}:*", count=200):
        client.delete(redis_key)


def _fp(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _seed_claim(store: ExecutionStore, run_id: str, fingerprint: str):
    store.create_or_get(
        record=ExecutionRecord(
            run_id=run_id,
            idempotency_key=f"idem-{run_id}",
            request_schema_version=1,
            request_fingerprint=fingerprint,
            queue_schema_version=1,
            status=ExecutionStatus.ACCEPTED,
            created_at=store.redis_time(),
            conversation_id=f"conversation-{run_id}",
            user_id_digest=digest_id("user-1"),
            group_id_digest=digest_id("group-1"),
        )
    )
    store.mark_queued(run_id, expected_status="accepted")
    outcome = store.claim_lease(
        run_id=run_id,
        conversation_id=f"conversation-{run_id}",
        owner="intent-worker",
    )
    assert outcome.claim is not None
    return outcome.claim


def _activate(store: ExecutionStore, claim):
    import threading

    return set_active_fence(
        ActiveAttemptFence(
            store=store,
            claim=claim,
            conversation_id=f"conversation-{claim.run_id}",
            cancel_event=threading.Event(),
        ),
        require=True,
    )


def _resolution(*, can_search: bool = True):
    from tests.unit_tests.group_agent_api.planner_fixtures import planner_for_message

    return resolve_intent(
        "有没有上海餐饮合伙人",
        execution_mode="group_agent",
        can_search=can_search,
        planner_model=planner_for_message("有没有上海餐饮合伙人"),
    )


def _content() -> str:
    return (
        '{"data":[{"id":"user-1","name":"公开用户"}],'
        '"_source_url":"https://www.aihehuo.com/ai/users.json?page=1"}'
    )


def test_snapshot_receipt_survive_attempt_and_are_adopted_without_new_budget(
    durable: ExecutionStore,
) -> None:
    run_id = "intent-run-adopt"
    fingerprint = _fp(run_id)
    first_claim = _seed_claim(durable, run_id, fingerprint)
    tokens = _activate(durable, first_claim)
    try:
        first = prepare_intent_execution(
            _resolution(),
            original_message="有没有上海餐饮合伙人",
            run_id=run_id,
            request_fingerprint=fingerprint,
        )
        reservation = first.reserve(
            resource="users", parameters={"page": 1}, tool_call_id="tool-1"
        )
        reserved_ledger = durable.get_receipt_ledger(run_id)
        assert isinstance(reserved_ledger["receipt_refs"], list)
        assert isinstance(reserved_ledger["audit_refs"], list)
        assert isinstance(reserved_ledger["adoption_refs"], list)
        receipt = first.commit_result(reservation, _content())
        assert first.required_steps_satisfied()
    finally:
        clear_active_fence(tokens)

    durable.schedule_retry(
        first_claim,
        conversation_id=f"conversation-{run_id}",
        next_attempt_at=durable.redis_time() - 1,
        error_code="worker_lost",
    )
    assert durable.promote_retry_wait(run_id) == "ok"
    durable.mark_queued(run_id, expected_status="enqueue_failed")
    second = durable.claim_lease(
        run_id=run_id,
        conversation_id=f"conversation-{run_id}",
        owner="intent-worker-2",
    ).claim
    assert second is not None
    tokens = _activate(durable, second)
    try:
        recovered = prepare_intent_execution(
            _resolution(),
            original_message="有没有上海餐饮合伙人",
            run_id=run_id,
            request_fingerprint=fingerprint,
        )
        assert recovered.required_steps_satisfied()
        assert recovered.adopted_result("users") is not None
        assert receipt.receipt_id in recovered.adopted_receipt_ids
        ledger = durable.get_receipt_ledger(run_id)
        assert ledger["budget"]["requests_reserved"] == 1
        assert ledger["adoption_refs"] == [
            {"receipt_id": receipt.receipt_id, "adopted_by_attempt_number": 2}
        ]
        from apps.group_agent_api.agent_factory.checks.intent_final_guard import (
            apply_intent_final_guard,
        )

        guarded = apply_intent_final_guard(
            reply="已经搜索到候选人【公开用户】。",
            execution=recovered,
            enabled=True,
        )
        assert not guarded.blocked
        assert "查询范围" in guarded.reply
        corrupted = durable.get_receipt_ledger(run_id)
        corrupted["budget"]["items_committed"] = 99
        durable._r.hset(  # noqa: SLF001
            durable.run_key(run_id),
            "receipt_ledger_json",
            json.dumps(corrupted, separators=(",", ":")),
        )
        guarded = apply_intent_final_guard(
            reply="已经搜索到候选人【公开用户】。",
            execution=recovered,
            enabled=True,
        )
        assert guarded.blocked
        assert "receipt_invalid" in guarded.violations
    finally:
        clear_active_fence(tokens)


def test_crashed_reservation_keeps_budget_and_revoked_permission_fails_closed(
    durable: ExecutionStore,
) -> None:
    run_id = "intent-run-crash"
    fingerprint = _fp(run_id)
    claim = _seed_claim(durable, run_id, fingerprint)
    tokens = _activate(durable, claim)
    try:
        execution = prepare_intent_execution(
            _resolution(),
            original_message="有没有上海餐饮合伙人",
            run_id=run_id,
            request_fingerprint=fingerprint,
        )
        execution.reserve(resource="users", parameters={}, tool_call_id="lost-result")
    finally:
        clear_active_fence(tokens)
    assert durable.get_receipt_ledger(run_id)["budget"]["requests_reserved"] == 1

    durable.schedule_retry(
        claim,
        conversation_id=f"conversation-{run_id}",
        next_attempt_at=durable.redis_time() - 1,
        error_code="worker_lost",
    )
    durable.promote_retry_wait(run_id)
    durable.mark_queued(run_id, expected_status="enqueue_failed")
    next_claim = durable.claim_lease(
        run_id=run_id,
        conversation_id=f"conversation-{run_id}",
        owner="intent-worker-2",
    ).claim
    assert next_claim is not None
    tokens = _activate(durable, next_claim)
    try:
        recovered = prepare_intent_execution(
            _resolution(),
            original_message="有没有上海餐饮合伙人",
            run_id=run_id,
            request_fingerprint=fingerprint,
        )
        with pytest.raises(QueryPlanError, match="query_budget_exhausted"):
            recovered.reserve(
                resource="users", parameters={}, tool_call_id="must-not-repeat"
            )
        with pytest.raises(QueryPlanError, match="permission_denied"):
            prepare_intent_execution(
                _resolution(can_search=False),
                original_message="有没有上海餐饮合伙人",
                run_id=run_id,
                request_fingerprint=fingerprint,
            )
    finally:
        clear_active_fence(tokens)


def test_cross_run_snapshot_ciphertext_and_corrupt_ledger_fail_closed(
    durable: ExecutionStore,
) -> None:
    first_run = "intent-run-source"
    first_fp = _fp(first_run)
    first_claim = _seed_claim(durable, first_run, first_fp)
    tokens = _activate(durable, first_claim)
    try:
        prepare_intent_execution(
            _resolution(),
            original_message="有没有上海餐饮合伙人",
            run_id=first_run,
            request_fingerprint=first_fp,
        )
        created_once = durable._r.hgetall(durable.run_key(first_run))  # noqa: SLF001
        prepare_intent_execution(
            _resolution(),
            original_message="有没有上海餐饮合伙人",
            run_id=first_run,
            request_fingerprint=first_fp,
        )
        recovered_once = durable._r.hgetall(durable.run_key(first_run))  # noqa: SLF001
        assert recovered_once["routing_snapshot_nonce_b64"] == created_once[
            "routing_snapshot_nonce_b64"
        ]
        assert recovered_once["routing_snapshot_ciphertext_b64"] == created_once[
            "routing_snapshot_ciphertext_b64"
        ]
    finally:
        clear_active_fence(tokens)

    second_run = "intent-run-target"
    second_fp = _fp(second_run)
    second_claim = _seed_claim(durable, second_run, second_fp)
    source = durable._r.hgetall(durable.run_key(first_run))  # noqa: SLF001
    copied = {
        key: value
        for key, value in source.items()
        if key.startswith("routing_snapshot_") or key == "receipt_ledger_json"
    }
    durable._r.hset(durable.run_key(second_run), mapping=copied)  # noqa: SLF001
    tokens = _activate(durable, second_claim)
    try:
        with pytest.raises(QueryPlanError, match="routing_snapshot_invalid"):
            prepare_intent_execution(
                _resolution(),
                original_message="有没有上海餐饮合伙人",
                run_id=second_run,
                request_fingerprint=second_fp,
            )
    finally:
        clear_active_fence(tokens)

    durable._r.hset(durable.run_key(first_run), "receipt_ledger_json", "{")  # noqa: SLF001
    # The original claim is still the authoritative running lease.
    tokens = _activate(durable, first_claim)
    try:
        with pytest.raises(QueryPlanError, match="receipt_invalid"):
            prepare_intent_execution(
                _resolution(),
                original_message="有没有上海餐饮合伙人",
                run_id=first_run,
                request_fingerprint=first_fp,
            )
    finally:
        clear_active_fence(tokens)


def test_durable_failure_commits_audit_only(durable: ExecutionStore) -> None:
    run_id = "intent-run-audit"
    fingerprint = _fp(run_id)
    claim = _seed_claim(durable, run_id, fingerprint)
    tokens = _activate(durable, claim)
    try:
        execution = prepare_intent_execution(
            _resolution(),
            original_message="有没有上海餐饮合伙人",
            run_id=run_id,
            request_fingerprint=fingerprint,
        )
        reservation = execution.reserve(
            resource="users", parameters={"page": 1}, tool_call_id="timed-out"
        )
        audit = execution.commit_failure(reservation, timed_out=True)
        assert audit.status == "timed_out"
        assert not execution.required_steps_satisfied()
        ledger = durable.get_receipt_ledger(run_id)
        assert ledger["receipt_refs"] == []
        assert ledger["audit_refs"] == [
            {
                "plan_step_id": "users-primary",
                "request_index": 1,
                "status": "timed_out",
            }
        ]
        assert ledger["budget"]["items_committed"] == 0
        assert ledger["budget"]["tool_elapsed_reserved_ms"] == 0
    finally:
        clear_active_fence(tokens)


def test_durable_budget_counts_unique_items_across_pages(
    durable: ExecutionStore,
) -> None:
    run_id = "intent-run-deduplicate"
    fingerprint = _fp(run_id)
    claim = _seed_claim(durable, run_id, fingerprint)
    tokens = _activate(durable, claim)
    try:
        execution = prepare_intent_execution(
            _resolution(),
            original_message="有没有上海餐饮合伙人",
            run_id=run_id,
            request_fingerprint=fingerprint,
        )
        first = execution.reserve(
            resource="users", parameters={"page": 1}, tool_call_id="page-1"
        )
        execution.commit_result(first, _content())
        second = execution.reserve(
            resource="users", parameters={"page": 2}, tool_call_id="page-2"
        )
        second_receipt = execution.commit_result(
            second,
            '{"data":[{"id":"user-1","name":"公开用户"},'
            '{"id":"user-2","name":"第二位用户"}],'
            '"_source_url":"https://www.aihehuo.com/ai/users.json?page=2"}',
        )
        assert second_receipt.result_count == 2
        assert execution.unique_result_count() == 2
        ledger = durable.get_receipt_ledger(run_id)
        assert ledger["budget"]["items_committed"] == 2
    finally:
        clear_active_fence(tokens)
