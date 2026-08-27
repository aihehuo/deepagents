"""BSD-01 P1/P2: mouth ingress reject → bb.brain.repair seam + helpers."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from apps.group_agent_api.agent_factory.brain_repair import (
    MOUTH_INGRESS_MAX_ATTEMPTS,
    MouthIngressRejected,
    apply_mouth_repair,
    decide_mouth_repair_action,
    emit_final_with_mouth_repair,
    format_ingress_deny,
    parse_mouth_reject_body,
    peel_final_payload,
    prepare_repaired_final,
)
from apps.group_agent_api.agent_factory.integrations.callback_client import (
    send_callback_event,
)


def test_format_ingress_deny_contains_reason_and_hint() -> None:
    block = format_ingress_deny(
        reason_code="unverified_fact_source",
        message="no profile",
        hint="drop candidate",
        repairable_by="orchestrator",
        reject_class="truth",
        fields=["candidates[0]"],
        attempt=2,
    )
    assert "unverified_fact_source" in block
    assert "micro_ingress_reject" in block
    assert "drop candidate" in block
    assert 'attempt="2"' in block


def test_peel_clears_candidates_for_unverified_fact() -> None:
    payload = {
        "reply": "找到了张三",
        "reply_mode": "recommendation",
        "candidates": [{"user_id": "1", "display_name": "张三"}],
        "match_status": "matched",
        "match": {"status": "matched", "candidates": [{"user_id": "1"}]},
    }
    out = peel_final_payload(payload, reason_code="unverified_fact_source")
    assert out["candidates"] == []
    assert out["match_status"] == "empty"
    assert out["reply_mode"] == "no_match"
    assert out["match"]["status"] == "empty"


def test_apply_mouth_repair_orchestrator_without_model() -> None:
    payload = {
        "reply": "dialogue with people",
        "reply_mode": "dialogue",
        "candidates": [{"user_id": "9"}],
        "match_status": "matched",
    }
    reject = MouthIngressRejected(
        "dialogue_mode_has_candidates",
        repairable_by="orchestrator",
        reject_class="consistency",
        message="dialogue must not carry candidates",
    )
    out = apply_mouth_repair(payload, reject=reject, model=None, attempt=2)
    assert out["candidates"] == []
    assert out["reply_mode"] == "dialogue"


def test_parse_mouth_reject_body_nested() -> None:
    body = {
        "disposition": "rejected",
        "error": "fact_value_mismatch",
        "reason_code": "fact_value_mismatch",
        "reject": {
            "class": "truth",
            "reason_code": "fact_value_mismatch",
            "repairable_by": "orchestrator",
            "fields": ["candidates[0].facts.doing"],
            "message": "doing mismatch",
            "hint": "drop fact",
        },
    }
    exc = parse_mouth_reject_body(body)
    assert exc is not None
    assert exc.reason_code == "fact_value_mismatch"
    assert exc.repairable_by == "orchestrator"
    assert exc.fields == ["candidates[0].facts.doing"]
    assert "mismatch" in exc.message


def test_decide_mouth_repair_action_abandon_when_none_or_exhausted() -> None:
    reject_none = MouthIngressRejected(
        "protocol_mode_mismatch",
        repairable_by="none",
    )
    assert (
        decide_mouth_repair_action(reject=reject_none, attempt=1) == "abandon"
    )
    reject_ok = MouthIngressRejected(
        "unverified_fact_source",
        repairable_by="orchestrator",
    )
    assert decide_mouth_repair_action(reject=reject_ok, attempt=1) == "repair"
    assert (
        decide_mouth_repair_action(
            reject=reject_ok,
            attempt=MOUTH_INGRESS_MAX_ATTEMPTS,
        )
        == "abandon"
    )


def test_prepare_repaired_final_abandons_empty_recommendation() -> None:
    # Peel does not rewrite reply_mode for unknown codes; empty recommendation
    # must still fall back to abandon dialogue (orchestrator guard).
    payload = {
        "reply": "推荐张三",
        "reply_mode": "recommendation",
        "candidates": [],
        "match_status": "matched",
        "protocol_version": "ga-grounding-v1",
        "run_id": "r1",
    }
    reject = MouthIngressRejected(
        "some_unknown_mouth_code",
        repairable_by="orchestrator",
    )
    out = prepare_repaired_final(payload, reject=reject, model=None, attempt=2)
    assert out["reply_mode"] == "dialogue"
    assert out["candidates"] == []
    assert "未经确认" in str(out.get("reply") or "")


@pytest.mark.asyncio
async def test_send_callback_event_raises_mouth_reject_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    class _Resp:
        status_code = 422

        def json(self) -> dict[str, Any]:
            return {
                "disposition": "rejected",
                "reason_code": "unverified_fact_source",
                "error": "unverified_fact_source",
                "reject": {
                    "class": "truth",
                    "reason_code": "unverified_fact_source",
                    "repairable_by": "orchestrator",
                    "message": "no profile",
                    "hint": "drop",
                    "fields": [],
                },
            }

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def post(self, *a: Any, **k: Any) -> _Resp:
            calls["n"] += 1
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setenv("GROUP_AGENT_CALLBACK_HMAC_SECRET", "test-secret-for-hmac")
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS",
        "http://127.0.0.1:3001",
    )

    with pytest.raises(MouthIngressRejected) as ei:
        await send_callback_event(
            callback_url="http://127.0.0.1:3001/group_agent_callbacks/r1",
            envelope_dict={
                "version": "GA-CALLBACK-V1",
                "run_id": "r1",
                "seq": 1,
                "event": "final",
                "payload": {},
            },
            secret="test-secret-for-hmac",
            max_retries=3,
        )
    assert ei.value.reason_code == "unverified_fact_source"
    assert calls["n"] == 1  # no transport-style retry of same reject


@pytest.mark.asyncio
async def test_emit_final_with_mouth_repair_same_seq_then_accept() -> None:
    """Seam loop: reject once → peel → accept on same logical seq."""
    payloads_seen: list[dict[str, Any]] = []
    seq_box = {"seq": 0}

    async def emit(event: str, payload: dict[str, Any]) -> bool:
        assert event == "final"
        seq_box["seq"] += 1
        used_seq = seq_box["seq"]
        payloads_seen.append(dict(payload))
        if len(payloads_seen) == 1:
            seq_box["seq"] -= 1
            raise MouthIngressRejected(
                "dialogue_mode_has_candidates",
                repairable_by="orchestrator",
            )
        assert used_seq == 1  # same seq after rewind
        assert payload.get("candidates") == []
        return True

    current = {
        "reply": "hi",
        "reply_mode": "dialogue",
        "candidates": [{"user_id": "1"}],
        "match_status": "matched",
    }
    ok = await emit_final_with_mouth_repair(
        emit_callback=emit,
        final_payload=current,
        model=None,
        run_id="r-test",
    )
    assert ok is True
    assert len(payloads_seen) == 2


@pytest.mark.asyncio
async def test_emit_final_with_mouth_repair_abandons_repairable_none() -> None:
    async def emit(event: str, payload: dict[str, Any]) -> bool:
        raise MouthIngressRejected(
            "internal_error",
            repairable_by="none",
        )

    with pytest.raises(RuntimeError, match="mouth_ingress_rejected:internal_error"):
        await emit_final_with_mouth_repair(
            emit_callback=emit,
            final_payload={"reply": "x", "reply_mode": "dialogue"},
            model=None,
        )
