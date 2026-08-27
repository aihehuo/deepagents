"""BSD-01 P1: mouth ingress reject → brain repair helpers + client parse."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from apps.group_agent_api.agent_factory.ingress_repair import (
    apply_mouth_repair,
    format_ingress_deny,
    peel_final_payload,
)
from apps.group_agent_api.agent_factory.integrations.callback_client import (
    MouthIngressRejected,
    parse_mouth_reject_body,
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
async def test_final_emit_repair_loop_same_seq_then_accept() -> None:
    """Orchestrator-shaped loop: reject once → peel → accept on same logical attempt."""
    from apps.group_agent_api.agent_factory.ingress_repair import (
        MOUTH_INGRESS_MAX_ATTEMPTS,
        apply_mouth_repair,
    )

    payloads_seen: list[dict[str, Any]] = []
    seq_box = {"seq": 0}

    async def emit(event: str, payload: dict[str, Any]) -> bool:
        assert event == "final"
        seq_box["seq"] += 1
        used_seq = seq_box["seq"]
        payloads_seen.append(dict(payload))
        if len(payloads_seen) == 1:
            # Simulate mouth reject + rewind
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
    mouth_attempt = 1
    while True:
        try:
            ok = await emit("final", current)
        except MouthIngressRejected as exc:
            assert mouth_attempt < MOUTH_INGRESS_MAX_ATTEMPTS
            mouth_attempt += 1
            current = apply_mouth_repair(current, reject=exc, model=None, attempt=mouth_attempt)
            continue
        assert ok is True
        break
    assert len(payloads_seen) == 2
    assert mouth_attempt == 2
