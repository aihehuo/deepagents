"""Brain-as-SUT: production brain + contract-shaped fakes (ear / hand / mouth).

Layers
------
1. Always-on structural tests (stub model): wiring + MatchResult / ack shapes.
2. Opt-in ``@pytest.mark.real_llm``: same harness, live ChatOpenAI, full
   multi-turn conversation (write profile → search → extra user turn).

Neighbors never leave process. LLM is the only real network when opt-in.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI

from apps.group_agent_api.agent_factory.profile_store import assert_profile_persisted
from tests.support.brain_sut import install_brain_sut, install_instrumented_real_model
from tests.support.brain_sut.contracts import (
    HandSearchScript,
    empty_result,
    matched_result,
    production_shaped_candidate,
)
from tests.support.brain_sut.fakes import FakeHand
from tests.support.req012_llm_budget import LLMBudgetRecorder

REAL_LLM = os.environ.get("GROUP_AGENT_REAL_LLM_TEST", "").strip() in {"1", "true", "yes"}

GROUP_ID = "group_l1_alpha"
USER_ID = "u105"
CONV_ID = "conv_brain_sut_full"

ROUND_1_PROFILE = (
    "大家好，我正在做 AI / LLM Agent 相关的创业项目，"
    "现在特别需要一位精通 Python、LangChain、PyTorch、LLM Agent 开发的技术合伙人，"
    "我这边可以提供业务拓展和客户资源方面的支持。"
)
ROUND_2_MATCH = (
    "需求已经比较清楚了：找懂 Python、LangChain 的技术合伙人。"
    "请帮我在本群搜索并推荐合适的人；如果有合适的，我愿意直接 @ 认识一下。"
)
ROUND_3_FOLLOWUP = (
    "刚才推荐的那位我挺感兴趣。请再跟我聊聊为什么适合，"
    "并给我一句可以在群里开口的话，帮我把这次交流继续下去。"
)


def _base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, model_mode: str) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", model_mode)
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv(
        "GROUP_AGENT_PRINCIPAL_HMAC_SECRET",
        "test_32byte_secret_for_brain_sut!!",
    )
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_HMAC_SECRET",
        "test_32byte_callback_secret_sut!!",
    )
    # Stub model needs a fixture level for tool-call profile dims; L1 is fine.
    if model_mode == "stub":
        monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")


# =====================================================================
# 1. Contract smoke — fakes speak production types (no LLM)
# =====================================================================


def test_fake_hand_good_and_bad_return_production_types() -> None:
    hand = FakeHand()
    hand.enqueue_matched(group_id=GROUP_ID, query="need python")
    good = hand.run_match(query="need python", group_id=GROUP_ID)
    assert good.status == "matched"
    assert len(good.candidates) == 1
    assert good.candidates[0]["user_id"] == "cand_poc_1"
    assert good.candidates[0]["doing"]["disclosure"] == "confirmed_public"

    hand.enqueue_rejected(message="v2_not_configured", status_code=422)
    bad = hand.run_match(query="need python", group_id=GROUP_ID)
    assert bad.status == "rejected"
    assert bad.candidates == []
    assert "http_error:" in bad.reason
    assert "v2_not_configured" in bad.reason


# =====================================================================
# 2. Structural PoC — production /chat + stub model + FakeHand scripts
# =====================================================================


@pytest.mark.l1
@pytest.mark.brain_sut
def test_brain_sut_good_match_with_stub_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Good scenario: hand returns matched → brain response carries candidate."""
    _base_env(monkeypatch, tmp_path, model_mode="stub")
    harness = install_brain_sut(monkeypatch)
    harness.hand.enqueue_matched(group_id=GROUP_ID)

    from apps.group_agent_api.app import app

    with TestClient(app) as client:
        res = client.post(
            "/chat",
            json={
                "user_id": USER_ID,
                "group_id": GROUP_ID,
                "conversation_id": "conv_brain_sut_good",
                "message": (
                    "我在做 LLM Agent，需要 Python/LangChain 技术合伙人，"
                    "我可以提供业务资源。请帮我在群里匹配。"
                ),
                "membership": "in_group",
                "run_match": True,
                "run_invite": False,
            },
        )

    assert res.status_code == 200, res.text
    body = res.json()
    assert body.get("match_status") == "matched", body
    cands = body.get("candidates") or []
    assert len(cands) >= 1, body
    assert cands[0].get("user_id") == "cand_poc_1"
    assert len(harness.hand.search_calls) >= 1
    last = harness.hand.search_calls[-1]
    assert last.get("query"), last
    slog = body.get("search_log") or {}
    assert slog.get("query"), slog


@pytest.mark.l1
@pytest.mark.brain_sut
def test_brain_sut_hand_rejected_stays_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BSD-01 P0: hand 422 rejected must not surface as empty/no_match."""
    _base_env(monkeypatch, tmp_path, model_mode="stub")
    harness = install_brain_sut(monkeypatch)
    harness.hand.enqueue_rejected(message="v2_not_configured", status_code=422)

    from apps.group_agent_api.app import app

    with TestClient(app) as client:
        res = client.post(
            "/chat",
            json={
                "user_id": USER_ID,
                "group_id": GROUP_ID,
                "conversation_id": "conv_brain_sut_bad",
                "message": "请帮我匹配一位 Python 技术合伙人。",
                "membership": "in_group",
                "run_match": True,
                "run_invite": False,
            },
        )

    assert res.status_code == 200, res.text
    body = res.json()
    assert body.get("match_status") == "rejected", body
    assert not (body.get("candidates") or []), "rejected must not yield candidates"
    assert len(harness.hand.search_calls) >= 1
    last = harness.hand.search_calls[-1]
    assert last.get("group_id") == GROUP_ID
    assert last.get("query"), last


@pytest.mark.l1
def test_brain_sut_script_queue_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Scripts are consumed in order — required for multi-turn scenarios later."""
    _base_env(monkeypatch, tmp_path, model_mode="stub")
    harness = install_brain_sut(monkeypatch)
    harness.hand.enqueue_search(
        HandSearchScript(
            kind="ok_empty",
            result=empty_result(query="q1", group_id=GROUP_ID, reason="first"),
        )
    )
    harness.hand.enqueue_search(
        HandSearchScript(
            kind="ok_matched",
            result=matched_result(
                query="q2",
                group_id=GROUP_ID,
                candidates=[
                    production_shaped_candidate(
                        user_id="cand_second",
                        group_id=GROUP_ID,
                        display_name="第二人",
                        doing_value="FastAPI",
                    )
                ],
                reason="second",
            ),
        )
    )

    r1 = harness.hand.run_match(query="q1", group_id=GROUP_ID)
    r2 = harness.hand.run_match(query="q2", group_id=GROUP_ID)
    assert r1.status == "empty" and r1.reason == "first"
    assert r2.status == "matched" and r2.candidates[0]["user_id"] == "cand_second"


@pytest.mark.l1
def test_brain_sut_stick_matched_reuses_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _base_env(monkeypatch, tmp_path, model_mode="stub")
    harness = install_brain_sut(monkeypatch)
    harness.hand.stick_matched(group_id=GROUP_ID)
    a = harness.hand.run_match(query="q", group_id=GROUP_ID)
    b = harness.hand.run_match(query="q2", group_id=GROUP_ID)
    assert a.status == b.status == "matched"
    assert a.candidates[0]["user_id"] == b.candidates[0]["user_id"] == "cand_poc_1"


# =====================================================================
# 3. Real LLM — full conversation (profile write → search → extra turn)
# =====================================================================


def _require_real_llm_optin() -> None:
    if not REAL_LLM:
        pytest.skip("set GROUP_AGENT_REAL_LLM_TEST=1 for brain_sut real-LLM conversation")
    for key in ("GROUP_AGENT_MODEL", "GROUP_AGENT_BASE_URL", "DASHSCOPE_API_KEY"):
        if not (os.environ.get(key) or "").strip():
            pytest.skip(f"missing {key} for brain_sut real-LLM conversation")


def _chat_payload(message: str, *, run_match: bool, run_invite: bool) -> dict[str, Any]:
    return {
        "user_id": USER_ID,
        "group_id": GROUP_ID,
        "conversation_id": CONV_ID,
        "message": message,
        "membership": "in_group",
        "run_match": run_match,
        "run_invite": run_invite,
        "willing_to_at": True,
    }


def _assert_no_sensitive_leak(body: dict[str, Any], label: str) -> None:
    raw = json.dumps(body)
    for field in ("phone", "wechat", "email", "private_notes"):
        assert f'"{field}": "' not in raw, f"{label} leaked {field}"


def _raise_timeout() -> None:
    raise TimeoutError("brain_sut real-LLM conversation timed out")


def _search_log_fields(body: dict[str, Any]) -> dict[str, Any]:
    raw = body.get("search_log") or {}
    if not isinstance(raw, dict):
        return {"called": False, "query": "", "rank_query": ""}
    query = str(raw.get("query") or "")
    rank_query = str(raw.get("rank_query") or "")
    return {
        "called": bool(query or raw.get("match_status")),
        "query": query,
        "rank_query": rank_query,
        "match_status": raw.get("match_status"),
        "match_reason": raw.get("match_reason"),
    }


def _round_record(
    *,
    number: int,
    title: str,
    user_message: str,
    body: dict[str, Any],
    latency_s: float,
    delta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "round": number,
        "title": title,
        "latency_s": round(latency_s, 3),
        "tokens": {
            "input": int(delta.get("input_tokens") or 0),
            "output": int(delta.get("output_tokens") or 0),
            "total": int(delta.get("tokens") or 0),
        },
        "llm_starts": delta["llm_starts"],
        "llm_ends": delta["llm_ends"],
        "tool_calls": dict(delta.get("tool_calls") or {}),
        "user_message": user_message,
        "assistant_reply": str(body.get("reply") or ""),
        "invite_text": body.get("invite_text") or None,
        "match_status": body.get("match_status"),
        "search": _search_log_fields(body),
        "llm_search_tool_calls": int((delta.get("tool_calls") or {}).get("search_candidates") or 0),
        "candidates": [
            c.get("user_id") for c in (body.get("candidates") or []) if isinstance(c, dict)
        ],
        "profile_persisted": body.get("profile_persisted"),
    }


def _format_conversation_log(rounds: list[dict[str, Any]], *, totals: dict[str, Any]) -> str:
    lines = [
        "=== brain_sut real-LLM conversation log ===",
        f"model={totals.get('model_class')}  llm_starts={totals.get('llm_starts')}  "
        f"tokens_total={totals.get('tokens_total')}  wall_s={totals.get('total_s')}",
        "",
    ]
    for rec in rounds:
        tok = rec["tokens"]
        lines.append(
            f"--- Round {rec['round']} · {rec['title']} ---"
        )
        lines.append(
            f"latency_s={rec['latency_s']:.3f}  "
            f"tokens_in={tok['input']}  tokens_out={tok['output']}  "
            f"tokens_total={tok['total']}  llm_starts={rec['llm_starts']}"
        )
        if rec["tool_calls"]:
            lines.append(f"tools={rec['tool_calls']}")
        search = rec.get("search") or {}
        if rec.get("llm_search_tool_calls") or search.get("called") or search.get("query"):
            lines.append(
                "SEARCH: llm_tool_calls="
                f"{rec.get('llm_search_tool_calls', 0)}  "
                f"query={search.get('query')!r}  "
                f"rank_query={search.get('rank_query')!r}"
            )
        elif rec.get("title", "").find("search") >= 0:
            lines.append("SEARCH: not called by model this round")
        lines.append(f"USER: {rec['user_message']}")
        lines.append(f"ASSISTANT: {rec['assistant_reply']}")
        if rec.get("invite_text"):
            lines.append(f"INVITE: {rec['invite_text']}")
        extra = []
        if rec.get("match_status"):
            extra.append(f"match_status={rec['match_status']}")
        if rec.get("candidates"):
            extra.append(f"candidates={rec['candidates']}")
        if rec.get("profile_persisted") is not None:
            extra.append(f"profile_persisted={rec['profile_persisted']}")
        if extra:
            lines.append("  " + "  ".join(extra))
        lines.append("")
    lines.append("=== end conversation log ===")
    return "\n".join(lines)


@pytest.mark.real_llm
@pytest.mark.brain_sut
def test_brain_sut_real_llm_full_conversation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Live ChatOpenAI drives a 3-turn user conversation; ear/hand/mouth stay fakes.

    Round 1 · write profile (save_group_profile tool → FakeHand persist)
    Round 2 · search + invite (FakeHand matched candidate)
    Round 3 · extra user turn (follow-up conversation, still real LLM)
    """
    _require_real_llm_optin()

    _base_env(monkeypatch, tmp_path, model_mode="real")
    monkeypatch.setenv(
        "GROUP_AGENT_PROVIDER", os.environ.get("GROUP_AGENT_PROVIDER") or "qwen"
    )
    monkeypatch.setenv("GROUP_AGENT_LLM_POLISH", "0")
    monkeypatch.setenv("GROUP_AGENT_MAX_TOKENS", os.environ.get("GROUP_AGENT_MAX_TOKENS") or "800")
    monkeypatch.setenv("GROUP_AGENT_TIMEOUT_S", os.environ.get("GROUP_AGENT_TIMEOUT_S") or "60")
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")

    recorder = LLMBudgetRecorder()
    live_model = install_instrumented_real_model(monkeypatch, recorder)
    assert isinstance(live_model, ChatOpenAI), type(live_model).__name__
    assert "ChatOpenAI" in (recorder.model_class or "")
    assert "Stub" not in (recorder.model_class or "")

    harness = install_brain_sut(monkeypatch)
    harness.hand.stick_matched(group_id=GROUP_ID)

    old_handler = signal.signal(signal.SIGALRM, lambda _s, _f: _raise_timeout())
    signal.alarm(300)
    try:
        from apps.group_agent_api.app import app

        with TestClient(app) as client:
            # ---- Round 1: establish profile ----
            snap0 = recorder.snapshot()
            t0 = time.time()
            res1 = client.post("/chat", json=_chat_payload(ROUND_1_PROFILE, run_match=False, run_invite=False))
            r1_s = time.time() - t0
            assert res1.status_code == 200, res1.text
            d1 = res1.json()
            r1 = LLMBudgetRecorder.delta(snap0, recorder.snapshot())

            assert r1["llm_starts"] >= 1, "R1 must invoke the live LLM"
            assert r1["tool_calls"].get("save_group_profile", 0) >= 1, (
                f"R1 must call save_group_profile, got {r1['tool_calls']}"
            )
            assert d1.get("profile_persisted") is True, "R1 profile_persisted"
            assert isinstance(d1.get("reply"), str) and d1["reply"].strip(), "R1 empty reply"
            assert len(harness.hand.profile_calls) >= 1, "R1 FakeHand.persist not used"
            r1_assert = assert_profile_persisted(tmp_path, USER_ID, GROUP_ID)
            assert r1_assert.ok and r1_assert.profile is not None, r1_assert.reason
            r1_snap = r1_assert.profile.to_storage_dict()
            for dim in ("doing", "need", "offer"):
                assert str(r1_snap.get(dim, {}).get("value") or "").strip(), f"R1 {dim} empty"
            _assert_no_sensitive_leak(d1, "R1")
            assert r1_s < 120, f"R1 latency={r1_s:.1f}s"

            # ---- Round 2: search + accommodation (invite) ----
            snap1 = recorder.snapshot()
            t1 = time.time()
            res2 = client.post("/chat", json=_chat_payload(ROUND_2_MATCH, run_match=True, run_invite=True))
            r2_s = time.time() - t1
            assert res2.status_code == 200, res2.text
            d2 = res2.json()
            r2 = LLMBudgetRecorder.delta(snap1, recorder.snapshot())

            assert r2["llm_starts"] >= 1, "R2 must invoke the live LLM"
            assert r2["tool_calls"].get("search_candidates", 0) >= 1, (
                f"R2 model must call search_candidates, got {r2['tool_calls']}"
            )
            slog = d2.get("search_log") or {}
            assert slog.get("query"), f"R2 search_log missing query: {slog}"
            assert len(harness.hand.search_calls) >= 1, "R2 FakeHand.search not used"
            last_search = harness.hand.search_calls[-1]
            assert last_search.get("group_id") == GROUP_ID
            assert last_search.get("query"), last_search
            assert d2.get("match_status") == "matched", d2.get("match_status")
            cands = d2.get("candidates") or []
            assert len(cands) >= 1, "R2 expected FakeHand candidate"
            assert cands[0].get("user_id") == "cand_poc_1"
            assert (cands[0].get("source_group_id") or cands[0].get("group_id")) == GROUP_ID
            assert USER_ID not in [c.get("user_id") for c in cands]
            assert d2.get("guard_blocked") is not True
            assert d2.get("invite_ok") is True, "R2 invite_ok"
            assert isinstance(d2.get("reply"), str) and d2["reply"].strip(), "R2 empty reply"
            _assert_no_sensitive_leak(d2, "R2")
            assert r2_s < 120, f"R2 latency={r2_s:.1f}s"

            # ---- Round 3: extra user conversation after the match ----
            snap2 = recorder.snapshot()
            t2 = time.time()
            res3 = client.post(
                "/chat", json=_chat_payload(ROUND_3_FOLLOWUP, run_match=False, run_invite=False)
            )
            r3_s = time.time() - t2
            assert res3.status_code == 200, res3.text
            d3 = res3.json()
            r3 = LLMBudgetRecorder.delta(snap2, recorder.snapshot())

            assert r3["llm_starts"] >= 1, "R3 follow-up must invoke the live LLM"
            assert d3.get("conversation_id") == CONV_ID
            assert isinstance(d3.get("reply"), str) and d3["reply"].strip(), "R3 empty reply"
            assert isinstance(d3.get("suggested_replies"), list)
            _assert_no_sensitive_leak(d3, "R3")
            assert r3_s < 120, f"R3 latency={r3_s:.1f}s"

            total_s = r1_s + r2_s + r3_s
            assert recorder.llm_starts >= 3, (
                f"expected ≥1 live start per round, got {recorder.llm_starts}"
            )
            assert recorder.model_class and "Stub" not in recorder.model_class
            assert "ChatOpenAI" in recorder.model_class

            rounds = [
                _round_record(
                    number=1,
                    title="write profile",
                    user_message=ROUND_1_PROFILE,
                    body=d1,
                    latency_s=r1_s,
                    delta=r1,
                ),
                _round_record(
                    number=2,
                    title="search + invite",
                    user_message=ROUND_2_MATCH,
                    body=d2,
                    latency_s=r2_s,
                    delta=r2,
                ),
                _round_record(
                    number=3,
                    title="extra user turn",
                    user_message=ROUND_3_FOLLOWUP,
                    body=d3,
                    latency_s=r3_s,
                    delta=r3,
                ),
            ]
            totals = {
                "model_class": recorder.model_class,
                "llm_starts": recorder.llm_starts,
                "llm_ends": recorder.llm_ends,
                "tokens_input": recorder.input_tokens,
                "tokens_output": recorder.output_tokens,
                "tokens_total": recorder.total_tokens,
                "tool_calls": dict(recorder.tool_call_counts),
                "profile_writes": len(harness.hand.profile_calls),
                "search_calls": len(harness.hand.search_calls),
                "total_s": round(total_s, 3),
            }
            log_text = _format_conversation_log(rounds, totals=totals)
            log_path = tmp_path / "brain_sut_conversation.log"
            log_path.write_text(log_text + "\n", encoding="utf-8")
            (tmp_path / "brain_sut_conversation.json").write_text(
                json.dumps({"totals": totals, "rounds": rounds}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            print("\n" + log_text + "\n")
            print(f"(also wrote {log_path})")
    except TimeoutError:
        pytest.fail("brain_sut real-LLM conversation timed out after 300s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
