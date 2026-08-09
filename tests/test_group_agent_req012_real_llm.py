"""REQ-012-FIX2: Group Agent 独立 Real-LLM + L1 Mock Scenario 验收（FIX2 版）。

⚠️ 仅当 GROUP_AGENT_REAL_LLM_TEST=1 时运行；消耗真实 LLM 额度。
⚠️ 禁部署 / 禁 push / 不启动 Docker / Micro / New API / callback。

FIX2 改进（在 FIX 基础上）：
  1. Budget guard uses raise_error=True so the REAL CallbackManager propagates
     the 13th-invocation block (proven no-network in the helper suite).
  2. Explicit OutcomeKind enum: provider/network vs internal vs budget vs
     isolation vs unknown-timeout (no single BLOCKED_EXTERNAL bucket).
  3. Runner emits structured PASSED / FAILED / BLOCKED_EXTERNAL.
  4. Round 1 AND Round 2 each independently assert a save_group_profile call.
  5. Guard/classification helpers live in tests.support and are covered by the
     DEFAULT no-network suite (test_group_agent_req012_helpers.py).
  6. Self-contained SIGALRM watchdog is the reliable 240s kill (no external
     pytest-timeout dependency required).
  7. Failure messages carry status/field-name/type only — no response body.

Usage:
    GROUP_AGENT_REAL_LLM_TEST=1 \
      GROUP_AGENT_PROVIDER=qwen \
      GROUP_AGENT_MODEL=... \
      GROUP_AGENT_BASE_URL=... \
      DASHSCOPE_API_KEY=... \
      pytest tests/test_group_agent_req012_real_llm.py -v -m real_llm
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apps.group_agent_api.agent_factory.model_builder import create_model
from apps.group_agent_api.agent_factory.profile_store import assert_profile_persisted
from apps.group_agent_api.agent_factory.disclosure import public_match_basis
from apps.group_agent_api.fixtures.human_audit import (
    HumanAuditCollector,
    audit_enabled,
)
from tests.support.req012_llm_budget import (
    MAX_LLM_INVOCATIONS,
    GroupAgentIsolationError,
    LLMBudgetRecorder,
    OutcomeKind,
    classify_exception,
    classify_http_response,
)

# =====================================================================
# 0. Opt-in gate
# =====================================================================

REAL_LLM_TEST = os.environ.get("GROUP_AGENT_REAL_LLM_TEST", "").strip() in {"1", "true", "yes"}


def require_real_llm_optin() -> None:
    if not REAL_LLM_TEST:
        pytest.skip("REQ-012: set GROUP_AGENT_REAL_LLM_TEST=1 to run real-LLM acceptance")


def _write_outcome(kind: OutcomeKind, detail: str = "") -> None:
    """Write a single structured outcome line to the runner's outcome file.

    The runner reads ONLY this file for its verdict — it never parses the full
    pytest output (FIX2 §3). The line contains only the OutcomeKind label plus a
    status/type diagnostic; no response body, prompt, or secret.
    """
    path = os.environ.get("GROUP_AGENT_REQ012_OUTCOME_FILE")
    if not path:
        return
    line = kind.value if not detail else f"{kind.value} {detail}"
    try:
        Path(path).write_text(line + "\n", encoding="utf-8")
    except OSError:
        pass


def _write_audit_metadata(metadata: dict[str, Any]) -> None:
    """Atomically hand safe report path/size/hash metadata to the shell runner."""
    raw_path = os.environ.get("GROUP_AGENT_REQ013_REPORT_META_FILE", "").strip()
    if not raw_path:
        return
    target = Path(raw_path)
    fd, temp_name = tempfile.mkstemp(
        prefix=".req013-meta-",
        suffix=".tmp",
        dir=target.parent,
    )
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        os.chmod(target, 0o600)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


# The recorder / budget guard / classification live in tests.support so the
# no-network helper tests (test_group_agent_req012_helpers.py) can exercise
# them in the DEFAULT gate — they must NOT skip with the real scenario.


# =====================================================================
# 2. HTTP response outcome check (FIX2 §2, §7 — enum classification)
# =====================================================================


def _check_http_response(response: Any, label: str) -> None:
    """Assert 200; otherwise fail with an explicit OutcomeKind label.

    Provider/network → BLOCKED_EXTERNAL:PROVIDER_NETWORK; our own budget or
    isolation guard → the corresponding FAILED:* label; anything else →
    FAILED:INTERNAL. Diagnostic carries status + error type only (no body).
    """
    kind, diag = classify_http_response(response)
    if kind is OutcomeKind.PASSED:
        return
    _write_outcome(kind, f"{label} {diag}")
    pytest.fail(f"{kind.value}: {label} {diag}")


# =====================================================================
# 3. Isolation Guards — HTTP clients must NOT be called
# =====================================================================


def _install_http_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace HTTP client functions with call-fail guards."""

    def guard(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise GroupAgentIsolationError(
            "REQ-012 isolation guard: HTTP client invoked"
        )

    async def async_guard(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        guard()

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.fetch_membership",
        guard,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_client.fetch_group_agent_match",
        guard,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.callback_client.send_callback_event",
        async_guard,
    )


# =====================================================================
# 4. Agent factory patching — inject instrumented model
# =====================================================================


def _install_instrumented_model(monkeypatch: pytest.MonkeyPatch, recorder: LLMBudgetRecorder) -> None:
    """Create a real model, instrument it, and inject into the agent factory."""
    import importlib

    instrumented_model = create_model()
    recorder.attach(instrumented_model)

    startup_mod = importlib.import_module("apps.group_agent_api.app.startup")
    original_create_agent = startup_mod.create_agent  # type: ignore[attr-defined]

    def patched_create_agent(
        *,
        base_dir: Path | None = None,
        model: Any | None = None,
        checkpointer: Any | None = None,
    ) -> tuple[Any, Any, Path]:
        return original_create_agent(
            base_dir=base_dir,
            model=instrumented_model,
            checkpointer=checkpointer,
        )

    monkeypatch.setattr(startup_mod, "create_agent", patched_create_agent)


# =====================================================================
# 5. Scenario constants
# =====================================================================

SCENARIO_USER_ID = "u105"
SCENARIO_GROUP_ID = "group_l1_alpha"
SCENARIO_CONVERSATION_ID = "conv_req012"

ROUND_1_MESSAGE = (
    "大家好，我正在做 AI / LLM Agent 相关的创业项目，"
    "现在特别需要一位精通 Python、LangChain、PyTorch、LLM Agent 开发的技术负责人，"
    "我这边可以提供业务拓展和客户资源方面的支持。"
)

ROUND_2_MESSAGE = (
    "补充一下具体的技术方向：我们主要用 Python + LangChain + FastAPI 搭建 LLM Agent 系统，"
    "也会用到 PyTorch，需要能独立负责后端架构设计、全职投入的 Python 工程师，"
    "合作方式可以谈，希望能快速启动。"
)

ROUND_3_MESSAGE = (
    "我再明确下需求：找懂 Python、LangChain、PyTorch、LLM Agent 的技术负责人。"
    "群里有没有合适的人选？麻烦帮我从本群推荐一下，"
    "如果有合适的，我愿意直接 @ 他本人认识一下。"
)


# =====================================================================
# 6. Test
# =====================================================================

@pytest.mark.real_llm
class TestReq012RealLLMThreeRoundScenario:
    """REQ-012-FIX2: 3-round real-LLM + L1 fixture acceptance scenario."""

    def test_three_rounds(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Execute the full 3-round real-LLM scenario with all FIX improvements."""
        require_real_llm_optin()

        # ---- Setup: env + isolation guards + instrumented model ----
        monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
        monkeypatch.setenv("GROUP_AGENT_ENV", "test")
        monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
        monkeypatch.setenv("GROUP_AGENT_PROVIDER", "qwen")
        monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")
        monkeypatch.setenv("GROUP_AGENT_MAX_TOKENS", "800")
        monkeypatch.setenv("GROUP_AGENT_TIMEOUT_S", "60")
        monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
        monkeypatch.setenv("GROUP_AGENT_LLM_POLISH", "0")
        monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test_32byte_secret_for_req012!!")

        _install_http_guards(monkeypatch)
        recorder = LLMBudgetRecorder()
        _install_instrumented_model(monkeypatch, recorder)
        audit = HumanAuditCollector(
            enabled=audit_enabled(),
            run_id=f"req012_{int(time.time())}_{os.getpid()}",
            provider=os.environ.get("GROUP_AGENT_PROVIDER", ""),
            model=os.environ.get("GROUP_AGENT_MODEL", ""),
            base_url_configured=bool(os.environ.get("GROUP_AGENT_BASE_URL", "")),
            fixture_level="L1",
            group_id=SCENARIO_GROUP_ID,
            caller_id=SCENARIO_USER_ID,
        )

        # ---- Hard 240s wall-clock watchdog (FIX2 §6) ----
        # Self-contained SIGALRM kill — no external pytest-timeout plugin.
        # On fire, raises TimeoutError → classified as BLOCKED_EXTERNAL:UNKNOWN_TIMEOUT.
        old_handler = signal.signal(signal.SIGALRM, lambda s, f: _raise_timeout())
        signal.alarm(240)

        try:
            from apps.group_agent_api.app import app

            with TestClient(app) as client:
                # =====================================================
                # Round 1: Establish profile
                # =====================================================
                snap_before = recorder.snapshot()
                r1_start = time.time()
                res1 = client.post(
                    "/chat",
                    json={
                        "user_id": SCENARIO_USER_ID,
                        "group_id": SCENARIO_GROUP_ID,
                        "conversation_id": SCENARIO_CONVERSATION_ID,
                        "message": ROUND_1_MESSAGE,
                        "membership": "in_group",
                        "run_match": False,
                        "run_invite": False,
                    },
                )
                r1_latency = time.time() - r1_start
                snap_after = recorder.snapshot()
                _check_http_response(res1, "R1")
                d1 = res1.json()

                # Business oracles: profile persisted
                assert d1.get("profile_persisted") is True, "R1 profile_persisted"
                assert d1.get("assert_attempts", 0) <= 2, f"R1 assert_attempts={d1.get('assert_attempts')}"

                # Profile snapshot (FIX §3)
                r1_assert = assert_profile_persisted(tmp_path, SCENARIO_USER_ID, SCENARIO_GROUP_ID)
                assert r1_assert.ok, f"R1 profile assertion: {r1_assert.reason}"
                assert r1_assert.profile is not None, "R1 profile None"
                r1_snapshot = r1_assert.profile.to_storage_dict()
                for dim in ("doing", "need", "offer"):
                    assert r1_snapshot.get(dim, {}).get("value", "").strip(), f"R1 {dim} empty"

                assert r1_latency < 120, f"R1 latency={r1_latency:.1f}s"
                r1_delta = LLMBudgetRecorder.delta(snap_before, snap_after)
                # FIX2 §4: Round 1 must itself execute save_group_profile ≥1.
                assert r1_delta["tool_calls"].get("save_group_profile", 0) >= 1, (
                    "R1 must execute save_group_profile at least once "
                    f"(delta tool_calls={r1_delta['tool_calls']})"
                )
                if audit.enabled:
                    audit.capture_round(
                        number=1,
                        user_input=ROUND_1_MESSAGE,
                        reply=str(d1.get("reply", "") or ""),
                        llm_delta=r1_delta,
                        latency_s=r1_latency,
                        profile_before=None,
                        profile_after=r1_snapshot,
                    )

                # =====================================================
                # Round 2: Profile evolution
                # =====================================================
                snap_before = recorder.snapshot()
                r2_start = time.time()
                res2 = client.post(
                    "/chat",
                    json={
                        "user_id": SCENARIO_USER_ID,
                        "group_id": SCENARIO_GROUP_ID,
                        "conversation_id": SCENARIO_CONVERSATION_ID,
                        "message": ROUND_2_MESSAGE,
                        "membership": "in_group",
                        "run_match": False,
                        "run_invite": False,
                    },
                )
                r2_latency = time.time() - r2_start
                snap_after = recorder.snapshot()
                _check_http_response(res2, "R2")
                d2 = res2.json()

                # Business oracles: profile persisted
                assert d2.get("profile_persisted") is True, "R2 profile_persisted"

                # Compare R2 snapshot with R1 (FIX §3)
                r2_assert = assert_profile_persisted(tmp_path, SCENARIO_USER_ID, SCENARIO_GROUP_ID)
                assert r2_assert.ok, f"R2 profile assertion: {r2_assert.reason}"
                assert r2_assert.profile is not None, "R2 profile None"
                r2_storage = r2_assert.profile.to_storage_dict()
                assert r2_storage["updated_at"] != r1_snapshot["updated_at"], (
                    "R2 must produce a real profile update: updated_at unchanged"
                )
                for dim in ("doing", "need", "offer"):
                    assert r2_storage.get(dim, {}).get("value", "").strip(), f"R2 {dim} empty"

                assert r2_latency < 120, f"R2 latency={r2_latency:.1f}s"
                r2_delta = LLMBudgetRecorder.delta(snap_before, snap_after)
                # FIX2 §4: Round 2 must itself execute save_group_profile ≥1
                # (independent of Round 1's call).
                assert r2_delta["tool_calls"].get("save_group_profile", 0) >= 1, (
                    "R2 must execute save_group_profile at least once "
                    f"(delta tool_calls={r2_delta['tool_calls']})"
                )
                if audit.enabled:
                    audit.capture_round(
                        number=2,
                        user_input=ROUND_2_MESSAGE,
                        reply=str(d2.get("reply", "") or ""),
                        llm_delta=r2_delta,
                        latency_s=r2_latency,
                        profile_before=r1_snapshot,
                        profile_after=r2_storage,
                    )

                # =====================================================
                # Round 3: Match + invite
                # =====================================================
                snap_before = recorder.snapshot()
                r3_start = time.time()
                res3 = client.post(
                    "/chat",
                    json={
                        "user_id": SCENARIO_USER_ID,
                        "group_id": SCENARIO_GROUP_ID,
                        "conversation_id": SCENARIO_CONVERSATION_ID,
                        "message": ROUND_3_MESSAGE,
                        "membership": "in_group",
                        "run_match": True,
                        "run_invite": True,
                        "willing_to_at": True,
                    },
                )
                r3_latency = time.time() - r3_start
                snap_after = recorder.snapshot()
                _check_http_response(res3, "R3")
                d3 = res3.json()
                assert r3_latency < 120, f"R3 latency={r3_latency:.1f}s"
                r3_delta = LLMBudgetRecorder.delta(snap_before, snap_after)

                # ------- Business oracles (FIX §7: field names only) -------

                # Match (REQ-012 §7.2)
                candidates = d3.get("candidates", [])
                cand_ids = [c.get("user_id") for c in candidates]
                assert len(cand_ids) <= 3, f"candidates<=3 got {len(cand_ids)}"
                assert "u101" in cand_ids, "u101 must be in candidates"
                for fb in ("u201", "u202"):
                    assert fb not in cand_ids, f"forbidden {fb} in candidates"
                assert SCENARIO_USER_ID not in cand_ids, "caller must not self-match"
                assert d3.get("guard_blocked") is not True, "guard_blocked"

                # Per-candidate group-ID strong oracle (FIX3 §3): every
                # candidate MUST carry a group identity equal to the trusted
                # scenario group — this catches any wrong/foreign group id, not
                # just the two known forbidden users.
                for c in candidates:
                    cgid = c.get("source_group_id") or c.get("group_id")
                    assert cgid is not None, (
                        f"candidate {c.get('user_id')} missing source_group_id/group_id"
                    )
                    assert cgid == SCENARIO_GROUP_ID, (
                        f"candidate {c.get('user_id')} group={cgid} != {SCENARIO_GROUP_ID}"
                    )
                    assert public_match_basis(c), (
                        f"candidate {c.get('user_id')} missing public match basis"
                    )

                # Invite (REQ-012 §7.3)
                assert d3.get("invite_ok") is True, "invite_ok must be True"
                mentioned = d3.get("mentioned_user_ids", [])
                for uid in mentioned:
                    assert uid in cand_ids, f"mentioned {uid} not in candidates"
                assert "u201" not in mentioned, "forbidden mention u201"
                assert "u202" not in mentioned, "forbidden mention u202"
                if d3.get("delivery_kind") == "directed":
                    assert "u101" in mentioned, "directed must mention u101"

                # Sensitive field leak (field names only, FIX §7)
                _raw = json.dumps(d3)
                for sf in ("phone", "wechat", "email", "private_notes"):
                    assert f'"{sf}": "' not in _raw, f"sensitive {sf} leaked"

                # ------- Budget checks -------
                total_time = r1_latency + r2_latency + r3_latency
                assert recorder.llm_starts <= MAX_LLM_INVOCATIONS, (
                    f"budget: llm_starts={recorder.llm_starts} > {MAX_LLM_INVOCATIONS}"
                )
                assert total_time < 240, f"budget: total_time={total_time:.1f}s > 240s"

                # ------- Evidence summary (FIX §4) -------
                save_calls = recorder.tool_call_counts.get("save_group_profile", 0)
                # Aggregate: at least R1 + R2 each did one save (§4).
                assert save_calls >= 2, (
                    f"save_group_profile total={save_calls} < 2 (R1 and R2 each must save)"
                )
                evidence = {
                    "model_class": recorder.model_class,
                    "provider": os.environ.get("GROUP_AGENT_PROVIDER", ""),
                    "model_name": os.environ.get("GROUP_AGENT_MODEL", ""),
                    "base_url_configured": bool(os.environ.get("GROUP_AGENT_BASE_URL", "")),
                    "total_llm_starts": recorder.llm_starts,
                    "total_llm_ends": recorder.llm_ends,
                    "total_tool_calls": dict(recorder.tool_call_counts),
                    "total_tokens": recorder.total_tokens,
                    "save_group_profile_calls": save_calls,
                    "round1_save_group_profile": r1_delta["tool_calls"].get("save_group_profile", 0),
                    "round2_save_group_profile": r2_delta["tool_calls"].get("save_group_profile", 0),
                    "total_time_s": round(total_time, 2),
                    "outcome": OutcomeKind.PASSED.value,
                    "budget_llm_starts": f"{recorder.llm_starts}<={MAX_LLM_INVOCATIONS}",
                    "budget_time_s": f"{total_time:.1f}<240",
                    "round1": {
                        "latency_s": round(r1_latency, 2),
                        "llm_starts": r1_delta["llm_starts"],
                        "llm_ends": r1_delta["llm_ends"],
                        "tool_calls": r1_delta["tool_calls"],
                        "tokens": r1_delta["tokens"],
                        "profile_persisted": d1.get("profile_persisted"),
                        "assert_attempts": d1.get("assert_attempts"),
                    },
                    "round2": {
                        "latency_s": round(r2_latency, 2),
                        "llm_starts": r2_delta["llm_starts"],
                        "llm_ends": r2_delta["llm_ends"],
                        "tool_calls": r2_delta["tool_calls"],
                        "tokens": r2_delta["tokens"],
                        "profile_persisted": d2.get("profile_persisted"),
                        "profile_updated_at_changed": (
                            r2_storage["updated_at"] != r1_snapshot["updated_at"]
                        ),
                    },
                    "round3": {
                        "latency_s": round(r3_latency, 2),
                        "llm_starts": r3_delta["llm_starts"],
                        "llm_ends": r3_delta["llm_ends"],
                        "tool_calls": r3_delta["tool_calls"],
                        "tokens": r3_delta["tokens"],
                        "match_status": d3.get("match_status"),
                        "delivery_kind": d3.get("delivery_kind"),
                        "candidates": cand_ids,
                        "mentioned": mentioned,
                        "invite_ok": d3.get("invite_ok"),
                        "invite_violations": d3.get("invite_violations", []),
                        "guard_blocked": d3.get("guard_blocked"),
                    },
                }

                if audit.enabled:
                    audit.capture_round(
                        number=3,
                        user_input=ROUND_3_MESSAGE,
                        reply=str(d3.get("reply", "") or ""),
                        llm_delta=r3_delta,
                        latency_s=r3_latency,
                        profile_before=r2_storage,
                        profile_after=r2_storage,
                        candidates=candidates,
                        invite_text=str(d3.get("invite_text", "") or ""),
                        mentioned_user_ids=mentioned,
                        invite_ok=d3.get("invite_ok"),
                        guard_blocked=d3.get("guard_blocked"),
                    )
                    audit_report = audit.build_report(
                        total_llm_invocations=recorder.llm_starts,
                        total_tokens=recorder.total_tokens,
                        total_time_s=total_time,
                        machine_oracles={
                            "profile_persisted_r1": d1.get("profile_persisted") is True,
                            "profile_persisted_r2": d2.get("profile_persisted") is True,
                            "profile_updated": (
                                r2_storage["updated_at"] != r1_snapshot["updated_at"]
                            ),
                            "candidate_count_lte_3": len(candidates) <= 3,
                            "all_candidates_have_public_basis": all(
                                bool(public_match_basis(candidate))
                                for candidate in candidates
                            ),
                            "all_mentioned_have_public_basis": all(
                                bool(
                                    public_match_basis(
                                        next(
                                            candidate
                                            for candidate in candidates
                                            if candidate.get("user_id") == user_id
                                        )
                                    )
                                )
                                for user_id in mentioned
                            ),
                            "current_group_only": all(
                                (c.get("source_group_id") or c.get("group_id"))
                                == SCENARIO_GROUP_ID
                                for c in candidates
                            ),
                            "caller_not_self_matched": SCENARIO_USER_ID not in cand_ids,
                            "u101_present": "u101" in cand_ids,
                            "known_foreign_candidate_count": sum(
                                uid in {"u201", "u202"} for uid in cand_ids
                            ),
                            "invite_ok": d3.get("invite_ok") is True,
                            "guard_not_blocked": d3.get("guard_blocked") is not True,
                            "sensitive_leak_count": 0,
                        },
                    )
                    assert audit_report is not None
                    audit_result = audit.write_report(audit_report)
                    _write_audit_metadata(audit_result.safe_metadata())

                self.evidence = evidence  # type: ignore[attr-defined]
                _write_outcome(OutcomeKind.PASSED)

                # Safe print: no secrets, no response bodies
                print("\n=== REQ-012-FIX2 Real-LLM Acceptance Evidence ===")
                for key, val in evidence.items():
                    if key in ("round1", "round2", "round3"):
                        print(f"  {key}:")
                        for k, v in val.items():
                            print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {val}")
                print("===============================================\n")

        except TimeoutError:
            _write_outcome(OutcomeKind.TIMEOUT, "overall test timeout 240s")
            pytest.fail(OutcomeKind.TIMEOUT.value + ": overall test timeout 240s")
        except Exception as exc:
            # Classify the raised exception by its EXACT type (walking causes),
            # never by message prose (FIX3 §1).
            kind = classify_exception(exc)
            if kind is OutcomeKind.INTERNAL:
                _write_outcome(OutcomeKind.INTERNAL, type(exc).__name__)
                raise
            _write_outcome(kind, type(exc).__name__)
            pytest.fail(f"{kind.value}: {type(exc).__name__}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _raise_timeout() -> None:
    raise TimeoutError("BLOCKED_EXTERNAL: overall test timeout 240s")
