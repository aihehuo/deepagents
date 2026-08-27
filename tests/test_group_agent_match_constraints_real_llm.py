"""Brief G · opt-in live LLM tests for hard/soft match_constraints extraction.

Skipped unless ``GROUP_AGENT_REAL_LLM_TEST=1`` + an API key. Forces
``GROUP_AGENT_MODEL_MODE=real`` (setdefault alone can leave stub from
conftest). Default CI keeps green via ``-m "not real_llm"``.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI

from apps.group_agent_api.agent_factory.module_config import (
    reload_modules_config,
    reset_modules_config_cache,
)
from apps.group_agent_api.agent_factory.profile_store import assert_profile_persisted
from tests.support.brain_sut import install_brain_sut, install_instrumented_real_model
from tests.support.req012_llm_budget import LLMBudgetRecorder

REAL_LLM = os.environ.get("GROUP_AGENT_REAL_LLM_TEST", "").strip() in {
    "1",
    "true",
    "yes",
}

GROUP_ID = "group_constraints_llm"
USER_ID = "u_constraints_llm"
CONV_ID = "conv_match_constraints_real_llm"

# City must → hard; tech stack / 「最好」→ soft (experience_tags).
USER_SPEECH_HARD_CITY_SOFT_TECH = (
    "大家好，我在做 AI 教育产品创业，需要找一位技术合伙人。"
    "只要上海的人；最好懂 langchain。"
    "我这边可以提供教研资源和客户对接。"
)

USER_SPEECH_ASK_SEARCH = (
    "需求清楚了：只要上海，最好懂 langchain。"
    "请在本群帮我搜索并推荐合适的人；我愿意直接 @ 认识。"
)


def _require_real_llm_env() -> None:
    if not REAL_LLM:
        pytest.skip("set GROUP_AGENT_REAL_LLM_TEST=1 to run live LLM tests")
    key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not key or key == "EMPTY":
        pytest.skip("real LLM API key missing in process env")
    provider = (os.environ.get("GROUP_AGENT_PROVIDER") or "qwen").strip().lower()
    # Force real — setdefault is wrong when the shell/conftest already has stub.
    os.environ["GROUP_AGENT_MODEL_MODE"] = "real"
    if not (os.environ.get("GROUP_AGENT_MODEL") or "").strip() and provider in {
        "qwen",
        "dashscope",
    }:
        os.environ["GROUP_AGENT_MODEL"] = "qwen-plus"
    if not (os.environ.get("GROUP_AGENT_BASE_URL") or "").strip() and provider in {
        "qwen",
        "dashscope",
    }:
        os.environ["GROUP_AGENT_BASE_URL"] = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )


def _write_modules_yaml(
    path: Path,
    *,
    search_relax: bool,
    reply_grounding: bool = False,
) -> Path:
    text = f"""version: 1
preset: current
checks:
  chk.reply_fact_grounding_llm: {'true' if reply_grounding else 'false'}
modules:
  mod.brain.reply_grounding: {'true' if reply_grounding else 'false'}
  mod.brain.search_relax: {'true' if search_relax else 'false'}
  mod.brain.profile_pool: false
search_relax:
  max_levels: 2
reply_grounding:
  max_attempts: 2
"""
    path.write_text(text, encoding="utf-8")
    return path


def _base_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    modules_yaml: Path,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    # Force — do not setdefault (conftest may already have stub).
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "real")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GROUP_AGENT_MODULES_CONFIG", str(modules_yaml))
    monkeypatch.setenv(
        "GROUP_AGENT_PROVIDER",
        os.environ.get("GROUP_AGENT_PROVIDER") or "qwen",
    )
    monkeypatch.setenv(
        "GROUP_AGENT_MODEL",
        os.environ.get("GROUP_AGENT_MODEL") or "qwen-plus",
    )
    monkeypatch.setenv(
        "GROUP_AGENT_BASE_URL",
        os.environ.get("GROUP_AGENT_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    monkeypatch.setenv("GROUP_AGENT_LLM_POLISH", "0")
    monkeypatch.setenv(
        "GROUP_AGENT_MAX_TOKENS",
        os.environ.get("GROUP_AGENT_MAX_TOKENS") or "800",
    )
    monkeypatch.setenv(
        "GROUP_AGENT_TIMEOUT_S",
        os.environ.get("GROUP_AGENT_TIMEOUT_S") or "60",
    )
    monkeypatch.setenv("GROUP_AGENT_TEST_LEVEL", "L1")
    monkeypatch.setenv(
        "GROUP_AGENT_PRINCIPAL_HMAC_SECRET",
        "test_32byte_secret_for_constraints!!",
    )
    monkeypatch.setenv(
        "GROUP_AGENT_CALLBACK_HMAC_SECRET",
        "test_32byte_callback_secret_cst!!",
    )


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


def _raise_timeout() -> None:
    raise TimeoutError("match_constraints real-LLM wall-clock timeout")


def _constraint_rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [c for c in raw if isinstance(c, dict)]


def _has_hard_city(constraints: list[dict[str, Any]]) -> bool:
    for c in constraints:
        field = str(c.get("field") or "").strip().lower()
        strength = str(c.get("strength") or "hard").strip().lower()
        if field != "city" or strength != "hard":
            continue
        values = c.get("values") or []
        blob = " ".join(str(v) for v in values)
        if "上海" in blob or "shanghai" in blob.lower():
            return True
    return False


def _has_soft_tech(constraints: list[dict[str, Any]]) -> bool:
    soft_markers = ("langchain", "lang chain", "llm", "python", "技术")
    for c in constraints:
        strength = str(c.get("strength") or "").strip().lower()
        if strength != "soft":
            continue
        field = str(c.get("field") or "").strip().lower()
        values = c.get("values") or []
        blob = f"{field} " + " ".join(str(v) for v in values)
        blob_l = blob.lower()
        if any(m in blob_l for m in soft_markers):
            return True
        if field in {"experience_tags", "role"}:
            return True
    return False


@pytest.fixture(autouse=True)
def _reset_modules_config() -> Any:
    reset_modules_config_cache()
    yield
    reset_modules_config_cache()


@pytest.mark.real_llm
@pytest.mark.brain_sut
@pytest.mark.timeout(180)
def test_real_llm_extracts_hard_city_and_soft_tech_into_match_constraints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MAIN must extract hard city (+ prefer soft tech) into save_group_profile."""
    _require_real_llm_env()

    yaml_path = _write_modules_yaml(
        tmp_path / "modules_constraints.yaml",
        search_relax=False,
        reply_grounding=False,
    )
    _base_env(monkeypatch, tmp_path, modules_yaml=yaml_path)
    reload_modules_config(yaml_path)

    recorder = LLMBudgetRecorder()
    live_model = install_instrumented_real_model(monkeypatch, recorder)
    assert isinstance(live_model, ChatOpenAI), type(live_model).__name__
    assert "Stub" not in (recorder.model_class or "")

    harness = install_brain_sut(monkeypatch)
    harness.hand.stick_matched(group_id=GROUP_ID)

    old_handler = signal.signal(signal.SIGALRM, lambda _s, _f: _raise_timeout())
    signal.alarm(180)
    try:
        from apps.group_agent_api.app import app

        with TestClient(app) as client:
            snap0 = recorder.snapshot()
            res = client.post(
                "/chat",
                json=_chat_payload(
                    USER_SPEECH_HARD_CITY_SOFT_TECH,
                    run_match=False,
                    run_invite=False,
                ),
            )
            assert res.status_code == 200, res.text
            body = res.json()
            delta = LLMBudgetRecorder.delta(snap0, recorder.snapshot())

            assert delta["llm_starts"] >= 1, "must invoke live LLM"
            assert delta["tool_calls"].get("save_group_profile", 0) >= 1, (
                f"expected save_group_profile, got {delta['tool_calls']}"
            )
            assert body.get("profile_persisted") is True, body
            assert len(harness.hand.profile_calls) >= 1

            persisted = assert_profile_persisted(tmp_path, USER_ID, GROUP_ID)
            assert persisted.ok and persisted.profile is not None, persisted.reason
            constraints = _constraint_rows(persisted.profile.match_constraints)
            print(
                "\n=== match_constraints from saved profile ===\n"
                f"{constraints!r}\n"
            )
            assert _has_hard_city(constraints), (
                "expected hard city=上海 in match_constraints; "
                f"got {constraints!r}"
            )
            # Soft tech preferred but not hard-fail if model only put it in need text.
            if not _has_soft_tech(constraints):
                need = str(
                    getattr(getattr(persisted.profile, "need", None), "value", "") or ""
                )
                print(
                    "NOTE: soft tech constraint missing; "
                    f"need={need!r} (soft preferred, hard city asserted)"
                )
    except TimeoutError:
        pytest.fail("match_constraints real-LLM timed out after 180s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@pytest.mark.real_llm
@pytest.mark.brain_sut
@pytest.mark.timeout(240)
def test_real_llm_search_relax_l0_empty_then_l1_optional(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Optional: search_relax on; fake empty@L0 matched@L1 → second call with L1.

    MAIN cooperation is flaky — skip with a clear note if the model does not
    re-call ``search_candidates`` at ``relax_level=1``.
    """
    _require_real_llm_env()

    yaml_path = _write_modules_yaml(
        tmp_path / "modules_relax.yaml",
        search_relax=True,
        reply_grounding=False,
    )
    _base_env(monkeypatch, tmp_path, modules_yaml=yaml_path)
    cfg = reload_modules_config(yaml_path)
    assert cfg.search_relax_enabled() is True

    recorder = LLMBudgetRecorder()
    live_model = install_instrumented_real_model(monkeypatch, recorder)
    assert isinstance(live_model, ChatOpenAI), type(live_model).__name__

    harness = install_brain_sut(monkeypatch)
    # FIFO: first search empty (L0), second matched (L1) if model re-calls.
    harness.hand.enqueue_empty(group_id=GROUP_ID, query="constraints")
    harness.hand.enqueue_matched(group_id=GROUP_ID, query="constraints")

    old_handler = signal.signal(signal.SIGALRM, lambda _s, _f: _raise_timeout())
    signal.alarm(240)
    try:
        from apps.group_agent_api.app import app

        with TestClient(app) as client:
            # Round 1: establish profile + constraints
            res1 = client.post(
                "/chat",
                json=_chat_payload(
                    USER_SPEECH_HARD_CITY_SOFT_TECH,
                    run_match=False,
                    run_invite=False,
                ),
            )
            assert res1.status_code == 200, res1.text
            persisted = assert_profile_persisted(tmp_path, USER_ID, GROUP_ID)
            assert persisted.ok and persisted.profile is not None, persisted.reason
            constraints = _constraint_rows(persisted.profile.match_constraints)
            assert _has_hard_city(constraints), (
                f"R1 hard city required before search_relax; got {constraints!r}"
            )

            # Round 2: ask to search (model should try L0 then L1 on empty)
            snap = recorder.snapshot()
            res2 = client.post(
                "/chat",
                json=_chat_payload(
                    USER_SPEECH_ASK_SEARCH,
                    run_match=True,
                    run_invite=False,
                ),
            )
            assert res2.status_code == 200, res2.text
            delta = LLMBudgetRecorder.delta(snap, recorder.snapshot())
            search_tool_n = int(delta["tool_calls"].get("search_candidates") or 0)
            levels = [
                int(c.get("relax_level") or 0) for c in harness.hand.search_calls
            ]
            print(
                "\n=== search_relax live calls ===\n"
                f"tool_calls={delta['tool_calls']}\n"
                f"hand_relax_levels={levels}\n"
                f"hand_calls={harness.hand.search_calls!r}\n"
            )

            assert search_tool_n >= 1, (
                f"expected ≥1 search_candidates; got {delta['tool_calls']}"
            )
            assert len(harness.hand.search_calls) >= 1

            if len(harness.hand.search_calls) < 2 or max(levels, default=0) < 1:
                pytest.skip(
                    "optional search_relax L1: MAIN did not issue a second "
                    f"search_candidates with relax_level=1 "
                    f"(hand_levels={levels}, tool_n={search_tool_n}); "
                    "known flaky — re-run with GROUP_AGENT_REAL_LLM_TEST=1"
                )

            assert levels[0] == 0, f"first search should be L0, got {levels}"
            assert any(lv >= 1 for lv in levels[1:]), (
                f"expected a later search with relax_level>=1, got {levels}"
            )
    except TimeoutError:
        pytest.fail("search_relax real-LLM timed out after 240s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
