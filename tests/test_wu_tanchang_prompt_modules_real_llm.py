"""Real-LLM checks for each Wu Tanchang content module (opt-in).

Skipped unless ``WU_E2E_REAL_LLM=1`` (same gate as ``test_wu_tanchang_api_e2e``).
Default CI stays green: ``pytest -m "not real_llm"`` or unset env.

Each test turns **only one** module on, then asks the live model to use that
module's prompt (and tools, when the module has any):

- profile: conversation tools only — recap + one follow-up, then 一、交谈要点
- eight_dim_anchors: write 二/三 with 1–3 of the eight anchors, no scores
- kb_cases: must call kb_analyst / kb_semantic_search and cite a hotpot note
- kb_cases off (profile-only + 要案例): must not search or write section 四

Run::

    WU_E2E_REAL_LLM=1 PYTHONPATH=. .venv/bin/pytest \\
      tests/test_wu_tanchang_prompt_modules_real_llm.py -v -s --tb=short
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from apps.wu_tanchang_api.agent_factory.agent import (
    create_agent,
    register_active_agent,
    unregister_active_agent,
)
from apps.wu_tanchang_api.agent_factory.prompt_modules import MODULE_TITLES
from apps.wu_tanchang_api.config import WuAgentConfig, WuPromptModules, load_env_file

REAL_LLM = os.environ.get("WU_E2E_REAL_LLM", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

API_DIR = Path(__file__).resolve().parents[1] / "apps" / "wu_tanchang_api"
REAL_WORKSPACE = API_DIR / "workspace"

EIGHT_DIM_NAMES = (
    "行业赛道",
    "品类战略",
    "客群",
    "菜单",
    "品牌叙事",
    "品牌",
    "体验",
    "峰值",
    "动线",
    "运营",
    "服务标准",
    "盈利",
    "投资测算",
    "模型迭代",
)


class _OkResponse:
    """Minimal requests.Response stand-in so save_meeting_prep and search_log succeed."""

    status_code = 201
    text = '{"ok":true,"id":1}'

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"ok": True, "id": 1}


PROFILE_SCRIPT = (
    "你好，我想在上海开一家火锅店，预算大概80万。我做过两年后厨管理，没自己开过店。"
    "最担心选址和客单价撑不住。",
    "还在想法阶段，团队就我自己，对标品牌还没定。",
    "真正卡住的是选址和定价，希望面谈时帮我判断能不能做、客单大概要定多少。",
    "目标是先活下来，计划半年内选址装修。深度已经够了，请立即输出会议准备材料，不要再提问。",
)

ANCHOR_USER = """下面是已经收集好的用户画像，请按八维锚点模块写出会议准备材料（只要二、三两节）：

城市：上海
创业背景：餐饮后厨管理两年，未自己开过店
选择赛道：火锅
阶段：想法期
定义问题：选址和客单价能否撑住
聚焦目标：先让单店活下来
实施计划：半年内选址装修，预算约80万

请挂 1–3 个有证据的八维锚点，说明见面准备什么。不要打分，不要写成最终方案。"""

CASES_USER = (
    "我打算在上海开火锅店，预算80万，有后厨背景。吴探长是否有相关的案例？"
    "请检索知识库后回答，点名命中的品牌，有探店笔记网页就写成 Markdown 链接。"
)


def _require_real_llm() -> None:
    if not REAL_LLM:
        pytest.skip("set WU_E2E_REAL_LLM=1 to run live Wu Tanchang module tests")


def _only(module_id: str) -> WuPromptModules:
    return WuPromptModules(
        profile=module_id == "profile",
        eight_dim_anchors=module_id == "eight_dim_anchors",
        kb_cases=module_id == "kb_cases",
    )


def _prepare_backend(tmp_path: Path) -> Path:
    """Copy persona + modules; symlink real kb so search tools can run."""
    backend_root = tmp_path / "backend"
    dest = backend_root / "workspace"
    dest.mkdir(parents=True)
    for name in ("IDENTITY.md", "SOUL.md", "AGENTS.md", "USER.md"):
        shutil.copy2(REAL_WORKSPACE / name, dest / name)
    shutil.copytree(REAL_WORKSPACE / "modules", dest / "modules")
    os.symlink(REAL_WORKSPACE / "kb", dest / "kb")
    return backend_root


def _ai_text(messages: list[Any]) -> str:
    chunks: list[str] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chunks.append(str(block.get("text") or ""))
                elif isinstance(block, str):
                    chunks.append(block)
    return "\n".join(chunks)


def _tool_names(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                if isinstance(call, dict):
                    names.append(str(call.get("name") or ""))
                else:
                    names.append(str(getattr(call, "name", "") or ""))
        if isinstance(msg, ToolMessage):
            names.append(str(getattr(msg, "name", "") or ""))
    return [n for n in names if n]


def _last_ai(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    str(b.get("text") if isinstance(b, dict) else b) for b in content
                )
    return _ai_text(messages)


@pytest.fixture
def module_agent_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Point config/env at the real app; isolate checkpoints under tmp."""
    _require_real_llm()
    env_path = API_DIR / ".env"
    monkeypatch.setenv("WU_API_CONFIG", str(API_DIR / "config.json"))
    monkeypatch.setenv("WU_API_ENV_FILE", str(env_path))
    load_env_file(env_path)
    key = os.environ.get("WU_DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        pytest.fail("WU_DASHSCOPE_API_KEY or DASHSCOPE_API_KEY is required")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        monkeypatch.setenv("DASHSCOPE_API_KEY", key)

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    monkeypatch.setattr(
        "apps.wu_tanchang_api.agent_factory.agent.default_runtime_dir",
        lambda: runtime_dir,
    )
    monkeypatch.setattr(
        "requests.post",
        lambda *_a, **_k: _OkResponse(),
    )
    yield _prepare_backend(tmp_path)


def _make_agent(backend_root: Path, module_id: str, agent_name: str) -> Any:
    cfg = WuAgentConfig(
        name=agent_name,
        provider="qwen",
        model="deepseek-v4-flash",
        max_tokens=4000,
        workspace="workspace",
        modules=_only(module_id),
    )
    agent, _ckpt = create_agent(backend_root=backend_root, agent_config=cfg)
    return agent


def _invoke(
    agent: Any,
    *,
    thread_id: str,
    text: str,
    agent_name: str,
) -> dict[str, Any]:
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "user_id": "9001",
            "user_a_id": "9001",
            "calendar_id": "9001",
            "callback_url": "http://localhost:3001/wu_tanchang_callbacks/",
        },
        "recursion_limit": 80,
    }
    register_active_agent(thread_id, agent)
    try:
        invoke = getattr(agent, "invoke", None)
        if invoke is None:
            import asyncio

            result = asyncio.run(
                agent.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)
            )
        else:
            result = invoke({"messages": [HumanMessage(content=text)]}, config=config)
    finally:
        unregister_active_agent(thread_id)
    assert isinstance(result, dict)
    print(f"\n[{agent_name} tools]", _tool_names(result.get("messages") or []))
    print(f"[{agent_name} last AI]\n", _last_ai(result.get("messages") or [])[:2000])
    return result


@pytest.mark.real_llm
@pytest.mark.integration
@pytest.mark.timeout(180)
def test_real_llm_profile_module_asks_then_writes_section_one(
    module_agent_env: Path,
) -> None:
    """画像模块开启：先追问，凑齐后写出「一、交谈要点」，不跑知识库、不打八维分。"""
    agent_name = "mod_profile"
    agent = _make_agent(module_agent_env, "profile", agent_name)
    tid = f"wt::{agent_name}::modtest::profile"
    messages: list[Any] = []
    for turn, user in enumerate(PROFILE_SCRIPT, start=1):
        result = _invoke(
            agent, thread_id=tid, text=user, agent_name=f"{agent_name}-t{turn}"
        )
        messages = result.get("messages") or []
        if turn == 1:
            reply = _last_ai(messages)
            assert "上海" in reply or "火锅" in reply
            assert ("？" in reply) or ("吗" in reply) or ("呢" in reply)
            assert "八维诊断评分" not in reply
            assert "0-10" not in reply and "0–10" not in reply

    tools = _tool_names(messages)
    assert "kb_semantic_search" not in tools
    assert "task" not in tools
    final = _ai_text(messages)
    assert "一、" in final
    assert "交谈要点" in final or "城市" in final
    assert "上海" in final
    assert "火锅" in final
    assert "四、可参考案例" not in final


@pytest.mark.real_llm
@pytest.mark.integration
@pytest.mark.timeout(180)
def test_real_llm_eight_dim_module_hangs_one_to_three_anchors(
    module_agent_env: Path,
) -> None:
    """八维模块开启：用户可见的初步框架挂 1–3 个锚点，当议程而不是打分诊断。"""
    agent_name = "mod_anchors"
    agent = _make_agent(module_agent_env, "eight_dim_anchors", agent_name)
    tid = f"wt::{agent_name}::modtest::anchors"
    result = _invoke(agent, thread_id=tid, text=ANCHOR_USER, agent_name=agent_name)
    text = _ai_text(result.get("messages") or [])
    assert "二、" in text
    assert "三、" in text
    hits = [name for name in EIGHT_DIM_NAMES if name in text]
    assert hits, f"expected at least one eight-dim name, got excerpt: {text[:800]}"
    full_labels = [
        "行业赛道与品类战略",
        "市场细分与目标客群定位",
        "业务设计与菜单工程",
        "品牌叙事与价值传递",
        "全触点体验与峰值设计",
        "空间场景与高效动线",
        "运营系统与服务标准",
        "投资测算与模型迭代",
    ]
    stuffed = sum(1 for label in full_labels if label in text)
    assert stuffed <= 3, f"too many full eight-dim labels ({stuffed}): should hang 1–3"
    assert "0-10分" not in text and "0–10分" not in text
    assert "八维诊断评分" not in text
    assert "选址" in text or "客单" in text or "客单价" in text


@pytest.mark.real_llm
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_real_llm_kb_cases_module_searches_and_cites_hotpot_note(
    module_agent_env: Path,
) -> None:
    """案例模块开启：必须走检索工具，命中火锅时不得说没有，并尽量带短链。"""
    kb_db = REAL_WORKSPACE / "kb" / "kb.db"
    kb_chroma = REAL_WORKSPACE / "kb" / "chroma"
    if not kb_db.exists() or not kb_chroma.exists():
        pytest.skip(
            "local kb.db / chroma missing; run kb_build_db.py and kb_build_vectors.py"
        )

    agent_name = "mod_cases"
    agent = _make_agent(module_agent_env, "kb_cases", agent_name)
    tid = f"wt::{agent_name}::modtest::cases"
    result = _invoke(agent, thread_id=tid, text=CASES_USER, agent_name=agent_name)
    messages = result.get("messages") or []
    tools = _tool_names(messages)
    used_search = any(
        name in {"task", "kb_semantic_search", "get_note_content"} for name in tools
    )
    assert used_search, f"kb_cases module must use search tools, got {tools}"
    text = _ai_text(messages)
    assert "没有" not in text or "火锅" in text
    denied = ("没有火锅" in text) or ("没有该品类" in text) or ("暂无该品类" in text)
    assert not denied, f"search should have hotpot notes, model denied: {text[:800]}"
    assert ("大虎" in text) or ("火锅" in text)
    assert re.search(r"\[[^\]]+ · 看探店笔记\]\(https://xhslink\.cn/o/[^)]+\)", text)


@pytest.mark.real_llm
@pytest.mark.integration
@pytest.mark.timeout(180)
def test_real_llm_kb_cases_off_does_not_search_or_cite_section_four(
    module_agent_env: Path,
) -> None:
    """案例模块关闭：用户要案例时不得调用 kb_analyst，也不得写「四、可参考案例」。"""
    agent_name = "mod_cases_off"
    agent = _make_agent(module_agent_env, "profile", agent_name)
    assert agent.uses_prompt_modules is True
    assert agent.prompt_modules.kb_cases is False
    tid = f"wt::{agent_name}::modtest::cases_off"
    result = _invoke(agent, thread_id=tid, text=CASES_USER, agent_name=agent_name)
    tools = _tool_names(result.get("messages") or [])
    assert "task" not in tools
    assert "kb_semantic_search" not in tools
    assert "get_note_content" not in tools
    text = _ai_text(result.get("messages") or [])
    assert "四、可参考案例" not in text
    assert "xhslink.cn" not in text


def test_module_titles_cover_three_blocks() -> None:
    """Sanity: keep real-LLM file aligned with the three shipped modules."""
    assert set(MODULE_TITLES) == {"profile", "eight_dim_anchors", "kb_cases"}
