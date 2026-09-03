# Each module also has an opt-in live-LLM test in
# tests/test_wu_tanchang_prompt_modules_real_llm.py (WU_E2E_REAL_LLM=1).

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from apps.wu_tanchang_api.agent_factory.prompt_modules import (
    WU_PROMPT_MODULES_OFF,
    compose_frontend_rules,
    compose_module_orchestration,
    load_enabled_module_files,
    meeting_prep_forbidden_headings,
    meeting_prep_required_headings,
    modules_for_run,
    register_prompt_modules,
)
from apps.wu_tanchang_api.config import (
    WuPromptModules,
    load_agent_registry,
    parse_prompt_modules,
)


def test_parse_prompt_modules_defaults_all_on() -> None:
    modules = parse_prompt_modules(None)
    assert modules.profile is True
    assert modules.eight_dim_anchors is True
    assert modules.kb_cases is True


def test_parse_prompt_modules_can_disable_one() -> None:
    modules = parse_prompt_modules({"kb_cases": False, "profile": True})
    assert modules.profile is True
    assert modules.eight_dim_anchors is True
    assert modules.kb_cases is False


def test_agent_module_overrides_merge_with_module_defaults() -> None:
    registry = load_agent_registry(
        {
            "agents": {
                "defaults": {
                    "provider": "qwen",
                    "model": "qwen-flash",
                    "modules": {
                        "profile": False,
                        "eight_dim_anchors": False,
                        "kb_cases": False,
                    },
                },
                "list": [{"name": "default", "modules": {"profile": True}}],
            }
        }
    )
    assert registry is not None
    assert registry.agents["default"].modules == WuPromptModules(
        profile=True,
        eight_dim_anchors=False,
        kb_cases=False,
    )


def test_meeting_prep_forbidden_headings_match_disabled_modules() -> None:
    forbidden = meeting_prep_forbidden_headings(WuPromptModules(kb_cases=False))
    assert forbidden == ("四、",)


def test_absent_modules_object_parses_all_off_for_non_wu_profiles() -> None:
    registry = load_agent_registry(
        {
            "agents": {
                "defaults": {"provider": "qwen", "model": "qwen-flash"},
                "list": [
                    {
                        "name": "default",
                        "modules": {
                            "profile": True,
                            "eight_dim_anchors": True,
                            "kb_cases": True,
                        },
                    },
                    {"name": "aihehuo", "workspace": "workspace_aihehuo"},
                ],
            }
        }
    )
    assert registry is not None
    assert registry.agents["default"].modules.kb_cases is True
    assert registry.agents["aihehuo"].modules == WU_PROMPT_MODULES_OFF


def test_shipped_config_limits_modules_to_wu_front_agent() -> None:
    import json

    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "wu_tanchang_api"
        / "config.json"
    )
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "modules" not in data["agents"]["defaults"]
    by_name = {entry["name"]: entry for entry in data["agents"]["list"]}
    assert by_name["default"]["modules"]["kb_cases"] is True
    assert by_name["aihehuo"]["modules"]["kb_cases"] is False
    assert by_name["owner"]["modules"]["profile"] is False

    registry = load_agent_registry({"agents": data["agents"]})
    assert registry is not None
    assert registry.agents["default"].modules == WuPromptModules()
    assert registry.agents["aihehuo"].modules == WU_PROMPT_MODULES_OFF
    assert registry.agents["owner"].modules == WU_PROMPT_MODULES_OFF


def test_meeting_prep_headings_follow_enabled_modules() -> None:
    all_on = meeting_prep_required_headings(WuPromptModules())
    assert all_on == ("一、", "二、", "三、", "四、")

    no_cases = meeting_prep_required_headings(WuPromptModules(kb_cases=False))
    assert no_cases == ("一、", "二、", "三、")
    assert "四、" not in no_cases

    anchors_only = meeting_prep_required_headings(
        WuPromptModules(profile=False, eight_dim_anchors=True, kb_cases=False)
    )
    assert anchors_only == ("二、", "三、")


def test_orchestration_banner_marks_disabled_modules() -> None:
    text = compose_module_orchestration(WuPromptModules(kb_cases=False))
    assert "可参考案例（`kb_cases`）：**关闭**" in text
    assert "禁止**调用 `kb_analyst`" in text
    skeleton = text.split("## 会议准备材料格式", 1)[-1]
    assert "四、可参考案例" not in skeleton
    assert "一、交谈要点" in skeleton
    assert "二、初步建议" in skeleton


def test_load_enabled_module_files_skips_disabled(tmp_path: Path) -> None:
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()
    (modules_dir / "profile.md").write_text("PROFILE_BODY", encoding="utf-8")
    (modules_dir / "eight_dim_anchors.md").write_text("ANCHOR_BODY", encoding="utf-8")
    (modules_dir / "kb_cases.md").write_text("CASES_BODY", encoding="utf-8")

    loaded = load_enabled_module_files(
        tmp_path, WuPromptModules(profile=True, eight_dim_anchors=False, kb_cases=True)
    )
    assert "PROFILE_BODY" in loaded
    assert "CASES_BODY" in loaded
    assert "ANCHOR_BODY" not in loaded
    assert "MODULE:eight_dim_anchors" not in loaded


def test_save_meeting_prep_does_not_require_disabled_sections() -> None:
    from apps.wu_tanchang_api.agent_factory.agent import save_meeting_prep

    register_prompt_modules("default", WuPromptModules(kb_cases=False))
    body = (
        "【会议准备】\n一、要点上海火锅\n二、建议挂客群与模型\n三、话题见面看选址\n"
        + ("x" * 400)
    )
    config = {
        "configurable": {"thread_id": "wt::default::u1::c1"},
        "metadata": {
            "user_id": "1",
            "calendar_id": "2",
            "callback_url": "http://malicious-attacker.com/wu_tanchang_callbacks/",
        },
    }
    res = save_meeting_prep.invoke({"body": body}, config=config)
    # Reaches callback allowlist (no 四、 required) rather than incomplete-body.
    assert "不完整" not in res
    assert "未被授权" in res
    register_prompt_modules("default", WuPromptModules())


def test_save_meeting_prep_rejects_disabled_sections() -> None:
    from apps.wu_tanchang_api.agent_factory.agent import save_meeting_prep

    register_prompt_modules("default", WuPromptModules(kb_cases=False))
    body = (
        "【会议准备】\n一、要点上海火锅\n二、建议挂客群与模型\n三、话题见面看选址\n四、案例\n"
        + ("x" * 400)
    )
    config = {
        "configurable": {"thread_id": "wt::default::u1::c1"},
        "metadata": {
            "user_id": "1",
            "calendar_id": "2",
            "callback_url": "http://malicious-attacker.com/wu_tanchang_callbacks/",
        },
    }
    try:
        res = save_meeting_prep.invoke({"body": body}, config=config)
        assert "未启用却出现：四、" in res
    finally:
        register_prompt_modules("default", WuPromptModules())


def test_modules_for_run_prefers_active_agent_over_thread_name() -> None:
    from types import SimpleNamespace

    from apps.wu_tanchang_api.agent_factory.agent import (
        register_active_agent,
        unregister_active_agent,
    )

    tid = "wt::yc01::u1::c1"
    expected = WuPromptModules(profile=True, eight_dim_anchors=False, kb_cases=False)
    register_active_agent(tid, SimpleNamespace(prompt_modules=expected))
    try:
        assert modules_for_run({"configurable": {"thread_id": tid}}) == expected
    finally:
        unregister_active_agent(tid)


def test_create_agent_omits_kb_subagent_when_cases_disabled(tmp_path: Path) -> None:
    from apps.wu_tanchang_api.agent_factory.agent import create_agent
    from apps.wu_tanchang_api.config import WuAgentConfig

    for name in ["IDENTITY.md", "SOUL.md", "AGENTS.md", "USER.md"]:
        (tmp_path / name).write_text("dummy", encoding="utf-8")
    (tmp_path / "modules").mkdir()

    with (
        patch(
            "apps.wu_tanchang_api.agent_factory.agent.default_runtime_dir",
            return_value=tmp_path / "runtime",
        ),
        patch(
            "apps.wu_tanchang_api.agent_factory.agent.create_deep_agent"
        ) as mock_create,
        patch(
            "apps.wu_tanchang_api.agent_factory.agent.create_model",
            return_value=object(),
        ),
    ):
        cfg = WuAgentConfig(
            name="no_cases",
            model="qwen-flash",
            provider="qwen",
            max_tokens=800,
            workspace="workspace",
            modules=WuPromptModules(kb_cases=False),
        )
        create_agent(backend_root=tmp_path, agent_config=cfg)
        _args, kwargs = mock_create.call_args
        subagents = kwargs.get("subagents", [])
        assert all(s.get("name") != "kb_analyst" for s in subagents)


def test_non_module_workspace_keeps_legacy_frontend_rules(tmp_path: Path) -> None:
    from apps.wu_tanchang_api.agent_factory.agent import create_agent
    from apps.wu_tanchang_api.config import WuAgentConfig

    workspace = tmp_path / "workspace_aihehuo"
    workspace.mkdir()
    for name in ["IDENTITY.md", "SOUL.md", "AGENTS.md", "USER.md"]:
        (workspace / name).write_text("AIHEHUO_PERSONA", encoding="utf-8")

    with (
        patch(
            "apps.wu_tanchang_api.agent_factory.agent.default_runtime_dir",
            return_value=tmp_path / "runtime",
        ),
        patch(
            "apps.wu_tanchang_api.agent_factory.agent.create_deep_agent"
        ) as mock_create,
        patch(
            "apps.wu_tanchang_api.agent_factory.agent.create_model",
            return_value=object(),
        ),
    ):
        cfg = WuAgentConfig(
            name="aihehuo",
            model="qwen-flash",
            provider="qwen",
            max_tokens=800,
            workspace="workspace_aihehuo",
            modules=WuPromptModules(
                profile=True, eight_dim_anchors=True, kb_cases=False
            ),
        )
        agent, _ckpt = create_agent(backend_root=tmp_path, agent_config=cfg)

    prompt = mock_create.call_args.kwargs["system_prompt"]
    assert "AIHEHUO_PERSONA" in prompt
    assert "【模块编排】" not in prompt
    assert "八维锚点" not in prompt
    assert "body 必须含完整四节" in prompt
    assert agent.uses_prompt_modules is False
    assert agent.prompt_modules == WU_PROMPT_MODULES_OFF
    subagents = mock_create.call_args.kwargs.get("subagents", [])
    assert any(s.get("name") == "kb_analyst" for s in subagents)


def test_legacy_frontend_rules_do_not_reference_module_orchestration() -> None:
    text = compose_frontend_rules(WuPromptModules(), uses_prompt_modules=False)
    assert "【模块编排】" not in text
    assert "body 必须含完整四节" in text


def test_modules_for_run_ignores_inherited_flags_when_workspace_is_legacy() -> None:
    from types import SimpleNamespace

    from apps.wu_tanchang_api.agent_factory.agent import (
        register_active_agent,
        unregister_active_agent,
    )

    tid = "wt::yc01::u1::c1"
    leaked = WuPromptModules(profile=True, eight_dim_anchors=True, kb_cases=False)
    register_active_agent(
        tid,
        SimpleNamespace(uses_prompt_modules=False, prompt_modules=leaked),
    )
    try:
        assert modules_for_run({"configurable": {"thread_id": tid}}) == (
            WU_PROMPT_MODULES_OFF
        )
    finally:
        unregister_active_agent(tid)


def test_legacy_workspace_save_keeps_four_sections_even_if_wu_cases_off() -> None:
    """Dynamic YC/爱合伙 thread ids must not inherit Wu forbidden-heading rules."""
    from types import SimpleNamespace

    from apps.wu_tanchang_api.agent_factory.agent import (
        register_active_agent,
        save_meeting_prep,
        unregister_active_agent,
    )

    tid = "wt::yc01::u1::c1"
    register_active_agent(
        tid,
        SimpleNamespace(
            uses_prompt_modules=False,
            prompt_modules=WuPromptModules(kb_cases=False),
        ),
    )
    body = (
        "【会议准备】\n一、要点上海火锅\n二、建议挂客群与模型\n三、话题见面看选址\n四、案例\n"
        + ("x" * 400)
    )
    config = {
        "configurable": {"thread_id": tid},
        "metadata": {
            "user_id": "1",
            "calendar_id": "2",
            "callback_url": "http://malicious-attacker.com/wu_tanchang_callbacks/",
        },
    }
    try:
        res = save_meeting_prep.invoke({"body": body}, config=config)
        assert "未启用却出现" not in res
        assert "不完整" not in res
        assert "未被授权" in res
    finally:
        unregister_active_agent(tid)


def test_wu_save_follows_modules_even_when_thread_name_is_workspace_id() -> None:
    """Active Wu graph wins over thread ids like wt::yc01::… that are not config names."""
    from types import SimpleNamespace

    from apps.wu_tanchang_api.agent_factory.agent import (
        register_active_agent,
        save_meeting_prep,
        unregister_active_agent,
    )

    tid = "wt::yc01::u1::c1"
    register_active_agent(
        tid,
        SimpleNamespace(
            uses_prompt_modules=True,
            prompt_modules=WuPromptModules(kb_cases=False),
        ),
    )
    body = (
        "【会议准备】\n一、要点上海火锅\n二、建议挂客群与模型\n三、话题见面看选址\n四、案例\n"
        + ("x" * 400)
    )
    try:
        res = save_meeting_prep.invoke(
            {"body": body},
            config={
                "configurable": {"thread_id": tid},
                "metadata": {
                    "user_id": "1",
                    "calendar_id": "2",
                    "callback_url": "http://malicious-attacker.com/wu_tanchang_callbacks/",
                },
            },
        )
        assert "未启用却出现：四、" in res
    finally:
        unregister_active_agent(tid)
