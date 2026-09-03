"""Composable, on/off prompt modules for the Wu Tanchang front agent.

Three content modules sit beside the orchestrator persona (AGENTS.md):

- profile: 七维画像勾画
- eight_dim_anchors: 八维锚点 / 给用户看的初步建议
- kb_cases: 知识库探店笔记引用
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from apps.wu_tanchang_api.config import WuPromptModules

_logger = logging.getLogger("uvicorn.error")

MODULE_IDS = ("profile", "eight_dim_anchors", "kb_cases")

MODULE_FILES: dict[str, str] = {
    "profile": "profile.md",
    "eight_dim_anchors": "eight_dim_anchors.md",
    "kb_cases": "kb_cases.md",
}

MODULE_TITLES: dict[str, str] = {
    "profile": "画像勾画",
    "eight_dim_anchors": "八维锚点（初步建议）",
    "kb_cases": "可参考案例",
}


# Test/direct-call fallback only. Runtime save_meeting_prep must read the
# active agent instance — thread IDs use workspace Agent ids (yc01), not
# config.json profile names (default).
_AGENT_MODULES: dict[str, WuPromptModules] = {}

WU_PROMPT_MODULES_OFF = WuPromptModules(
    profile=False, eight_dim_anchors=False, kb_cases=False
)

LEGACY_MEETING_PREP_HEADINGS = ("一、", "二、", "三、", "四、")


def register_prompt_modules(agent_name: str, modules: WuPromptModules) -> None:
    """Remember modules for later tool validation (save_meeting_prep)."""
    _AGENT_MODULES[agent_name] = modules
    _logger.info(
        "[Agent] prompt modules for %s: profile=%s eight_dim=%s kb_cases=%s",
        agent_name,
        modules.profile,
        modules.eight_dim_anchors,
        modules.kb_cases,
    )


def modules_for_run(config: RunnableConfig | None = None) -> WuPromptModules:
    """Resolve Wu modules from the active agent graph, not the thread name.

    Non-Wu workspaces set ``uses_prompt_modules=False``; those return all-off
    so inherited config flags cannot leak into save validation.
    Unknown lookups also return all-off (never silently all-on).
    """
    if not config:
        return WU_PROMPT_MODULES_OFF
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "")
    if thread_id:
        from apps.wu_tanchang_api.agent_factory.agent import get_active_agent

        active_agent = get_active_agent(thread_id)
        if active_agent is not None:
            uses = getattr(active_agent, "uses_prompt_modules", None)
            if uses is False:
                return WU_PROMPT_MODULES_OFF
            active_modules = getattr(active_agent, "prompt_modules", None)
            if isinstance(active_modules, WuPromptModules):
                return active_modules
    parts = thread_id.split("::")
    name = parts[1] if len(parts) > 2 else "default"
    return _AGENT_MODULES.get(name, WU_PROMPT_MODULES_OFF)


def meeting_prep_required_headings(modules: WuPromptModules) -> tuple[str, ...]:
    """Section prefixes the meeting-prep body must contain for enabled modules."""
    headings: list[str] = []
    if modules.profile:
        headings.append("一、")
    if modules.eight_dim_anchors:
        headings.extend(["二、", "三、"])
    if modules.kb_cases:
        headings.append("四、")
    return tuple(headings)


def meeting_prep_forbidden_headings(modules: WuPromptModules) -> tuple[str, ...]:
    """Section prefixes that must not appear when the matching module is off."""
    required = set(meeting_prep_required_headings(modules))
    return tuple(h for h in LEGACY_MEETING_PREP_HEADINGS if h not in required)


def meeting_prep_policy_for_run(
    config: RunnableConfig | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(required, forbidden)`` headings for ``save_meeting_prep``.

    Wu front-agent (``uses_prompt_modules=True``): required + forbidden follow
    the enabled modules. Legacy YC / 爱合伙 workspaces keep the old four-section
    contract and do not treat Wu-disabled headings as forbidden.
    """
    thread_id = ""
    if config:
        thread_id = str((config.get("configurable") or {}).get("thread_id") or "")
        if thread_id:
            from apps.wu_tanchang_api.agent_factory.agent import get_active_agent

            active_agent = get_active_agent(thread_id)
            if active_agent is not None:
                uses = getattr(active_agent, "uses_prompt_modules", None)
                if uses is False:
                    return LEGACY_MEETING_PREP_HEADINGS, ()
                active_modules = getattr(active_agent, "prompt_modules", None)
                if isinstance(active_modules, WuPromptModules):
                    return (
                        meeting_prep_required_headings(active_modules),
                        meeting_prep_forbidden_headings(active_modules),
                    )
    parts = thread_id.split("::")
    name = parts[1] if len(parts) > 2 else "default"
    if name in _AGENT_MODULES:
        modules = _AGENT_MODULES[name]
        return (
            meeting_prep_required_headings(modules),
            meeting_prep_forbidden_headings(modules),
        )
    return LEGACY_MEETING_PREP_HEADINGS, ()


def load_enabled_module_files(workspace_path: Path, modules: WuPromptModules) -> str:
    """Concatenate enabled module markdown from workspace/modules/."""
    modules_dir = workspace_path / "modules"
    parts: list[str] = []
    for module_id in modules.enabled_ids():
        path = modules_dir / MODULE_FILES[module_id]
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            _logger.warning("[Agent] Missing prompt module file: %s", path)
            continue
        if not content:
            continue
        title = MODULE_TITLES[module_id]
        parts.append(f"=== MODULE:{module_id} {title} ===\n{content}")
    return "\n\n".join(parts)


def compose_module_orchestration(modules: WuPromptModules) -> str:
    """Inject a machine-readable on/off table plus the meeting-prep skeleton."""
    lines = ["=== 模块编排（以本块为准，未开启的模块整段跳过） ===", ""]
    for module_id in MODULE_IDS:
        state = "开启" if modules.is_enabled(module_id) else "关闭"
        lines.append(f"- {MODULE_TITLES[module_id]}（`{module_id}`）：**{state}**")
    lines.append("")
    if not modules.profile:
        lines.append(
            "- 画像模块已关闭：不要按七维深度追问；用户说到哪用到哪，尽快进入已开启模块的产出。"
        )
    if not modules.eight_dim_anchors:
        lines.append(
            "- 八维锚点模块已关闭：材料里不要写「二、初步建议 / 三、建议讨论话题」，不要对用户列出八维。"
        )
    if not modules.kb_cases:
        lines.append(
            "- 案例模块已关闭：**禁止**调用 `kb_analyst`，材料里不要写「四、可参考案例」，用户要案例时说明本轮未启用案例检索。"
        )
    if modules.kb_cases and not modules.profile:
        lines.append(
            "- 无画像模块时，调用 `kb_analyst` 仍须把用户已说的城市/品类/问题原样传给它。"
        )
    lines.append("")
    lines.append(compose_meeting_prep_skeleton(modules))
    return "\n".join(lines)


def compose_meeting_prep_skeleton(modules: WuPromptModules) -> str:
    """Meeting-prep format for currently enabled modules only."""
    required = meeting_prep_required_headings(modules)
    if not required:
        return (
            "## 会议准备材料格式\n\n"
            "当前未启用任何内容模块。向用户确认是否约谈即可，不要编造画像、锚点或案例。"
        )

    sections: list[str] = ["【会议准备 · 供面谈参考】", ""]
    if modules.profile:
        sections.extend(
            [
                "一、交谈要点",
                "城市：…",
                "创业背景：…",
                "选择赛道：…",
                "阶段：想法期 / 筹备中 / 已开店 / 瓶颈期",
                "定义问题：…",
                "聚焦目标：…",
                "实施计划：…",
                "（已开店/瓶颈时补：门店名、客单、几家/面积、招牌与利润来源、自认优势）",
                "",
            ]
        )
    if modules.eight_dim_anchors:
        sections.extend(
            [
                "二、初步建议（八维锚点）",
                "用 3–6 句说明见面会从哪 1–3 个锚点展开，每条能回指画像里的原话。",
                "这是给用户看的面谈框架，不是最终方案，不打分。",
                "",
                "三、建议讨论话题",
                "每条对应一个已挂锚点：为何挂在这里、见面建议准备什么。",
                "1. 【锚点名】…",
                "2. 【锚点名】…",
                "",
            ]
        )
    if modules.kb_cases:
        sections.extend(
            [
                "四、可参考案例",
                "（来自 kb_analyst，1–2 个品牌；有探店笔记网页 URL 时写成 "
                "`[品牌 · 看探店笔记](完整URL)`）",
                "",
            ]
        )

    heading_hint = "、".join(required)
    return (
        "## 会议准备材料格式\n\n"
        f"**聊天回复与 `save_meeting_prep` 的 body 必须含【会议准备】以及本节列出的标题"
        f"（{heading_hint}）**，未开启的模块不要编造对应节。写精炼，禁止截断。\n\n"
        "```\n" + "\n".join(sections).rstrip() + "\n```"
    )


def compose_frontend_rules(
    modules: WuPromptModules,
    *,
    has_aihehuo: bool = False,
    uses_prompt_modules: bool = True,
) -> str:
    """Front-agent standing rules that depend on enabled modules."""
    persona_rule = (
        "- 严格按照人格文件与【模块编排】行事；关闭的模块整段跳过"
        if uses_prompt_modules
        else "- 严格按照人格文件中的指导行事"
    )
    rules = [
        persona_rule,
        "- 你**没有**文件系统访问权限，无法直接读取任何工作区文件",
    ]
    if has_aihehuo:
        rules.append(
            "- 如果用户询问爱合伙平台的相关数据、博客、微信群、项目或常见问题等信息，你可以随时调用子代理 `aihehuo_cruncher` 来获取最新数据并回复用户。**注意：在给用户的任何消息中，绝对禁止提及子代理 `aihehuo_cruncher` 的名字或说明是从该子代理获取的数据，直接将获取到的分析或信息以你自己的口吻加工整理后回复给用户。**"
        )
    if not uses_prompt_modules:
        rules.extend(
            [
                "- **当你生成会议准备材料后，必须先将材料以文字完整呈现给用户，然后调用 `save_meeting_prep` 工具保存材料，最后调用 `mark_material_delivered` 工具标记完成。顺序不可颠倒。**",
                "- `save_meeting_prep` 的 body 必须含完整四节（一/二/三/四），禁止截断；聊天呈现同样须四节写完；若保存失败须补全后重试，禁止未保存成功就 `mark_material_delivered`。",
            ]
        )
        return "\n".join(rules)
    if modules.profile:
        rules.append(
            "- **禁止**用户仅聊 1–2 轮、七维仍停留在标签层时就写会议材料"
            + ("或调用 kb_analyst" if modules.kb_cases else "")
        )
    if modules.kb_cases:
        rules.append(
            "- 用户主动要案例时：用 kb_analyst 的 1–2 个品牌作答，有 URL 必须写成 `[品牌 · 看探店笔记](完整URL)`"
        )
        rules.append("- 检索已命中用户品类时，禁止对用户说「库里没有该品类笔记」")
    else:
        rules.append("- **禁止**调用 kb_analyst 或编造探店案例")
    rules.extend(
        [
            "- **当你生成会议准备材料后，必须先将材料以文字完整呈现给用户，然后调用 `save_meeting_prep` 工具保存材料，最后调用 `mark_material_delivered` 工具标记完成。顺序不可颠倒。**",
            "- `save_meeting_prep` 的 body 必须含【会议准备】及【模块编排】要求的各节；若保存失败须按缺节补全后重试，禁止未保存成功就 `mark_material_delivered`。",
            "- 材料交付后：先问是否约吴探长本人；同意后说明吴探长会联系用户——**禁止**引导用户自找约谈日程/预约入口；暂不约则礼貌收尾，不再深入探讨",
        ]
    )
    return "\n".join(rules)
