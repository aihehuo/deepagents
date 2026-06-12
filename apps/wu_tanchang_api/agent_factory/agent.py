"""Agent factory for Wu Tanchang pre-consultation API.

Architecture:
  Front-end agent (前置助手)
    - Reads personality files (IDENTITY.md, SOUL.md, AGENTS.md) from workspace
    - Has NO direct KB access (no SkillsMiddleware)
    - When KB insights are needed, calls kb_analyst sub-agent via task() tool
    - Goal: collect info, produce structured outline for Wu Tanchang

  KB sub-agent (kb_analyst)
    - Has SkillsMiddleware + FilesystemMiddleware for full KB access
    - Reads METHOD.md, PLAYBOOK.md, index.json, chunks
    - Returns analysis in Wu Tanchang's voice
"""

from __future__ import annotations

import logging
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.accountant import AccountantMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents.middleware.prompt_logging import PromptLoggingMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

from langchain_core.tools import tool

from apps.wu_tanchang_api.agent_factory.model_builder import create_model
from apps.wu_tanchang_api.agent_factory.utils import default_runtime_dir
from apps.wu_tanchang_api.checkpointer import DiskBackedInMemorySaver
from apps.wu_tanchang_api.config import WuAgentConfig

_logger = logging.getLogger("uvicorn.error")

# =========================================================================
# KB sub-agent prompt: reads KB files and returns analysis in Wu Tanchang's voice
# =========================================================================
KB_ANALYST_PROMPT = """你是吴探长团队的知识库分析师。

## 职责

你接收前置助手转交的用户诉求（包括品类、预算、城市、痛点等），然后：
1. 读取 `kb/METHOD.md` 和 `kb/PLAYBOOK.md` 了解分析方法论。
2. 使用 `wu-tanchang-kb` 技能检索 `kb/index.json` 查找相关案例。
3. 读取匹配的 `kb/chunks/brands/{id}/` 维度文件获取案例详情。
4. 提供高度精炼的业务建议，作为前置助手生成会议材料的内部参考。

## ⚠️ 防泄露与精简规则 (极其重要)

- **禁止泄露任何技术或文件系统信息**：输出内容中**绝对不能**出现任何文件路径（如 `kb/chunks/...`、`kb/index.json`）、文件名（如 `dimension-...`、`.md`、`.json`）、技能名称（如 `wu-tanchang-kb`）、或者关于你如何搜索、读取文件及“分块 (chunks)”的系统行为描述。
- **禁止包含任何操作指令**：不要写类似“需要读取的 chunks”、“建议你自行前往以下路径读取”、“收录状态：库内已收录”等面向系统的提示。
- **字数严格限制**：输出的总字数**必须严格控制在 300 字以内**。探讨方向和案例介绍只保留最核心的商业逻辑和动作，禁止大段背景叙述或废话。
- **输出必须高度精炼**：避免长篇大论或虚套客套话，不输出无意义的步骤信息，直接切入核心商业洞察与案例参考。

## 输出内容

结合用户信息和知识库，仅输出以下两个部分：

1. **吴探长可能探讨的方向** (2-3 个方向)：
   - 基于用户当前阶段、预算及困惑，提炼出最关键的探讨方向（如定位卡位、产品线设计、成本控制、流量模型等），并给出简要的商业逻辑解释。
2. **可参考案例** (1-2 个最契合的库内案例)：
   - 列出匹配的品牌名称（直接写品牌名，如“荣家黄鱼面”、“肉肉大米”）。
   - 核心商业动作（该品牌具体做了什么）。
   - 核心参考价值（为什么对这位用户有参考意义）。

## 引用规则

- 引用品牌前必须检索 `kb/index.json` 确认收录。
- 仅对库内品牌写「吴探长探店 XX 时」。
- 数据必须来自 chunks 内容，不得凭空捏造。

## 语言

全部用中文。输出用自然的商业分析语言，不带任何技术痕迹，不需要 JSON 格式。
"""

# =========================================================================
# Front-end agent prompt: personality files are pre-loaded at startup
# =========================================================================
FRONTEND_SYSTEM_PROMPT_TEMPLATE = """你是一个通用智能助手。

## 你的人格

以下是你的人格定义文件内容，请在对话中严格遵守：

{persona_content}

## 基本规则

- 严格按照人格文件中的指导行事
- 你**没有**文件系统访问权限，无法直接读取任何工作区文件
- 你的工作流程：收集信息 → 调用 kb_analyst（一次）→ 产出材料 → 引导预约
- **不要做分析、不要给建议、不要做预算拆解**
- **当你生成会议准备材料后，必须先将材料以文字完整呈现给用户，再调用 `mark_material_delivered` 工具标记完成。顺序不可颠倒。**
- 材料交付后，只引导预约，不再深入探讨
"""


@tool
def mark_material_delivered() -> str:
    """先把会议准备材料的文字内容完整输出到回复中呈现给用户，然后调用此工具标记材料已完成交付。顺序不可颠倒：先呈现材料，再调此工具。

    调用此工具后，系统会知道该用户的对话已完成前置阶段。
    """
    return "material_delivered"


def _load_persona_files(workspace_path: Path) -> str:
    """Load all .md personality files from workspace root into a single string.

    Args:
        workspace_path: Path to the workspace directory.

    Returns:
        Concatenated content of all personality .md files.
    """
    # Only load the core persona files; skip kb/, memory/ directories
    persona_files = ["IDENTITY.md", "SOUL.md", "AGENTS.md", "USER.md"]
    parts: list[str] = []
    for name in persona_files:
        md_file = workspace_path / name
        try:
            content = md_file.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"=== {md_file.name} ===\n{content}")
        except OSError:
            _logger.warning("[Agent] Could not read persona file: %s", md_file)
    return "\n\n".join(parts)


def create_agent(
    *,
    backend_root: Path,
    provider: str = "qwen",
    agent_config: WuAgentConfig | None = None,
) -> tuple[object, Path]:
    """Create the front-end agent with a KB-backed sub-agent.

    Args:
        backend_root: Filesystem backend root (deployed workspace).
        provider: LLM provider name.
        agent_config: Optional resolved agent config for per-agent settings.

    Returns:
        Tuple of (agent graph, checkpoints path).
    """
    effective_provider = agent_config.provider if agent_config else provider
    effective_max_tokens = agent_config.max_tokens if agent_config else 800

    model = create_model(
        provider=effective_provider,
        model_name_suffix="MAIN_AGENT_MODEL",
        log_prefix="[Agent]",
        max_tokens=effective_max_tokens,
        model_name_override=agent_config.model if agent_config else None,
    )

    runtime_dir = default_runtime_dir()
    checkpoints_path = runtime_dir / "checkpoints.pkl"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # Filesystem backend for workspace access
    backend = FilesystemBackend(root_dir=str(backend_root), virtual_mode=True)

    # Pre-load persona files from workspace root into system prompt
    persona_content = _load_persona_files(backend_root)
    _logger.info("[Agent] Loaded %d persona file(s)", persona_content.count("===") if persona_content else 0)

    # ------------------------------------------------------------------
    # Sub-agent: KB analyst with full KB access
    # ------------------------------------------------------------------
    kb_subagent = {
        "name": "kb_analyst",
        "description": "检索吴探长知识库，查找餐饮案例、商业分析和参考数据",
        "model": model,
        "tools": [],
        "system_prompt": KB_ANALYST_PROMPT,
        "middleware": [
            AccountantMiddleware(max_tool_calls=6),
            SkillsMiddleware(backend=backend, sources=["/skills/"]),
            FilesystemMiddleware(backend=backend),
        ],
    }

    # ------------------------------------------------------------------
    # Front-end agent middleware: NO FilesystemMiddleware
    # Personality is pre-loaded in system prompt above.
    # ------------------------------------------------------------------
    middleware = [
        AccountantMiddleware(max_tool_calls=20),
        LanguageDetectionMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[kb_subagent],
        ),
        PromptLoggingMiddleware(),
    ]

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    system_prompt = FRONTEND_SYSTEM_PROMPT_TEMPLATE.format(persona_content=persona_content)

    agent = create_deep_agent(
        model=model,
        tools=[mark_material_delivered],
        backend=backend,
        checkpointer=checkpointer,
        subagents=[],
        middleware=middleware,
        system_prompt=system_prompt,
    )

    _logger.info(
        "[Agent] Created (backend=%s, checkpoints=%s, provider=%s, model=%s)",
        backend_root,
        checkpoints_path,
        effective_provider,
        agent_config.model if agent_config else "default",
    )
    return agent, checkpoints_path
