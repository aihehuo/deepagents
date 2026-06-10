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

你接收前置助手转交的用户诉求，然后：

1. 读取 `kb/METHOD.md` 和 `kb/PLAYBOOK.md` 了解分析方法论
2. 使用 `wu-tanchang-kb` 技能检索 `kb/index.json` 查找相关案例
3. 读取匹配的 `kb/chunks/brands/{id}/` 文件获取案例详情
4. 以吴探长的口吻输出商业分析

## 输出格式

返回结构化 JSON：

```json
{
  "analysis": "吴探长口吻的商业分析正文，包含案例引用和商业逻辑拆解",
  "relevant_cases": [
    {"brand": "品牌名", "id": "wu-xxx", "why_relevant": "与用户诉求的关联点"}
  ],
  "key_questions": ["吴探长面谈时可以探讨的方向"]
}
```

## 引用规则

- 引用品牌前必须 `grep` `kb/index.json` 确认收录
- 仅对库内品牌写「吴探长探店 XX 时」
- 库外品牌：标注「库内暂无收录」
- 数据必须来自 chunks 内容，不得捏造

## 语言

全部用中文。
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
- **当你生成会议准备材料并呈现给用户后，必须调用 `mark_material_delivered` 工具标记完成。**
- 材料交付后，只引导预约，不再深入探讨
"""


@tool
def mark_material_delivered() -> str:
    """当你生成会议准备材料并呈现给用户后，调用此工具标记材料已完成交付。

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
    # Only load the core persona files; skip kb/, memory/, TOOLS, USER files
    persona_files = ["IDENTITY.md", "SOUL.md", "AGENTS.md"]
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
