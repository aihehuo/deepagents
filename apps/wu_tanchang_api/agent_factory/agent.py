"""General-purpose agent factory for Wu Tanchang API.

The agent is general-purpose. Its persona and behavior are defined by
markdown files in the workspace. The agent reads these files via its
filesystem tools to understand its role.
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

from apps.wu_tanchang_api.agent_factory.model_builder import create_model
from apps.wu_tanchang_api.agent_factory.utils import default_runtime_dir
from apps.wu_tanchang_api.checkpointer import DiskBackedInMemorySaver
from apps.wu_tanchang_api.config import WuAgentConfig

_logger = logging.getLogger("uvicorn.error")

# Minimal system prompt: tell the agent to read persona files from workspace
AGENT_SYSTEM_PROMPT = """你是一个通用智能助手。

## 重要：首先阅读你的人格文件

在开始任何对话之前，你必须先使用 `read_file` 工具阅读工作区中的人格定义文件。
这些 markdown 文件定义了你的角色、行为准则和对话风格。

**步骤：**
1. 使用 `ls` 查看工作区目录结构
2. 找到并阅读所有 `.md` 人格文件（如 `intake/` 目录下的文件）
3. 根据人格文件中的定义来塑造你的行为和回复风格

## 基本规则

- 严格按照人格文件中的指导行事
- 可以使用工作区中的知识库（`kb/` 目录）和技能（`skills/` 目录）
- 保持对话自然流畅
"""


def create_agent(
    *,
    backend_root: Path,
    provider: str = "qwen",
    agent_config: WuAgentConfig | None = None,
) -> tuple[object, Path]:
    """Create a general-purpose agent that reads persona from workspace.

    Args:
        backend_root: Filesystem backend root (deployed workspace).
        provider: LLM provider name.
        agent_config: Optional resolved agent config for per-agent settings.

    Returns:
        Tuple of (agent graph, checkpoints path).
    """
    effective_provider = agent_config.provider if agent_config else provider
    effective_max_tokens = agent_config.intake_max_tokens if agent_config else 800

    model = create_model(
        provider=effective_provider,
        model_name_suffix="MAIN_AGENT_MODEL",
        log_prefix="[Agent]",
        max_tokens=effective_max_tokens,
        model_name_override=agent_config.intake_model if agent_config else None,
    )

    runtime_dir = default_runtime_dir()
    checkpoints_path = runtime_dir / "checkpoints.pkl"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # Filesystem backend for workspace access
    backend = FilesystemBackend(root_dir=str(backend_root), virtual_mode=True)

    middleware = [
        AccountantMiddleware(max_tool_calls=20),
        LanguageDetectionMiddleware(),
        FilesystemMiddleware(backend=backend),
        SkillsMiddleware(backend=backend, sources=["/skills/"]),
        PromptLoggingMiddleware(),
    ]

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        checkpointer=checkpointer,
        subagents=[],
        middleware=middleware,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    _logger.info(
        "[Agent] Created (backend=%s, checkpoints=%s, provider=%s, model=%s)",
        backend_root,
        checkpoints_path,
        effective_provider,
        agent_config.intake_model if agent_config else "default",
    )
    return agent, checkpoints_path
