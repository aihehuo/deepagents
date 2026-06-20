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
import threading
from datetime import UTC
from pathlib import Path
from typing import Any

from apps.wu_tanchang_api.agent_factory.kb_search import (
    get_note_content,
    kb_semantic_search,
)
from apps.wu_tanchang_api.agent_factory.model_builder import create_model
from apps.wu_tanchang_api.agent_factory.utils import (
    default_runtime_dir,
    get_workspace_owner_name,
)
from apps.wu_tanchang_api.checkpointer import DiskBackedInMemorySaver
from apps.wu_tanchang_api.config import WuAgentConfig
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.accountant import AccountantMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents.middleware.prompt_logging import PromptLoggingMiddleware

_logger = logging.getLogger("uvicorn.error")

_ACTIVE_AGENTS_LOCK = threading.Lock()
_ACTIVE_AGENTS: dict[str, Any] = {}


def register_active_agent(thread_id: str, agent: Any) -> None:
    """Register an active agent instance by thread ID."""
    with _ACTIVE_AGENTS_LOCK:
        _ACTIVE_AGENTS[thread_id] = agent


def unregister_active_agent(thread_id: str) -> None:
    """Unregister an active agent instance by thread ID."""
    with _ACTIVE_AGENTS_LOCK:
        _ACTIVE_AGENTS.pop(thread_id, None)


def get_active_agent(thread_id: str) -> Any | None:
    """Retrieve an active agent instance by thread ID."""
    with _ACTIVE_AGENTS_LOCK:
        return _ACTIVE_AGENTS.get(thread_id)


# =========================================================================
# KB sub-agent prompt: reads KB files and returns analysis in Wu Tanchang's voice
# =========================================================================
KB_ANALYST_PROMPT = """你是协助分析知识库的专业分析师。

## 职责

你接收前置助手转交的用户诉求（包括品类、预算、城市、痛点等），然后：
1. 读取 `kb/METHOD.md` 和 `kb/PLAYBOOK.md` 了解分析方法论。
2. 用 `kb_semantic_search` 工具做语义召回（query=用户诉求，k=5）。
3. 对返回的 note_id，用 `wu-tanchang-kb` 技能 grep `kb/index.json` 验证收录——
    只有 grep 命中的案例/品牌才能引用。
4. 对验证通过 of 1-2 个最契合条目，使用 `get_note_content` 工具读取其完整笔记内容。
5. 提供高度精炼的业务建议，作为前置助手生成会议材料的内部参考。

## 检索原则

- `kb_semantic_search` 是**召回**工具，返回的 note_id 是候选；它**不能**单独作为引用依据。
- `kb/index.json` 的 grep 是**收录硬护栏**——语义检索命中但 index.json 没收录的案例，必须当作未收录处理。
- 同一轮里 `kb_semantic_search` 最多调用 2 次（首次结果偏离时换 query 重试一次）。

## ⚠️ 防泄露与精简规则 (极其重要)

- **禁止泄露任何技术或文件系统信息**：输出内容中**绝对不能**出现任何文件路径（如 `kb/...`、`kb/index.json`）、文件名（如 `.md`、`.json`）、技能名称（如 `wu-tanchang-kb`）、或者关于你如何搜索、调用工具及“读取数据库”的系统行为描述。
- **禁止包含任何操作指令**：不要写类似“需要读取的 chunks”、“建议你自行前往以下路径读取”、“收录状态：库内已收录”等面向系统的提示。
- **字数严格限制**：输出的总字数**必须严格控制在 300 字以内**。探讨方向和案例介绍只保留最核心的商业逻辑 and 动作，禁止大段背景叙述或废话。
- **输出必须高度精炼**：避免长篇大论或极简客套话，不输出无意义的步骤信息，直接切入核心商业洞察与案例参考。

## 输出内容

结合用户信息和知识库，仅输出以下两个部分：

1. **可能探讨的方向** (2-3 个方向)：
   - 基于用户当前阶段及困惑，提炼出最关键的探讨方向，并给出简要的商业逻辑解释。
2. **可参考案例/方法论** (1-2 个最契合的库内案例/方法论)：
   - 列出匹配的品牌或方法论名称。
   - 核心商业动作/去风险要点（该案例/方法论具体指出了什么）。
   - 核心参考价值（为什么对这位用户有参考意义）。

## 引用规则

- 引用案例前必须检索 `kb/index.json` 确认收录。
- 仅对库内条目引用，且数据必须来自真实文件内容，不得凭空捏造。

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
- **当你生成会议准备材料后，必须先将材料以文字完整呈现给用户，然后调用 `save_meeting_prep` 工具保存材料，最后调用 `mark_material_delivered` 工具标记完成。顺序不可颠倒。**
- 材料交付后，只引导预约，不再深入探讨
"""


OWNER_SYSTEM_PROMPT_TEMPLATE = """你是一个通用智能助手。

## 你的人格

以下是你的人格定义文件内容，请在对话中严格遵守：

{persona_content}

## 基本规则

- 严格按照人格文件中的指导行事，你是{owner_name}的专属AI数据决策助理。
- 你的工作是帮助{owner_name}掌握客户进展、统计数据、分析客户需求。
- **请务必在需要时调用提供的工具（get_consultation_stats, list_recent_clients, get_client_detail）来获取和汇总数据。不要凭空虚构任何客户统计或明细信息！**
- 回复{owner_name}时要专业、高效、结构化，通常使用中文。
"""


@tool
def mark_material_delivered() -> str:
    """先把会议准备材料的文字内容完整输出到回复中呈现给用户，然后调用此工具标记材料已完成交付。顺序不可颠倒：先呈现材料，再调此工具。

    调用此工具后，系统会知道该用户的对话已完成前置阶段。
    """
    return "material_delivered"


@tool(parse_docstring=True)
def save_meeting_prep(
    body: str,
    user_a_id: int | None = None,
    user_b_id: int | None = None,
    *,
    config: RunnableConfig,
) -> str:
    """保存生成的两位用户之间的会面准备材料。
    当会议材料（Markdown）生成完毕后，必须在调用 mark_material_delivered 之前调用此工具进行保存。

    Args:
        body: 生成的会面准备材料正文，支持 Markdown 格式。
        user_a_id: 第一位用户的 ID。如果不传，将自动从上下文 metadata 中读取。
        user_b_id: 第二位用户的 ID。如果不传，将自动从上下文 metadata 中读取。
    """
    import os
    from datetime import datetime

    import requests

    metadata = config.get("metadata") or {}

    # Extract user IDs
    effective_user_a = user_a_id or metadata.get("user_a_id") or metadata.get("user_id")
    effective_user_b = (
        user_b_id or metadata.get("user_b_id") or metadata.get("calendar_id")
    )

    if not effective_user_a or not effective_user_b:
        _logger.warning(
            "[AgentTool] save_meeting_prep missing user_a_id or user_b_id in metadata/args"
        )
        return "保存失败：未在上下文或参数中找到 user_a_id 或 user_b_id"

    # Enforce body size limit (S4)
    if len(body) > 50000:
        return "保存失败：准备材料正文超长（最大限制为 50KB）"

    # Determine API endpoint URL
    callback_url = metadata.get("callback_url")
    if callback_url:
        if "/wu_tanchang_callbacks/" in callback_url:
            base_url = (
                callback_url.split("/wu_tanchang_callbacks/")[0]
                + "/wu_tanchang_callbacks"
            )
        else:
            base_url = callback_url.rstrip("/")
        api_url = f"{base_url}/meeting_preps"
    else:
        raw_allowed = os.environ.get("WU_CALLBACK_ALLOWED_BASE_URLS")
        base_urls = (
            [base.strip() for base in raw_allowed.split(",") if base.strip()]
            if raw_allowed
            else [
                "http://host.docker.internal:3001/wu_tanchang_callbacks/",
                "http://localhost:3001/wu_tanchang_callbacks/",
                "http://127.0.0.1:3001/wu_tanchang_callbacks/",
            ]
        )
        base_url = base_urls[0].rstrip("/")
        api_url = f"{base_url}/meeting_preps"

    # Validate callback URL against allowed list (S4)
    from apps.wu_tanchang_api.app.callbacks import (
        CallbackUrlError,
        validate_callback_url,
    )

    try:
        validate_callback_url(api_url)
    except CallbackUrlError as exc:
        _logger.warning(
            "[AgentTool] Unauthorized callback API url: %s, error: %s", api_url, exc
        )
        return f"保存失败：出网接口地址未被授权 ({exc})"

    # Fetch token
    token = os.environ.get("WU_TANCHANG_CALLBACK_TOKEN") or os.environ.get(
        "WU_CALLBACK_AGENT_KEY"
    )
    headers = {"X-Agent-Key": token or "", "Content-Type": "application/json"}

    # Validate integer IDs (S6)
    try:
        user_a_int = int(effective_user_a)
        user_b_int = int(effective_user_b)
    except (ValueError, TypeError):
        _logger.warning(
            "[AgentTool] Invalid user ID format: user_a=%s, user_b=%s",
            effective_user_a,
            effective_user_b,
        )
        return "保存失败：用户 ID 格式不正确（必须为整数）"

    # Prepare payload
    author = metadata.get("agent_name")
    if not author:
        # Determine default author name based on workspace
        from apps.wu_tanchang_api.agent_factory.agent import get_active_agent
        from apps.wu_tanchang_api.agent_factory.utils import get_workspace_agent_id

        config_configurable = config.get("configurable") or {}
        tid = config_configurable.get("thread_id")
        active_agent = get_active_agent(tid) if tid else None
        workspace_name = (
            getattr(active_agent, "workspace_name", "workspace")
            if active_agent
            else "workspace"
        )

        backend_root = Path(__file__).resolve().parent.parent
        workspace_path = backend_root / workspace_name
        agent_id = get_workspace_agent_id(workspace_path)

        if "yc01" in agent_id:
            author = "yc"
        elif "andy01" in agent_id:
            author = "wu_tanchang"
        else:
            author = agent_id.replace("_owner", "")

    payload = {
        "user_a_id": user_a_int,
        "user_b_id": user_b_int,
        "author": author,
        "body": body,
        "prepared_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }

    try:
        _logger.info(
            "[AgentTool] Calling create_meeting_prep: url=%s, user_a=%s, user_b=%s, author=%s",
            api_url,
            effective_user_a,
            effective_user_b,
            author,
        )
        response = requests.post(api_url, json=payload, headers=headers, timeout=15)
        if response.status_code == 201:
            _logger.info("[AgentTool] Meeting prep saved successfully.")
            return "meeting_prep_saved"
        _logger.error(
            "[AgentTool] Failed to save meeting prep: status_code=%s, response=%s",
            response.status_code,
            response.text,
        )
        return f"保存失败：后端返回状态码 {response.status_code}"
    except Exception as e:
        _logger.exception("[AgentTool] Connection error calling save_meeting_prep")
        return f"保存失败：网络或接口调用异常 {type(e).__name__}: {e!s}"


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
    app_root = Path(__file__).resolve().parent.parent
    backend = FilesystemBackend(
        root_dir=str(backend_root),
        virtual_mode=True,
        allowed_symlink_roots=[app_root],
    )

    # Pre-load persona files from workspace root into system prompt
    effective_workspace = agent_config.workspace if agent_config else "workspace"
    workspace_path = (
        backend_root / effective_workspace
        if (backend_root / effective_workspace).exists()
        else backend_root
    )
    persona_content = _load_persona_files(workspace_path)
    _logger.info(
        "[Agent] Loaded %d persona file(s) from %s",
        persona_content.count("===") if persona_content else 0,
        workspace_path,
    )

    # ------------------------------------------------------------------
    # Sub-agent: KB analyst with full KB access
    # ------------------------------------------------------------------
    from deepagents.middleware.filesystem import FilesystemPermission

    kb_skills = []
    # Load app-level default skills if they exist
    default_skills_path = workspace_path / "skills" / "default"
    if default_skills_path.exists():
        for item in default_skills_path.iterdir():
            if item.is_dir():
                kb_skills.append(f"/{effective_workspace}/skills/default/{item.name}/")
        kb_skills.append(f"/{effective_workspace}/skills/default/")

    # Load tenant-level local KB skills dynamically
    local_kb_path = workspace_path / "skills" / "local_kb"
    local_skills_path = workspace_path / "skills" / "local"
    if local_kb_path.exists():
        kb_skills.append(f"/{effective_workspace}/skills/local_kb/")
        try:
            local_kb_skills = [item.name for item in local_kb_path.iterdir() if item.is_dir()]
            _logger.info(
                "[Agent] Detected %d local KB skill(s) on disk for workspace %s: %s",
                len(local_kb_skills),
                effective_workspace,
                local_kb_skills,
            )
        except Exception as e:
            _logger.warning("[Agent] Error scanning local KB skills: %s", e)
    elif local_skills_path.exists():
        kb_skills.append(f"/{effective_workspace}/skills/local/")
        try:
            local_skills = [item.name for item in local_skills_path.iterdir() if item.is_dir()]
            _logger.info(
                "[Agent] Detected %d local skill(s) on disk (fallback) for workspace %s: %s",
                len(local_skills),
                effective_workspace,
                local_skills,
            )
        except Exception as e:
            _logger.warning("[Agent] Error scanning local fallback skills: %s", e)

    _logger.info("[Agent] Configuring kb_skills for %s: %s", effective_workspace, kb_skills)

    # Load tenant-level local Aihehuo skills dynamically
    aihehuo_skills = []
    local_aihehuo_path = workspace_path / "skills" / "local_aihehuo"
    if local_aihehuo_path.exists():
        aihehuo_skills.append(f"/{effective_workspace}/skills/local_aihehuo/")
        try:
            local_aihehuo_skills = [item.name for item in local_aihehuo_path.iterdir() if item.is_dir()]
            _logger.info(
                "[Agent] Detected %d local Aihehuo skill(s) on disk for workspace %s: %s",
                len(local_aihehuo_skills),
                effective_workspace,
                local_aihehuo_skills,
            )
        except Exception as e:
            _logger.warning("[Agent] Error scanning local Aihehuo skills: %s", e)

    _logger.info("[Agent] Configuring aihehuo_skills for %s: %s", effective_workspace, aihehuo_skills)

    # Enforce strict multi-tenant isolation:
    # 1. Format prompt to use tenant-specific path
    formatted_kb_prompt = KB_ANALYST_PROMPT.replace(
        "kb/", f"/{effective_workspace}/kb/"
    )

    # 2. Add filesystem permissions: deny all access to root /kb and deny write access to all kb/skills directories
    kb_permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=[
                "/kb/**",
            ],
            mode="deny",
        ),
        FilesystemPermission(
            operations=["write"],
            paths=[
                "/workspace*/kb/**",
                "/workspace*/skills/**",
                "/skills/**",
            ],
            mode="deny",
        ),
    ]

    kb_subagent = {
        "name": "kb_analyst",
        "description": "检索知识库，查找商业方法论、案例和参考数据",
        "model": model,
        "tools": [kb_semantic_search, get_note_content],
        "system_prompt": formatted_kb_prompt,
        "skills": kb_skills,
        "permissions": kb_permissions,
        "middleware": [
            AccountantMiddleware(max_tool_calls=6),
            PromptLoggingMiddleware(),
        ],
    }

    # ------------------------------------------------------------------
    # Determine agent configuration: owner mode vs. client front-end mode
    # ------------------------------------------------------------------
    is_owner = agent_config and agent_config.name == "owner"

    if is_owner:
        from apps.wu_tanchang_api.agent_factory.owner_tools import (
            get_client_detail,
            get_consultation_stats,
            list_recent_clients,
        )

        owner_name = get_workspace_owner_name(workspace_path)

        tools = [get_consultation_stats, list_recent_clients, get_client_detail]
        system_prompt = OWNER_SYSTEM_PROMPT_TEMPLATE.format(
            persona_content=persona_content,
            owner_name=owner_name,
        )
        middleware = [
            AccountantMiddleware(max_tool_calls=6),
            LanguageDetectionMiddleware(),
            PromptLoggingMiddleware(),
        ]
        subagents = []
    else:
        tools = [mark_material_delivered, save_meeting_prep]
        
        # Build dynamic front-end system prompt based on aihehuo_skills availability
        frontend_rules = [
            "- 严格按照人格文件中的指导行事",
            "- 你**没有**文件系统访问权限，无法直接读取任何工作区文件",
            "- 你的工作流程：收集信息 → 调用 kb_analyst（一次）→ 产出材料 → 引导预约",
        ]
        if aihehuo_skills:
            frontend_rules.append(
                "- 如果用户询问爱合伙平台的相关数据、博客、微信群、项目或常见问题等信息，你可以随时调用子代理 `aihehuo_cruncher` 来获取最新数据并回复用户。"
            )
        frontend_rules.extend([
            "- **不要做分析、不要给建议、不要做预算拆解**",
            "- **当你生成会议准备材料后，必须先将材料以文字完整呈现给用户，然后调用 `save_meeting_prep` 工具保存材料，最后调用 `mark_material_delivered` 工具标记完成。顺序不可颠倒。**",
            "- 材料交付后，只引导预约，不再深入探讨",
        ])
        
        rules_text = "\n".join(frontend_rules)
        dynamic_frontend_prompt = f"""你是一个通用智能助手。

## 你的人格

以下是你的人格定义文件内容，请在对话中严格遵守：

{{persona_content}}

## 基本规则

{rules_text}
"""
        system_prompt = dynamic_frontend_prompt.format(
            persona_content=persona_content
        )
        middleware = [
            AccountantMiddleware(max_tool_calls=20),
            LanguageDetectionMiddleware(),
            PromptLoggingMiddleware(),
        ]
        
        subagents = [kb_subagent]
        if aihehuo_skills:
            aihehuo_cruncher_prompt = """你是协助获取和分析爱合伙相关数据的专业分析助理。
你拥有调用爱合伙官方博客、常见问题（FAQS）、微信群数据、用户行为和周趋势等公开数据接口技能。
请根据主助手的要求，调用最合适的技能/工具来检索数据，并直接用中文向主代理提供结构化 and 简练的分析结果。
"""
            aihehuo_cruncher = {
                "name": "aihehuo_cruncher",
                "description": "调用爱合伙数据接口技能获取博客、问答、用户、项目、微信群等数据，进行汇总分析",
                "model": model,
                "tools": [],
                "system_prompt": aihehuo_cruncher_prompt,
                "skills": aihehuo_skills,
                "permissions": kb_permissions,
                "middleware": [
                    AccountantMiddleware(max_tool_calls=10),
                    PromptLoggingMiddleware(),
                ],
            }
            subagents.append(aihehuo_cruncher)

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    agent = create_deep_agent(
        model=model,
        tools=tools,
        backend=backend,
        checkpointer=checkpointer,
        subagents=subagents,
        middleware=middleware,
        system_prompt=system_prompt,
    )
    try:
        agent.workspace_name = effective_workspace
    except AttributeError:
        pass

    _logger.info(
        "[Agent] Created (backend=%s, checkpoints=%s, provider=%s, model=%s)",
        backend_root,
        checkpoints_path,
        effective_provider,
        agent_config.model if agent_config else "default",
    )
    return agent, checkpoints_path
