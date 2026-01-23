"""Business cofounder agent creation (legacy single-agent mode)."""

from __future__ import annotations

import logging
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.accountant import AccountantMiddleware
from deepagents.middleware.aihehuo import AihehuoMiddleware
from deepagents.middleware.artifacts import ArtifactsMiddleware
from deepagents.middleware.asset_upload import AssetUploadMiddleware
from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware
from deepagents.middleware.business_idea_tracker import BusinessIdeaTrackerMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents.subagent_presets import (
    build_aihehuo_subagent_from_env,
    build_coder_subagent_from_env,
)
from deepagents_cli.skills.middleware import SkillsMiddleware

from apps.business_cofounder_api.agent_factory.memory import ApiMemoryMiddleware
from apps.business_cofounder_api.agent_factory.middleware_builder import (
    VirtualPathSkillsMiddleware,
    build_routing_middleware,
)
from apps.business_cofounder_api.agent_factory.model_builder import create_model
from apps.business_cofounder_api.agent_factory.utils import copy_example_skills_if_missing
from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


def create_business_cofounder_agent(
    *, 
    agent_id: str, 
    provider: str = "qwen",
    user_id: str | None = None,  # Deprecated: kept for backward compatibility, not used (middleware handles it)
    conversation_id: str | None = None,  # Deprecated: kept for backward compatibility, not used (middleware handles it)
) -> tuple[object, Path]:
    """Create the Business Co-Founder deep agent (shared across users; state isolated by thread_id).

    Memory paths are injected dynamically via ApiMemoryMiddleware based on user_id and
    conversation_id from request metadata. The user_id and conversation_id parameters
    are deprecated and kept only for backward compatibility.

    Args:
        agent_id: Identifier for the agent
        provider: Model provider to use ("qwen" or "deepseek", default: "qwen")
        user_id: Deprecated - not used (middleware extracts from metadata)
        conversation_id: Deprecated - not used (middleware extracts from thread_id)

    Returns:
        (agent_graph, checkpoints_path)
    """
    # Create model
    model = create_model(
        provider=provider,
        model_name_suffix="MAIN_AGENT_MODEL",
        log_prefix="[ModelConfig]",
        set_max_input_tokens=True,
    )

    base_dir = Path.home() / ".deepagents" / "business_cofounder_api"
    skills_dir = base_dir / "skills"
    checkpoints_path = base_dir / "checkpoints.pkl"
    agent_md_path = base_dir / "agent.md"
    docs_dir = base_dir / "docs"

    base_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: Memory directories are created on-demand by the middleware when needed

    # CLI-created agents typically have ~/.deepagents/<agent_id>/agent.md injected into prompts.
    # Our API runs without the CLI, so we support an API-local agent.md at:
    #   ~/.deepagents/business_cofounder_api/agent.md
    # If missing, we create a small default template.
    if not agent_md_path.exists():
        agent_md_path.write_text(
            """# Business Co-Founder Agent (API)

## Operating mode
- You are running behind an HTTP API.
- You must follow the BusinessIdeaTrackerMiddleware sequential unlock rules.
- When you complete a milestone, call the corresponding mark_* tool.

## Output style
- Be concise, structured, and action-oriented.
- Prefer bullet points and clear section headers.

## File outputs
- When asked to generate a document (HTML/Markdown/etc.), you MUST use the filesystem tools.
- Save all generated documents to the docs folder:
  - `~/.deepagents/business_cofounder_api/docs/` (this path exists in the API runtime)
""",
            encoding="utf-8",
        )

    agent_md = agent_md_path.read_text(encoding="utf-8").strip()
    global_memory_prefix = f"<agent_md>\\n{agent_md}\\n</agent_md>\\n\\n" if agent_md else ""

    copy_example_skills_if_missing(dest_skills_dir=skills_dir)

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    # Use virtual_mode=True for security (sandbox to base_dir)
    # This prevents path traversal and ensures all file operations stay within base_dir.
    # The agent can write anywhere within base_dir using virtual paths (e.g., /docs/, /skills/, etc.)
    # Since skills_dir and docs_dir are both under base_dir, they're accessible via /skills/ and /docs/
    backend = FilesystemBackend(root_dir=str(base_dir), virtual_mode=True)

    # Subagents use the same provider as the main agent, but with their own model names
    # Get provider from model config (we need to parse it again to get provider)
    from deepagents.model_config import parse_model_config
    model_config = parse_model_config(
        provider=provider,
        model_name_suffix="MAIN_AGENT_MODEL",
        default_provider=provider,
    )
    
    coder_subagent = build_coder_subagent_from_env(
        tools=None, name="coder", provider=model_config.provider
    )
    aihehuo_subagent = build_aihehuo_subagent_from_env(
        tools=None, name="aihehuo", provider=model_config.provider
    )
    
    subagents = []
    if coder_subagent is not None:
        subagents.append(coder_subagent)
    if aihehuo_subagent is not None:
        subagents.append(aihehuo_subagent)
    
    # Create SkillsMiddleware with path conversion wrapper
    # This converts absolute skill paths to virtual paths (/skills/{skill_name}/SKILL.md)
    # so they work with virtual_mode=True backend (rooted at base_dir)
    base_skills_middleware = SkillsMiddleware(
        skills_dir=skills_dir,
        assistant_id=agent_id,
        project_skills_dir=None,
    )
    
    virtual_skills_middleware = VirtualPathSkillsMiddleware(base_skills_middleware, skills_dir)
    
    # Create API memory middleware to inject user/conversation memory paths dynamically
    api_memory_middleware = ApiMemoryMiddleware(base_dir=base_dir)
    
    middleware = [
        AccountantMiddleware(),  # Enforces tool call limit (default: 25) and tracks token usage
        LanguageDetectionMiddleware(),
        BusinessIdeaTrackerMiddleware(),
        BusinessIdeaDevelopmentMiddleware(strict_todo_sync=True),
        virtual_skills_middleware,
        api_memory_middleware,  # Injects memory paths based on user_id/conversation_id from metadata
        AihehuoMiddleware(),  # Provides aihehuo_search_members and aihehuo_search_ideas tools
        AssetUploadMiddleware(
            backend_root=str(base_dir),  # Backend root is base_dir, not cwd
            docs_dir=str(docs_dir),  # Preferred location for documents (agent can write anywhere in base_dir)
        ),  # Provides upload_asset tool
        ArtifactsMiddleware(),  # Provides add_artifact tool to track uploaded artifact URLs
    ]
    
    # Add routing middleware if we have subagents
    routing_middleware = build_routing_middleware(
        coder_subagent=coder_subagent,
        aihehuo_subagent=aihehuo_subagent,
    )
    if routing_middleware:
        middleware.append(routing_middleware)

    # Build system prompt with global memory (user/conversation memory injected dynamically by middleware)
    system_prompt_parts = []
    if global_memory_prefix:
        system_prompt_parts.append(global_memory_prefix)
    system_prompt_parts.append("You are a business co-founder assistant helping entrepreneurs develop their startup ideas.")
    
    system_prompt = "\n".join(system_prompt_parts)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        checkpointer=checkpointer,
        subagents=subagents,
        middleware=middleware,
        system_prompt=system_prompt,
    )

    return agent, checkpoints_path
