from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.aihehuo import AihehuoMiddleware
from deepagents.middleware.artifacts import ArtifactsMiddleware
from deepagents.middleware.asset_upload import AssetUploadMiddleware
from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware
from deepagents.middleware.business_idea_tracker import BusinessIdeaTrackerMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents.middleware.routing import (
    SubagentRoutingMiddleware,
    SubagentRouteRule,
    _looks_like_aihehuo_search_task,
    _looks_like_coding_task,
)
from deepagents.model_config import parse_model_config
from deepagents.subagent_presets import (
    build_aihehuo_subagent_from_env,
    build_coder_subagent_from_env,
)
from collections.abc import Awaitable, Callable

from deepagents_cli.skills.middleware import SkillsMiddleware, SkillsState, SkillsStateUpdate
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langgraph.runtime import Runtime

from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


def _copy_example_skills_if_missing(*, dest_skills_dir: Path) -> None:
    """Copy deepagents-cli packaged example skills into dest_skills_dir (no overwrite)."""
    # deepagents_cli/... -> deepagents-cli root -> examples/skills
    # Use find_spec to avoid importing the full CLI package (which may pull optional deps).
    import importlib.util

    spec = importlib.util.find_spec("deepagents_cli")
    if spec is None or not spec.origin:
        return
    cli_root = Path(spec.origin).resolve().parent.parent
    src = cli_root / "examples" / "skills"
    if not src.exists():
        return

    dest_skills_dir.mkdir(parents=True, exist_ok=True)
    for skill_dir in sorted(src.iterdir()):
        if not skill_dir.is_dir():
            continue
        if not (skill_dir / "SKILL.md").exists():
            continue
        dest = dest_skills_dir / skill_dir.name
        if dest.exists():
            continue
        shutil.copytree(skill_dir, dest)


def _mask_sensitive_value(value: str | None, show_chars: int = 8) -> str:
    """Mask a sensitive value for logging (show first N chars and last 4 chars)."""
    if not value:
        return "(not set)"
    if len(value) <= show_chars + 4:
        return "***"  # Too short to mask meaningfully
    return f"{value[:show_chars]}...{value[-4:]}"


def _mask_url(url: str | None) -> str:
    """Mask URL for logging (show full URL as it's less sensitive than API keys)."""
    if not url:
        return "(not set)"
    # For URLs, show the full URL since domain/path is not sensitive
    # Only mask query parameters if present
    if "?" in url:
        base_url, query = url.split("?", 1)
        return f"{base_url}?***"
    return url


def create_business_cofounder_agent(*, agent_id: str, provider: str = "qwen") -> tuple[object, Path]:
    """Create the Business Co-Founder deep agent (shared across users; state isolated by thread_id).

    Args:
        agent_id: Identifier for the agent
        provider: Model provider to use ("qwen" or "deepseek", default: "qwen")

    Returns:
        (agent_graph, checkpoints_path)
    """
    # Model configuration using new provider-specific design:
    # - supported_model_providers: comma-separated list (e.g., "deepseek,qwen")
    # - Provider-specific: [PROVIDER]_BASE_URL, [PROVIDER]_API_KEY (e.g., QWEN_BASE_URL, DEEPSEEK_API_KEY)
    # - Model name: [PROVIDER]_MAIN_AGENT_MODEL (e.g., QWEN_MAIN_AGENT_MODEL="qwen-plus")
    # - Shared config: MODEL_API_MAX_TOKENS, MODEL_API_TEMPERATURE, MODEL_API_TIMEOUT_S
    
    model_config = parse_model_config(
        provider=provider,
        model_name_suffix="MAIN_AGENT_MODEL",
        default_provider=provider,
    )

    # Log model configuration during initialization
    _logger.info("[ModelConfig] Model provider configuration:")
    _logger.info("  Provider: %s", model_config.provider)
    _logger.info("  Model: %s", model_config.model)
    _logger.info("  Base URL: %s", _mask_url(model_config.base_url))
    _logger.info("  API Key: %s", _mask_sensitive_value(model_config.api_key))
    _logger.info("  Max Tokens: %s", model_config.max_tokens)
    _logger.info("  Timeout: %ss", model_config.timeout_s)
    if model_config.temperature is not None:
        _logger.info("  Temperature: %s", model_config.temperature)

    # Create model based on provider
    if model_config.provider == "qwen":
        from langchain_openai import ChatOpenAI  # lazy import (avoid import-time side effects in tests)
        
        model_kwargs: dict[str, object] = {
            "model": model_config.model,
            "max_tokens": model_config.max_tokens,
            "timeout": model_config.timeout_s,
        }
        if model_config.temperature is not None:
            model_kwargs["temperature"] = model_config.temperature
        if model_config.base_url:
            model_kwargs["base_url"] = model_config.base_url
        if model_config.api_key:
            model_kwargs["api_key"] = model_config.api_key
        
        model = ChatOpenAI(**model_kwargs)
    else:
        # DeepSeek / Anthropic-compatible proxy
        model_kwargs: dict[str, object] = {
            "model": model_config.model,
            "max_tokens": model_config.max_tokens,
            "timeout": model_config.timeout_s,
        }
        if model_config.temperature is not None:
            model_kwargs["temperature"] = model_config.temperature
        if model_config.base_url:
            model_kwargs["base_url"] = model_config.base_url
        if model_config.api_key:
            model_kwargs["api_key"] = model_config.api_key
        
        model = ChatAnthropic(**model_kwargs)

    base_dir = Path.home() / ".deepagents" / "business_cofounder_api"
    skills_dir = base_dir / "skills"
    checkpoints_path = base_dir / "checkpoints.pkl"
    agent_md_path = base_dir / "agent.md"
    docs_dir = base_dir / "docs"

    base_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

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
    memory_prefix = f"<agent_md>\\n{agent_md}\\n</agent_md>\\n\\n" if agent_md else ""

    _copy_example_skills_if_missing(dest_skills_dir=skills_dir)

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    # Use virtual_mode=True for security (sandbox to base_dir)
    # This prevents path traversal and ensures all file operations stay within base_dir.
    # The agent can write anywhere within base_dir using virtual paths (e.g., /docs/, /skills/, etc.)
    # Since skills_dir and docs_dir are both under base_dir, they're accessible via /skills/ and /docs/
    backend = FilesystemBackend(root_dir=str(base_dir), virtual_mode=True)

    # Subagents use the same provider as the main agent, but with their own model names
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
    
    # Wrapper to convert absolute skill paths to virtual paths
    class VirtualPathSkillsMiddleware(AgentMiddleware):
        """Wrapper that converts absolute skill paths to virtual paths for virtual_mode backend."""
        
        state_schema = SkillsState
        
        def __init__(self, base_middleware: SkillsMiddleware, skills_dir: Path):
            self.base_middleware = base_middleware
            self.skills_dir = Path(skills_dir).expanduser().resolve()
            
            # Discover and log skills during initialization
            from deepagents_cli.skills.load import list_skills
            _logger.info("[VirtualPathSkillsMiddleware] Initializing...")
            _logger.info("  Skills directory: %s", self.skills_dir)
            
            # Load skills to discover what's available
            skills = list_skills(
                user_skills_dir=self.skills_dir,
                project_skills_dir=None,
            )
            
            if skills:
                _logger.info("[VirtualPathSkillsMiddleware] Discovered %d skill(s):", len(skills))
                for skill in skills:
                    # Convert to virtual path for display
                    virtual_path = self._convert_skill_path_to_virtual(skill["path"])
                    _logger.info("  - %s: %s", skill['name'], skill['description'])
                    _logger.info("    → Virtual path: %s", virtual_path)
            else:
                _logger.info("[VirtualPathSkillsMiddleware] No skills discovered in %s", self.skills_dir)
            _logger.info("[VirtualPathSkillsMiddleware] Initialization complete.")
        
        def _convert_skill_path_to_virtual(self, absolute_path: str) -> str:
            """Convert absolute skill path to virtual path (/skills/{skill_name}/SKILL.md)."""
            try:
                skill_path = Path(absolute_path).resolve()
                # Check if path is within skills_dir
                if skill_path.is_relative_to(self.skills_dir):
                    # Get relative path from skills_dir
                    relative = skill_path.relative_to(self.skills_dir)
                    # Convert to virtual path: /skills/{skill_name}/SKILL.md
                    return f"/skills/{relative}"
                # If not in skills_dir, return as-is (shouldn't happen, but fail safe)
                return absolute_path
            except (ValueError, OSError):
                # If path resolution fails, return as-is
                return absolute_path
        
        def before_agent(self, state: SkillsState, runtime: Runtime) -> SkillsStateUpdate | None:
            """Load skills and convert paths to virtual paths."""
            result = self.base_middleware.before_agent(state, runtime)
            if result and "skills_metadata" in result:
                # Convert all skill paths to virtual paths
                converted_skills = []
                for skill in result["skills_metadata"]:
                    converted_skill = dict(skill)
                    converted_skill["path"] = self._convert_skill_path_to_virtual(skill["path"])
                    converted_skills.append(converted_skill)
                
                return SkillsStateUpdate(skills_metadata=converted_skills)
            return result
        
        def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
            """Delegate to base middleware (paths already converted in before_agent)."""
            return self.base_middleware.wrap_model_call(request, handler)
        
        async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
            """Delegate to base middleware (paths already converted in before_agent)."""
            return await self.base_middleware.awrap_model_call(request, handler)
    
    virtual_skills_middleware = VirtualPathSkillsMiddleware(base_skills_middleware, skills_dir)
    
    middleware = [
        LanguageDetectionMiddleware(),
        BusinessIdeaTrackerMiddleware(),
        BusinessIdeaDevelopmentMiddleware(strict_todo_sync=True),
        virtual_skills_middleware,
        AihehuoMiddleware(),  # Provides aihehuo_search_members and aihehuo_search_ideas tools
        AssetUploadMiddleware(
            backend_root=str(base_dir),  # Backend root is base_dir, not cwd
            docs_dir=str(docs_dir),  # Preferred location for documents (agent can write anywhere in base_dir)
        ),  # Provides upload_asset tool
        ArtifactsMiddleware(),  # Provides add_artifact tool to track uploaded artifact URLs
    ]
    
    # Combine routing rules into a single SubagentRoutingMiddleware to avoid duplicate middleware instances
    routing_rules = []
    if coder_subagent is not None:
        routing_rules.append(
            SubagentRouteRule(
                name="code_html_to_coder",
                subagent_type="coder",
                should_route=lambda text, _req: _looks_like_coding_task(text),
                instruction="""## Routing hint (code/HTML)

If the user is asking for **code** (including **HTML/CSS/JS**, scripts, Dockerfiles, config changes, refactors, or debugging), you **MUST** delegate the heavy lifting to the `coder` subagent (unless it is truly trivial, e.g. a <5-line snippet with no repo edits):
- Use the `task` tool with `subagent_type="coder"`.
- In the task description, include: the goal, relevant constraints, file paths, and the exact expected output.
- The subagent can use the same tools (files/execute/etc.) to implement changes; then you summarize results to the user.
""",
            )
        )
    if aihehuo_subagent is not None:
        routing_rules.append(
            SubagentRouteRule(
                name="aihehuo_search_to_aihehuo",
                subagent_type="aihehuo",
                should_route=lambda text, _req: _looks_like_aihehuo_search_task(text),
                instruction="""## Routing hint (AI He Huo search)

If the user is asking to **find co-founders, partners, investors, or search the AI He Huo (爱合伙) platform**, you **MUST** delegate the search to the `aihehuo` subagent:
- Use the `task` tool with `subagent_type="aihehuo"`.
- In the task description, include:
  - The business idea or requirements
  - What types of people are needed (technical co-founder, business co-founder, investors, domain experts)
  - Any specific criteria or constraints
- The subagent has specialized AI He Huo search tools and will perform multiple targeted searches.
- After the subagent completes the search, summarize the findings and recommendations for the user.
""",
            )
        )
    
    if routing_rules:
        middleware.append(SubagentRoutingMiddleware(rules=routing_rules))

    agent = create_deep_agent(
        model=model,
        backend=backend,
        checkpointer=checkpointer,
        subagents=subagents,
        middleware=middleware,
        system_prompt=memory_prefix + "You are a business co-founder assistant helping entrepreneurs develop their startup ideas.",
    )

    return agent, checkpoints_path


