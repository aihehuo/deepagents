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
from typing import TypedDict, NotRequired, cast

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


def _get_user_memory_path(base_dir: Path, user_id: str) -> Path:
    """Get the path to user-level memory file.
    
    Args:
        base_dir: Base directory for the API (~/.deepagents/business_cofounder_api)
        user_id: User identifier
        
    Returns:
        Path to user memory file: base_dir/users/{user_id}/agent.md
    """
    user_dir = base_dir / "users" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / "agent.md"


def _get_conversation_memory_path(base_dir: Path, user_id: str, conversation_id: str) -> Path:
    """Get the path to conversation-level memory file.
    
    Args:
        base_dir: Base directory for the API (~/.deepagents/business_cofounder_api)
        user_id: User identifier
        conversation_id: Conversation identifier
        
    Returns:
        Path to conversation memory file: base_dir/users/{user_id}/conversations/{conversation_id}/agent.md
    """
    conversation_dir = base_dir / "users" / user_id / "conversations" / conversation_id
    conversation_dir.mkdir(parents=True, exist_ok=True)
    return conversation_dir / "agent.md"


def _ensure_memory_directories_exist(base_dir: Path, user_id: str | None, conversation_id: str | None) -> None:
    """Ensure memory directories exist for the given user and conversation.
    
    Args:
        base_dir: Base directory for the API
        user_id: User identifier (optional)
        conversation_id: Conversation identifier (optional)
    """
    if user_id:
        user_dir = base_dir / "users" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        if conversation_id:
            conversation_dir = user_dir / "conversations" / conversation_id
            conversation_dir.mkdir(parents=True, exist_ok=True)


class ApiMemoryState(AgentState):
    """State for API memory middleware."""
    
    user_id: NotRequired[str]
    """User identifier for memory paths."""
    
    conversation_id: NotRequired[str]
    """Conversation identifier for memory paths."""


def _build_memory_documentation(user_id: str | None, conversation_id: str | None) -> str:
    """Build memory structure documentation for the system prompt.
    
    Args:
        user_id: User identifier (optional)
        conversation_id: Conversation identifier (optional)
        
    Returns:
        Memory documentation string to include in system prompt
    """
    if not user_id:
        # No user context, return minimal documentation
        return ""
    
    # Build virtual paths (relative to base_dir, with leading /)
    user_memory_path = f"/users/{user_id}/agent.md"
    conversation_memory_path = f"/users/{user_id}/conversations/{conversation_id}/agent.md" if conversation_id else None
    
    memory_docs = f"""
## Long-term Memory

Your memory is organized in two tiers:

**User Memory**: `{user_memory_path}`
- Stores user preferences, communication style, general behavior
- Shared across ALL conversations for this user
- Update when user provides feedback or changes preferences

"""
    
    if conversation_memory_path:
        memory_docs += f"""**Conversation Memory**: `{conversation_memory_path}`
- Stores business idea context, progress, decisions for THIS conversation
- Isolated per conversation
- Update as the business idea develops

"""
    
    memory_docs += """**Accessing Memory**:
- Read: Use `read_file` tool with the memory path (e.g., `read_file '/users/{user_id}/agent.md'`)
- Update: Use `edit_file` or `write_file` tools with the memory path
- Check what exists: Use `ls` tool to list directories (e.g., `ls '/users/{user_id}'`)

**When to Read Memory**:
- At conversation start: Check both user and conversation memory if available
- When user asks about preferences: Read user memory
- When continuing a business idea: Read conversation memory
- When user references past work: Search conversation memory files

**When to Update Memory**:
- User feedback on style/behavior → Update user memory
- Business idea progress → Update conversation memory
- User preferences change → Update user memory
- Important decisions or context → Update conversation memory

**Memory File Format**:
- Memory files are Markdown (`.md`) files
- You can read and write them using standard filesystem tools
- If a memory file doesn't exist, you can create it with `write_file`

**Important Notes**:
- User memory persists across all conversations for this user
- Conversation memory is isolated per conversation
- Always use absolute virtual paths starting with `/` when accessing memory files
- The global agent.md at the API root provides default instructions, but user/conversation memory takes precedence for specific context

"""
    
    return memory_docs


class ApiMemoryMiddleware(AgentMiddleware):
    """Middleware that injects user/conversation memory paths into system prompt dynamically.
    
    This middleware reads user_id and conversation_id from request metadata and injects
    memory documentation into the system prompt so the agent knows where to find/update
    user-level and conversation-level memory.
    """
    
    state_schema = ApiMemoryState
    
    def __init__(self, base_dir: Path):
        """Initialize the API memory middleware.
        
        Args:
            base_dir: Base directory for the API (~/.deepagents/business_cofounder_api)
        """
        self.base_dir = base_dir
        # Test memory file write/read during initialization to verify filesystem permissions
        self._test_memory_file_access()
    
    def before_agent(
        self,
        state: ApiMemoryState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Extract user_id and conversation_id from runtime config metadata.
        
        Also ensures memory directories exist for the user and conversation.
        
        Args:
            state: Current agent state
            runtime: Runtime context with config
            
        Returns:
            Updated state with user_id and conversation_id if available
        """
        updates: dict[str, Any] = {}
        
        # Extract metadata from config
        config = runtime.config if hasattr(runtime, "config") else {}
        metadata = config.get("metadata", {})
        
        user_id = metadata.get("user_id")
        conversation_id = None
        
        if user_id and isinstance(user_id, str):
            updates["user_id"] = user_id
            # Extract conversation_id from thread_id if available
            # thread_id format: "bc::{user_id}::{conversation_id}"
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "")
            if thread_id.startswith("bc::") and "::" in thread_id[4:]:
                parts = thread_id.split("::")
                if len(parts) >= 3:
                    conversation_id = parts[2]
                    updates["conversation_id"] = conversation_id
            
            # Ensure memory directories exist
            _ensure_memory_directories_exist(self.base_dir, user_id, conversation_id)
        
        return updates if updates else None
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject memory documentation into system prompt based on user/conversation context.
        
        Args:
            request: The model request being processed
            handler: The handler function to call with the modified request
            
        Returns:
            The model response from the handler
        """
        # Get user_id and conversation_id from state
        state = cast("ApiMemoryState", request.state)
        user_id = state.get("user_id")
        conversation_id = state.get("conversation_id")
        
        # Build memory documentation if we have user context
        memory_docs = _build_memory_documentation(user_id, conversation_id)
        
        if memory_docs:
            # Append memory documentation to system prompt
            if request.system_prompt:
                system_prompt = request.system_prompt + memory_docs
            else:
                system_prompt = memory_docs
            
            return handler(request.override(system_prompt=system_prompt))
        
        # No user context, pass through unchanged
        return handler(request)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        # Get user_id and conversation_id from state
        state = cast("ApiMemoryState", request.state)
        user_id = state.get("user_id")
        conversation_id = state.get("conversation_id")
        
        # Build memory documentation if we have user context
        memory_docs = _build_memory_documentation(user_id, conversation_id)
        
        if memory_docs:
            # Append memory documentation to system prompt
            if request.system_prompt:
                system_prompt = request.system_prompt + memory_docs
            else:
                system_prompt = memory_docs
            
            return await handler(request.override(system_prompt=system_prompt))
        
        # No user context, pass through unchanged
        return await handler(request)
    
    def _test_memory_file_access(self) -> None:
        """Test write/read functionality for memory files during initialization.
        
        This writes a temporary test memory file, reads it back, and verifies
        that filesystem write permissions work correctly for memory files.
        """
        try:
            # Create a test user directory
            test_user_id = "__init_test__"
            test_user_dir = self.base_dir / "users" / test_user_id
            test_user_dir.mkdir(parents=True, exist_ok=True)
            
            # Test file path
            test_memory_file = test_user_dir / "agent.md"
            
            # Generate test content with timestamp
            from datetime import datetime
            timestamp = datetime.utcnow().isoformat() + "Z"
            test_content = f"""# Memory File Access Test

This is a test file created during ApiMemoryMiddleware initialization.

Timestamp: {timestamp}

This file verifies that:
- Memory directories can be created
- Memory files can be written
- Memory files can be read
- Filesystem permissions are correct

This file will be automatically deleted after the test.
"""
            
            _logger.info("[ApiMemoryMiddleware] Testing memory file write/read access...")
            _logger.info("  Test memory file: %s", test_memory_file)
            
            # Test write
            try:
                test_memory_file.write_text(test_content, encoding="utf-8")
                _logger.info("[ApiMemoryMiddleware] ✓ Write test passed")
            except Exception as write_err:
                _logger.error(
                    "[ApiMemoryMiddleware] ❌ WRITE TEST FAILED: %s: %s",
                    type(write_err).__name__,
                    str(write_err),
                )
                return
            
            # Test read
            try:
                read_content = test_memory_file.read_text(encoding="utf-8")
                if read_content == test_content:
                    _logger.info("[ApiMemoryMiddleware] ✓ Read test passed")
                    _logger.info("  Read content verified (%d bytes)", len(read_content))
                    _logger.info("[ApiMemoryMiddleware] ✓ All memory file access tests passed")
                else:
                    _logger.warning(
                        "[ApiMemoryMiddleware] ⚠️  READ TEST WARNING: Content mismatch"
                    )
                    _logger.warning("  Expected length: %d bytes", len(test_content))
                    _logger.warning("  Actual length: %d bytes", len(read_content))
            except Exception as read_err:
                _logger.error(
                    "[ApiMemoryMiddleware] ❌ READ TEST FAILED: %s: %s",
                    type(read_err).__name__,
                    str(read_err),
                )
                return
            
            # Clean up test file and directory
            try:
                test_memory_file.unlink(missing_ok=True)
                # Only remove directory if it's empty (don't remove if user has other files)
                try:
                    test_user_dir.rmdir()
                    _logger.info("[ApiMemoryMiddleware] ✓ Cleanup test passed")
                except OSError:
                    # Directory not empty or other error - that's fine, leave it
                    pass
            except Exception as cleanup_err:
                _logger.warning(
                    "[ApiMemoryMiddleware] ⚠️  Cleanup warning (non-fatal): %s: %s",
                    type(cleanup_err).__name__,
                    str(cleanup_err),
                )
                
        except Exception as e:
            _logger.error(
                "[ApiMemoryMiddleware] ❌ MEMORY FILE TEST ERROR: %s: %s",
                type(e).__name__,
                str(e),
            )
            import traceback
            _logger.debug("[ApiMemoryMiddleware] Traceback:\n%s", traceback.format_exc())


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
    
    # Create API memory middleware to inject user/conversation memory paths dynamically
    api_memory_middleware = ApiMemoryMiddleware(base_dir=base_dir)
    
    middleware = [
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


