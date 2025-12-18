from __future__ import annotations

import os
import shutil
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware
from deepagents.middleware.business_idea_tracker import BusinessIdeaTrackerMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents.middleware.routing import (
    SubagentRoutingMiddleware,
    SubagentRouteRule,
    _looks_like_aihehuo_search_task,
    _looks_like_coding_task,
)
from deepagents.subagent_presets import (
    build_aihehuo_subagent_from_env,
    build_coder_subagent_from_env,
)
from deepagents_cli.skills.middleware import SkillsMiddleware
from langchain_anthropic import ChatAnthropic

from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver
from apps.business_cofounder_api.docs_backend import DocsOnlyWriteBackend


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


def create_business_cofounder_agent(*, agent_id: str) -> tuple[object, Path]:
    """Create the Business Co-Founder deep agent (shared across users; state isolated by thread_id).

    Returns:
        (agent_graph, checkpoints_path)
    """
    # Model configuration (single generic env var set):
    # - MODEL_API_PROVIDER: deepseek | qwen
    # - MODEL_API_KEY / MODEL_BASE_URL / MODEL_NAME
    # - MODEL_API_MAX_TOKENS / MODEL_API_TEMPERATURE / MODEL_API_TIMEOUT_S (optional)
    provider = (os.environ.get("MODEL_API_PROVIDER") or "deepseek").strip().lower()
    base_url = os.environ.get("MODEL_BASE_URL")
    api_key = os.environ.get("MODEL_API_KEY")

    model_name = os.environ.get("MODEL_NAME")
    if not model_name:
        model_name = "qwen-plus" if provider == "qwen" else "deepseek-chat"

    max_tokens_env = os.environ.get("MODEL_API_MAX_TOKENS") or "20000"
    timeout_env = os.environ.get("MODEL_API_TIMEOUT_S") or "180.0"
    temperature_env = os.environ.get("MODEL_API_TEMPERATURE")

    try:
        max_tokens = int(max_tokens_env)
    except ValueError:
        max_tokens = 20000
    try:
        timeout_s = float(timeout_env)
    except ValueError:
        timeout_s = 180.0

    temperature: float | None = None
    if temperature_env is not None and temperature_env != "":
        try:
            temperature = float(temperature_env)
        except ValueError:
            temperature = None

    if provider == "qwen":
        # Qwen (DashScope) OpenAI-compatible mode
        from langchain_openai import ChatOpenAI  # lazy import (avoid import-time side effects in tests)

        openai_kwargs: dict[str, object] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "timeout": timeout_s,
        }
        if temperature is not None:
            openai_kwargs["temperature"] = temperature
        if base_url:
            openai_kwargs["base_url"] = base_url
        if api_key:
            openai_kwargs["api_key"] = api_key
        
        # Note: stream_options can only be set when stream: true, so we don't set it here.
        # Usage metadata is typically included by default in both streaming and non-streaming responses.

        model = ChatOpenAI(**openai_kwargs)
    else:
        # DeepSeek / Anthropic-compatible proxy
        anthropic_kwargs: dict[str, object] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "timeout": timeout_s,
        }
        if temperature is not None:
            anthropic_kwargs["temperature"] = temperature
        if base_url:
            anthropic_kwargs["base_url"] = base_url
        if api_key:
            anthropic_kwargs["api_key"] = api_key

        model = ChatAnthropic(**anthropic_kwargs)

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

    # IMPORTANT: FilesystemBackend virtual_mode=False so SkillsMiddleware absolute paths work.
    # Wrap it so all writes/edits are forced into docs_dir (prevents writing to / or /home/user).
    backend = DocsOnlyWriteBackend(
        backend=FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=False),
        docs_dir=docs_dir,
    )

    coder_subagent = build_coder_subagent_from_env(tools=None, name="coder")
    aihehuo_subagent = build_aihehuo_subagent_from_env(tools=None, name="aihehuo")
    
    subagents = []
    if coder_subagent is not None:
        subagents.append(coder_subagent)
    if aihehuo_subagent is not None:
        subagents.append(aihehuo_subagent)
    
    middleware = [
        LanguageDetectionMiddleware(),
        BusinessIdeaTrackerMiddleware(),
        BusinessIdeaDevelopmentMiddleware(strict_todo_sync=True),
        SkillsMiddleware(
            skills_dir=skills_dir,
            assistant_id=agent_id,
            project_skills_dir=None,
        ),
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


