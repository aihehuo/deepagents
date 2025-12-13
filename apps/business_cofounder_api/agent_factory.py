from __future__ import annotations

import os
import shutil
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware
from deepagents.middleware.business_idea_tracker import BusinessIdeaTrackerMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents_cli.skills.middleware import SkillsMiddleware
from langchain_anthropic import ChatAnthropic

from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver


def _copy_example_skills_if_missing(*, dest_skills_dir: Path) -> None:
    """Copy deepagents-cli packaged example skills into dest_skills_dir (no overwrite)."""
    # deepagents_cli/... -> deepagents-cli root -> examples/skills
    import deepagents_cli

    cli_root = Path(deepagents_cli.__file__).resolve().parent.parent
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
    # Model configuration:
    # - Defaults to env-driven settings so you can point at DeepSeek proxy or Anthropic.
    # - Falls back to a reasonable default model name if env is not set.
    model_name = os.environ.get("BC_API_MODEL") or os.environ.get("ANTHROPIC_MODEL") or "deepseek-chat"
    max_tokens = int(os.environ.get("BC_API_MAX_TOKENS") or "20000")
    temperature_env = os.environ.get("BC_API_TEMPERATURE")

    model_kwargs: dict[str, object] = {
        "model": model_name,
        "max_tokens": max_tokens,
    }
    if temperature_env is not None:
        try:
            model_kwargs["temperature"] = float(temperature_env)
        except ValueError:
            pass

    # If you're using an OpenAI-compatible or Anthropic-compatible proxy (e.g. DeepSeek),
    # these env vars are typically how ChatAnthropic is configured.
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if base_url:
        model_kwargs["base_url"] = base_url
    if api_key:
        model_kwargs["api_key"] = api_key

    model = ChatAnthropic(**model_kwargs)

    base_dir = Path.home() / ".deepagents" / "business_cofounder_api"
    skills_dir = base_dir / "skills"
    checkpoints_path = base_dir / "checkpoints.pkl"

    _copy_example_skills_if_missing(dest_skills_dir=skills_dir)

    checkpointer = DiskBackedInMemorySaver(file_path=checkpoints_path)

    # IMPORTANT: FilesystemBackend virtual_mode=False so SkillsMiddleware absolute paths work.
    backend = FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=False)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        checkpointer=checkpointer,
        middleware=[
            LanguageDetectionMiddleware(),
            BusinessIdeaTrackerMiddleware(),
            BusinessIdeaDevelopmentMiddleware(),
            SkillsMiddleware(
                skills_dir=skills_dir,
                assistant_id=agent_id,
                project_skills_dir=None,
            ),
        ],
        system_prompt="You are a business co-founder assistant helping entrepreneurs develop their startup ideas.",
    )

    return agent, checkpoints_path


