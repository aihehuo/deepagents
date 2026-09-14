"""Identical member permission graph bootstrap for API and worker processes."""

from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from apps.group_agent_api.agent_factory.agent import create_agent, create_member_agent
from apps.group_agent_api.agent_factory.model_builder import create_model
from apps.group_agent_api.checkpointer import DiskBackedInMemorySaver


def build_member_graphs(*, runtime: Path, model: BaseChatModel | None = None) -> dict[str, Any]:
    """Build every permission variant with one shared model and checkpointer.

    !!! warning
        This application bootstrap is experimental.

    Args:
        runtime: Host runtime containing installed skills and checkpoints.
        model: Optional host model override.

    Returns:
        Graph and checkpoint fields accepted by application state.
    """
    model = model or create_model()
    checkpoint_path = runtime / "checkpoints.pkl"
    checkpointer = DiskBackedInMemorySaver(file_path=checkpoint_path)
    agent, admin, path = create_agent(base_dir=runtime, model=model, checkpointer=checkpointer)
    graphs = {"agent": agent, "admin_agent": admin, "checkpoints_path": str(path)}
    for name, search, save in (
        ("search_only_agent", True, False),
        ("write_only_agent", False, True),
        ("no_tool_agent", False, False),
    ):
        graphs[name] = create_member_agent(
            runtime=runtime,
            model=model,
            checkpointer=checkpointer,
            include_search=search,
            include_save=save,
        )
    return graphs
