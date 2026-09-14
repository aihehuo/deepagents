"""Portable host adapter; authentication and transport belong to the host."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

NAME = "save-ai-user-profile"
MANIFEST = {
    "name": NAME,
    "version": "1.0.0",
    "kind": "write",
    "tools": ["save_group_profile"],
    "requires": ["trusted_session"],
}


def validate_installation(*, skills_dir: Path) -> None:
    """Require a complete installed write package.

    Args:
        skills_dir: Host filesystem directory containing the write package.

    Raises:
        ValueError: The package is missing, incomplete, or has an invalid manifest.
        OSError: Installed metadata cannot be read.
    """
    package = skills_dir / NAME
    required = [package / filename for filename in ("SKILL.md", "adapter.py", "package.json")]
    if package.is_symlink() or not all(path.is_file() and not path.is_symlink() for path in required):
        msg = f"Install the complete {NAME} write package before enabling it"
        raise ValueError(msg)
    if json.loads((package / "package.json").read_text(encoding="utf-8")) != MANIFEST:
        msg = f"Invalid installed write package manifest: {NAME}"
        raise ValueError(msg)


def create_profile_write_tool(*, skills_dir: Path, writer: BaseTool) -> BaseTool:
    """Wrap the host writer with explicit per-invocation write authorization.

    The host must authenticate the session and sanitize client metadata before
    setting `can_write_profile`, `user_id`, and `group_id` in RunnableConfig.
    Installing a package never grants authorization by itself.

    !!! warning
        This host integration is experimental. Never use unfiltered client
        metadata as the trusted authorization config.

    Args:
        skills_dir: Installed write-package source on the host filesystem.
        writer: Host `save_group_profile` tool that enforces identity, persistence,
            fencing, and acknowledgements. Identity must not be a model argument.

    Returns:
        A compatible tool with the host writer's content-only argument schema.

    Raises:
        ValueError: Installation or the host writer's schema is invalid.
        OSError: Installed metadata cannot be read.
    """
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import StructuredTool

    validate_installation(skills_dir=skills_dir)
    schema = writer.args_schema
    reserved = {
        "user_id",
        "group_id",
        "unionid",
        "user_token",
        "group_token",
        "run_id",
        "can_write_profile",
        "config",
        "base_dir",
        "attempt_id",
        "fencing_token",
        "execution_binding",
        "source_message_id",
        "source_message_text",
    }
    if writer.name != "save_group_profile" or schema is None or isinstance(schema, dict):
        msg = "Provide a content-only save_group_profile host tool with a model schema"
        raise ValueError(msg)
    if reserved.intersection(schema.model_fields):
        msg = "Host writer must inject identity and authorization through trusted config"
        raise ValueError(msg)

    def write(*, config: RunnableConfig, **content: Any) -> str:
        metadata = config.get("metadata") or {}
        permission = metadata.get("can_write_profile", False)
        allowed = permission is True or (isinstance(permission, str) and permission.strip().lower() in {"true", "1", "yes", "on", "enabled"})
        if not allowed:
            return "error: write_capability_disabled"
        if not str(metadata.get("user_id") or "").strip() or not str(metadata.get("group_id") or "").strip():
            return "error: missing trusted user_id or group_id"
        # Revalidate at the side-effect boundary, including deterministic saves.
        try:
            validate_installation(skills_dir=skills_dir)
        except (ValueError, OSError) as error:
            return f"error: write_skill_unavailable:{type(error).__name__}"
        return str(writer.invoke(content, config=config))

    # RunnableConfig must be resolvable by LangChain even when this module is
    # copied outside the repository and optional dependencies are imported late.
    write.__annotations__["config"] = RunnableConfig
    return StructuredTool.from_function(
        write,
        name=writer.name,
        args_schema=schema,
        description=writer.description
        + " Only the authenticated session owner may write. Identity is server-injected; an error or superseded result is not a successful update.",
    )
