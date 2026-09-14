"""Compose authenticated member writes separately from public read skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.tools import BaseTool

import os

READ_ONLY_PROMPT = "\n本会话没有画像写入工具或写权限。不要尝试保存画像，也不要声称已保存或更新。\n"  # noqa: RUF001
WRITE_PROMPT = (
    "\n画像写入使用已安装的 save-ai-user-profile 技能。只保存当前可信会话用户的本人画像；"  # noqa: RUF001  # Chinese punctuation in the user-facing prompt.
    "工具返回 error 或 profile_superseded 时不能声称本次更新成功。\n"
)


def profile_write_mode() -> str:
    """Resolve the independently configurable member write mode.

    Returns:
        legacy, skillpack, or disabled.

    Raises:
        ValueError: The configured mode is invalid.
    """
    mode = os.environ.get("GROUP_AGENT_WRITE_MODE", "legacy").strip().lower()
    if mode not in {"legacy", "skillpack", "disabled"}:
        msg = "GROUP_AGENT_WRITE_MODE must be legacy, skillpack, or disabled"
        raise ValueError(msg)
    return mode


def member_write_configuration(*, runtime: Path, enabled: bool, writer: BaseTool) -> tuple[list[BaseTool], list[str], str]:
    """Select the member write package without changing read configuration.

    Args:
        runtime: Existing member filesystem backend root.
        enabled: Whether this member graph is permitted to expose writes.
        writer: Existing trusted host profile writer.

    Returns:
        Write tools, virtual skill sources, and the applicable write policy.

    Raises:
        ValueError: Mode or installed package is invalid.
        OSError: Installed package metadata cannot be read.
    """
    mode = profile_write_mode()
    if not enabled or mode == "disabled":
        return [], [], READ_ONLY_PROMPT
    if mode == "legacy":
        return [writer], [], ""

    from skillpacks.aihehuo_write.adapter import create_profile_write_tool

    tool = create_profile_write_tool(skills_dir=runtime / "skills" / "aihehuo_write", writer=writer)
    return [tool], ["/skills/aihehuo_write/"], WRITE_PROMPT
