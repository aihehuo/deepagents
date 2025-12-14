"""Skills module for deepagents CLI.

Keep this package import-light.

The Business Co-Founder API imports `deepagents_cli.skills.middleware.SkillsMiddleware`
but should not require the full CLI stack (dotenv/rich/prompt_toolkit/etc.). Historically,
importing this package would eagerly import `commands.py`, which pulls in `deepagents_cli.config`
and optional CLI dependencies.

So we provide lazy wrappers for CLI-only helpers, while still exporting SkillsMiddleware.
"""

from __future__ import annotations

from typing import Any

from deepagents_cli.skills.middleware import SkillsMiddleware


def execute_skills_command(*args: Any, **kwargs: Any) -> Any:
    from deepagents_cli.skills.commands import execute_skills_command as _execute

    return _execute(*args, **kwargs)


def setup_skills_parser(*args: Any, **kwargs: Any) -> Any:
    from deepagents_cli.skills.commands import setup_skills_parser as _setup

    return _setup(*args, **kwargs)


__all__ = ["SkillsMiddleware", "execute_skills_command", "setup_skills_parser"]
