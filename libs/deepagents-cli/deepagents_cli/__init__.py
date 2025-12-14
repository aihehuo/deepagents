"""DeepAgents CLI.

Important: keep this module lightweight.

Some runtime environments (like the Business Co-Founder API server) import
`deepagents_cli.skills.*` for skill loading, but do NOT need the full CLI stack
(rich/prompt_toolkit/tavily/etc.).

So we avoid importing `deepagents_cli.main` at import-time and provide a lazy
wrapper instead.
"""

from __future__ import annotations

from typing import Any


def cli_main(*args: Any, **kwargs: Any) -> Any:
    from deepagents_cli.main import cli_main as _cli_main

    return _cli_main(*args, **kwargs)


__all__ = ["cli_main"]
