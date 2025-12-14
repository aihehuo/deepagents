"""Routing middleware for delegating work to subagents.

This module is intentionally simple and extensible:
- Start with a single "coder" route for code/HTML tasks.
- Later add more rules/subagents without changing call sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage


def _last_human_text(request: ModelRequest) -> str:
    for msg in reversed(request.messages):
        if isinstance(msg, HumanMessage):
            return msg.text or ""
    return ""


_DEFAULT_CODE_KEYWORDS = (
    "code",
    "coding",
    "implement",
    "implementation",
    "bug",
    "fix",
    "refactor",
    "function",
    "class",
    "python",
    "typescript",
    "javascript",
    "react",
    "node",
    "dockerfile",
    "sql",
    "api",
    "endpoint",
    "html",
    "css",
)

_DEFAULT_CODE_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"```"),  # fenced code block
    re.compile(r"<(html|div|span|body|head|script|style)\b", re.IGNORECASE),
    re.compile(r"\b(CSS|HTML|JS|TS|TSX|JSX)\b", re.IGNORECASE),
)


def _looks_like_coding_task(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    if any(k in low for k in _DEFAULT_CODE_KEYWORDS):
        return True
    return any(rx.search(t) is not None for rx in _DEFAULT_CODE_REGEXES)


@dataclass(frozen=True)
class SubagentRouteRule:
    """A simple routing rule that injects guidance to delegate to a subagent."""

    name: str
    subagent_type: str
    should_route: Callable[[str, ModelRequest], bool]
    instruction: str


class SubagentRoutingMiddleware(AgentMiddleware):
    """Injects light routing guidance for when to delegate to a subagent via `task`.

    This middleware does NOT force tool calls. It only appends a short system message
    that nudges the orchestrator to delegate when a rule matches.
    """

    def __init__(self, *, rules: Sequence[SubagentRouteRule]) -> None:
        super().__init__()
        self._rules = list(rules)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        user_text = _last_human_text(request)
        additions: list[str] = []
        for rule in self._rules:
            try:
                if rule.should_route(user_text, request):
                    additions.append(rule.instruction)
            except Exception:  # noqa: BLE001
                # Routing must never break the agent.
                continue

        if not additions:
            return handler(request)

        extra = "\n\n".join(additions).strip()
        if not extra:
            return handler(request)

        current = request.system_prompt or ""
        merged = (current + "\n\n" + extra).strip() if current else extra
        return handler(request.override(system_message=SystemMessage(content=merged)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async variant required when agents are invoked via `ainvoke()` / `astream()`."""
        user_text = _last_human_text(request)
        additions: list[str] = []
        for rule in self._rules:
            try:
                if rule.should_route(user_text, request):
                    additions.append(rule.instruction)
            except Exception:  # noqa: BLE001
                continue

        if not additions:
            return await handler(request)

        extra = "\n\n".join(additions).strip()
        if not extra:
            return await handler(request)

        current = request.system_prompt or ""
        merged = (current + "\n\n" + extra).strip() if current else extra
        return await handler(request.override(system_message=SystemMessage(content=merged)))


def build_default_coder_routing_middleware(*, coder_subagent_type: str = "coder") -> SubagentRoutingMiddleware:
    """Factory for the initial "code/HTML -> coder subagent" routing behavior."""

    instruction = f"""## Routing hint (code/HTML)

If the user is asking for **code** (including **HTML/CSS/JS**, scripts, Dockerfiles, config changes, refactors, or debugging), you **MUST** delegate the heavy lifting to the `coder` subagent (unless it is truly trivial, e.g. a <5-line snippet with no repo edits):
- Use the `task` tool with `subagent_type="{coder_subagent_type}"`.
- In the task description, include: the goal, relevant constraints, file paths, and the exact expected output.
- The subagent can use the same tools (files/execute/etc.) to implement changes; then you summarize results to the user.
"""

    rules = [
        SubagentRouteRule(
            name="code_html_to_coder",
            subagent_type=coder_subagent_type,
            should_route=lambda text, _req: _looks_like_coding_task(text),
            instruction=instruction,
        )
    ]
    return SubagentRoutingMiddleware(rules=rules)


