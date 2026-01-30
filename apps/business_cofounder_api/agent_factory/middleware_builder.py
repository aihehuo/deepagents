"""Middleware building utilities for agent factory."""

from __future__ import annotations

from deepagents.middleware.routing import (
    SubagentRouteRule,
    SubagentRoutingMiddleware,
    _looks_like_aihehuo_search_task,
    _looks_like_coding_task,
)


def build_routing_middleware(
    *,
    coder_subagent: object | None,
    aihehuo_subagent: object | None,
) -> SubagentRoutingMiddleware | None:
    """Build routing middleware for subagents.
    
    Args:
        coder_subagent: Coder subagent instance (if available)
        aihehuo_subagent: Aihehuo subagent instance (if available)
    
    Returns:
        SubagentRoutingMiddleware if any subagents are available, None otherwise
    """
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
        return SubagentRoutingMiddleware(rules=routing_rules)
    return None
