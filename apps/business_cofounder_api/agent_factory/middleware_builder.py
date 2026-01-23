"""Middleware building utilities for agent factory."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from deepagents.middleware.routing import (
    SubagentRouteRule,
    SubagentRoutingMiddleware,
    _looks_like_aihehuo_search_task,
    _looks_like_coding_task,
)
from deepagents_cli.skills.middleware import SkillsMiddleware, SkillsState, SkillsStateUpdate
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.runtime import Runtime

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


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
