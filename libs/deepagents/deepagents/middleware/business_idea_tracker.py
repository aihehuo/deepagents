"""Middleware for tracking whether a business idea has been materialized in the conversation.

This middleware tracks the completion status of business ideas, allowing skills like
business-idea-evaluation to check if an idea has already been identified and avoid
re-evaluation once a complete idea is found.
"""

from collections.abc import Awaitable, Callable
from typing import Annotated, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command


class BusinessIdeaState(AgentState):
    """State for tracking business idea completion."""

    business_idea_complete: NotRequired[bool]
    """Whether a complete business idea has been identified in the conversation."""

    materialized_business_idea: NotRequired[str | None]
    """Summary or description of the materialized business idea, if one has been identified."""


class BusinessIdeaStateUpdate(TypedDict):
    """A state update for the business idea tracker middleware."""

    business_idea_complete: NotRequired[bool]
    """Whether a complete business idea has been identified."""

    materialized_business_idea: NotRequired[str | None]
    """Summary or description of the materialized business idea."""


MARK_BUSINESS_IDEA_COMPLETE_TOOL_DESCRIPTION = """Mark a business idea as complete and materialized.

Use this tool when you have identified a complete business idea through the business-idea-evaluation skill or other means. Once marked as complete, the business-idea-evaluation skill should not be used again in this conversation.

Args:
    idea_summary: A concise summary of the materialized business idea (1-3 sentences).

This tool updates the agent state to indicate that a complete business idea has been identified, preventing unnecessary re-evaluation."""


def _get_mark_business_idea_complete_tool() -> StructuredTool:
    """Create a tool that marks a business idea as complete."""
    
    def mark_business_idea_complete(
        idea_summary: Annotated[
            str,
            "A concise summary of the materialized business idea (1-3 sentences).",
        ],
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Mark a business idea as complete and materialized.
        
        Args:
            idea_summary: A concise summary of the materialized business idea.
            tool_call_id: Injected tool call ID.
        
        Returns:
            Command with state update marking the idea as complete.
        """
        return Command(
            update={
                "business_idea_complete": True,
                "materialized_business_idea": idea_summary,
                "messages": [
                    ToolMessage(
                        content=f"Marked business idea as complete: {idea_summary}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    return StructuredTool.from_function(
        func=mark_business_idea_complete,
        name="mark_business_idea_complete",
        description=MARK_BUSINESS_IDEA_COMPLETE_TOOL_DESCRIPTION,
    )


BUSINESS_IDEA_TRACKER_SYSTEM_PROMPT = """## Business Idea Tracking

The conversation may contain a materialized business idea. If a complete business idea has already been identified in this conversation, it is stored in the agent state.

**Important for business-idea-evaluation skill:**
- Before using the business-idea-evaluation skill, check if a complete business idea has already been identified
- If a business idea has already been materialized, do NOT use the business-idea-evaluation skill again
- The skill should only be used until a complete business idea is identified
- Once an idea is complete, use the `mark_business_idea_complete` tool to record it
- After marking an idea as complete, the conversation moves to the post-idea phase and business-idea-evaluation becomes irrelevant

**How to mark an idea as complete:**
- When you identify a complete business idea (through business-idea-evaluation or other means), use the `mark_business_idea_complete` tool
- Provide a concise summary (1-3 sentences) of the materialized idea
- This will update the agent state and prevent future use of business-idea-evaluation skill"""


class BusinessIdeaTrackerMiddleware(AgentMiddleware):
    """Middleware for tracking whether a business idea has been materialized.

    This middleware:
    1. Tracks whether a complete business idea has been identified
    2. Stores a summary of the materialized idea
    3. Provides system prompt guidance for skills to check this state
    4. Provides a tool to mark ideas as complete

    Example:
        ```python
        from deepagents.middleware.business_idea_tracker import BusinessIdeaTrackerMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            model="anthropic:claude-sonnet-4-20250514",
            middleware=[BusinessIdeaTrackerMiddleware()],
        )
        ```
    """

    state_schema = BusinessIdeaState

    def __init__(
        self,
        *,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the BusinessIdeaTrackerMiddleware.

        Args:
            system_prompt_template: Optional custom system prompt template.
                If None, uses the default template.
        """
        self.system_prompt_template = (
            system_prompt_template or BUSINESS_IDEA_TRACKER_SYSTEM_PROMPT
        )
        self.tools = [_get_mark_business_idea_complete_tool()]

    def before_agent(
        self,
        state: BusinessIdeaState,
        runtime: Runtime,
    ) -> BusinessIdeaStateUpdate | None:
        """Initialize business idea tracking state if not present.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with business_idea_complete initialized to False if not present.
        """
        # Initialize to False if not already set
        # Note: We initialize both fields to ensure they exist in state
        if "business_idea_complete" not in state:
            return BusinessIdeaStateUpdate(
                business_idea_complete=False,
                materialized_business_idea=None,
            )

        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject business idea tracking guidance into the system prompt.

        This runs on every model call to ensure the tracking information is always available.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Get current state
        state = cast("BusinessIdeaState", request.state)
        business_idea_complete = state.get("business_idea_complete", False)
        materialized_idea = state.get("materialized_business_idea")

        # Build context-aware prompt
        if business_idea_complete and materialized_idea:
            # Idea is complete - provide summary with strong instructions
            idea_context = (
                f"\n\n**CRITICAL: Business Idea Status**\n"
                f"A complete business idea has already been identified and materialized:\n\n"
                f"{materialized_idea}\n\n"
                f"**REQUIRED ACTION**: \n"
                f"- DO NOT read the business-idea-evaluation SKILL.md file\n"
                f"- DO NOT use the business-idea-evaluation skill\n"
                f"- DO NOT re-evaluate the idea\n"
                f"- DO NOT call mark_business_idea_complete again\n"
                f"- The business-idea-evaluation skill is no longer relevant\n"
                f"- Proceed directly with next steps for the already-identified idea"
            )
        elif business_idea_complete:
            # Idea is complete but no summary stored
            idea_context = (
                "\n\n**CRITICAL: Business Idea Status**\n"
                "A complete business idea has already been identified in this conversation.\n\n"
                "**REQUIRED ACTION**: \n"
                "- DO NOT read the business-idea-evaluation SKILL.md file\n"
                "- DO NOT use the business-idea-evaluation skill\n"
                "- DO NOT re-evaluate the idea\n"
                "- DO NOT call mark_business_idea_complete again\n"
                "- The business-idea-evaluation skill is no longer relevant\n"
                "- Proceed directly with next steps for the already-identified idea"
            )
        else:
            # No complete idea yet
            idea_context = "\n\n**Current Status**: No complete business idea has been identified yet.\n\n**Action**: You may use the business-idea-evaluation skill to evaluate the user's input."

        full_prompt = self.system_prompt_template + idea_context

        if request.system_prompt:
            new_system_prompt = request.system_prompt + "\n\n" + full_prompt
        else:
            new_system_prompt = full_prompt

        return handler(request.override(system_prompt=new_system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject business idea tracking guidance into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Get current state
        state = cast("BusinessIdeaState", request.state)
        business_idea_complete = state.get("business_idea_complete", False)
        materialized_idea = state.get("materialized_business_idea")

        # Build context-aware prompt
        if business_idea_complete and materialized_idea:
            # Idea is complete - provide summary with strong instructions
            idea_context = (
                f"\n\n**CRITICAL: Business Idea Status**\n"
                f"A complete business idea has already been identified and materialized:\n\n"
                f"{materialized_idea}\n\n"
                f"**REQUIRED ACTION**: \n"
                f"- DO NOT read the business-idea-evaluation SKILL.md file\n"
                f"- DO NOT use the business-idea-evaluation skill\n"
                f"- DO NOT re-evaluate the idea\n"
                f"- DO NOT call mark_business_idea_complete again\n"
                f"- The business-idea-evaluation skill is no longer relevant\n"
                f"- Proceed directly with next steps for the already-identified idea"
            )
        elif business_idea_complete:
            # Idea is complete but no summary stored
            idea_context = (
                "\n\n**CRITICAL: Business Idea Status**\n"
                "A complete business idea has already been identified in this conversation.\n\n"
                "**REQUIRED ACTION**: \n"
                "- DO NOT read the business-idea-evaluation SKILL.md file\n"
                "- DO NOT use the business-idea-evaluation skill\n"
                "- DO NOT re-evaluate the idea\n"
                "- DO NOT call mark_business_idea_complete again\n"
                "- The business-idea-evaluation skill is no longer relevant\n"
                "- Proceed directly with next steps for the already-identified idea"
            )
        else:
            # No complete idea yet
            idea_context = "\n\n**Current Status**: No complete business idea has been identified yet.\n\n**Action**: You may use the business-idea-evaluation skill to evaluate the user's input."

        full_prompt = self.system_prompt_template + idea_context

        if request.system_prompt:
            new_system_prompt = request.system_prompt + "\n\n" + full_prompt
        else:
            new_system_prompt = full_prompt

        return await handler(request.override(system_prompt=new_system_prompt))

