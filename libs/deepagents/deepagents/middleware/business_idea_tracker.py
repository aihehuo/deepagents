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
    """State for tracking business idea completion and skill progression milestones."""

    business_idea_complete: NotRequired[bool]
    """Whether a complete business idea has been identified in the conversation."""

    materialized_business_idea: NotRequired[str | None]
    """Summary or description of the materialized business idea, if one has been identified."""

    persona_clarified: NotRequired[bool]
    """Whether the user persona has been clarified."""

    painpoint_enhanced: NotRequired[bool]
    """Whether the pain point has been enhanced."""

    pitch_created: NotRequired[bool]
    """Whether a 60-second pitch has been created."""

    pricing_optimized: NotRequired[bool]
    """Whether baseline pricing and optimization has been completed."""


class BusinessIdeaStateUpdate(TypedDict):
    """A state update for the business idea tracker middleware."""

    business_idea_complete: NotRequired[bool]
    """Whether a complete business idea has been identified."""

    materialized_business_idea: NotRequired[str | None]
    """Summary or description of the materialized business idea."""

    persona_clarified: NotRequired[bool]
    """Whether the user persona has been clarified."""

    painpoint_enhanced: NotRequired[bool]
    """Whether the pain point has been enhanced."""

    pitch_created: NotRequired[bool]
    """Whether a 60-second pitch has been created."""

    pricing_optimized: NotRequired[bool]
    """Whether baseline pricing and optimization has been completed."""


MARK_BUSINESS_IDEA_COMPLETE_TOOL_DESCRIPTION = """Mark a business idea as complete and materialized.

Use this tool when you have identified a complete business idea through the business-idea-evaluation skill or other means. Once marked as complete, the business-idea-evaluation skill should not be used again in this conversation, and the persona-clarification skill will unlock.

Args:
    idea_summary: A concise summary of the materialized business idea (1-3 sentences).

This tool updates the agent state to indicate that a complete business idea has been identified, preventing unnecessary re-evaluation."""


MARK_PERSONA_CLARIFIED_TOOL_DESCRIPTION = """Mark that the user persona has been clarified.

Use this tool after using the persona-clarification skill to mark that the persona has been clarified. This unlocks the painpoint-enhancement skill.

Args:
    confirmation: A brief confirmation message (e.g., "Persona clarified").

This tool updates the agent state to indicate that the persona has been clarified."""


MARK_PAINPOINT_ENHANCED_TOOL_DESCRIPTION = """Mark that the pain point has been enhanced.

Use this tool after using the painpoint-enhancement skill to mark that the pain point has been enhanced. This, combined with persona_clarified, unlocks the 60s-pitch-creation skill.

Args:
    confirmation: A brief confirmation message (e.g., "Pain point enhanced").

This tool updates the agent state to indicate that the pain point has been enhanced."""


MARK_PITCH_CREATED_TOOL_DESCRIPTION = """Mark that a 60-second pitch has been created.

Use this tool after using the 60s-pitch-creation skill to mark that the pitch has been created. This unlocks the baseline-pricing-and-optimization skill.

Args:
    confirmation: A brief confirmation message (e.g., "Pitch created").

This tool updates the agent state to indicate that the pitch has been created."""


MARK_PRICING_OPTIMIZED_TOOL_DESCRIPTION = """Mark that baseline pricing and optimization has been completed.

Use this tool after using the baseline-pricing-and-optimization skill to mark that pricing optimization has been completed. This unlocks the business-model-pivot-exploration skill.

Args:
    confirmation: A brief confirmation message (e.g., "Pricing optimized").

This tool updates the agent state to indicate that pricing optimization has been completed."""


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


def _get_mark_persona_clarified_tool() -> StructuredTool:
    """Create a tool that marks persona as clarified."""
    
    def mark_persona_clarified(
        confirmation: Annotated[
            str,
            "A brief confirmation message (e.g., 'Persona clarified').",
        ] = "Persona clarified",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Mark that the user persona has been clarified.
        
        Args:
            confirmation: A brief confirmation message.
            tool_call_id: Injected tool call ID.
        
        Returns:
            Command with state update marking persona as clarified.
        """
        return Command(
            update={
                "persona_clarified": True,
                "messages": [
                    ToolMessage(
                        content=f"Marked persona as clarified: {confirmation}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    return StructuredTool.from_function(
        func=mark_persona_clarified,
        name="mark_persona_clarified",
        description=MARK_PERSONA_CLARIFIED_TOOL_DESCRIPTION,
    )


def _get_mark_painpoint_enhanced_tool() -> StructuredTool:
    """Create a tool that marks pain point as enhanced."""
    
    def mark_painpoint_enhanced(
        confirmation: Annotated[
            str,
            "A brief confirmation message (e.g., 'Pain point enhanced').",
        ] = "Pain point enhanced",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Mark that the pain point has been enhanced.
        
        Args:
            confirmation: A brief confirmation message.
            tool_call_id: Injected tool call ID.
        
        Returns:
            Command with state update marking pain point as enhanced.
        """
        return Command(
            update={
                "painpoint_enhanced": True,
                "messages": [
                    ToolMessage(
                        content=f"Marked pain point as enhanced: {confirmation}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    return StructuredTool.from_function(
        func=mark_painpoint_enhanced,
        name="mark_painpoint_enhanced",
        description=MARK_PAINPOINT_ENHANCED_TOOL_DESCRIPTION,
    )


def _get_mark_pitch_created_tool() -> StructuredTool:
    """Create a tool that marks pitch as created."""
    
    def mark_pitch_created(
        confirmation: Annotated[
            str,
            "A brief confirmation message (e.g., 'Pitch created').",
        ] = "Pitch created",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Mark that a 60-second pitch has been created.
        
        Args:
            confirmation: A brief confirmation message.
            tool_call_id: Injected tool call ID.
        
        Returns:
            Command with state update marking pitch as created.
        """
        return Command(
            update={
                "pitch_created": True,
                "messages": [
                    ToolMessage(
                        content=f"Marked pitch as created: {confirmation}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    return StructuredTool.from_function(
        func=mark_pitch_created,
        name="mark_pitch_created",
        description=MARK_PITCH_CREATED_TOOL_DESCRIPTION,
    )


def _get_mark_pricing_optimized_tool() -> StructuredTool:
    """Create a tool that marks pricing optimization as completed."""
    
    def mark_pricing_optimized(
        confirmation: Annotated[
            str,
            "A brief confirmation message (e.g., 'Pricing optimized').",
        ] = "Pricing optimized",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Mark that baseline pricing and optimization has been completed.
        
        Args:
            confirmation: A brief confirmation message.
            tool_call_id: Injected tool call ID.
        
        Returns:
            Command with state update marking pricing optimization as completed.
        """
        return Command(
            update={
                "pricing_optimized": True,
                "messages": [
                    ToolMessage(
                        content=f"Marked pricing optimization as completed: {confirmation}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    
    return StructuredTool.from_function(
        func=mark_pricing_optimized,
        name="mark_pricing_optimized",
        description=MARK_PRICING_OPTIMIZED_TOOL_DESCRIPTION,
    )


BUSINESS_IDEA_TRACKER_SYSTEM_PROMPT = """## Business Idea Tracking & Skill Unlock Conditions

The conversation follows a sequential skill progression. Skills are unlocked only when their prerequisites are met. Each skill must be completed and marked before the next skill can be unlocked.

**Sequential Skill Progression:**

1. **business-idea-evaluation** (UNLOCKED when: business_idea_complete == False)
   - This is the FIRST skill to use when the user shares an idea
   - Use this skill to evaluate whether the user's input contains a complete business idea
   - LOCKED when: business_idea_complete == True (idea already identified)
   - When complete, call `mark_business_idea_complete` tool → unlocks persona-clarification

2. **persona-clarification** (UNLOCKED when: business_idea_complete == True)
   - Use this skill after a business idea is identified to clarify the target user persona
   - Helps transform vague user descriptions into clear, detailed personas
   - LOCKED when: business_idea_complete == False (no idea identified yet)
   - When complete, call `mark_persona_clarified` tool → unlocks painpoint-enhancement

3. **painpoint-enhancement** (UNLOCKED when: persona_clarified == True)
   - Use this skill after persona is clarified to strengthen and enhance the pain point
   - Enhances pain points using six emotional-resonance dimensions
   - LOCKED when: persona_clarified == False (persona not clarified yet)
   - When complete, call `mark_painpoint_enhanced` tool → unlocks 60s-pitch-creation (with persona)

4. **60s-pitch-creation** (UNLOCKED when: persona_clarified == True AND painpoint_enhanced == True)
   - Use this skill after both persona is clarified AND pain point is enhanced
   - Creates a 60-second entrepreneurial pitch
   - LOCKED when: persona_clarified == False OR painpoint_enhanced == False
   - When complete, call `mark_pitch_created` tool → unlocks baseline-pricing-and-optimization

5. **baseline-pricing-and-optimization** (UNLOCKED when: pitch_created == True)
   - Use this skill after the 60-second pitch has been created
   - Establishes baseline pricing and generates pricing optimization tactics
   - LOCKED when: pitch_created == False (pitch not created yet)
   - When complete, call `mark_pricing_optimized` tool → unlocks business-model-pivot-exploration

6. **business-model-pivot-exploration** (UNLOCKED when: pricing_optimized == True)
   - Use this skill after baseline pricing and optimization is completed
   - Explores seven business model archetypes (Retail, Service, Brokerage, etc.)
   - LOCKED when: pricing_optimized == False (pricing not optimized yet)

**Milestone Marking Tools:**
- `mark_business_idea_complete`: Mark business idea as complete (unlocks persona-clarification)
- `mark_persona_clarified`: Mark persona as clarified (unlocks painpoint-enhancement)
- `mark_painpoint_enhanced`: Mark pain point as enhanced (unlocks 60s-pitch-creation)
- `mark_pitch_created`: Mark pitch as created (unlocks baseline-pricing-and-optimization)
- `mark_pricing_optimized`: Mark pricing optimization as completed (unlocks business-model-pivot-exploration)

**Important:** Always call the appropriate milestone marking tool after completing each skill to unlock the next skill in the sequence."""


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
        self.tools = [
            _get_mark_business_idea_complete_tool(),
            _get_mark_persona_clarified_tool(),
            _get_mark_painpoint_enhanced_tool(),
            _get_mark_pitch_created_tool(),
            _get_mark_pricing_optimized_tool(),
        ]

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
        # Initialize all milestone fields to False if not already set
        updates = {}
        if "business_idea_complete" not in state:
            updates["business_idea_complete"] = False
            updates["materialized_business_idea"] = None
        if "persona_clarified" not in state:
            updates["persona_clarified"] = False
        if "painpoint_enhanced" not in state:
            updates["painpoint_enhanced"] = False
        if "pitch_created" not in state:
            updates["pitch_created"] = False
        if "pricing_optimized" not in state:
            updates["pricing_optimized"] = False
        
        return BusinessIdeaStateUpdate(**updates) if updates else None

    def _get_skill_unlock_status(self, state: BusinessIdeaState) -> str:
        """Determine which skills are unlocked based on current state.
        
        Args:
            state: Current agent state.
            
        Returns:
            Formatted string showing skill unlock status.
        """
        business_idea_complete = state.get("business_idea_complete", False)
        persona_clarified = state.get("persona_clarified", False)
        painpoint_enhanced = state.get("painpoint_enhanced", False)
        pitch_created = state.get("pitch_created", False)
        pricing_optimized = state.get("pricing_optimized", False)
        
        # Determine unlock status for each skill
        business_idea_eval_unlocked = not business_idea_complete
        persona_clarification_unlocked = business_idea_complete
        painpoint_enhancement_unlocked = persona_clarified
        pitch_creation_unlocked = persona_clarified and painpoint_enhanced
        pricing_optimization_unlocked = pitch_created
        pivot_exploration_unlocked = pricing_optimized
        
        status_lines = []
        
        # business-idea-evaluation
        if business_idea_eval_unlocked:
            status_lines.append("✅ UNLOCKED: business-idea-evaluation (use this to evaluate the user's input for a complete idea)")
        else:
            status_lines.append("🔒 LOCKED: business-idea-evaluation (idea already complete, do NOT use)")
        
        # persona-clarification
        if persona_clarification_unlocked:
            if persona_clarified:
                status_lines.append("✅ UNLOCKED: persona-clarification (persona already clarified)")
            else:
                status_lines.append("✅ UNLOCKED: persona-clarification (use this after business idea is complete)")
        else:
            status_lines.append("🔒 LOCKED: persona-clarification (requires business_idea_complete == True)")
        
        # painpoint-enhancement
        if painpoint_enhancement_unlocked:
            if painpoint_enhanced:
                status_lines.append("✅ UNLOCKED: painpoint-enhancement (pain point already enhanced)")
            else:
                status_lines.append("✅ UNLOCKED: painpoint-enhancement (use this after persona is clarified)")
        else:
            status_lines.append("🔒 LOCKED: painpoint-enhancement (requires persona_clarified == True)")
        
        # 60s-pitch-creation
        if pitch_creation_unlocked:
            if pitch_created:
                status_lines.append("✅ UNLOCKED: 60s-pitch-creation (pitch already created)")
            else:
                status_lines.append("✅ UNLOCKED: 60s-pitch-creation (use this after persona clarified AND painpoint enhanced)")
        else:
            missing = []
            if not persona_clarified:
                missing.append("persona_clarified")
            if not painpoint_enhanced:
                missing.append("painpoint_enhanced")
            status_lines.append(f"🔒 LOCKED: 60s-pitch-creation (requires {' AND '.join(missing)})")
        
        # baseline-pricing-and-optimization
        if pricing_optimization_unlocked:
            if pricing_optimized:
                status_lines.append("✅ UNLOCKED: baseline-pricing-and-optimization (pricing already optimized)")
            else:
                status_lines.append("✅ UNLOCKED: baseline-pricing-and-optimization (use this after pitch is created)")
        else:
            status_lines.append("🔒 LOCKED: baseline-pricing-and-optimization (requires pitch_created == True)")
        
        # business-model-pivot-exploration
        if pivot_exploration_unlocked:
            status_lines.append("✅ UNLOCKED: business-model-pivot-exploration (use this after pricing optimization is complete)")
        else:
            status_lines.append("🔒 LOCKED: business-model-pivot-exploration (requires pricing_optimized == True)")
        
        return "\n".join(status_lines)

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
        
        # Get skill unlock status
        skill_unlock_status = self._get_skill_unlock_status(state)

        # Build context-aware prompt with skill unlock status
        if business_idea_complete and materialized_idea:
            # Idea is complete - provide summary with skill unlock information
            idea_context = (
                f"\n\n**CRITICAL: Business Idea Status & Skill Availability**\n"
                f"A complete business idea has already been identified and materialized:\n\n"
                f"{materialized_idea}\n\n"
                f"**SKILL UNLOCK STATUS:**\n"
                f"{skill_unlock_status}\n\n"
                f"**REQUIRED ACTION**: \n"
                f"- DO NOT read the business-idea-evaluation SKILL.md file\n"
                f"- DO NOT attempt a tool/function call named `business-idea-evaluation`\n"
                f"- DO NOT re-evaluate the idea\n"
                f"- DO NOT call mark_business_idea_complete again\n"
                f"- Use appropriate unlocked skills based on the user's request and current progression state"
            )
        elif business_idea_complete:
            # Idea is complete but no summary stored
            idea_context = (
                "\n\n**CRITICAL: Business Idea Status & Skill Availability**\n"
                "A complete business idea has already been identified in this conversation.\n\n"
                "**SKILL UNLOCK STATUS:**\n"
                f"{skill_unlock_status}\n\n"
                "**REQUIRED ACTION**: \n"
                "- DO NOT read the business-idea-evaluation SKILL.md file\n"
                "- DO NOT attempt a tool/function call named `business-idea-evaluation`\n"
                "- DO NOT re-evaluate the idea\n"
                "- DO NOT call mark_business_idea_complete again\n"
                "- Use appropriate unlocked skills based on the user's request and current progression state"
            )
        else:
            # No complete idea yet
            idea_context = (
                "\n\n**Current Status & Skill Availability**\n"
                "No complete business idea has been identified yet.\n\n"
                "**SKILL UNLOCK STATUS:**\n"
                f"{skill_unlock_status}\n\n"
                "**ACTION**: The business idea is not complete yet.\n"
                "- Read the `business-idea-evaluation` skill's `SKILL.md` using `read_file` (see Skills list for the exact path)\n"
                "- Follow the instructions in that SKILL.md in your normal response\n"
                "- If and only if the idea is complete, call `mark_business_idea_complete` with an idea summary\n"
                "Once an idea is identified and marked complete, other skills will unlock sequentially."
            )

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
        
        # Get skill unlock status
        skill_unlock_status = self._get_skill_unlock_status(state)

        # Build context-aware prompt with skill unlock status
        if business_idea_complete and materialized_idea:
            # Idea is complete - provide summary with skill unlock information
            idea_context = (
                f"\n\n**CRITICAL: Business Idea Status & Skill Availability**\n"
                f"A complete business idea has already been identified and materialized:\n\n"
                f"{materialized_idea}\n\n"
                f"**SKILL UNLOCK STATUS:**\n"
                f"{skill_unlock_status}\n\n"
                f"**REQUIRED ACTION**: \n"
                f"- DO NOT read the business-idea-evaluation SKILL.md file\n"
                f"- DO NOT attempt a tool/function call named `business-idea-evaluation`\n"
                f"- DO NOT re-evaluate the idea\n"
                f"- DO NOT call mark_business_idea_complete again\n"
                f"- Use appropriate unlocked skills based on the user's request and current progression state"
            )
        elif business_idea_complete:
            # Idea is complete but no summary stored
            idea_context = (
                "\n\n**CRITICAL: Business Idea Status & Skill Availability**\n"
                "A complete business idea has already been identified in this conversation.\n\n"
                "**SKILL UNLOCK STATUS:**\n"
                f"{skill_unlock_status}\n\n"
                "**REQUIRED ACTION**: \n"
                "- DO NOT read the business-idea-evaluation SKILL.md file\n"
                "- DO NOT attempt a tool/function call named `business-idea-evaluation`\n"
                "- DO NOT re-evaluate the idea\n"
                "- DO NOT call mark_business_idea_complete again\n"
                "- Use appropriate unlocked skills based on the user's request and current progression state"
            )
        else:
            # No complete idea yet
            idea_context = (
                "\n\n**Current Status & Skill Availability**\n"
                "No complete business idea has been identified yet.\n\n"
                "**SKILL UNLOCK STATUS:**\n"
                f"{skill_unlock_status}\n\n"
                "**ACTION**: The business idea is not complete yet.\n"
                "- Read the `business-idea-evaluation` skill's `SKILL.md` using `read_file` (see Skills list for the exact path)\n"
                "- Follow the instructions in that SKILL.md in your normal response\n"
                "- If and only if the idea is complete, call `mark_business_idea_complete` with an idea summary\n"
                "Once an idea is identified and marked complete, other skills will unlock sequentially."
            )

        full_prompt = self.system_prompt_template + idea_context

        if request.system_prompt:
            new_system_prompt = request.system_prompt + "\n\n" + full_prompt
        else:
            new_system_prompt = full_prompt

        return await handler(request.override(system_prompt=new_system_prompt))

