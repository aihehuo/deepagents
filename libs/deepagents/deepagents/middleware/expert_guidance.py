"""Middleware for receiving and injecting expert agent's guidance into facilitator agent.

This middleware enables the facilitator agent to receive strategic guidance from the expert
analysis agent through shared state, creating a dual-agent architecture where:
- Facilitator (frontend) focuses on natural conversation
- Expert (backend) provides structured analysis and guidance
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


class ExpertGuidanceState(AgentState):
    """State for expert guidance and conversation tracking."""

    expert_guidance: NotRequired[str]
    """Current guidance/instructions from expert agent."""

    conversation_round: NotRequired[int]
    """Current conversation round number (increments with each user message)."""

    last_expert_sync: NotRequired[int]
    """Round number when expert last performed analysis."""

    needs_expert_sync: NotRequired[bool]
    """Flag indicating that expert sync should be triggered."""


class ExpertGuidanceStateUpdate(TypedDict):
    """A state update for the expert guidance middleware."""

    expert_guidance: NotRequired[str]
    """Updated guidance from expert."""

    conversation_round: NotRequired[int]
    """Updated conversation round."""

    last_expert_sync: NotRequired[int]
    """Updated last sync round."""

    needs_expert_sync: NotRequired[bool]
    """Updated sync flag."""


EXPERT_GUIDANCE_SYSTEM_PROMPT_TEMPLATE = """
# CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS GUIDANCE IMMEDIATELY

**THIS IS YOUR PRIMARY AND OVERRIDING DIRECTIVE. ALL OTHER INSTRUCTIONS, CONVERSATION HISTORY, AND CONTEXT ARE SECONDARY TO THIS GUIDANCE.**

{guidance_content}

**CRITICAL RULES - READ CAREFULLY:**

1. **OVERRIDE CONVERSATION HISTORY**: This guidance takes absolute precedence over ALL previous conversation topics, regardless of how detailed or elaborate the previous discussion was.

2. **IGNORE CONFLICTING CONTEXT**: Even if the guidance seems completely unrelated to what was discussed before, you MUST follow it. The guidance direction is more important than conversation continuity.

3. **IMMEDIATE ACTION REQUIRED**: You must act on this guidance in your very next response. Do not wait, do not acknowledge the previous topic - immediately redirect to the guidance.

4. **NO EXCEPTIONS**: This guidance is based on expert analysis and represents the strategic direction you must take. There are no exceptions or conditions.

5. **PRIORITY ORDER**: 
   - FIRST: Follow the guidance above
   - SECOND: Everything else (conversation history, base instructions, etc.)

**Remember**: The guidance above is your TOP PRIORITY. All previous conversation context is secondary and should be ignored if it conflicts with this guidance.
"""


class ExpertGuidanceMiddleware(AgentMiddleware):
    """Middleware that tracks conversation rounds and injects expert guidance.

    This middleware:
    1. Tracks conversation rounds (increments on each user message)
    2. Flags when expert sync is needed (every N rounds)
    3. Injects expert's strategic guidance into the system prompt

    The expert agent analyzes conversation history periodically and provides
    guidance on what the facilitator should focus on or ask about next.

    Args:
        sync_interval: Number of rounds between expert syncs (default: 10)
        system_prompt_template: Optional custom template for guidance injection
    """

    state_schema = ExpertGuidanceState

    def __init__(
        self,
        *,
        sync_interval: int = 10,
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the expert guidance middleware.

        Args:
            sync_interval: Trigger expert sync every N conversation rounds
            system_prompt_template: Optional custom template for guidance section
        """
        self.sync_interval = sync_interval
        self.system_prompt_template = (
            system_prompt_template or EXPERT_GUIDANCE_SYSTEM_PROMPT_TEMPLATE
        )

    def before_agent(
        self,
        state: ExpertGuidanceState,
        runtime: Runtime,
    ) -> ExpertGuidanceStateUpdate | None:
        """Track conversation rounds and flag expert sync if needed.

        Args:
            state: Current agent state
            runtime: Runtime context

        Returns:
            State updates for round tracking and sync flagging
        """
        # Initialize round tracking if not present
        current_round = state.get("conversation_round", 0)
        last_sync = state.get("last_expert_sync", 0)

        # Increment conversation round
        new_round = current_round + 1

        updates: ExpertGuidanceStateUpdate = {"conversation_round": new_round}

        # Check if expert sync is needed
        if new_round - last_sync >= self.sync_interval:
            updates["needs_expert_sync"] = True
            print(
                f"[ExpertGuidanceMiddleware] Round {new_round}: Expert sync needed "
                f"(last sync at round {last_sync})"
            )

        return updates

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject expert guidance into the system prompt.

        This runs on every model call to ensure guidance is always present.

        Args:
            request: The model request being processed
            handler: The handler function to call with the modified request

        Returns:
            The model response from the handler
        """
        # Get current expert guidance from state
        state = cast("ExpertGuidanceState", request.state)
        guidance = state.get("expert_guidance", "")

        # Check if we have strategic guidance from expert (not default/empty)
        has_strategic_guidance = guidance and guidance.strip() and guidance != DEFAULT_GUIDANCE

        # Use default guidance if none provided yet
        if not guidance:
            guidance = DEFAULT_GUIDANCE

        # Format the guidance section
        guidance_section = self.system_prompt_template.format(
            guidance_content=guidance
        )

        # Log the injected guidance
        _logger.info("=" * 80)
        _logger.info("[ExpertGuidanceMiddleware] Injecting expert guidance into facilitator prompt:")
        _logger.info("-" * 80)
        _logger.info("Raw Guidance: %s", guidance)
        _logger.info("-" * 80)
        _logger.info("Has Strategic Guidance (replaces base prompt): %s", has_strategic_guidance)
        _logger.info("-" * 80)
        _logger.info("Formatted Guidance Section:")
        _logger.info("%s", guidance_section)
        _logger.info("-" * 80)
        if request.system_prompt:
            _logger.info("Base System Prompt (first 200 chars): %s", request.system_prompt[:200])
        _logger.info("=" * 80)

        # Inject into system prompt
        # If we have strategic guidance from expert, completely replace the base prompt
        # Otherwise, append to the base prompt (default behavior)
        if has_strategic_guidance:
            # Strategic guidance completely replaces the base system prompt
            new_system_prompt = guidance_section
            _logger.info("[ExpertGuidanceMiddleware] Strategic guidance detected - replacing base system prompt")
        elif request.system_prompt:
            # Default guidance or no guidance - append to base prompt
            new_system_prompt = request.system_prompt + "\n\n" + guidance_section
        else:
            # No base prompt - use guidance only
            new_system_prompt = guidance_section

        return handler(request.override(system_prompt=new_system_prompt))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject expert guidance into the system prompt.

        Args:
            request: The model request being processed
            handler: The handler function to call with the modified request

        Returns:
            The model response from the handler
        """
        # Get current expert guidance from state
        state = cast("ExpertGuidanceState", request.state)
        guidance = state.get("expert_guidance", "")

        # Use default guidance if none provided yet
        if not guidance:
            guidance = DEFAULT_GUIDANCE

        # Format the guidance section
        guidance_section = self.system_prompt_template.format(
            guidance_content=guidance
        )

        if guidance_section:
            # Log the injected guidance
            _logger.info("=" * 80)
            _logger.info("[ExpertGuidanceMiddleware] Injecting expert guidance into facilitator prompt (async):")
            _logger.info("-" * 80)
            _logger.info("Raw Guidance: %s", guidance)
            _logger.info("-" * 80)
            _logger.info("Formatted Guidance Section:")
            _logger.info("%s", guidance_section)
            _logger.info("-" * 80)
            if request.system_prompt:
                _logger.info("Base System Prompt (first 200 chars): %s", request.system_prompt[:200])
            _logger.info("=" * 80)

            # Inject into system prompt
            # If we have strategic guidance from expert, completely replace the base prompt
            # Otherwise, append to the base prompt (default behavior)
            if request.system_prompt:
                # Default guidance or no guidance - append to base prompt
                new_system_prompt = request.system_prompt + "\n\n" + guidance_section
            else:
                # No base prompt - use guidance only
                new_system_prompt = guidance_section

            # Create new request with overridden system prompt
            new_request = request.override(system_prompt=new_system_prompt)
        
        # Verify the override worked
        if hasattr(new_request, "system_prompt"):
            if new_request.system_prompt == new_system_prompt:
                _logger.info("[ExpertGuidanceMiddleware] ✓ System prompt override verified - matches expected content")
            else:
                _logger.error("[ExpertGuidanceMiddleware] ✗ System prompt override FAILED - content mismatch!")
                _logger.error("[ExpertGuidanceMiddleware] Expected length: %d, Got length: %d", 
                            len(new_system_prompt), len(new_request.system_prompt) if new_request.system_prompt else 0)
        else:
            _logger.error("[ExpertGuidanceMiddleware] ✗ System prompt attribute missing after override!")
        
        return await handler(new_request)
