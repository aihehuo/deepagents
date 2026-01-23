"""State schemas for dual-agent architecture.

This module defines the shared state structure used by both facilitator (frontend)
and expert (backend) agents in the dual-agent architecture.

The canvas is a domain-agnostic JSON structure - its schema is defined by the expert's
prompt, and the backend treats it as an opaque blob. The frontend is responsible for
interpreting and displaying the canvas based on the expertise type.
"""

from typing import Any, NotRequired

from langchain.agents.middleware.types import AgentState


class DualAgentState(AgentState):
    """Shared state between facilitator (frontend) and expert (backend) agents.

    This state is shared across both agents and enables them to coordinate:
    - Facilitator tracks conversation rounds and receives guidance
    - Expert analyzes conversations and provides strategic direction
    - Canvas data is synchronized with frontend for visualization
    
    The canvas is domain-agnostic - its structure is defined by the expert's prompt
    and treated as an opaque JSON blob by the backend.
    """

    # ========== Conversation Tracking ==========

    conversation_round: NotRequired[int]
    """Current round number in the conversation (increments with each user message)."""

    last_expert_sync: NotRequired[int]
    """Round number when expert last performed analysis."""

    needs_expert_sync: NotRequired[bool]
    """Flag indicating that expert sync should be triggered."""

    # ========== Expert Configuration ==========

    expertise_type: NotRequired[str]
    """Type of expertise being used (e.g., 'business_cofounder', 'education_mentor').
    
    This determines which expertise template is loaded for expert analysis.
    Defaults to 'business_cofounder' if not specified.
    """

    # ========== Expert → Facilitator Communication ==========

    expert_guidance: NotRequired[str]
    """Strategic guidance from expert to facilitator agent.
    
    This tells the facilitator what to focus on, what questions to ask,
    or what areas to explore in upcoming conversations.
    """

    canvas: NotRequired[dict[str, Any]]
    """Canvas data - domain-agnostic JSON structure from expert analysis.
    
    The canvas structure is defined by the expert's prompt and varies by expertise type.
    The backend treats this as an opaque blob - it just stores and syncs it.
    The frontend is responsible for interpreting and displaying the canvas.
    
    Example structures:
    - Business: {"stage": "...", "completeness": {...}, "milestones": [...]}
    - Education: {"level": "...", "topics": [...], "progress": {...}}
    - Health: {"goals": [...], "metrics": {...}, "plan": [...]}
    """

    analysis_timestamp: NotRequired[str]
    """ISO 8601 timestamp of last expert analysis."""

    # ========== Memory & User Context ==========
    # (These fields are used by ApiMemoryMiddleware)

    user_id: NotRequired[str]
    """User identifier for memory paths."""

    conversation_id: NotRequired[str]
    """Conversation identifier for memory paths."""

    # ========== Artifacts ==========
    # (Used by ArtifactsMiddleware)

    artifacts: NotRequired[list[dict]]
    """List of artifact metadata (uploaded files, reports, etc.)."""

    # ========== Accounting ==========
    # (Used by AccountantMiddleware)

    tool_call_count: NotRequired[int]
    """Total tool calls made in this conversation."""

    total_input_tokens: NotRequired[int]
    """Total input tokens used."""

    total_output_tokens: NotRequired[int]
    """Total output tokens used."""

    # ========== Language Detection ==========
    # (Used by LanguageDetectionMiddleware)

    detected_language: NotRequired[str]
    """Detected language code (e.g., 'en', 'zh')."""
    
    # ========== Domain-Specific State (Backward Compatibility) ==========
    # These fields are kept for backward compatibility with existing middleware
    # (e.g., BusinessIdeaTrackerMiddleware). They can be gradually migrated to
    # be part of the canvas or removed entirely.
    
    domain_state: NotRequired[dict[str, Any]]
    """Optional domain-specific state for backward compatibility.
    
    This allows existing middleware to continue working while we migrate
    to the pure canvas approach. New code should use the canvas instead.
    """


# Type alias for backward compatibility
BusinessCofounderState = DualAgentState
