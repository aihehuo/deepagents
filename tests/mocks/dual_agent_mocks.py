"""Mock utilities for dual-agent testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from deepagents.state import DualAgentState


@dataclass
class MockExpertAgent:
    """Mock expert agent that returns pre-defined canvas and guidance."""

    response: dict[str, Any] = field(default_factory=dict)
    last_ainvoke_input: dict[str, Any] | None = None
    last_ainvoke_config: dict[str, Any] | None = None
    call_count: int = 0

    def set_response(self, response: dict[str, Any]) -> None:
        """Set the response the mock should return."""
        self.response = response

    async def ainvoke(
        self, input: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock ainvoke that returns pre-configured response."""
        self.last_ainvoke_input = input
        self.last_ainvoke_config = config
        self.call_count += 1

        # Default response if not set
        if not self.response:
            self.response = {
                "expert_guidance": "Default guidance for testing",
                "canvas": {
                    "current_stage": "test_stage",
                    "insights": ["Test insight"],
                },
            }

        # Format response as an AI message
        import json

        content = json.dumps(self.response, ensure_ascii=False)
        return {"messages": [AIMessage(content=content)]}


@dataclass
class MockFacilitatorAgent:
    """Mock facilitator agent for testing."""

    response_content: str = "Mock facilitator response"
    last_ainvoke_input: dict[str, Any] | None = None
    last_ainvoke_config: dict[str, Any] | None = None
    call_count: int = 0

    async def ainvoke(
        self, input: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock ainvoke that echoes or returns configured response."""
        self.last_ainvoke_input = input
        self.last_ainvoke_config = config
        self.call_count += 1

        # Echo user message or return configured response
        user_msg = ""
        msgs = input.get("messages") or []
        if msgs:
            user_msg = str(getattr(msgs[-1], "content", ""))

        content = self.response_content or f"Echo: {user_msg}"
        return {"messages": [AIMessage(content=content)]}


@dataclass
class MockCheckpointer:
    """Mock checkpointer for testing."""

    saved_states: dict[str, DualAgentState] = field(default_factory=dict)
    deleted_threads: list[str] = field(default_factory=list)

    def get(self, thread_id: str) -> DualAgentState | None:
        """Get state for thread."""
        return self.saved_states.get(thread_id)

    async def aget(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Async get state for thread from config."""
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is None:
            return None
        
        state = self.saved_states.get(thread_id)
        if state is None:
            return None
        
        # Return checkpoint with channel_values containing the state
        return {"channel_values": state}

    def put(self, thread_id: str, state: DualAgentState) -> None:
        """Save state for thread."""
        self.saved_states[thread_id] = state

    def delete_thread(self, thread_id: str) -> None:
        """Delete thread."""
        self.deleted_threads.append(thread_id)
        if thread_id in self.saved_states:
            del self.saved_states[thread_id]


def create_mock_dual_agent_state(**kwargs) -> DualAgentState:
    """Factory for creating test state objects with defaults.
    
    Args:
        **kwargs: Override default state fields
        
    Returns:
        DualAgentState with test-friendly defaults
    """
    default_state: DualAgentState = {
        "messages": [],
        "conversation_round": 0,
        "needs_expert_sync": False,
        "last_expert_sync": None,
        "expertise_type": "business_cofounder",
        "expert_guidance": None,
        "canvas": None,
    }

    # Override with provided kwargs
    default_state.update(kwargs)

    return default_state


def create_sample_expertise_file(
    expertise_dir: Path,
    name: str = "test_expertise",
    description: str = "Test expertise",
    canvas_template: str | None = None,
    content: str = "# Test Content\n\nSample expertise content.",
) -> Path:
    """Create a sample expertise .md file for testing.
    
    Args:
        expertise_dir: Directory to create file in
        name: Expertise name
        description: Expertise description
        canvas_template: Canvas template JSON (defaults to simple structure)
        content: Markdown content body
        
    Returns:
        Path to created file
    """
    if canvas_template is None:
        canvas_template = """{
  "field1": "value1",
  "insights": [],
  "gaps": []
}"""

    file_content = f"""---
name: {name}
description: {description}
canvas_template: |
  {canvas_template}
---

{content}
"""

    expertise_file = expertise_dir / f"{name}.md"
    expertise_file.write_text(file_content, encoding="utf-8")

    return expertise_file


def create_business_cofounder_expertise(expertise_dir: Path) -> Path:
    """Create realistic business_cofounder.md expertise file for testing."""
    canvas_template = """{
  "current_stage": "idea_exploration",
  "completeness": {
    "idea_description": 0,
    "target_customer": 0,
    "pain_point": 0,
    "solution": 0,
    "value_proposition": 0,
    "business_model": 0
  },
  "next_milestones": [],
  "insights": [],
  "gaps": [],
  "strengths": []
}"""

    content = """# Business Co-Founder Expertise

You are an expert business mentor analyzing conversations between a facilitator and entrepreneur.

## Your Role

Analyze conversations to:
1. Extract business insights and key information
2. Track progress through the entrepreneurial journey
3. Generate structured assessments (canvas data)
4. Provide strategic guidance to the facilitator

## Canvas Structure

Generate a canvas with:
- current_stage: Current phase
- completeness: Scores (0-100)
- next_milestones: Upcoming goals
- insights: Key learnings
- gaps: Areas needing exploration
- strengths: Identified advantages
"""

    return create_sample_expertise_file(
        expertise_dir=expertise_dir,
        name="business_cofounder",
        description="Business co-founder expertise for entrepreneurial guidance",
        canvas_template=canvas_template,
        content=content,
    )


def create_education_mentor_expertise(expertise_dir: Path) -> Path:
    """Create education_mentor.md expertise file for testing."""
    canvas_template = """{
  "learning_level": "beginner",
  "subject_area": "python",
  "topics_mastered": [],
  "topics_in_progress": [],
  "learning_style": "visual",
  "pace": "moderate",
  "insights": [],
  "gaps": [],
  "strengths": []
}"""

    content = """# Education Mentor Expertise

You are an expert education mentor analyzing learning conversations.

## Your Role

Analyze conversations to:
1. Assess learning progress and level
2. Identify learning style and pace
3. Track mastered and in-progress topics
4. Provide educational guidance

## Canvas Structure

- learning_level: Current proficiency
- subject_area: Focus area
- topics_mastered: Completed topics
- topics_in_progress: Active learning
- insights: Key observations
- gaps: Areas needing attention
- strengths: Learning advantages
"""

    return create_sample_expertise_file(
        expertise_dir=expertise_dir,
        name="education_mentor",
        description="Education mentoring expertise for learning guidance",
        canvas_template=canvas_template,
        content=content,
    )
