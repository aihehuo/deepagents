"""Unit tests for DualAgentState schema."""

from __future__ import annotations

from typing import Any

import pytest

from deepagents.state import DualAgentState


class TestDualAgentState:
    """Test DualAgentState TypedDict schema."""

    def test_state_has_expertise_type_field(self) -> None:
        """Should accept expertise_type field."""
        state: DualAgentState = {
            "messages": [],
            "expertise_type": "business_cofounder",
        }

        assert state["expertise_type"] == "business_cofounder"

    def test_state_has_canvas_field(self) -> None:
        """Should accept canvas as dict[str, Any]."""
        canvas_data = {
            "current_stage": "idea_exploration",
            "completeness": {"idea": 75, "customer": 50},
            "insights": ["Good progress", "Need more validation"],
        }

        state: DualAgentState = {
            "messages": [],
            "canvas": canvas_data,
        }

        assert state["canvas"] == canvas_data
        assert state["canvas"]["current_stage"] == "idea_exploration"

    def test_state_has_expert_guidance_field(self) -> None:
        """Should accept expert_guidance as string."""
        state: DualAgentState = {
            "messages": [],
            "expert_guidance": "Focus on customer validation",
        }

        assert state["expert_guidance"] == "Focus on customer validation"

    def test_state_has_conversation_round_field(self) -> None:
        """Should accept conversation_round as int."""
        state: DualAgentState = {
            "messages": [],
            "conversation_round": 15,
        }

        assert state["conversation_round"] == 15

    def test_state_has_needs_expert_sync_field(self) -> None:
        """Should accept needs_expert_sync as bool."""
        state: DualAgentState = {
            "messages": [],
            "needs_expert_sync": True,
        }

        assert state["needs_expert_sync"] is True

    def test_state_has_last_expert_sync_field(self) -> None:
        """Should accept last_expert_sync as float timestamp."""
        state: DualAgentState = {
            "messages": [],
            "last_expert_sync": 1234567890.123,
        }

        assert state["last_expert_sync"] == 1234567890.123

    def test_canvas_accepts_arbitrary_json(self) -> None:
        """Canvas should accept any JSON structure (domain-agnostic)."""
        # Business canvas
        business_canvas = {
            "current_stage": "idea_exploration",
            "completeness": {"idea": 80},
            "insights": ["test"],
        }

        state1: DualAgentState = {
            "messages": [],
            "canvas": business_canvas,
        }

        assert state1["canvas"]["current_stage"] == "idea_exploration"

        # Education canvas
        education_canvas = {
            "learning_level": "beginner",
            "topics_mastered": ["Python basics"],
            "learning_style": "visual",
        }

        state2: DualAgentState = {
            "messages": [],
            "canvas": education_canvas,
        }

        assert state2["canvas"]["learning_level"] == "beginner"

        # Health canvas
        health_canvas = {
            "goals": ["lose weight", "build muscle"],
            "progress": {"weight": -5, "strength": 10},
            "habits": ["exercise", "healthy eating"],
        }

        state3: DualAgentState = {
            "messages": [],
            "canvas": health_canvas,
        }

        assert len(state3["canvas"]["goals"]) == 2

    def test_canvas_accepts_nested_structures(self) -> None:
        """Canvas should handle deeply nested structures."""
        nested_canvas = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep",
                        "list": [1, 2, 3],
                    }
                }
            }
        }

        state: DualAgentState = {
            "messages": [],
            "canvas": nested_canvas,
        }

        assert state["canvas"]["level1"]["level2"]["level3"]["value"] == "deep"

    def test_canvas_accepts_array_of_objects(self) -> None:
        """Canvas should handle arrays of objects."""
        canvas_with_arrays = {
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"},
            ],
            "metadata": {"count": 2},
        }

        state: DualAgentState = {
            "messages": [],
            "canvas": canvas_with_arrays,
        }

        assert len(state["canvas"]["items"]) == 2
        assert state["canvas"]["items"][0]["name"] == "Item 1"

    def test_state_fields_are_optional(self) -> None:
        """Most fields should be optional (NotRequired)."""
        # Minimal valid state (only messages is required in base AgentState)
        state: DualAgentState = {
            "messages": [],
        }

        # Should not raise error
        assert "messages" in state

    def test_state_with_all_fields(self) -> None:
        """Should accept state with all dual-agent fields."""
        state: DualAgentState = {
            "messages": [],
            "expertise_type": "business_cofounder",
            "expert_guidance": "Focus on validation",
            "canvas": {"stage": "exploration"},
            "conversation_round": 12,
            "needs_expert_sync": False,
            "last_expert_sync": 1234567890.0,
        }

        assert state["expertise_type"] == "business_cofounder"
        assert state["expert_guidance"] == "Focus on validation"
        assert state["canvas"]["stage"] == "exploration"
        assert state["conversation_round"] == 12
        assert state["needs_expert_sync"] is False
        assert state["last_expert_sync"] == 1234567890.0

    def test_canvas_with_unicode(self) -> None:
        """Canvas should handle Unicode content."""
        canvas_unicode = {
            "stage": "探索阶段",
            "insights": ["技术背景很好", "需要更多验证"],
            "emoji": "🚀",
        }

        state: DualAgentState = {
            "messages": [],
            "canvas": canvas_unicode,
        }

        assert state["canvas"]["stage"] == "探索阶段"
        assert "🚀" in state["canvas"]["emoji"]

    def test_canvas_with_null_values(self) -> None:
        """Canvas should handle null/None values."""
        canvas_with_nulls = {
            "field1": None,
            "field2": "value",
            "field3": None,
        }

        state: DualAgentState = {
            "messages": [],
            "canvas": canvas_with_nulls,
        }

        assert state["canvas"]["field1"] is None
        assert state["canvas"]["field2"] == "value"

    def test_canvas_empty_dict(self) -> None:
        """Canvas should accept empty dict."""
        state: DualAgentState = {
            "messages": [],
            "canvas": {},
        }

        assert state["canvas"] == {}

    def test_expertise_type_different_domains(self) -> None:
        """Should accept different expertise types."""
        domains = [
            "business_cofounder",
            "education_mentor",
            "health_coach",
            "career_advisor",
            "custom_expertise",
        ]

        for domain in domains:
            state: DualAgentState = {
                "messages": [],
                "expertise_type": domain,
            }
            assert state["expertise_type"] == domain

    def test_state_backward_compatibility(self) -> None:
        """Should maintain backward compatibility with domain_state."""
        # domain_state is for backward compatibility with existing middleware
        state: DualAgentState = {
            "messages": [],
            "domain_state": {
                "old_field": "old_value",
            },
        }

        assert state.get("domain_state") == {"old_field": "old_value"}
