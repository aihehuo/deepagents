"""Unit tests for Expert Guidance Middleware."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from deepagents.middleware.expert_guidance import ExpertGuidanceMiddleware
from deepagents.state import DualAgentState


class TestExpertGuidanceMiddleware:
    """Test Expert Guidance Middleware functionality."""

    def test_initial_state_conversation_round(self) -> None:
        """Should initialize and increment conversation_round."""
        middleware = ExpertGuidanceMiddleware()
        state: DualAgentState = {
            "messages": [],
        }
        runtime = Mock(spec=Runtime)

        # First call should increment from 0 to 1
        updates = middleware.before_agent(state, runtime)

        assert updates is not None
        assert updates["conversation_round"] == 1

    def test_conversation_round_increments(self) -> None:
        """Should increment conversation_round on each call."""
        middleware = ExpertGuidanceMiddleware()
        runtime = Mock(spec=Runtime)
        
        state: DualAgentState = {
            "messages": [HumanMessage(content="Hello")],
            "conversation_round": 5,
            "needs_expert_sync": False,
        }

        updates = middleware.before_agent(state, runtime)

        assert updates is not None
        assert updates["conversation_round"] == 6

    def test_conversation_round_from_zero(self) -> None:
        """Should handle missing conversation_round (starts at 0)."""
        middleware = ExpertGuidanceMiddleware()
        runtime = Mock(spec=Runtime)
        
        state: DualAgentState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there"),
            ],
        }

        updates = middleware.before_agent(state, runtime)

        assert updates is not None
        assert updates["conversation_round"] == 1

    def test_needs_expert_sync_flag_after_10_rounds(self) -> None:
        """Should set needs_expert_sync=True after 10 rounds."""
        middleware = ExpertGuidanceMiddleware()
        runtime = Mock(spec=Runtime)
        
        # SYNC_INTERVAL is 10 by default
        # Round 9 -> 10, last_sync = 0, so 10 - 0 >= 10
        state: DualAgentState = {
            "messages": [HumanMessage(content=f"Message {i}") for i in range(20)],
            "conversation_round": 9,
            "needs_expert_sync": False,
            "last_expert_sync": 0,
        }

        updates = middleware.before_agent(state, runtime)

        assert updates is not None
        assert updates["conversation_round"] == 10
        assert updates.get("needs_expert_sync") is True

    def test_needs_expert_sync_flag_at_multiples_of_10(self) -> None:
        """Should set needs_expert_sync=True at rounds 10, 20, 30, etc."""
        middleware = ExpertGuidanceMiddleware()
        runtime = Mock(spec=Runtime)
        
        # Test round 19 -> 20, last_sync = 10
        state: DualAgentState = {
            "messages": [HumanMessage(content=f"Message {i}") for i in range(40)],
            "conversation_round": 19,
            "needs_expert_sync": False,
            "last_expert_sync": 10,
        }

        updates = middleware.before_agent(state, runtime)

        assert updates is not None
        assert updates["conversation_round"] == 20
        assert updates.get("needs_expert_sync") is True

    def test_no_sync_flag_before_10_rounds(self) -> None:
        """Should not set needs_expert_sync before reaching 10 rounds."""
        middleware = ExpertGuidanceMiddleware()
        runtime = Mock(spec=Runtime)
        
        state: DualAgentState = {
            "messages": [HumanMessage(content=f"Message {i}") for i in range(8)],
            "conversation_round": 4,
            "needs_expert_sync": False,
            "last_expert_sync": 0,
        }

        updates = middleware.before_agent(state, runtime)

        assert updates is not None
        assert "needs_expert_sync" not in updates or updates.get("needs_expert_sync") is not True

    def test_expert_guidance_injection_into_prompt(self) -> None:
        """Should inject expert_guidance into facilitator system prompt."""
        middleware = ExpertGuidanceMiddleware()
        
        guidance_text = "Focus on customer validation and pain point exploration."
        
        state: DualAgentState = {
            "messages": [],
            "expert_guidance": guidance_text,
            "conversation_round": 10,
        }

        # Create a mock request
        base_prompt = "You are a facilitator."
        request = ModelRequest(
            state=state,
            system_prompt=base_prompt,
            model=Mock(),
            messages=[],
        )
        
        # Mock handler to capture the modified request
        modified_request = None
        def mock_handler(req: ModelRequest) -> Any:
            nonlocal modified_request
            modified_request = req
            return Mock()  # Return mock response

        # Call wrap_model_call
        middleware.wrap_model_call(request, mock_handler)

        # Check the modified prompt
        assert modified_request is not None
        assert guidance_text in modified_request.system_prompt
        assert "Strategic Guidance" in modified_request.system_prompt

    def test_default_guidance_when_none_provided(self) -> None:
        """Should use default guidance if expert_guidance not set."""
        middleware = ExpertGuidanceMiddleware()
        
        state: DualAgentState = {
            "messages": [],
            "conversation_round": 5,
        }

        base_prompt = "You are a facilitator."
        request = ModelRequest(
            state=state,
            system_prompt=base_prompt,
            model=Mock(),
            messages=[],
        )
        
        modified_request = None
        def mock_handler(req: ModelRequest) -> Any:
            nonlocal modified_request
            modified_request = req
            return Mock()

        middleware.wrap_model_call(request, mock_handler)

        # Should have default guidance
        assert modified_request is not None
        assert "Strategic Guidance" in modified_request.system_prompt
        assert len(modified_request.system_prompt) > len(base_prompt)

    def test_guidance_injection_with_empty_string(self) -> None:
        """Should use default guidance if expert_guidance is empty string."""
        middleware = ExpertGuidanceMiddleware()
        
        state: DualAgentState = {
            "messages": [],
            "expert_guidance": "",
            "conversation_round": 10,
        }

        base_prompt = "You are a facilitator."
        request = ModelRequest(
            state=state,
            system_prompt=base_prompt,
            model=Mock(),
            messages=[],
        )
        
        modified_request = None
        def mock_handler(req: ModelRequest) -> Any:
            nonlocal modified_request
            modified_request = req
            return Mock()

        middleware.wrap_model_call(request, mock_handler)

        # Should use default guidance
        assert modified_request is not None
        assert "Strategic Guidance" in modified_request.system_prompt

    def test_state_persistence_across_rounds(self) -> None:
        """Should maintain state across multiple rounds."""
        middleware = ExpertGuidanceMiddleware()
        runtime = Mock(spec=Runtime)
        
        # Initial state
        state: DualAgentState = {
            "messages": [],
            "conversation_round": 0,
            "needs_expert_sync": False,
            "last_expert_sync": 0,
        }

        # Round 1
        updates1 = middleware.before_agent(state, runtime)
        state.update(updates1 or {})
        
        assert state["conversation_round"] == 1

        # Advance to round 5
        state["conversation_round"] = 5
        updates5 = middleware.before_agent(state, runtime)
        state.update(updates5 or {})
        
        assert state["conversation_round"] == 6

    def test_custom_sync_interval(self) -> None:
        """Should respect custom SYNC_INTERVAL setting."""
        # Create middleware with custom interval (5 rounds)
        middleware = ExpertGuidanceMiddleware(sync_interval=5)
        runtime = Mock(spec=Runtime)
        
        state: DualAgentState = {
            "messages": [HumanMessage(content=f"Message {i}") for i in range(10)],
            "conversation_round": 4,
            "needs_expert_sync": False,
            "last_expert_sync": 0,
        }

        updates = middleware.before_agent(state, runtime)

        assert updates is not None
        assert updates["conversation_round"] == 5
        assert updates.get("needs_expert_sync") is True

    def test_middleware_handles_missing_fields_gracefully(self) -> None:
        """Should handle state missing optional fields."""
        middleware = ExpertGuidanceMiddleware()
        runtime = Mock(spec=Runtime)
        
        # State with minimal fields
        state: DualAgentState = {
            "messages": [HumanMessage(content="Hello")],
        }

        # Should not raise error
        updates = middleware.before_agent(state, runtime)

        assert updates is not None
        assert "conversation_round" in updates
        assert isinstance(updates["conversation_round"], int)

    def test_expert_guidance_format_in_prompt(self) -> None:
        """Should format expert guidance properly in system prompt."""
        middleware = ExpertGuidanceMiddleware()
        
        guidance = "Test guidance with specific instructions."
        
        state: DualAgentState = {
            "messages": [],
            "expert_guidance": guidance,
            "conversation_round": 12,
        }

        base_prompt = "Base prompt."
        request = ModelRequest(
            state=state,
            system_prompt=base_prompt,
            model=Mock(),
            messages=[],
        )
        
        modified_request = None
        def mock_handler(req: ModelRequest) -> Any:
            nonlocal modified_request
            modified_request = req
            return Mock()

        middleware.wrap_model_call(request, mock_handler)

        # Check formatting
        assert modified_request is not None
        assert guidance in modified_request.system_prompt
        assert "Strategic Guidance" in modified_request.system_prompt
        # Should have more content than original
        assert len(modified_request.system_prompt) > len(base_prompt) + len(guidance)
