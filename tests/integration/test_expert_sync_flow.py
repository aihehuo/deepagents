"""Integration tests for expert sync flow."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from apps.business_cofounder_api.expert_sync import (
    trigger_and_update_expert,
    trigger_expert_analysis,
)
from deepagents.state import DualAgentState
from tests.mocks.dual_agent_mocks import (
    MockCheckpointer,
    MockExpertAgent,
    create_mock_dual_agent_state,
    create_sample_expertise_file,
)


class TestExpertSyncFlow:
    """Test expert sync flow integration."""

    @pytest.mark.asyncio
    async def test_trigger_expert_analysis_success(
        self, tmp_path: Path, fake_expert_agent: MockExpertAgent
    ) -> None:
        """Should trigger expert, get analysis, and return state updates."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()
        create_sample_expertise_file(expertise_dir, name="test_expertise")

        # Setup state
        state: DualAgentState = create_mock_dual_agent_state(
            conversation_round=10,
            needs_expert_sync=True,
            expertise_type="test_expertise",
        )

        # Add messages
        for i in range(10):
            state["messages"].append(HumanMessage(content=f"User message {i}"))
            state["messages"].append(AIMessage(content=f"AI response {i}"))

        # Set expert response
        fake_expert_agent.set_response({
            "expert_guidance": "Focus on validation",
            "canvas": {"stage": "exploration", "insights": ["test"]},
        })

        # Execute
        result = await trigger_expert_analysis(
            state=state,
            expert_agent=fake_expert_agent,
            conversation_history=state["messages"][-10:],
            thread_id="test_thread",
            expertise_dir=expertise_dir,
        )

        # Verify
        assert result["expert_guidance"] == "Focus on validation"
        assert result["canvas"]["stage"] == "exploration"
        assert fake_expert_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_trigger_expert_with_expertise_template(
        self, tmp_path: Path, fake_expert_agent: MockExpertAgent
    ) -> None:
        """Should load expertise template and include in prompt."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()

        canvas_template = '{"field1": "value1", "insights": []}'
        create_sample_expertise_file(
            expertise_dir,
            name="custom_expertise",
            canvas_template=canvas_template,
        )

        state: DualAgentState = create_mock_dual_agent_state(
            conversation_round=10,
            expertise_type="custom_expertise",
        )
        state["messages"] = [
            HumanMessage(content="Test"),
            AIMessage(content="Response"),
        ]

        fake_expert_agent.set_response({
            "expert_guidance": "Test",
            "canvas": {"field1": "value1"},
        })

        result = await trigger_expert_analysis(
            state=state,
            expert_agent=fake_expert_agent,
            conversation_history=state["messages"],
            thread_id="test_thread",
            expertise_dir=expertise_dir,
        )

        # Should have called expert agent
        assert fake_expert_agent.call_count == 1
        # Check that input contained the canvas template
        assert fake_expert_agent.last_ainvoke_input is not None

    @pytest.mark.asyncio
    async def test_trigger_and_update_expert_full_flow(
        self, tmp_path: Path, fake_expert_agent: MockExpertAgent, fake_checkpointer: MockCheckpointer
    ) -> None:
        """Should execute full sync: trigger -> analyze -> update state."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()
        create_sample_expertise_file(expertise_dir, name="business_cofounder")

        # Setup state with 10+ messages
        state: DualAgentState = create_mock_dual_agent_state(
            conversation_round=10,
            needs_expert_sync=True,
            expertise_type="business_cofounder",
        )

        for i in range(12):
            state["messages"].append(HumanMessage(content=f"Message {i}"))
            state["messages"].append(AIMessage(content=f"Response {i}"))

        # Set expert response
        fake_expert_agent.set_response({
            "expert_guidance": "Focus on customer validation",
            "canvas": {
                "current_stage": "idea_exploration",
                "insights": ["Good progress"],
            },
        })

        # Give fake_checkpointer a method to update state
        fake_checkpointer.put("test_thread", state)

        # Execute full flow
        await trigger_and_update_expert(
            thread_id="test_thread",
            state=state,
            expert_agent=fake_expert_agent,
            checkpointer=fake_checkpointer,
            expertise_dir=expertise_dir,
        )

        # Verify state was updated
        updated_state = fake_checkpointer.get("test_thread")
        assert updated_state is not None
        assert updated_state["expert_guidance"] == "Focus on customer validation"
        assert updated_state["canvas"]["current_stage"] == "idea_exploration"
        assert updated_state["last_expert_sync"] is not None

    @pytest.mark.asyncio
    async def test_expert_sync_updates_canvas(
        self, tmp_path: Path, fake_expert_agent: MockExpertAgent
    ) -> None:
        """Should update canvas field in state with expert output."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()
        create_sample_expertise_file(expertise_dir, name="test_expertise")

        state: DualAgentState = create_mock_dual_agent_state(
            canvas={"old_field": "old_value"},
            expertise_type="test_expertise",
        )
        state["messages"] = [HumanMessage(content="Test")]

        new_canvas = {
            "new_field": "new_value",
            "insights": ["Insight 1", "Insight 2"],
        }
        fake_expert_agent.set_response({
            "expert_guidance": "Test",
            "canvas": new_canvas,
        })

        result = await trigger_expert_analysis(
            state=state,
            expert_agent=fake_expert_agent,
            conversation_history=state["messages"],
            thread_id="test_thread",
            expertise_dir=expertise_dir,
        )

        # Should replace canvas
        assert result["canvas"] == new_canvas
        assert "old_field" not in result["canvas"]

    @pytest.mark.asyncio
    async def test_expert_sync_updates_guidance(
        self, tmp_path: Path, fake_expert_agent: MockExpertAgent
    ) -> None:
        """Should update expert_guidance field in state."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()
        create_sample_expertise_file(expertise_dir, name="test_expertise")

        state: DualAgentState = create_mock_dual_agent_state(
            expert_guidance="Old guidance",
            expertise_type="test_expertise",
        )
        state["messages"] = [HumanMessage(content="Test")]

        fake_expert_agent.set_response({
            "expert_guidance": "New strategic guidance",
            "canvas": {},
        })

        result = await trigger_expert_analysis(
            state=state,
            expert_agent=fake_expert_agent,
            conversation_history=state["messages"],
            thread_id="test_thread",
            expertise_dir=expertise_dir,
        )

        assert result["expert_guidance"] == "New strategic guidance"

    @pytest.mark.asyncio
    async def test_expert_sync_without_expertise_dir(
        self, fake_expert_agent: MockExpertAgent
    ) -> None:
        """Should handle None expertise_dir gracefully."""
        state: DualAgentState = create_mock_dual_agent_state(
            expertise_type="business_cofounder",
        )
        state["messages"] = [HumanMessage(content="Test")]

        fake_expert_agent.set_response({
            "expert_guidance": "Guidance",
            "canvas": {"field": "value"},
        })

        # Should not raise error
        result = await trigger_expert_analysis(
            state=state,
            expert_agent=fake_expert_agent,
            conversation_history=state["messages"],
            thread_id="test_thread",
            expertise_dir=None,
        )

        assert result["expert_guidance"] == "Guidance"

    @pytest.mark.asyncio
    async def test_expert_sync_handles_malformed_response(
        self, tmp_path: Path, fake_expert_agent: MockExpertAgent
    ) -> None:
        """Should handle malformed expert responses gracefully."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()
        create_sample_expertise_file(expertise_dir, name="test_expertise")

        state: DualAgentState = create_mock_dual_agent_state(
            expertise_type="test_expertise",
        )
        state["messages"] = [HumanMessage(content="Test")]

        # Override to return malformed response
        async def bad_ainvoke(input, config):
            return {"messages": [AIMessage(content="Not valid JSON {")]}

        fake_expert_agent.ainvoke = bad_ainvoke

        # Should handle gracefully with fallback
        result = await trigger_expert_analysis(
            state=state,
            expert_agent=fake_expert_agent,
            conversation_history=state["messages"],
            thread_id="test_thread",
            expertise_dir=expertise_dir,
        )

        # Should have fallback values
        assert "expert_guidance" in result
        assert "canvas" in result

    @pytest.mark.asyncio
    async def test_expert_sync_preserves_other_state_fields(
        self, tmp_path: Path, fake_expert_agent: MockExpertAgent, fake_checkpointer: MockCheckpointer
    ) -> None:
        """Should not overwrite unrelated state fields."""
        expertise_dir = tmp_path / "expertise"
        expertise_dir.mkdir()
        create_sample_expertise_file(expertise_dir, name="test_expertise")

        state: DualAgentState = create_mock_dual_agent_state(
            conversation_round=10,
            needs_expert_sync=True,
            expertise_type="test_expertise",
        )
        state["messages"] = [HumanMessage(content="Test")]
        # Add custom field
        state["custom_field"] = "should_persist"  # type: ignore[typeddict-unknown-key]

        fake_expert_agent.set_response({
            "expert_guidance": "Guidance",
            "canvas": {"field": "value"},
        })

        fake_checkpointer.put("test_thread", state)

        await trigger_and_update_expert(
            thread_id="test_thread",
            state=state,
            expert_agent=fake_expert_agent,
            checkpointer=fake_checkpointer,
            expertise_dir=expertise_dir,
        )

        updated_state = fake_checkpointer.get("test_thread")
        # Custom field should still be there
        assert updated_state.get("custom_field") == "should_persist"  # type: ignore[typeddict-item]
        assert updated_state["conversation_round"] == 10
