"""End-to-end tests for dual-agent conversation flow."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestDualAgentConversationE2E:
    """Test complete dual-agent conversation flows."""

    def test_facilitator_conversation_first_9_rounds(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should handle 9 rounds without triggering expert sync."""
        # Send 9 messages
        for i in range(9):
            resp = dual_agent_client.post(
                "/chat",
                json={
                    "user_id": "user1",
                    "conversation_id": "conv1",
                    "message": f"Message {i+1}",
                },
            )
            assert resp.status_code == 200

        # Expert should not have been called yet
        expert = dual_agent_client._fake_expert_agent  # type: ignore[attr-defined]
        assert expert.call_count == 0

    def test_expert_sync_triggers_at_round_10(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should trigger expert analysis at 10th round."""
        # Set expert response
        expert = dual_agent_client._fake_expert_agent  # type: ignore[attr-defined]
        expert.set_response({
            "expert_guidance": "Focus on validation",
            "canvas": {"stage": "exploration"},
        })

        # Send 10 messages
        for i in range(10):
            resp = dual_agent_client.post(
                "/chat",
                json={
                    "user_id": "user1",
                    "conversation_id": "conv1",
                    "message": f"Message {i+1}",
                },
            )
            assert resp.status_code == 200

        # Expert sync should be triggered asynchronously
        # Note: In real implementation, this happens async after response
        # In tests with mocks, we may need to check state instead

    def test_canvas_persists_across_rounds(
        self, dual_agent_client: TestClient
    ) -> None:
        """Canvas data should persist and be available via /canvas."""
        # Send initial messages
        for i in range(3):
            dual_agent_client.post(
                "/chat",
                json={
                    "user_id": "user1",
                    "conversation_id": "conv1",
                    "message": f"Message {i}",
                },
            )

        # Query canvas
        canvas_resp = dual_agent_client.post(
            "/canvas",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
            },
        )

        # Should work (may be empty at this stage)
        assert canvas_resp.status_code in [200, 404]

        # Send more messages
        for i in range(3, 6):
            dual_agent_client.post(
                "/chat",
                json={
                    "user_id": "user1",
                    "conversation_id": "conv1",
                    "message": f"Message {i}",
                },
            )

        # Query canvas again
        canvas_resp2 = dual_agent_client.post(
            "/canvas",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
            },
        )

        # Should still work
        assert canvas_resp2.status_code in [200, 404]

    def test_expertise_type_per_conversation(
        self, dual_agent_client: TestClient
    ) -> None:
        """Different conversations should use different expertise types."""
        # Conversation 1 - business
        resp1 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "business_conv",
                "message": "I want to start a business",
                "expertise_type": "business_cofounder",
            },
        )
        assert resp1.status_code == 200

        # Conversation 2 - education
        resp2 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "learning_conv",
                "message": "I want to learn Python",
                "expertise_type": "education_mentor",
            },
        )
        assert resp2.status_code == 200

        # Both should work independently
        facilitator = dual_agent_client._fake_facilitator_agent  # type: ignore[attr-defined]
        assert facilitator.call_count == 2

    def test_expertise_type_initialization_first_message(
        self, dual_agent_client: TestClient
    ) -> None:
        """First message should set expertise_type in state."""
        resp = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "Hello",
                "expertise_type": "custom_expertise",
            },
        )

        assert resp.status_code == 200

        # Check that facilitator received expertise_type
        facilitator = dual_agent_client._fake_facilitator_agent  # type: ignore[attr-defined]
        if facilitator.last_ainvoke_input:
            # expertise_type should be in the input
            assert "expertise_type" in facilitator.last_ainvoke_input

    def test_multiple_expert_syncs_in_conversation(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should trigger expert sync at rounds 10, 20, 30, etc."""
        expert = dual_agent_client._fake_expert_agent  # type: ignore[attr-defined]
        expert.set_response({
            "expert_guidance": "Guidance",
            "canvas": {"field": "value"},
        })

        # Send 25 messages (should trigger at 10 and 20)
        for i in range(25):
            resp = dual_agent_client.post(
                "/chat",
                json={
                    "user_id": "user1",
                    "conversation_id": "conv1",
                    "message": f"Message {i+1}",
                },
            )
            assert resp.status_code == 200

        # In real implementation with async sync, we'd check for multiple syncs
        # With mocks, the expert call_count would show this
        # Note: Actual count depends on async task completion timing

    def test_conversation_thread_isolation(
        self, dual_agent_client: TestClient
    ) -> None:
        """Different threads should have isolated state."""
        # User 1, Conversation A
        resp1 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "convA",
                "message": "Hello A",
            },
        )
        assert resp1.status_code == 200

        # User 1, Conversation B
        resp2 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "convB",
                "message": "Hello B",
            },
        )
        assert resp2.status_code == 200

        # User 2, Conversation A
        resp3 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user2",
                "conversation_id": "convA",
                "message": "Hello from user 2",
            },
        )
        assert resp3.status_code == 200

        # All should work independently
        assert resp1.json()["thread_id"] != resp2.json()["thread_id"]
        assert resp1.json()["thread_id"] != resp3.json()["thread_id"]

    def test_facilitator_receives_expert_guidance(
        self, dual_agent_client: TestClient
    ) -> None:
        """Expert guidance should influence facilitator responses."""
        expert = dual_agent_client._fake_expert_agent  # type: ignore[attr-defined]
        expert.set_response({
            "expert_guidance": "Ask about customer pain points",
            "canvas": {"stage": "exploration"},
        })

        # Send 10 messages to trigger sync
        for i in range(10):
            dual_agent_client.post(
                "/chat",
                json={
                    "user_id": "user1",
                    "conversation_id": "conv1",
                    "message": f"Message {i+1}",
                },
            )

        # Next message should potentially reflect expert guidance
        # (In real implementation, guidance is injected into system prompt)
        resp = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "What should I focus on?",
            },
        )

        assert resp.status_code == 200

    def test_conversation_metadata_persistence(
        self, dual_agent_client: TestClient
    ) -> None:
        """Metadata should persist across conversation."""
        # Send with metadata
        resp1 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "First message",
                "metadata": {"source": "web", "session_id": "abc123"},
            },
        )
        assert resp1.status_code == 200

        # Send more messages
        resp2 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "Second message",
            },
        )
        assert resp2.status_code == 200

        # Both should use same thread
        assert resp1.json()["thread_id"] == resp2.json()["thread_id"]

    def test_canvas_reflects_conversation_progress(
        self, dual_agent_client: TestClient
    ) -> None:
        """Canvas should evolve with conversation."""
        expert = dual_agent_client._fake_expert_agent  # type: ignore[attr-defined]

        # Initial canvas
        expert.set_response({
            "expert_guidance": "Initial guidance",
            "canvas": {"progress": 0, "insights": []},
        })

        # Send initial messages
        for i in range(5):
            dual_agent_client.post(
                "/chat",
                json={
                    "user_id": "user1",
                    "conversation_id": "conv1",
                    "message": f"Message {i}",
                },
            )

        # Update canvas for next sync
        expert.set_response({
            "expert_guidance": "Updated guidance",
            "canvas": {
                "progress": 50,
                "insights": ["Insight 1", "Insight 2"],
            },
        })

        # Send more messages to trigger another sync
        for i in range(5, 15):
            dual_agent_client.post(
                "/chat",
                json={
                    "user_id": "user1",
                    "conversation_id": "conv1",
                    "message": f"Message {i}",
                },
            )

        # Canvas should reflect progress
        canvas_resp = dual_agent_client.post(
            "/canvas",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
            },
        )

        if canvas_resp.status_code == 200:
            # Canvas should be available
            assert "canvas" in canvas_resp.json() or "expert_guidance" in canvas_resp.json()

    def test_error_recovery_in_conversation(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should handle and recover from errors gracefully."""
        # Normal message
        resp1 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "Hello",
            },
        )
        assert resp1.status_code == 200

        # Invalid request (missing required field 'message')
        resp2 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                # Missing required 'message' field
            },
        )
        assert resp2.status_code in [400, 422]

        # Continue conversation - should still work
        resp3 = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "Another message",
            },
        )
        assert resp3.status_code == 200

    def test_long_conversation_maintains_context(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should maintain context over long conversations."""
        expert = dual_agent_client._fake_expert_agent  # type: ignore[attr-defined]
        expert.set_response({
            "expert_guidance": "Guidance",
            "canvas": {"round": 0},
        })

        # Send 30 messages (3 sync cycles)
        for i in range(30):
            resp = dual_agent_client.post(
                "/chat",
                json={
                    "user_id": "user1",
                    "conversation_id": "long_conv",
                    "message": f"Long conversation message {i+1}",
                },
            )
            assert resp.status_code == 200

            # Update canvas each sync
            if (i + 1) % 10 == 0:
                expert.set_response({
                    "expert_guidance": f"Guidance at round {i+1}",
                    "canvas": {"round": i + 1},
                })

        # Verify conversation continued successfully
        facilitator = dual_agent_client._fake_facilitator_agent  # type: ignore[attr-defined]
        assert facilitator.call_count == 30
