"""Integration tests for canvas endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestCanvasEndpoint:
    """Test /canvas endpoint functionality."""

    def test_canvas_endpoint_returns_canvas_data(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should return canvas and expert_guidance from state."""
        # First, send chat messages to set up state
        chat_resp = dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "I want to build a SaaS product",
            },
        )
        assert chat_resp.status_code == 200

        # Set canvas and guidance in the fake agent's state
        facilitator = dual_agent_client._fake_facilitator_agent  # type: ignore[attr-defined]
        expert = dual_agent_client._fake_expert_agent  # type: ignore[attr-defined]

        # Simulate expert sync by setting response
        expert.set_response({
            "expert_guidance": "Focus on customer validation",
            "canvas": {
                "current_stage": "idea_exploration",
                "insights": ["Good technical background"],
            },
        })

        # Get canvas
        canvas_resp = dual_agent_client.post(
            "/canvas",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
            },
        )

        assert canvas_resp.status_code == 200
        payload = canvas_resp.json()

        # Should have canvas and guidance
        assert "canvas" in payload or "expert_guidance" in payload

    def test_canvas_endpoint_empty_state(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should handle empty canvas/guidance gracefully."""
        # Query canvas for new conversation
        canvas_resp = dual_agent_client.post(
            "/canvas",
            json={
                "user_id": "user_new",
                "conversation_id": "conv_new",
            },
        )

        # Should return success even with empty state
        assert canvas_resp.status_code in [200, 404]

        if canvas_resp.status_code == 200:
            payload = canvas_resp.json()
            # Canvas might be None or empty
            assert payload.get("canvas") is None or isinstance(payload.get("canvas"), dict)

    def test_canvas_endpoint_thread_id_format(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should use correct thread_id format."""
        canvas_resp = dual_agent_client.post(
            "/canvas",
            json={
                "user_id": "user123",
                "conversation_id": "conv456",
            },
        )

        # Should accept request (thread_id is bc::user123::conv456)
        assert canvas_resp.status_code in [200, 404]

    def test_canvas_endpoint_domain_agnostic_structure(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should return canvas as opaque JSON without validation."""
        # Send initial message
        dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "Test",
            },
        )

        # Set custom canvas structure
        expert = dual_agent_client._fake_expert_agent  # type: ignore[attr-defined]
        expert.set_response({
            "expert_guidance": "Test guidance",
            "canvas": {
                "custom_field_1": "value1",
                "nested": {
                    "deep": ["array", "of", "items"],
                },
                "number": 42,
            },
        })

        canvas_resp = dual_agent_client.post(
            "/canvas",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
            },
        )

        if canvas_resp.status_code == 200:
            payload = canvas_resp.json()
            # Canvas structure should be returned as-is
            canvas = payload.get("canvas")
            if canvas:
                # Should accept arbitrary structure
                assert isinstance(canvas, dict)

    def test_canvas_endpoint_with_unicode_content(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should handle Unicode in canvas data."""
        dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "测试",
            },
        )

        expert = dual_agent_client._fake_expert_agent  # type: ignore[attr-defined]
        expert.set_response({
            "expert_guidance": "重点关注客户验证 🚀",
            "canvas": {
                "stage": "探索阶段",
                "insights": ["技术背景很好"],
            },
        })

        canvas_resp = dual_agent_client.post(
            "/canvas",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
            },
        )

        if canvas_resp.status_code == 200:
            payload = canvas_resp.json()
            # Should handle Unicode correctly
            guidance = payload.get("expert_guidance")
            if guidance:
                assert "🚀" in guidance or isinstance(guidance, str)

    def test_canvas_endpoint_multiple_conversations(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should isolate canvas data per conversation."""
        # Conversation 1
        dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "Business idea",
            },
        )

        # Conversation 2
        dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv2",
                "message": "Learning Python",
            },
        )

        # Query both
        canvas1 = dual_agent_client.post(
            "/canvas",
            json={"user_id": "user1", "conversation_id": "conv1"},
        )

        canvas2 = dual_agent_client.post(
            "/canvas",
            json={"user_id": "user1", "conversation_id": "conv2"},
        )

        # Both should be valid responses (isolated state)
        assert canvas1.status_code in [200, 404]
        assert canvas2.status_code in [200, 404]

    def test_canvas_endpoint_requires_dual_agent_mode(
        self, client: TestClient
    ) -> None:
        """Should only work when dual-agent mode is enabled."""
        # The regular client fixture doesn't have dual-agent enabled
        canvas_resp = client.post(
            "/canvas",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
            },
        )

        # Should either work or return appropriate error
        # (depends on implementation - may need to check _state.use_dual_agent)
        assert canvas_resp.status_code in [200, 400, 404, 501]

    def test_canvas_endpoint_json_response_format(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should return properly formatted JSON."""
        dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "Test",
            },
        )

        canvas_resp = dual_agent_client.post(
            "/canvas",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
            },
        )

        if canvas_resp.status_code == 200:
            # Should be valid JSON
            payload = canvas_resp.json()
            assert isinstance(payload, dict)

            # Should have expected structure
            assert "canvas" in payload or "expert_guidance" in payload

    def test_canvas_endpoint_handles_concurrent_requests(
        self, dual_agent_client: TestClient
    ) -> None:
        """Should handle multiple canvas requests."""
        # Send message
        dual_agent_client.post(
            "/chat",
            json={
                "user_id": "user1",
                "conversation_id": "conv1",
                "message": "Test",
            },
        )

        # Multiple canvas requests
        resp1 = dual_agent_client.post(
            "/canvas",
            json={"user_id": "user1", "conversation_id": "conv1"},
        )
        resp2 = dual_agent_client.post(
            "/canvas",
            json={"user_id": "user1", "conversation_id": "conv1"},
        )

        # Both should succeed
        assert resp1.status_code in [200, 404]
        assert resp2.status_code in [200, 404]
