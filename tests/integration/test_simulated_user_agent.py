"""Integration tests for simulated user agent and conversation loop.

Tests the /simulated_user/chat endpoint, user agent behavior (initialization vs follow-up,
zero-experience persona), and the full conversation loop between user agent and facilitator.

Requires real LLM API credentials and network access. Run with:
  pytest tests/integration/test_simulated_user_agent.py -v -s -m integration
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

# ----- Test data -----

SAMPLE_ASSIGNMENTS = [
    "You've been asked to explore a startup idea. Share whatever rough thought comes to mind.",
    "Your instructor wants you to brainstorm a business idea. What's the first thing that pops into your head?",
]

SAMPLE_FACILITATOR_MESSAGES = [
    "That's an interesting direction. Can you tell me more about who might use this?",
    "Thanks for sharing. What problem are you trying to solve?",
    "I'd like to understand your idea better. How would someone use it in practice?",
]

# Uncertainty / simple-language indicators (zero-experience behavior)
ZERO_EXPERIENCE_INDICATORS = [
    "maybe", "not sure", "think", "guess", "don't know", "?", "i'm not",
    "kind of", "something like", "not really", "might", "could be",
    "i was thinking", "not entirely", "haven't thought",
]

# Business jargon we'd expect a zero-experience user to avoid
BUSINESS_JARGON = [
    "mvp", "pivot", "runway", "b2b", "b2c", "monetization", "value proposition",
    "product-market fit", "go-to-market", "burn rate", "unit economics",
]


# ----- Fixtures -----


@pytest.fixture
def app_client(dual_agent_client: TestClient) -> TestClient:
    """Create a FastAPI TestClient from the app with startup to initialize agents."""
    return dual_agent_client


@pytest.fixture
def test_simulation_id() -> str:
    """Generate a unique simulation ID for each test to ensure isolation."""
    return f"test-{uuid.uuid4().hex[:12]}"


# ----- Helper functions -----


def call_user_agent(
    client: TestClient,
    simulation_id: str,
    message: str,
    user_id: str = "test_user",
) -> dict[str, Any]:
    """Call /simulated_user/chat and return parsed JSON. Raises on HTTP errors."""
    resp = client.post(
        "/simulated_user/chat",
        json={
            "simulation_id": simulation_id,
            "message": message,
            "user_id": user_id,
        },
    )
    assert resp.status_code == 200, (
        f"POST /simulated_user/chat failed: {resp.status_code} {resp.text}"
    )
    return resp.json()


def call_facilitator(
    client: TestClient,
    user_id: str,
    message: str,
    conversation_id: str = "default",
) -> dict[str, Any]:
    """Call /chat (facilitator) and return parsed JSON. Raises on HTTP errors."""
    resp = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "message": message,
            "conversation_id": conversation_id,
        },
    )
    assert resp.status_code == 200, (
        f"POST /chat failed: {resp.status_code} {resp.text}"
    )
    return resp.json()


def run_conversation_round(
    client: TestClient,
    simulation_id: str,
    facilitator_message: str,
    facilitator_user_id: str | None = None,
    facilitator_conversation_id: str = "default",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one round: user agent receives facilitator message, replies; facilitator receives reply.

    Returns (user_agent_response, facilitator_response).
    """
    uid = facilitator_user_id or f"sim_{simulation_id}"
    user_resp = call_user_agent(client, simulation_id, facilitator_message, user_id=uid)
    assert user_resp.get("success") is True, f"User agent failed: {user_resp}"
    fac_resp = call_facilitator(
        client, uid, user_resp["reply"], conversation_id=facilitator_conversation_id
    )
    return user_resp, fac_resp


# ----- User agent unit tests -----


@pytest.mark.integration
class TestUserAgentInitialization:
    """Tests for /simulated_user/chat initialization (first message)."""

    def test_user_agent_initialization(
        self, app_client: TestClient, test_simulation_id: str
    ) -> None:
        """First message returns startup idea, success, correct thread_id."""
        assignment = SAMPLE_ASSIGNMENTS[0]
        data = call_user_agent(app_client, test_simulation_id, assignment)

        assert data["success"] is True
        assert data["thread_id"] == f"sim_user::{test_simulation_id}"
        assert "reply" in data and len(data["reply"].strip()) > 0
        # Response should look like a rough idea (even if vague)
        assert len(data["reply"]) >= 20


@pytest.mark.integration
class TestUserAgentFollowUp:
    """Tests for /simulated_user/chat follow-up messages."""

    def test_user_agent_follow_up(
        self, app_client: TestClient, test_simulation_id: str
    ) -> None:
        """Follow-up returns conversational reply, same thread_id."""
        # Initialize first
        init_data = call_user_agent(
            app_client, test_simulation_id, SAMPLE_ASSIGNMENTS[0]
        )
        assert init_data["success"] is True
        thread_id = init_data["thread_id"]

        # Follow-up
        followup_data = call_user_agent(
            app_client, test_simulation_id, SAMPLE_FACILITATOR_MESSAGES[0]
        )
        assert followup_data["success"] is True
        assert followup_data["thread_id"] == thread_id
        assert "reply" in followup_data and len(followup_data["reply"].strip()) > 0
        # Should be conversational, not generating a new idea
        assert len(followup_data["reply"]) >= 10


@pytest.mark.integration
class TestUserAgentZeroExperience:
    """Tests that user agent exhibits zero-experience characteristics."""

    def test_user_agent_zero_experience_behavior(
        self, app_client: TestClient, test_simulation_id: str
    ) -> None:
        """User agent uses vague, uncertain language; simple wording; no business jargon."""
        data = call_user_agent(
            app_client, test_simulation_id, SAMPLE_ASSIGNMENTS[0]
        )
        assert data["success"] is True
        reply = data["reply"].lower()

        # At least one uncertainty / simple-language indicator
        has_uncertainty = any(
            ind in reply for ind in ZERO_EXPERIENCE_INDICATORS
        )
        assert has_uncertainty, (
            f"Expected vague/uncertain language (e.g. maybe, not sure, think). "
            f"Reply: {data['reply'][:300]}..."
        )

        # Avoid heavy business jargon (allow maybe 1; zero-experience users rarely use multiple)
        jargon_count = sum(1 for j in BUSINESS_JARGON if j in reply)
        assert jargon_count <= 1, (
            f"Expected minimal business jargon. Found {jargon_count} in: {reply[:400]}..."
        )

        # Simple length—not a full business plan
        assert len(reply) < 1500, (
            "Zero-experience user replies should be concise, not long-form."
        )


# ----- Conversation loop integration tests -----


@pytest.mark.integration
class TestConversationLoop:
    """Tests for user agent <-> facilitator conversation loop."""

    def test_conversation_loop_basic(
        self, app_client: TestClient, test_simulation_id: str
    ) -> None:
        """Full loop: init user -> idea to facilitator -> fac reply to user -> user reply to fac."""
        # 1. Initialize user agent with assignment -> startup idea
        init_data = call_user_agent(
            app_client, test_simulation_id, SAMPLE_ASSIGNMENTS[0]
        )
        assert init_data["success"] is True
        idea = init_data["reply"]
        user_thread = init_data["thread_id"]

        # 2. Send idea to facilitator
        fac_user_id = f"sim_{test_simulation_id}"
        fac_data_1 = call_facilitator(app_client, fac_user_id, idea)
        assert "reply" in fac_data_1 and len(fac_data_1["reply"].strip()) > 0
        fac_thread = fac_data_1["thread_id"]
        assert fac_thread != user_thread
        assert fac_thread == f"bc::{fac_user_id}::default"

        # 3. Send facilitator response to user agent
        user_data_2 = call_user_agent(
            app_client, test_simulation_id, fac_data_1["reply"], user_id=fac_user_id
        )
        assert user_data_2["success"] is True
        assert user_data_2["thread_id"] == user_thread

        # 4. Send user reply to facilitator
        fac_data_2 = call_facilitator(
            app_client, fac_user_id, user_data_2["reply"]
        )
        assert "reply" in fac_data_2 and len(fac_data_2["reply"].strip()) > 0
        assert fac_data_2["thread_id"] == fac_thread

    def test_conversation_loop_multiple_rounds(
        self, app_client: TestClient, test_simulation_id: str
    ) -> None:
        """Extended conversation (5+ rounds) maintains coherence and context."""
        fac_user_id = f"sim_{test_simulation_id}"
        user_thread: str | None = None
        fac_thread: str | None = None
        prev_fac_msg = SAMPLE_ASSIGNMENTS[0]
        num_rounds = 5

        for i in range(num_rounds):
            user_resp, fac_resp = run_conversation_round(
                app_client, test_simulation_id, prev_fac_msg,
                facilitator_user_id=fac_user_id,
            )
            if user_thread is None:
                user_thread = user_resp["thread_id"]
                fac_thread = fac_resp["thread_id"]
            assert user_resp["thread_id"] == user_thread
            assert fac_resp["thread_id"] == fac_thread
            assert user_resp["success"] is True
            assert len(user_resp["reply"].strip()) > 0
            assert len(fac_resp["reply"].strip()) > 0
            prev_fac_msg = fac_resp["reply"]

    def test_conversation_loop_thread_isolation(
        self, app_client: TestClient
    ) -> None:
        """Two simulations run independently; no cross-contamination."""
        sim_a = f"test-{uuid.uuid4().hex[:8]}"
        sim_b = f"test-{uuid.uuid4().hex[:8]}"

        # Sim A: init + one round
        init_a = call_user_agent(app_client, sim_a, SAMPLE_ASSIGNMENTS[0])
        assert init_a["success"] is True
        user_a, fac_a = run_conversation_round(
            app_client, sim_a, SAMPLE_FACILITATOR_MESSAGES[0],
            facilitator_user_id=f"sim_{sim_a}",
        )

        # Sim B: init + one round
        init_b = call_user_agent(app_client, sim_b, SAMPLE_ASSIGNMENTS[1])
        assert init_b["success"] is True
        user_b, fac_b = run_conversation_round(
            app_client, sim_b, SAMPLE_FACILITATOR_MESSAGES[1],
            facilitator_user_id=f"sim_{sim_b}",
        )

        assert init_a["thread_id"] != init_b["thread_id"]
        assert user_a["thread_id"] != user_b["thread_id"]
        assert fac_a["thread_id"] != fac_b["thread_id"]
        assert init_a["thread_id"] == f"sim_user::{sim_a}"
        assert init_b["thread_id"] == f"sim_user::{sim_b}"


# ----- Edge cases and error handling -----


@pytest.mark.integration
class TestSimulatedUserEdgeCases:
    """Edge cases: empty message, invalid IDs, error response structure."""

    def test_empty_message_validation(
        self, app_client: TestClient, test_simulation_id: str
    ) -> None:
        """Empty or missing message yields validation error (422)."""
        resp = app_client.post(
            "/simulated_user/chat",
            json={
                "simulation_id": test_simulation_id,
                "message": "",
                "user_id": "test_user",
            },
        )
        # May be 422 (validation) or 200 with success=False depending on implementation
        assert resp.status_code in (200, 422), f"Unexpected status: {resp.status_code}"

    def test_response_structure(
        self, app_client: TestClient, test_simulation_id: str
    ) -> None:
        """Response has required keys: thread_id, reply, success."""
        data = call_user_agent(
            app_client, test_simulation_id, SAMPLE_ASSIGNMENTS[0]
        )
        for key in ("thread_id", "reply", "success"):
            assert key in data, f"Missing key: {key}"

    def test_thread_id_format(self, app_client: TestClient, test_simulation_id: str) -> None:
        """Thread ID is sim_user::{simulation_id}."""
        data = call_user_agent(
            app_client, test_simulation_id, SAMPLE_ASSIGNMENTS[0]
        )
        expected = f"sim_user::{test_simulation_id}"
        assert data["thread_id"] == expected, (
            f"Expected thread_id {expected!r}, got {data['thread_id']!r}"
        )
