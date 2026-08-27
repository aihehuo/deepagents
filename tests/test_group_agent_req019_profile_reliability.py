"""Permanent automated tests for REQ-019 profile persistence reliability & failure reasons.

Verifies:
1. First round missing tool call rescued by FORCE_SAVE_PROMPT in second round.
2. Two rounds persistence failure returns profile_status=failed, actionable persistence_failure_reason, and fail-closed match/invite.
3. Desensitization of failure reasons, logs, and callbacks (no tokens, secrets, HMACs, URLs, or tracebacks).
4. Strict fail-closed matching & invite behavior when profile persistence fails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from apps.group_agent_api.agent_factory.agent import FORCE_SAVE_PROMPT
from apps.group_agent_api.agent_factory.profile_store import assert_profile_persisted
from apps.group_agent_api.app.async_manager import (
    determine_persistence_failure_reason,
    execute_async_run,
)
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.principal import (
    SessionPrincipal,
)
from apps.group_agent_api.app.session import (
    TrustedSession,
)
from apps.group_agent_api.app.state import AppState


def _dummy_session(
    tier: str = "in_group",
    user_id: str = "u_req019",
    group_id: str = "g_req019",
) -> TrustedSession:
    from apps.group_agent_api.agent_factory.capability import CapabilityTier

    return TrustedSession(
        principal=SessionPrincipal(user_id=user_id, unionid="un_19", user_token="tok_19", source="header"),
        group_id=group_id,
        group_token="gtok_19",
        membership=MembershipResult(tier=CapabilityTier(tier), source="token"),
    )


# ---------------------------------------------------------------------------
# 1. Message Trace Reason Extraction Unit Tests
# ---------------------------------------------------------------------------


def test_determine_persistence_failure_reason_tool_not_called():
    """Verify tool_not_called and force_save_failed:tool_not_called when LLM makes no save_group_profile call."""
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="What are you working on?"),
    ]
    assert determine_persistence_failure_reason(messages, 0, attempt=1) == "tool_not_called"

    messages_round2 = [
        HumanMessage(content="Hello"),
        AIMessage(content="What are you working on?"),
        HumanMessage(content=FORCE_SAVE_PROMPT),
        AIMessage(content="I still didn't call the tool."),
    ]
    assert (
        determine_persistence_failure_reason(messages_round2, 0, attempt=2)
        == "force_save_failed:tool_not_called"
    )


def test_determine_persistence_failure_reason_validation_error():
    """Verify validation_error returned when save_group_profile returns semantic projection error."""
    messages = [
        HumanMessage(content="I need a dev"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_group_profile",
                    "args": {"doing": "Looking for frontend dev", "need": "Dev", "offer": "Equity"},
                    "id": "tc1",
                }
            ],
        ),
        ToolMessage(
            content="error: semantic_projection:doing_describes_need; resubmit_required",
            name="save_group_profile",
            tool_call_id="tc1",
        ),
    ]
    reason = determine_persistence_failure_reason(messages, 0, attempt=1)
    assert reason == "validation_error"


def test_determine_persistence_failure_reason_remote_ack_failed():
    """Verify remote_ack_failed returned when save_group_profile returns Micro database error."""
    messages = [
        HumanMessage(content="Building AI app"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_group_profile",
                    "args": {"doing": "Building AI app", "need": "Co-founder", "offer": "Python"},
                    "id": "tc2",
                }
            ],
        ),
        ToolMessage(
            content="error: profile_database:transport_error:ConnectionRefusedError",
            name="save_group_profile",
            tool_call_id="tc2",
        ),
    ]
    reason = determine_persistence_failure_reason(messages, 0, attempt=1)
    assert reason == "remote_ack_failed"


def test_determine_persistence_failure_reason_ignores_unrelated_tool_error():
    """An unrelated ToolMessage error must not be attributed to profile persistence."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_group_profile",
                    "args": {},
                    "id": "save-call",
                },
                {
                    "name": "unrelated_tool",
                    "args": {},
                    "id": "other-call",
                },
            ],
        ),
        ToolMessage(
            content="ok: saved profile",
            name="save_group_profile",
            tool_call_id="save-call",
        ),
        ToolMessage(
            content="error: unrelated failure with secret details",
            name="unrelated_tool",
            tool_call_id="other-call",
        ),
    ]

    assert (
        determine_persistence_failure_reason(messages, 0, attempt=2)
        == "force_save_failed"
    )


# ---------------------------------------------------------------------------
# 2. Rescue Test: Both Model Rounds Miss Tool -> Harness Saves Deterministically
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_round_missing_tool_call_rescued_by_force_save(tmp_path: Path):
    """The harness invokes the real tool; the mock model never writes the profile."""
    mock_agent = MagicMock(spec=["ainvoke"])
    state = AppState(agent=mock_agent, base_dir=tmp_path)
    session = _dummy_session(tier="in_group", user_id="u_rescue", group_id="g_rescue")

    call_count = 0

    async def mock_ainvoke(input_dict: dict[str, Any], config: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        msgs = input_dict.get("messages", [])
        return {
            "messages": [
                *msgs,
                AIMessage(content="I did not call the profile tool."),
            ]
        }

    mock_agent.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    state.agent = mock_agent

    emitted_events: list[dict[str, Any]] = []

    async def capture_callback(
        *, callback_url: str, envelope_dict: dict[str, Any], **kwargs: Any
    ) -> bool:
        emitted_events.append(envelope_dict)
        return True

    req = AsyncCallRequest(
        user_id="u_rescue",
        unionid="un_rescue",
        group_id="g_rescue",
        conversation_id="conv_rescue",
        run_id="run_rescue",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks",
        idempotency_key="idempotency_rescue",
        message=(
            "正在推进爱合伙群智能体产品，希望连接熟悉社群运营与 AI "
            "智能体落地的伙伴，可以提供产品设计和技术协作"
        ),
        run_match=False,
        run_invite=False,
    )

    with patch(
        "apps.group_agent_api.app.async_manager.send_callback_event",
        side_effect=capture_callback,
    ):
        await execute_async_run(
            req=req,
            session=session,
            state=state,
            tid="t_rescue",
            slot=None,
        )

    # Assertions
    assert call_count == 2, "Agent should be invoked twice (initial + FORCE_SAVE_PROMPT)"
    final_event = [event for event in emitted_events if event["event"] == "final"]
    assert len(final_event) == 1
    final_payload = final_event[0]["payload"]

    assert final_payload["profile_persisted"] is True
    assert final_payload["profile_status"] == "persisted"
    assert final_payload["persistence_failure_reason"] is None
    assert assert_profile_persisted(tmp_path, "u_rescue", "g_rescue").ok is True


# ---------------------------------------------------------------------------
# 3. Two Rounds Persistence Failure & Fail-Closed Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_rounds_failed_persistence_returns_actionable_reason_and_fails_closed(
    tmp_path: Path,
):
    """Verify when both rounds fail persistence, final callback has profile_status=failed, actionable persistence_failure_reason, and match/invite skipped."""
    mock_agent = MagicMock(spec=["ainvoke"])
    session = _dummy_session(tier="in_group", user_id="u_fail", group_id="g_fail")
    state = AppState(agent=mock_agent, base_dir=tmp_path)

    async def mock_ainvoke(input_dict: dict[str, Any], config: Any) -> dict[str, Any]:
        msgs = input_dict.get("messages", [])
        return {"messages": [*msgs, AIMessage(content="I refuse to call save_group_profile.")]}

    mock_agent.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    state.agent = mock_agent

    emitted_events: list[dict[str, Any]] = []

    async def capture_callback(
        *, callback_url: str, envelope_dict: dict[str, Any], **kwargs: Any
    ) -> bool:
        emitted_events.append(envelope_dict)
        return True

    req = AsyncCallRequest(
        user_id="u_fail",
        unionid="un_fail",
        group_id="g_fail",
        conversation_id="conv_fail",
        run_id="run_fail",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks",
        idempotency_key="idempotency_fail",
        message="Hello AI",
        run_match=True,
        run_invite=True,
    )

    with patch(
        "apps.group_agent_api.app.async_manager.send_callback_event",
        side_effect=capture_callback,
    ):
        with patch(
            "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
            side_effect=AssertionError("search tool should NOT run when profile fails!"),
        ):
            await execute_async_run(
                req=req,
                session=session,
                state=state,
                tid="t_fail",
                slot=None,
            )

    final_event = [event for event in emitted_events if event["event"] == "final"]
    assert len(final_event) == 1
    final_payload = final_event[0]["payload"]

    # Verify Failure Payload & Reason
    assert final_payload["profile_persisted"] is False
    assert final_payload["profile_status"] == "failed"
    assert final_payload["persistence_failure_reason"] == "force_save_failed:tool_not_called"

    # Verify Strict Fail-Closed (matching & invite skipped)
    assert final_payload["match_status"] == "skipped"
    assert final_payload["candidates"] == []
    assert final_payload["delivery_kind"] is None
    assert final_payload["invite_ok"] is None


# ---------------------------------------------------------------------------
# 4. Desensitization Test
# ---------------------------------------------------------------------------


def test_persistence_failure_reason_desensitization():
    """Verify persistence_failure_reason, logs, and callback payloads contain zero secrets, tokens, HMACs, URLs, or tracebacks."""
    raw_error_message = (
        "error: profile_database:transport_error:HTTP 500 at "
        "http://micro-web.example.invalid:3000/group_agent/profile "
        "Authorization: Bearer secret_token_123 HMAC=abcdef0123456789"
    )

    messages = [
        HumanMessage(content="User profile doing AI"),
        AIMessage(
            content="",
            tool_calls=[{"name": "save_group_profile", "args": {}, "id": "t_sens"}],
        ),
        ToolMessage(content=raw_error_message, name="save_group_profile", tool_call_id="t_sens"),
    ]

    reason = determine_persistence_failure_reason(messages, 0, attempt=2)

    # Must be a clean, canonical identifier
    assert reason == "remote_ack_failed"

    # Strict desensitization checks
    for secret_marker in [
        "secret_token",
        "HMAC",
        "http://",
        "Authorization",
        "Traceback",
        "AI",
    ]:
        assert secret_marker not in reason, f"Reason exposed sensitive marker '{secret_marker}'"


# ---------------------------------------------------------------------------
# 5. Model Tool Call Rescue & Reply Merging Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_round_missing_tool_call_rescued_by_model_tool_call(tmp_path: Path):
    """Verify round 2 model tool call triggers real save_group_profile.invoke to persist profile."""
    from apps.group_agent_api.agent_factory.agent import save_group_profile

    mock_agent = MagicMock(spec=["ainvoke"])
    state = AppState(agent=mock_agent, base_dir=tmp_path)
    session = _dummy_session(tier="in_group", user_id="u_tool_rescue", group_id="g_tool_rescue")

    call_count = 0

    async def mock_ainvoke(input_dict: dict[str, Any], config: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        msgs = input_dict.get("messages", [])
        if call_count == 1:
            return {"messages": [*msgs, AIMessage(content="Tell me more about your project.")]}

        # Round 2: Model calls real save_group_profile.invoke using tool args
        tool_args = {
            "doing": "Building AI app",
            "need": "Co-founder",
            "offer": "Python backend",
            "doing_disclosure": "inferred_unconfirmed",
            "need_disclosure": "inferred_unconfirmed",
            "offer_disclosure": "inferred_unconfirmed",
        }
        res_str = save_group_profile.invoke(tool_args, config)
        return {
            "messages": [
                *msgs,
                AIMessage(
                    content="Saved profile now.",
                    tool_calls=[{"name": "save_group_profile", "args": tool_args, "id": "tc_model"}],
                ),
                ToolMessage(content=res_str, name="save_group_profile", tool_call_id="tc_model"),
            ]
        }

    mock_agent.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    state.agent = mock_agent

    emitted_events: list[dict[str, Any]] = []

    async def capture_callback(
        *, callback_url: str, envelope_dict: dict[str, Any], **kwargs: Any
    ) -> bool:
        emitted_events.append(envelope_dict)
        return True

    req = AsyncCallRequest(
        user_id="u_tool_rescue",
        unionid="un_tool_rescue",
        group_id="g_tool_rescue",
        conversation_id="conv_tool_rescue",
        run_id="run_tool_rescue",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks",
        idempotency_key="idempotency_tool_rescue",
        message="Tell me how to start",
        run_match=False,
        run_invite=False,
    )

    with patch(
        "apps.group_agent_api.app.async_manager.send_callback_event",
        side_effect=capture_callback,
    ):
        await execute_async_run(
            req=req,
            session=session,
            state=state,
            tid="t_tool_rescue",
            slot=None,
        )

    assert call_count == 2
    final_event = [event for event in emitted_events if event["event"] == "final"]
    assert len(final_event) == 1
    final_payload = final_event[0]["payload"]

    assert final_payload["profile_persisted"] is True
    assert final_payload["profile_status"] == "persisted"
    assert final_payload["persistence_failure_reason"] is None
    assert assert_profile_persisted(tmp_path, "u_tool_rescue", "g_tool_rescue").ok is True


@pytest.mark.asyncio
async def test_retry_reply_merged_on_force_save_retry(tmp_path: Path):
    """Verify reply from second round FORCE_SAVE_PROMPT is merged into reply when profile fails persistence."""
    mock_agent = MagicMock(spec=["ainvoke"])
    state = AppState(agent=mock_agent, base_dir=tmp_path)
    session = _dummy_session(tier="in_group", user_id="u_merge", group_id="g_merge")

    call_count = 0

    async def mock_ainvoke(input_dict: dict[str, Any], config: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        msgs = input_dict.get("messages", [])
        if call_count == 1:
            return {"messages": [*msgs, AIMessage(content="First round assistant reply.")]}

        return {
            "messages": [
                *msgs,
                AIMessage(content="Second round assistant reply after save attempt."),
            ]
        }

    mock_agent.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    state.agent = mock_agent

    emitted_events: list[dict[str, Any]] = []

    async def capture_callback(
        *, callback_url: str, envelope_dict: dict[str, Any], **kwargs: Any
    ) -> bool:
        emitted_events.append(envelope_dict)
        return True

    req = AsyncCallRequest(
        user_id="u_merge",
        unionid="un_merge",
        group_id="g_merge",
        conversation_id="conv_merge",
        run_id="run_merge",
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks",
        idempotency_key="idempotency_merge",
        message="Help me match",
        run_match=False,
        run_invite=False,
    )

    with patch(
        "apps.group_agent_api.app.async_manager.send_callback_event",
        side_effect=capture_callback,
    ):
        await execute_async_run(
            req=req,
            session=session,
            state=state,
            tid="t_merge",
            slot=None,
        )

    final_event = [event for event in emitted_events if event["event"] == "final"]
    assert len(final_event) == 1
    reply = final_event[0]["payload"]["reply"]
    assert "First round assistant reply." in reply
    assert "Second round assistant reply after save attempt." in reply
