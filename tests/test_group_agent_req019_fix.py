"""Regression tests added by REQ-019-FIX.

The deterministic-save test intentionally imports only APIs that existed at
0f542d05 so the same test can prove the parent commit fails.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.agent_factory.profile_store import assert_profile_persisted
from apps.group_agent_api.app import async_manager
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState


def _session(*, user_id: str, group_id: str) -> TrustedSession:
    return TrustedSession(
        principal=SessionPrincipal(
            user_id=user_id,
            unionid=f"union-{user_id}",
            user_token="test-user-token",
            source="header",
        ),
        group_id=group_id,
        group_token="test-group-token",
        membership=MembershipResult(
            tier=CapabilityTier.in_group,
            source="token",
        ),
    )


def _request(*, user_id: str, group_id: str, message: str, run_id: str) -> AsyncCallRequest:
    return AsyncCallRequest(
        user_id=user_id,
        unionid=f"union-{user_id}",
        group_id=group_id,
        conversation_id=f"conversation-{run_id}",
        run_id=run_id,
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks",
        idempotency_key=f"idempotency-{run_id}",
        message=message,
        run_match=False,
        run_invite=False,
    )


def test_formal_async_chain_guarantees_save_after_both_model_attempts_miss(
    tmp_path: Path,
) -> None:
    """The harness, not the mock agent, must invoke the real persistence tool."""

    async def run_test() -> None:
        agent = MagicMock(spec=["ainvoke"])
        agent.ainvoke = AsyncMock(
            side_effect=lambda input_dict, config: {
                "messages": [
                    *input_dict["messages"],
                    AIMessage(content="No tool call was emitted."),
                ]
            }
        )
        state = AppState(agent=agent, base_dir=tmp_path)
        envelopes: list[dict[str, Any]] = []

        async def capture_callback(
            *, callback_url: str, envelope_dict: dict[str, Any], **kwargs: Any
        ) -> bool:
            envelopes.append(envelope_dict)
            return True

        with patch(
            "apps.group_agent_api.app.async_manager.send_callback_event",
            side_effect=capture_callback,
        ):
            await async_manager.execute_async_run(
                req=_request(
                    user_id="u-fix-save",
                    group_id="g-fix-save",
                    run_id="run-fix-save",
                    message=(
                        "正在推进爱合伙群智能体产品，希望连接熟悉社群运营与 AI "
                        "智能体落地的伙伴，可以提供产品设计和技术协作"
                    ),
                ),
                session=_session(user_id="u-fix-save", group_id="g-fix-save"),
                state=state,
                tid="thread-fix-save",
            )

        terminal = [item for item in envelopes if item["event"] == "final"]
        assert len(terminal) == 1
        assert terminal[0]["payload"]["profile_status"] == "persisted"
        assert terminal[0]["payload"]["profile_persisted"] is True
        assert agent.ainvoke.await_count == 2
        assert assert_profile_persisted(
            tmp_path, "u-fix-save", "g-fix-save"
        ).ok is True

    asyncio.run(run_test())


def test_formal_async_chain_merges_retry_reply(tmp_path: Path) -> None:
    """The second model reply remains visible when deterministic extraction is inapplicable."""

    async def run_test() -> None:
        agent = MagicMock(spec=["ainvoke"])
        history: list[Any] = []

        async def invoke(input_dict: dict[str, Any], config: Any) -> dict[str, Any]:
            history.extend(input_dict["messages"])
            reply = (
                "首轮回复。"
                if agent.ainvoke.await_count == 1
                else "强制保存轮回复。"
            )
            history.append(AIMessage(content=reply))
            return {"messages": list(history)}

        agent.ainvoke = AsyncMock(side_effect=invoke)
        state = AppState(agent=agent, base_dir=tmp_path)
        envelopes: list[dict[str, Any]] = []

        async def capture_callback(
            *, callback_url: str, envelope_dict: dict[str, Any], **kwargs: Any
        ) -> bool:
            envelopes.append(envelope_dict)
            return True

        with patch(
            "apps.group_agent_api.app.async_manager.send_callback_event",
            side_effect=capture_callback,
        ):
            await async_manager.execute_async_run(
                req=_request(
                    user_id="u-fix-reply",
                    group_id="g-fix-reply",
                    run_id="run-fix-reply",
                    message="信息尚不完整",
                ),
                session=_session(user_id="u-fix-reply", group_id="g-fix-reply"),
                state=state,
                tid="thread-fix-reply",
            )

        final_reply = next(
            item["payload"]["reply"]
            for item in envelopes
            if item["event"] == "final"
        )
        assert "首轮回复。" in final_reply
        assert "强制保存轮回复。" in final_reply

    asyncio.run(run_test())


def test_failure_classifier_correlates_save_tool_call_id_only() -> None:
    """Errors from other tools cannot overwrite the save tool's outcome."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "save_group_profile", "args": {}, "id": "save-id"},
                {"name": "search", "args": {}, "id": "search-id"},
            ],
        ),
        ToolMessage(
            content="ok: saved profile",
            tool_call_id="save-id",
        ),
        ToolMessage(
            content="error: unrelated backend failure",
            name="search",
            tool_call_id="search-id",
        ),
    ]

    assert (
        async_manager.determine_persistence_failure_reason(
            messages,
            0,
            attempt=2,
        )
        == "force_save_failed"
    )
