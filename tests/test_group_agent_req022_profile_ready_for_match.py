"""Regression tests for REQ-022.

Verifies that when a healthy profile already exists on disk, match and invite
pipelines proceed even if the model does not call save_group_profile in this turn.
"""

from __future__ import annotations

import asyncio

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.agent_factory.match_stub import MatchResult
from apps.group_agent_api.agent_factory.profile_store import (
    disk_profile_path,
)
from apps.group_agent_api.app import async_manager
from apps.group_agent_api.app.models import AsyncCallRequest, ChatRequest
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


def _async_request(*, user_id: str, group_id: str, message: str, run_id: str) -> AsyncCallRequest:
    return AsyncCallRequest(
        user_id=user_id,
        unionid=f"union-{user_id}",
        group_id=group_id,
        conversation_id=f"conversation-{run_id}",
        run_id=run_id,
        callback_url="http://micro-web.example.invalid:3000/group_agent_callbacks",
        idempotency_key=f"idempotency-{run_id}",
        message=message,
        run_match=True,
        run_invite=False,
    )


def _seed_profile(base_dir: Path, user_id: str, group_id: str) -> None:
    path = disk_profile_path(base_dir, user_id, group_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'{{\n'
        f'  "user_id": "{user_id}",\n'
        f'  "group_id": "{group_id}",\n'
        f'  "doing": {{"value": "推进 AI 项目", "disclosure": "confirmed_public"}},\n'
        f'  "need": {{"value": "寻找前端伙伴", "disclosure": "confirmed_public"}},\n'
        f'  "offer": {{"value": "提供后端架构", "disclosure": "confirmed_public"}},\n'
        f'  "updated_at": "2026-07-27T12:00:00.000000Z",\n'
        f'  "schema_version": 1\n'
        f'}}\n',
        encoding="utf-8",
    )


def test_hot_path_existing_profile_allows_match_without_save_call(
    tmp_path: Path,
) -> None:
    """Hot path: Pre-existing profile allows run_match even if model calls no save tool."""
    _seed_profile(tmp_path, "u-hot-1", "g-hot-1")

    async def run_test() -> None:
        agent = MagicMock(spec=["ainvoke"])
        agent.ainvoke = AsyncMock(
            side_effect=lambda input_dict, config: {
                "messages": [
                    *input_dict["messages"],
                    AIMessage(content="收到，正在为您在本群寻找互补伙伴。"),
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

        mock_match_result = MatchResult(
            status="matched",
            candidates=[
                {
                    "user_id": "u-cand-1",
                    "group_id": "g-hot-1",
                    "doing": {"value": "做前端与 UI 设计", "disclosure": "confirmed_public"},
                    "need": {"value": "找后端架构", "disclosure": "confirmed_public"},
                    "offer": {"value": "精通 Vue & Tailwind", "disclosure": "confirmed_public"},
                }
            ],
            query="test query",
            group_id="g-hot-1",
            reason="matched_via_vector_search",
        )

        with patch(
            "apps.group_agent_api.app.async_manager.send_callback_event",
            side_effect=capture_callback,
        ), patch(
            "apps.group_agent_api.app.async_manager.run_match",
            return_value=mock_match_result,
        ):
            await async_manager.execute_async_run(
                req=_async_request(
                    user_id="u-hot-1",
                    group_id="g-hot-1",
                    run_id="run-hot-1",
                    message="请帮我匹配互补的合作伙伴",
                ),
                session=_session(user_id="u-hot-1", group_id="g-hot-1"),
                state=state,
                tid="thread-hot-1",
            )

        terminal = [item for item in envelopes if item["event"] == "final"]
        assert len(terminal) == 1
        payload = terminal[0]["payload"]
        assert payload["profile_status"] == "persisted"
        assert payload["profile_persisted"] is True
        assert payload["match_status"] == "matched"
        assert payload["match_reason"] == "matched_via_vector_search"
        assert len(payload["candidates"]) == 1
        # No extra FORCE_SAVE retry should be invoked
        assert agent.ainvoke.await_count == 1

    asyncio.run(run_test())


def test_cold_start_no_profile_fails_and_skips_match(
    tmp_path: Path,
) -> None:
    """Cold start: Missing profile fails closed if model calls no save tool."""

    async def run_test() -> None:
        agent = MagicMock(spec=["ainvoke"])
        agent.ainvoke = AsyncMock(
            side_effect=lambda input_dict, config: {
                "messages": [
                    *input_dict["messages"],
                    AIMessage(content="未包含足够画像信息。"),
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
                req=_async_request(
                    user_id="u-cold-1",
                    group_id="g-cold-1",
                    run_id="run-cold-1",
                    message="请帮我匹配",
                ),
                session=_session(user_id="u-cold-1", group_id="g-cold-1"),
                state=state,
                tid="thread-cold-1",
            )

        terminal = [item for item in envelopes if item["event"] == "final"]
        assert len(terminal) == 1
        payload = terminal[0]["payload"]
        assert payload["profile_status"] == "failed"
        assert payload["profile_persisted"] is False
        assert payload["match_status"] == "skipped"
        assert payload["match_reason"] == "profile_persistence_failed"
        assert agent.ainvoke.await_count >= 2

    asyncio.run(run_test())


def test_hot_path_incremental_save_success(
    tmp_path: Path,
) -> None:
    """Hot path: Incremental update of existing profile succeeds and runs match."""
    _seed_profile(tmp_path, "u-inc-1", "g-inc-1")

    async def run_test() -> None:
        agent = MagicMock(spec=["ainvoke"])

        async def invoke_with_save(input_dict: dict[str, Any], config: Any) -> dict[str, Any]:
            # Simulate tool saving updated profile
            path = disk_profile_path(tmp_path, "u-inc-1", "g-inc-1")
            path.write_text(
                '{\n'
                '  "user_id": "u-inc-1",\n'
                '  "group_id": "g-inc-1",\n'
                '  "doing": {"value": "更新做 AI Agent", "disclosure": "confirmed_public"},\n'
                '  "need": {"value": "寻找前端伙伴", "disclosure": "confirmed_public"},\n'
                '  "offer": {"value": "提供全栈技术", "disclosure": "confirmed_public"},\n'
                '  "updated_at": "2026-07-28T00:00:00.000000Z",\n'
                '  "schema_version": 1\n'
                '}\n',
                encoding="utf-8",
            )
            return {
                "messages": [
                    *input_dict["messages"],
                    AIMessage(content="好的，已为您更新画像并开始匹配。"),
                ]
            }

        agent.ainvoke = AsyncMock(side_effect=invoke_with_save)
        state = AppState(agent=agent, base_dir=tmp_path)
        envelopes: list[dict[str, Any]] = []

        async def capture_callback(
            *, callback_url: str, envelope_dict: dict[str, Any], **kwargs: Any
        ) -> bool:
            envelopes.append(envelope_dict)
            return True

        mock_match_result = MatchResult(
            status="empty",
            candidates=[],
            query="test query",
            group_id="g-inc-1",
            reason="empty_pool",
        )

        with patch(
            "apps.group_agent_api.app.async_manager.send_callback_event",
            side_effect=capture_callback,
        ), patch(
            "apps.group_agent_api.app.async_manager.run_match",
            return_value=mock_match_result,
        ):
            await async_manager.execute_async_run(
                req=_async_request(
                    user_id="u-inc-1",
                    group_id="g-inc-1",
                    run_id="run-inc-1",
                    message="更新做 AI Agent 寻前端",
                ),
                session=_session(user_id="u-inc-1", group_id="g-inc-1"),
                state=state,
                tid="thread-inc-1",
            )

        terminal = [item for item in envelopes if item["event"] == "final"]
        assert len(terminal) == 1
        payload = terminal[0]["payload"]
        assert payload["profile_status"] == "persisted"
        assert payload["match_status"] == "empty"

    asyncio.run(run_test())


def test_sync_chat_hot_path_allows_match_without_save(
    tmp_path: Path,
) -> None:
    """Sync endpoint: Existing profile allows match without save tool call."""
    _seed_profile(tmp_path, "u-sync-1", "g-sync-1")

    async def run_test() -> None:
        from apps.group_agent_api.app.endpoints.chat import chat

        agent = MagicMock(spec=["ainvoke"])
        agent.ainvoke = AsyncMock(
            side_effect=lambda input_dict, config: {
                "messages": [
                    *input_dict["messages"],
                    AIMessage(content="同步匹配回复。"),
                ]
            }
        )
        state = AppState(agent=agent, base_dir=tmp_path)
        req = ChatRequest(
            user_id="u-sync-1",
            unionid="union-u-sync-1",
            group_id="g-sync-1",
            message="请帮我匹配",
            run_match=True,
            run_invite=False,
        )
        sess = _session(user_id="u-sync-1", group_id="g-sync-1")

        mock_match_result = MatchResult(
            status="empty",
            candidates=[],
            query="test query",
            group_id="g-sync-1",
            reason="empty_pool",
        )

        with patch(
            "apps.group_agent_api.app.endpoints.chat.resolve_trusted_session",
            AsyncMock(return_value=sess),
        ), patch(
            "apps.group_agent_api.app.endpoints.chat.run_match",
            return_value=mock_match_result,
        ):
            resp = await chat(req=req, state=state)

        assert resp.profile_status == "persisted"
        assert resp.match_status == "empty"
        assert resp.match_reason == "empty_pool"

    asyncio.run(run_test())


def test_hot_path_save_tool_error_still_allows_match_using_existing_healthy_profile(
    tmp_path: Path,
) -> None:
    """Contract §1.4 Silent Reuse: Existing healthy profile allows match even if save tool fails."""
    _seed_profile(tmp_path, "u-err-1", "g-err-1")

    async def run_test() -> None:
        agent = MagicMock(spec=["ainvoke"])

        async def invoke_with_tool_error(input_dict: dict[str, Any], config: Any) -> dict[str, Any]:
            # Model attempts save tool call, but tool returns error or fails to change disk file
            return {
                "messages": [
                    *input_dict["messages"],
                    AIMessage(content="尝试保存更新画像失败：error: validation_error"),
                ]
            }

        agent.ainvoke = AsyncMock(side_effect=invoke_with_tool_error)
        state = AppState(agent=agent, base_dir=tmp_path)
        envelopes: list[dict[str, Any]] = []

        async def capture_callback(
            *, callback_url: str, envelope_dict: dict[str, Any], **kwargs: Any
        ) -> bool:
            envelopes.append(envelope_dict)
            return True

        mock_match_result = MatchResult(
            status="empty",
            candidates=[],
            query="test query",
            group_id="g-err-1",
            reason="empty_pool",
        )

        with patch(
            "apps.group_agent_api.app.async_manager.send_callback_event",
            side_effect=capture_callback,
        ), patch(
            "apps.group_agent_api.app.async_manager.run_match",
            return_value=mock_match_result,
        ):
            await async_manager.execute_async_run(
                req=_async_request(
                    user_id="u-err-1",
                    group_id="g-err-1",
                    run_id="run-err-1",
                    message="更新画像并帮我匹配",
                ),
                session=_session(user_id="u-err-1", group_id="g-err-1"),
                state=state,
                tid="thread-err-1",
            )

        terminal = [item for item in envelopes if item["event"] == "final"]
        assert len(terminal) == 1
        payload = terminal[0]["payload"]
        # Silent reuse contract: pre-existing profile is valid on disk, so profile_status remains persisted & match runs
        assert payload["profile_status"] == "persisted"
        assert payload["profile_persisted"] is True
        assert payload["match_status"] == "empty"
        assert payload["match_reason"] == "empty_pool"

    asyncio.run(run_test())
