"""Frozen host policy and complete model-input tests without network calls."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from apps.group_agent_api.agent_factory.context.host_prompts import HOST_DAY, HOST_IDENTITY, HOST_NIGHT, HOST_SAFETY
from apps.group_agent_api.agent_factory.context.venue import VenueHostMiddleware
from apps.group_agent_api.agent_factory.host_context import HostContext, active_host_context
from apps.group_agent_api.agent_factory.venue_greeting import execute_venue_greeting
from apps.group_agent_api.app.models import AsyncCallRequest
from tests.test_group_agent_venue_da001 import REQUESTS, _refingerprint, _session
from tests.test_group_agent_venue_da002 import BareGreetingModel

POLICY_VECTORS = json.loads((Path(__file__).parent / "fixtures/venue_entry/cafe-host-role.json").read_text())["policy_vectors"]


@pytest.mark.parametrize("vector", POLICY_VECTORS, ids=[item["id"] for item in POLICY_VECTORS])
def test_all_frozen_policy_vectors(vector: dict[str, Any], tmp_path: Path) -> None:
    raw = copy.deepcopy(next(item["request"] for item in REQUESTS if item["name"] == vector["base"]))
    for path, value in vector["set"].items():
        target = raw
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    _refingerprint(raw)
    if "expected_sha256" in vector:
        assert raw["request_fingerprint"] == vector["expected_sha256"]
    expected = vector["expected"]
    if expected in {"schema_invalid", "binding_mismatch", "space_not_available"}:
        # Deep rejects these at strict schema; Micro owns the earlier stable
        # client-facing binding/space error codes in the shared fixture.
        with pytest.raises(ValueError):
            AsyncCallRequest.model_validate(raw)
        return
    req = AsyncCallRequest.model_validate(raw)
    if expected == "unsupported_bucket":
        with pytest.raises(ValueError, match="not an admin bucket"):
            HostContext.from_request(req, principal=_session(req).principal, base_dir=tmp_path)
        return
    host = HostContext.from_request(req, principal=_session(req).principal, base_dir=tmp_path)
    assert host.is_cafe_host == (expected == "cafe_host")


def _request(*, index: int = 0, version: int = 2) -> dict[str, Any]:
    body = copy.deepcopy(REQUESTS[index]["request"])
    body["venue_context"].update(host_config_version=version, context_policy_version=version)
    if version == 2:
        body.update(run_match=False, run_invite=False, willing_to_at=False)
    return _refingerprint(body)


def _host(tmp_path: Path, *, index: int = 0, version: int = 2) -> HostContext:
    req = AsyncCallRequest.model_validate(_request(index=index, version=version))
    return HostContext.from_request(req, principal=_session(req).principal, base_dir=tmp_path)


@pytest.mark.parametrize(("index", "expected"), [(0, "4838ee66320028cf1853b6b88d51c407ff921efd678939ac31ce842706f857ce"), (1, "3887d7526a0b383fdd1ae594a0a48052f78051bb61ced52fb46971672c53f0b9"), (2, "65f2ad3401651e446f9cea2cf97041abbcfac917ae4556b684af9ba4ad74b397")])
def test_frozen_host_vectors_keep_full_signed_body(index: int, expected: str) -> None:
    raw = _request(index=index)
    assert raw["request_fingerprint"] == expected
    assert AsyncCallRequest.model_validate(raw).model_dump() == raw


@pytest.mark.parametrize("pair", [(1, 2), (2, 1), (3, 3), ("2", "2"), (True, True), (2.0, 2.0)])
def test_reject_mixed_unknown_or_coerced_versions(pair: tuple[Any, Any]) -> None:
    raw = _request()
    raw["venue_context"].update(host_config_version=pair[0], context_policy_version=pair[1])
    with pytest.raises(ValueError):
        AsyncCallRequest.model_validate(_refingerprint(raw))


@pytest.mark.parametrize("flag", ["run_match", "run_invite", "willing_to_at"])
def test_new_text_rejects_execution_flag_escalation(flag: str) -> None:
    raw = _request(index=2)
    raw[flag] = True
    with pytest.raises(ValueError):
        AsyncCallRequest.model_validate(_refingerprint(raw))


@pytest.mark.parametrize("index", [0, 1])
def test_complete_model_input_replaces_conflicting_base_and_checkpoint_systems(tmp_path: Path, index: int) -> None:
    host = _host(tmp_path, index=index)
    old = SystemMessage(content="LEGACY_FORCE_SAVE_SEARCH", name="known_profile")
    pending = AIMessage(content="", tool_calls=[{"id": "old", "name": "search_candidates", "args": {}}])
    tool = ToolMessage(content="OLD_PRIVATE_TOOL_DATA", tool_call_id="old")
    user = HumanMessage(content="随便聊聊，不需要方案")
    request = ModelRequest(model=MagicMock(), messages=[old, pending, tool, user], system_message=SystemMessage(content="LEGACY_ADVISOR_BASE"), tools=[])
    with active_host_context(host):
        prepared = VenueHostMiddleware().prepare(request)
    assert prepared.messages == [user]
    assert HOST_SAFETY in prepared.system_message.text
    assert HOST_IDENTITY in prepared.system_message.text
    assert (HOST_DAY if index == 0 else HOST_NIGHT) in prepared.system_message.text
    rendered = prepared.system_message.text + "".join(str(item.content) for item in prepared.messages)
    for forbidden in ("LEGACY_", "OLD_PRIVATE_TOOL_DATA", "三维齐备后必须调用", "一个可行下一步", "挖需求 + 画像落库"):
        assert forbidden not in rendered
    assert request.messages == [old, pending, tool, user]
    assert request.system_message.text == "LEGACY_ADVISOR_BASE"


async def test_full_greeting_model_request_is_host_not_advice_menu(tmp_path: Path) -> None:
    host = _host(tmp_path)
    model = BareGreetingModel()
    await execute_venue_greeting(host=host, model=model, emit_callback=AsyncMock(return_value=True))
    call = model.calls[0]
    assert call["kwargs"]["tools"] == []
    assert len(call["messages"]) == 1
    content = call["messages"][0].content
    assert HOST_IDENTITY in content and HOST_DAY in content
    assert "不询问进展" in content and "选项可为空" in content
    assert "一个可行下一步" not in content
    assert "明确请你帮忙时认真回答" in content
    assert "当前没有可分享的其他客人故事源" in content


async def test_parallel_versions_do_not_change_shared_prompt_or_requests(tmp_path: Path) -> None:
    middleware = VenueHostMiddleware()

    async def prepare(version: int) -> str:
        request = ModelRequest(model=MagicMock(), messages=[HumanMessage(content="hello")], system_message=SystemMessage(content="LEGACY_BASE"), tools=[])
        with active_host_context(_host(tmp_path, version=version)):
            await asyncio.sleep(0)
            return middleware.prepare(request).system_message.text

    old, new = await asyncio.gather(prepare(1), prepare(2))
    assert "LEGACY_BASE" in old and HOST_IDENTITY not in old
    assert "LEGACY_BASE" not in new and HOST_IDENTITY in new
