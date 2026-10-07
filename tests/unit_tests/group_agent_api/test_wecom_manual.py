"""Manual WeCom actions must never confer automatic tool permissions."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from apps.group_agent_api.app.wecom_manual import (
    PROFILE_SKILL, READ_SKILLS, decode_plan, execute_wecom_run, validate_parameters,
)


@pytest.mark.parametrize("skill", list(READ_SKILLS) + [PROFILE_SKILL])
def test_all_skills_reject_identity_and_arbitrary_urls(skill):
    for raw in ({"user_id": "other"}, {"url": "http://localhost"}, {"config": "grant"}):
        with pytest.raises(ValueError):
            validate_parameters(skill, raw)


def test_parameter_validation_and_missing_profile():
    assert decode_plan("get-ai-cofounders", '```json\n{"q":" 上海技术合伙人 "}\n```') == {"q": "上海技术合伙人"}
    with pytest.raises(ValueError):
        validate_parameters(PROFILE_SKILL, {"doing": "创业"}, executing=True)
    with pytest.raises(ValueError):
        validate_parameters("get-ai-platform-help", {"keyword": ""}, executing=True)
    with pytest.raises(ValueError):
        validate_parameters("get-ai-communities", {"q": "上海"})
    with pytest.raises(ValueError):
        validate_parameters("get-ai-cofounders", {"q": "x" * 201})
    assert validate_parameters("get-ai-startup-digest", {}, executing=True) == {}


def fixtures(metadata, output="你好，请介绍一下你的项目。", *, can_search=True):
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=output)))
    req = SimpleNamespace(metadata={"source": "wecom_service", **metadata}, message="客户说想找上海技术合伙人", run_id="ga_test", conversation_id="isolated")
    session = SimpleNamespace(capabilities=SimpleNamespace(can_search=can_search, can_write_profile=False))
    state = SimpleNamespace(planner_model=model, quality_model=None, polish_model=None, base_dir=Path("/unused"))
    emit = AsyncMock()
    return req, session, state, emit


async def test_conversation_does_not_call_search_or_writer_even_when_user_requests_it():
    req, session, state, emit = fixtures({})
    with patch("skillpacks.aihehuo.client.fetch") as fetch:
        await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
    fetch.assert_not_called()
    assert emit.call_args.args[1]["profile_status"] == "disabled"
    assert emit.call_args.args[1]["candidates"] == []


async def test_planning_only_produces_parameters_and_never_queries():
    req, session, state, emit = fixtures({"wecom_service_action": "skill_plan", "wecom_service_skill": "get-ai-cofounders"}, '{"q":"上海技术合伙人"}')
    with patch("skillpacks.aihehuo.client.fetch") as fetch:
        await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
    fetch.assert_not_called()
    assert json.loads(emit.call_args.args[1]["reply"]) == {"q": "上海技术合伙人"}


@pytest.mark.parametrize("confirmed,can_search", [(False, True), (True, False)])
async def test_execution_requires_confirmation_and_capability(confirmed, can_search):
    req, session, state, emit = fixtures({"wecom_service_action": "skill_execute", "wecom_service_skill": "get-ai-cofounders", "wecom_service_parameters": {"q": "上海"}, "wecom_service_confirmed": confirmed}, can_search=can_search)
    with patch("skillpacks.aihehuo.client.fetch") as fetch:
        with pytest.raises(ValueError):
            await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
    fetch.assert_not_called()
    emit.assert_not_called()


async def test_confirmed_query_executes_once_with_exact_human_parameters():
    req, session, state, emit = fixtures({"wecom_service_action": "skill_execute", "wecom_service_skill": "get-ai-cofounders", "wecom_service_parameters": {"q": "上海技术合伙人"}, "wecom_service_confirmed": True})
    with patch("skillpacks.aihehuo.client.fetch", return_value='{"data": [{"id":"123","wechat_reachable":true,"title":"技术伙伴","url":"https://www.aihehuo.com/users/123"}]}') as fetch:
        await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
    fetch.assert_called_once()
    assert fetch.call_args.args[0].startswith("/ai/users.json?")
    assert "page=1&per=20" in fetch.call_args.args[0]
    state.planner_model.ainvoke.assert_awaited_once()


async def test_lost_lease_prevents_query():
    req, session, state, emit = fixtures({"wecom_service_action": "skill_execute", "wecom_service_skill": "get-ai-platform-help", "wecom_service_parameters": {"keyword": "注册"}, "wecom_service_confirmed": True})
    def gate(action):
        raise RuntimeError("lost lease")
    with patch("skillpacks.aihehuo.client.fetch") as fetch:
        with pytest.raises(RuntimeError):
            await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit, side_effect_gate=gate)
    fetch.assert_not_called()


async def test_wecom_dispatch_bypasses_app_agent_and_app_requests_keep_existing_pipeline():
    from apps.group_agent_api.app.async_manager import execute_async_run_core
    req = SimpleNamespace(request_schema_version=1, input_kind="user_message", metadata={"source": "wecom_service"})
    session = SimpleNamespace(principal=None)
    state = SimpleNamespace(base_dir=Path("/unused"))
    with patch("apps.group_agent_api.app.wecom_manual.execute_wecom_run", new_callable=AsyncMock) as manual, patch("apps.group_agent_api.app.async_manager._execute_core_agent", new_callable=AsyncMock) as app:
        await execute_async_run_core(req=req, session=session, state=state, tid="t", emit_callback=AsyncMock())
        manual.assert_awaited_once()
        app.assert_not_awaited()
        req.metadata = {"source": "group_agent_ui"}
        await execute_async_run_core(req=req, session=session, state=state, tid="t", emit_callback=AsyncMock())
        app.assert_awaited_once()


async def test_profile_write_requires_permission_even_after_staff_confirmation():
    req, session, state, emit = fixtures({"wecom_service_action": "skill_execute", "wecom_service_skill": PROFILE_SKILL, "wecom_service_parameters": {"doing": "开发产品", "need": "技术", "offer": "运营"}, "wecom_service_confirmed": True})
    with pytest.raises(ValueError, match="profile_write_forbidden"):
        await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
    emit.assert_not_called()


@pytest.mark.parametrize("tool_result,succeeded", [("ok: saved profile to database", True), ("error: fence_rejected", False), ("ok: profile_superseded; database kept newer version", False)])
async def test_profile_execution_uses_exact_confirmed_fields_and_checks_write_ack(tool_result, succeeded):
    fields = {"doing": "开发餐饮软件", "need": "技术合伙人", "offer": "运营经验"}
    req, session, state, emit = fixtures({"wecom_service_action": "skill_execute", "wecom_service_skill": PROFILE_SKILL, "wecom_service_parameters": fields, "wecom_service_confirmed": True})
    session.capabilities.can_write_profile = True
    session.principal = SimpleNamespace(user_id="123")
    session.group_id = "global"
    session.membership = SimpleNamespace(tier=SimpleNamespace(value="free"))
    writer = AsyncMock(return_value=tool_result)
    with patch("apps.group_agent_api.agent_factory.agent.save_group_profile", SimpleNamespace(ainvoke=writer)):
        if succeeded:
            await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
            assert emit.call_args.args[1]["profile_status"] == "persisted"
        else:
            with pytest.raises(ValueError, match="profile_write_not_committed"):
                await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
            emit.assert_not_called()
    writer.assert_awaited_once()
    assert writer.call_args.args[0] == fields
    assert writer.call_args.kwargs["config"]["metadata"]["user_id"] == "123"
    state.planner_model.ainvoke.assert_not_awaited()
