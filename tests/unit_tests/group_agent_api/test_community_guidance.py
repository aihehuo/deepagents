"""Only host-approved, customer-bound guidance can influence tool-free replies."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from apps.group_agent_api.app.community_guidance import validate_guidance
from apps.group_agent_api.app.models import validate_group_agent_metadata
from apps.group_agent_api.app.wecom_manual import execute_wecom_run


def fixture():
    value = {"schema_version": 1, "user_number": "123456", "strategy_version_id": 7, "strategy_revision": 2,
        "context_version_id": 5, "plan_id": 1, "status": "approved", "expires_at": "2099-10-23T00:00:00Z", "summary_state": "current",
        "summary": {"answered_questions": [{"text": "Already has a three-person team", "classification": "user_stated"}]},
        "directions": {"next_follow_up": "Confirm smallest pilot", "conversation_direction": "Who pays for delivered value?",
            "short_term_goal": "One explicit milestone", "do_not_repeat_or_assume": ["Do not ask team size again"]}}
    metadata = {"source": "wecom_service", "wecom_service_action": "conversation", "wecom_service_customer_number": "123456", "wecom_service_guidance": value}
    return value, metadata


def test_host_gate_rejects_public_client_injection_and_accepts_bound_async_snapshot():
    value, metadata = fixture()
    with pytest.raises(ValueError):
        validate_group_agent_metadata(metadata)
    assert validate_group_agent_metadata(metadata, allow_referral_context=True)["wecom_service_guidance"] == value


@pytest.mark.parametrize("changes", [{"status": "draft"}, {"user_number": "999"}, {"expires_at": "2020-01-01T00:00:00Z"},
    {"strategy_version_id": True}, {"summary_state": "stale"}, {"unknown": "value"}, {"directions": {"auto_send": True}}, {"schema_version": 2}])
def test_invalid_unreviewed_expired_or_wrong_customer_snapshot_rejected(changes):
    value, metadata = fixture()
    value.update(changes)
    with pytest.raises(ValueError):
        validate_guidance(value, metadata)


@pytest.mark.parametrize("action", ["group_conversation", "group_greeting", "skill_plan", "skill_execute"])
def test_personal_guidance_never_enters_group_or_tool_plans(action):
    value, metadata = fixture()
    metadata["wecom_service_action"] = action
    with pytest.raises(ValueError):
        validate_guidance(value, metadata)


def test_stale_summary_excluded_but_approved_abstract_directions_remain():
    value, metadata = fixture()
    value["summary_state"] = "stale"; value["summary"] = {}
    assert validate_guidance(value, metadata) == value
    assert validate_guidance(None, metadata) is None


async def test_approved_guidance_reaches_model_without_tools_or_fee_hypothesis():
    value, metadata = fixture()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="先确认最小试点。")))
    state = SimpleNamespace(planner_model=model, quality_model=None, polish_model=None)
    emit = AsyncMock()
    req = SimpleNamespace(metadata=metadata, message="用户最新诉求：先做一次小活动。")
    with patch("skillpacks.aihehuo.client.fetch") as fetch:
        await execute_wecom_run(req=req, session=SimpleNamespace(), state=state, tid="test", emit_callback=emit)
    fetch.assert_not_called()
    prompt = model.ainvoke.call_args.args[0]
    assert "Confirm smallest pilot" in prompt[0].content
    assert "已审核" not in prompt[1].content
    assert "用户最新诉求" in prompt[1].content
    assert "收费假设" in prompt[0].content  # Do not expose such internal hypotheses.
    assert "long_term_mutual_value_and_revenue" not in prompt[0].content
    emit.assert_awaited_once()
    assert emit.call_args.args[1]["profile_status"] == "disabled"
