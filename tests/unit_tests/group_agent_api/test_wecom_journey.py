"""Personal service convergence cannot confer permissions or leak internal summaries."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from apps.group_agent_api.app.models import validate_group_agent_metadata
from apps.group_agent_api.app.wecom_journey import decode_reply, validate_context, validate_journey
from apps.group_agent_api.app.wecom_manual import execute_wecom_run


def inputs():
    context = {"schema_version": 1, "user_number": "123456", "sources": {"wecom_turn:1:input:0": {"role": "user", "text": "找小程序技术伙伴和渠道合作方，我提供试点门店", "recorded_at": datetime.now(timezone.utc).isoformat()}}, "previous": None, "observed_at": datetime.now(timezone.utc).isoformat()}
    metadata = {"source": "wecom_service", "wecom_service_action": "conversation", "wecom_service_customer_number": "123456", "wecom_service_journey_context": context}
    return context, metadata


def summary():
    evidence = ["wecom_turn:1:input:0"]
    return {"schema_version": 1, "founder_profile": {"doing": "经营门店", "background": "", "offer": "试点门店", "clarity": "clear", "missing": [], "evidence_keys": evidence}, "has_need": "yes", "intent_evidence_keys": evidence, "needs": [
        {"id": "need_1", "title": "技术伙伴", "target": "小程序开发伙伴", "purpose": "完成门店试点", "conditions": "", "offer": "试点门店", "clarity": "clear", "missing": [], "evidence_keys": evidence},
        {"id": "need_2", "title": "渠道合作", "target": "渠道伙伴", "purpose": "", "conditions": "", "offer": "", "clarity": "unclear", "missing": ["目标渠道"], "evidence_keys": evidence},
    ], "needs_truncated": False, "next_action": "确认技术伙伴画像，再补渠道类型"}


def test_two_needs_remain_distinct_and_internal_summary_is_separate():
    context, _ = inputs()
    reply, progress = decode_reply(json.dumps({"reply": "先确认技术伙伴画像，是否可以准备筛选？", "service_journey": summary()}, ensure_ascii=False), context)
    assert "service_journey" not in reply
    assert len(progress["needs"]) == 2
    assert progress["needs"][0]["clarity"] == "clear"
    assert progress["needs"][1]["clarity"] == "unclear"


@pytest.mark.parametrize("mutate", [
    lambda value: value["needs"].append(copy.deepcopy(value["needs"][0])),
    lambda value: value["needs"][0].update(evidence_keys=["wecom_turn:999:input:0"]),
    lambda value: value["needs"][0].update(progress="introduced"),
    lambda value: value["founder_profile"].update(evidence_keys=[]),
    lambda value: value.update(needs=[value["needs"][0]] * 9),
    lambda value: value.update(has_need={}),
    lambda value: value["needs"][0].update(clarity=[]),
])
def test_invalid_analysis_keeps_reply_but_never_surfaces_unvalidated_state(mutate):
    context, _ = inputs()
    value = summary()
    mutate(value)
    reply, progress = decode_reply(json.dumps({"reply": "请确认想找谁。", "service_journey": value}), context)
    assert reply == "请确认想找谁。"
    assert progress is None
    with pytest.raises(ValueError):
        validate_journey(value, context["sources"])


@pytest.mark.parametrize("action", ["group_conversation", "group_greeting", "introduction", "skill_plan"])
def test_personal_analysis_rejected_for_other_modes(action):
    context, metadata = inputs()
    metadata["wecom_service_action"] = action
    with pytest.raises(ValueError):
        validate_context(context, metadata)


def test_public_client_cannot_inject_journey_or_other_customer_identity():
    context, metadata = inputs()
    with pytest.raises(ValueError):
        validate_group_agent_metadata(metadata)
    assert validate_group_agent_metadata(metadata, allow_referral_context=True) == metadata
    context["user_number"] = "other"
    with pytest.raises(ValueError):
        validate_context(context, metadata)


def test_plain_legacy_reply_has_no_fabricated_progress_and_json_never_leaks():
    context, _ = inputs()
    assert decode_reply("你好，请确认目标伙伴。", context) == ("你好，请确认目标伙伴。", None)
    for raw in ('{"service_journey":{}}', '{broken', '[{"service_journey":{}}]', '内部service_journey:...'):
        with pytest.raises(ValueError):
            decode_reply(raw, context)


async def test_one_call_generates_reply_and_summary_without_tools_or_profile_write():
    context, metadata = inputs()
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps({"reply": "找技术和渠道两类伙伴，对吗？", "service_journey": summary()}))))
    req = SimpleNamespace(metadata=metadata, message="客户原话", run_id="test", conversation_id="isolated")
    session = SimpleNamespace(capabilities=SimpleNamespace(can_search=True, can_write_profile=True))
    state = SimpleNamespace(planner_model=model, quality_model=None, polish_model=None, base_dir=Path("/unused"))
    emit = AsyncMock()
    with patch("skillpacks.aihehuo.client.fetch") as fetch:
        await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
    fetch.assert_not_called()
    model.ainvoke.assert_awaited_once()
    system = model.ainvoke.call_args.args[0][0].content
    assert "创业者本人画像" in system
    assert "每个追问" in system
    assert "2–3次" in system
    final = emit.call_args.args[1]
    assert final["profile_status"] == "disabled"
    assert len(final["service_journey"]["needs"]) == 2
    assert "service_journey" not in final["reply"]
