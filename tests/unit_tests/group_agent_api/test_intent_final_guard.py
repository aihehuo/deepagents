"""INTENT-DA-003 deterministic final action and receipt guard tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.group_agent_api.agent_factory.checks.intent_final_guard import (
    apply_intent_final_guard,
)
from apps.group_agent_api.agent_factory.intent_routing.execution import (
    prepare_intent_execution,
)
from apps.group_agent_api.agent_factory.intent_routing.resolver import resolve_intent
from tests.unit_tests.group_agent_api.planner_fixtures import (
    people_query,
    planner_for_message,
    planner_returning,
    platform_help_query,
    proposal,
)


def _execution(run_id: str = "guard-run"):
    message = "帮我找上海的技术合伙人"
    resolution = resolve_intent(
        message,
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(people_query("上海 技术合伙人")),
    )
    return prepare_intent_execution(
        resolution,
        original_message=message,
        run_id=run_id,
        request_fingerprint=None,
    )


def _commit(execution, items: list[dict[str, object]]):
    import json

    reservation = execution.reserve(
        resource="users", parameters={"page": 1}, tool_call_id="guard-tool"
    )
    return execution.commit_result(
        reservation,
        json.dumps(
            {
                "data": items,
                "_source_url": "https://www.aihehuo.com/ai/users.json?page=1",
            },
            ensure_ascii=False,
        ),
    )


def test_future_and_unverified_current_claims_are_blocked_when_routing_is_off() -> None:
    future = apply_intent_final_guard(
        reply="我正在搜索，请稍后看结果。",
        execution=None,
        enabled=True,
    )
    assert future.blocked
    assert "future_execution_promise" in future.violations
    assert "后台任务" in future.reply

    completed = apply_intent_final_guard(
        reply="已经搜索并找到了一位合伙人。",
        execution=None,
        enabled=True,
    )
    assert completed.blocked
    assert "unverified_search_claim" in completed.violations

    for claim in ("我正在查。", "我马上查。", "查询过了。", "稍后保存画像。"):
        result = apply_intent_final_guard(
            reply=claim,
            execution=None,
            enabled=True,
        )
        assert result.blocked, claim


@pytest.mark.parametrize(
    "reply",
    [
        "我没有说系统正在查询。",
        "“正在查询”只是状态文案。",
        "如果系统正在查询，应该如何展示？",
        "上一轮已经搜索过，本轮只讨论方法。",
        "上一轮已经发到群里，这里只回顾历史。",
        "历史记录写着候选人【李四】，不是本轮结果。",
        "如何找到合伙人？",
    ],
)
def test_negated_quoted_hypothetical_historical_and_method_claims_pass(
    reply: str,
) -> None:
    result = apply_intent_final_guard(
        reply=reply,
        execution=None,
        enabled=True,
    )
    assert not result.blocked
    assert result.reply == reply


def test_valid_receipt_authorizes_search_and_appends_scope_without_source_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    _commit(
        execution,
        [
            {
                "id": "u-1",
                "name": "张三",
                "city": "上海",
                "title": "技术合伙人",
                "bio": "公开资料",
            }
        ],
    )
    result = apply_intent_final_guard(
        reply=(
            "已经搜索，候选人【张三】的公开资料符合地域条件。"
            "来源：https://www.aihehuo.com/ai/users.json?page=1"
        ),
        execution=execution,
        enabled=True,
    )
    assert not result.blocked
    assert "查询范围：本次查询覆盖 1 页，检查 1 条去重公开记录" in result.reply
    assert "https://www.aihehuo.com/ai/users.json?page=1" not in result.reply
    assert "来源：" not in result.reply
    assert execution.receipts[0].source_urls[0].startswith(
        "https://www.aihehuo.com/ai/users.json"
    )


def test_invented_extra_entity_wrong_source_and_scope_overclaim_are_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    _commit(execution, [{"id": "u-1", "name": "张三"}])

    invented = apply_intent_final_guard(
        reply="候选人【张三】和候选人【李四】都值得进一步了解。",
        execution=execution,
        enabled=True,
    )
    assert invented.blocked
    assert "invented_public_entity" in invented.violations

    wrong_source = apply_intent_final_guard(
        reply="已经搜索。来源：https://evil.invalid/ai/users.json",
        execution=execution,
        enabled=True,
    )
    assert wrong_source.blocked
    assert "unsupported_source" in wrong_source.violations

    overclaim = apply_intent_final_guard(
        reply="这是全部人选，没有遗漏。",
        execution=execution,
        enabled=True,
    )
    assert overclaim.blocked
    assert "query_scope_overclaim" in overclaim.violations


def test_empty_receipt_cannot_be_described_as_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    _commit(execution, [])
    result = apply_intent_final_guard(
        reply="已经找到一位合伙人。",
        execution=execution,
        enabled=True,
    )
    assert result.blocked
    assert "empty_result_misrepresented" in result.violations
    assert "没有返回符合条件" in result.reply


@pytest.mark.parametrize(
    "reply",
    [
        "好的，我们来查北京的餐饮供应链创业者。",
        "明白，已切回上海。",
    ],
)
def test_completed_people_query_must_not_end_with_ack_only(
    monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    _commit(
        execution,
        [
            {
                "id": "101",
                "name": "张三",
                "city": "上海",
                "title": "技术合伙人",
                "bio": "餐饮供应链经验",
            }
        ],
    )
    result = apply_intent_final_guard(
        reply=reply,
        execution=execution,
        enabled=True,
    )
    assert result.blocked
    assert "execution_result_omitted" in result.violations
    assert "张三" in result.reply
    assert "公开查询已完成" in result.reply


def test_fallback_lists_search_results_without_rechecking_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public tool results are not rechecked against frozen constraints."""
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    _commit(
        execution,
        [
            {
                "id": "shenzhen-1",
                "name": "chen富",
                "city": "深圳",
                "bio": "供应链",
            }
        ],
    )
    result = apply_intent_final_guard(
        reply="好的，条件已记下。",
        execution=execution,
        enabled=True,
    )
    assert result.blocked
    assert "execution_result_omitted" in result.violations
    assert "chen富" in result.reply


def test_empty_query_must_explicitly_report_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    _commit(execution, [])
    result = apply_intent_final_guard(
        reply="好的，条件已更新。",
        execution=execution,
        enabled=True,
    )
    assert result.blocked
    assert "execution_result_omitted" in result.violations
    assert "没有返回符合条件" in result.reply


def test_cross_run_receipt_and_missing_adoption_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    current = _execution("current-run")
    foreign = _execution("foreign-run")
    foreign_receipt = _commit(foreign, [{"id": "u-1", "name": "张三"}])
    current.receipts.append(foreign_receipt)
    current.evidence_by_receipt[foreign_receipt.receipt_id] = (
        foreign.evidence_by_receipt[foreign_receipt.receipt_id]
    )
    result = apply_intent_final_guard(
        reply="已经搜索到候选人【张三】。",
        execution=current,
        enabled=True,
    )
    assert result.blocked
    assert "receipt_invalid" in result.violations

    adopted = _execution("adoption-run")
    receipt = _commit(adopted, [{"id": "u-2", "name": "王五"}])
    object.__setattr__(receipt, "produced_by_attempt_id", "older-attempt")
    result = apply_intent_final_guard(
        reply="已经搜索到候选人【王五】。",
        execution=adopted,
        enabled=True,
    )
    assert result.blocked
    assert "receipt_invalid" in result.violations


def test_profile_confirmation_requires_current_successful_save() -> None:
    rejected = apply_intent_final_guard(
        reply="你的画像已经更新并保存。",
        execution=None,
        profile_saved_this_turn=False,
        profile_persisted=True,
        enabled=True,
    )
    assert rejected.blocked
    assert "unverified_profile_claim" in rejected.violations

    accepted = apply_intent_final_guard(
        reply="你的画像已经更新并保存。",
        execution=None,
        profile_saved_this_turn=True,
        profile_persisted=True,
        enabled=True,
    )
    assert not accepted.blocked


def test_external_action_claim_never_uses_read_receipt_as_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    _commit(execution, [{"id": "u-1", "name": "张三"}])
    result = apply_intent_final_guard(
        reply="已经搜索到候选人【张三】，并帮你发到群里了。",
        execution=execution,
        enabled=True,
    )
    assert result.blocked
    assert "unverified_external_action" in result.violations


def test_guard_has_independent_explicit_kill_switch() -> None:
    original = "我正在搜索，请稍后。"
    result = apply_intent_final_guard(
        reply=original,
        execution=None,
        enabled=False,
    )
    assert not result.blocked
    assert result.reply == original


def test_guard_has_independent_yaml_switch(tmp_path: Path) -> None:
    from apps.group_agent_api.agent_factory.module_config import (
        reload_modules_config,
        reset_modules_config_cache,
    )

    config = tmp_path / "modules.yaml"
    config.write_text(
        "version: 1\npreset: test\nchecks:\n  chk.intent_final_guard: false\n",
        encoding="utf-8",
    )
    try:
        reload_modules_config(config)
        original = "我正在搜索，请稍后。"
        result = apply_intent_final_guard(reply=original, execution=None)
        assert not result.blocked
        assert result.reply == original
    finally:
        reset_modules_config_cache()


def test_repair_scrub_removes_ground_jargon_from_user_text() -> None:
    from apps.group_agent_api.agent_factory.checks.reply_grounding.leak import (
        reply_has_internal_context_leak,
        scrub_internal_context_leak,
    )
    from apps.group_agent_api.agent_factory.checks.reply_grounding.l0 import run_l0
    from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
        GroundBlock,
        ReplyGroundingInput,
    )

    leaked = "ground 中无 ID 为 9659712 的用户。公开资料里有几位相关人选。"
    cleaned = scrub_internal_context_leak(leaked)
    assert "ground" not in cleaned.lower()
    assert "9659712" not in cleaned  # dropped with the leaking clause
    assert "公开资料里有几位相关人选" in cleaned

    denial = run_l0(
        ReplyGroundingInput(
            reply=leaked,
            reply_mode="dialogue",
            ground=GroundBlock(),
        )
    )
    assert denial is not None
    assert denial.codes == ["internal_context_leak"]

    natural = "We need an on-the-ground operator in Shanghai."
    assert not reply_has_internal_context_leak(natural)
    assert scrub_internal_context_leak(natural) == natural

    assert reply_has_internal_context_leak("请核对 user_id=9659712")
    assert "user_id" not in scrub_internal_context_leak("请核对 user_id=9659712")


def test_shanghai_query_does_not_recheck_shenzhen_name_on_retired_guard_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    _commit(
        execution,
        [
            {
                "id": "shenzhen-1",
                "name": "chen富",
                "city": "深圳",
                "bio": "供应链",
            },
            {
                "id": "101",
                "name": "张三",
                "city": "上海",
                "title": "技术合伙人",
                "bio": "餐饮供应链经验",
            },
        ],
    )
    for reply in (
        "已经搜索，候选人【chen富】值得进一步了解。",
        "相关公开记录包括：chen富（ID: shenzhen-1）",
        "- chen富（ID: shenzhen-1），深圳，供应链\n没有完全匹配的人选。",
    ):
        blocked = apply_intent_final_guard(
            reply=reply,
            execution=execution,
            enabled=True,
        )
        assert not blocked.blocked, reply
        assert "chen富" in blocked.reply

    ok = apply_intent_final_guard(
        reply="已经搜索，候选人【张三】的公开资料符合地域条件。",
        execution=execution,
        enabled=True,
    )
    assert not ok.blocked
    assert "张三" in ok.reply
    assert "chen富" not in ok.reply
    assert "检查 2 条去重公开记录" in ok.reply


def test_job_title_and_nested_company_name_are_not_treated_as_people(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    execution = _execution()
    _commit(
        execution,
        [
            {
                "id": "u-1",
                "name": "张三",
                "city": "上海",
                "title": "技术合伙人",
                "company": {"name": "星火供应链"},
                "bio": "餐饮供应链经验",
            }
        ],
    )
    result = apply_intent_final_guard(
        reply="候选人【张三】是技术合伙人，有餐饮供应链经验，曾在星火供应链任职。",
        execution=execution,
        enabled=True,
    )
    assert not result.blocked
    assert "张三" in result.reply
    assert "技术合伙人" in result.reply
    assert "星火供应链" in result.reply


def test_source_url_strip_preserves_business_source_phrases() -> None:
    from apps.group_agent_api.agent_factory.checks.intent_final_guard.module import (
        _strip_user_visible_source_urls,
    )

    text = (
        "资金来源：自有资金，项目来源：https://www.aihehuo.com/ideas/123。"
        "收入来源：服务费。"
    )
    cleaned = _strip_user_visible_source_urls(text)
    assert "资金来源：自有资金" in cleaned
    assert "收入来源：服务费" in cleaned
    assert "https://www.aihehuo.com/" not in cleaned
    assert "项目来源：" not in cleaned


def test_platform_help_faq_is_not_treated_as_people_zero_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    import json

    resolution = resolve_intent(
        "如何发布餐饮项目招合伙人",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_for_message("如何发布餐饮项目招合伙人"),
    )
    assert resolution.decision.primary_intent == "platform_help"
    execution = prepare_intent_execution(
        resolution,
        original_message="如何发布餐饮项目招合伙人",
        run_id="faq-run",
        request_fingerprint=None,
    )
    reservation = execution.reserve(
        resource="faqs", parameters={"page": 1}, tool_call_id="faq-tool"
    )
    execution.commit_result(
        reservation,
        json.dumps(
            {
                "data": [{"title": "如何发布项目", "body": "打开发布入口填写信息"}],
                "_source_url": "https://www.aihehuo.com/ai/faqs.json?page=1",
            },
            ensure_ascii=False,
        ),
    )
    result = apply_intent_final_guard(
        reply="已经查询。如何发布项目：打开发布入口填写信息。",
        execution=execution,
        enabled=True,
    )
    assert not result.blocked
    assert "人选" not in result.reply
    assert "检查 1 条去重公开记录" in result.reply


def test_startup_digest_keeps_raw_result_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    import json

    resolution = resolve_intent(
        "看看本周创业周报和文章",
        execution_mode="group_agent",
        can_search=True,
        planner_model=planner_returning(proposal(intent="startup_digest")),
    )
    assert resolution.decision.primary_intent == "startup_digest"
    execution = prepare_intent_execution(
        resolution,
        original_message="看看本周创业周报和文章",
        run_id="digest-run",
        request_fingerprint=None,
    )
    reservation = execution.reserve(
        resource="weekly", parameters={}, tool_call_id="weekly-tool"
    )
    execution.commit_result(
        reservation,
        json.dumps(
            {
                "data": [{"title": "本周融资速递", "summary": "三家 AI 公司完成融资"}],
                "_source_url": "https://www.aihehuo.com/ai/weekly.json?page=1",
            },
            ensure_ascii=False,
        ),
    )
    result = apply_intent_final_guard(
        reply="已经查询。本周融资速递：三家 AI 公司完成融资。",
        execution=execution,
        enabled=True,
    )
    assert not result.blocked
    assert "没有找到完全满足当前条件的人选" not in result.reply
    assert "检查 1 条去重公开记录" in result.reply
