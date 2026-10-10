"""Investor lookup must preserve provider direction, identity and uncertainty."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from apps.group_agent_api.app.wecom_manual import validate_parameters
from apps.group_agent_api.app.wecom_recommendation import find_one
from apps.group_agent_api.app.wecom_match_review import review_candidates
from apps.group_agent_api.app.investor_client import fetch_investors


def card(identity, status="verified", summary="愿意出资，关注跨境电商"):
    return {"id": str(identity), "number": identity, "title": "候选" + str(identity), "summary": summary,
            "url": "https://www.aihehuo.com/users/" + str(identity), "wechat_reachable": True,
            "investment": {"status": status}}


def review(identity, status, evidence="愿意出资"):
    return {"id": str(identity), "status": status, "reason": "投资方向待核实", "evidence": [evidence], "missing": []}


@pytest.mark.parametrize("parameters", [{"investor_status": "rejected"}, {"investment_amount": "100000"},
    {"investor_status": "verified", "investment_amount": "01"}, {"investor_status": "verified", "investment_amount": "-1"}])
def test_invalid_investor_parameters(parameters):
    with pytest.raises(ValueError):
        validate_parameters("get-ai-cofounders", parameters)


def test_investor_parameters_only_belong_to_cofounder_skill():
    assert validate_parameters("get-ai-cofounders", {"investor_status": "verified", "investment_amount": "100000"}, executing=True)
    with pytest.raises(ValueError):
        validate_parameters("get-ai-opportunities", {"investor_status": "verified"})


async def test_fund_seeker_is_rejected_before_provider_recommendation():
    rows = [card(1, summary="主动寻求天使投资人"), card(2)]
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps([
        review(1, "rejected", "主动寻求天使投资人"), review(2, "matched")], ensure_ascii=False))))
    async def reviewer(cards):
        return await review_candidates(model, "找约10万元出资方", cards)
    with patch("apps.group_agent_api.app.investor_client.fetch_investors", return_value=json.dumps({"success": True, "data": rows, "meta": {"has_more": False}})), patch("skillpacks.aihehuo.client.fetch") as public:
        selected, scope = await find_one(resource="users", parameters={"investor_status": "verified_or_pending"}, excluded=[], reviewer=reviewer)
    assert selected["id"] == "2"
    assert scope["rejected"] == 1
    assert scope["reviewed"] == 2
    public.assert_not_called()


async def test_pending_application_cannot_become_verified_or_fully_matched():
    cards = [card(1, "pending")]
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps([review(1, "matched")], ensure_ascii=False))))
    result = await review_candidates(model, "找投资人", cards)
    assert result[0]["status"] == "needs_verification"
    assert "投资人身份尚未认证" in result[0]["missing"]
    assert cards[0]["investment"]["status"] == "pending"


async def test_invented_evidence_fails_closed():
    model = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content=json.dumps([review(1, "matched", "已有10万元可投入")], ensure_ascii=False))))
    with pytest.raises(ValueError, match="ungrounded_match_review"):
        await review_candidates(model, "找投资人", [card(1)])


async def test_all_rejected_returns_no_candidate_and_does_not_force_recommendation():
    async def reviewer(cards):
        return [review(c["id"], "rejected") for c in cards]
    with patch("apps.group_agent_api.app.investor_client.fetch_investors", return_value=json.dumps({"success": True, "data": [card(1)], "meta": {"has_more": False}})):
        selected, scope = await find_one(resource="users", parameters={"investor_status": "verified"}, excluded=[], reviewer=reviewer)
    assert selected is None
    assert scope["returned"] == 0


async def test_review_budget_is_twenty_and_duplicates_self_excluded():
    async def reviewer(cards):
        return [review(c["id"], "rejected") for c in cards]
    document = {"success": True, "data": [card(i) for i in range(1,21)], "meta": {"has_more": True}}
    with patch("apps.group_agent_api.app.investor_client.fetch_investors", return_value=json.dumps(document)) as fetch:
        selected, scope = await find_one(resource="users", parameters={"investor_status": "verified"}, excluded=[], reviewer=reviewer)
    assert selected is None and scope["reviewed"] == 20
    fetch.assert_called_once()


def test_protected_lookup_requires_key_and_fixed_origin(monkeypatch):
    monkeypatch.delenv("AIHEHUO_AGENT_KEY", raising=False)
    with pytest.raises(RuntimeError, match="auth_unconfigured"):
        fetch_investors({})
    monkeypatch.setenv("AIHEHUO_AGENT_KEY", "test-private-key")
    monkeypatch.setenv("AIHEHUO_PUBLIC_API_BASE_URL", "https://evil.example")
    with pytest.raises(ValueError, match="origin"):
        fetch_investors({})


def test_protected_transport_never_redirects_or_returns_contacts(monkeypatch):
    monkeypatch.setenv("AIHEHUO_AGENT_KEY", "test-private-key")
    monkeypatch.delenv("AIHEHUO_PUBLIC_API_BASE_URL", raising=False)
    response = SimpleNamespace(status=302)
    connection = SimpleNamespace(request=lambda *a, **kw: None, getresponse=lambda: response, close=lambda: None)
    with patch("http.client.HTTPSConnection", return_value=connection):
        with pytest.raises(RuntimeError, match="http_302"):
            fetch_investors({"investor_status": "verified"})


def test_selected_need_only_accepts_trusted_bounded_server_context():
    from apps.group_agent_api.app.models import validate_group_agent_metadata
    need = {"id": "need_capital", "title": "找出资人", "target": "出资合伙人", "purpose": "启动业务", "conditions": "10万元", "offer": "运营"}
    metadata = {"source": "wecom_service", "wecom_service_action": "skill_execute", "wecom_service_need": need}
    assert validate_group_agent_metadata(metadata, allow_referral_context=True) == metadata
    with pytest.raises(ValueError):
        validate_group_agent_metadata(metadata)
    with pytest.raises(ValueError):
        validate_group_agent_metadata({**metadata, "wecom_service_need": {**need, "token": "secret"}}, allow_referral_context=True)


async def test_old_capital_search_without_investor_filter_never_queries():
    from tests.unit_tests.group_agent_api.test_wecom_manual import fixtures
    from apps.group_agent_api.app.wecom_manual import execute_wecom_run
    req, session, state, emit = fixtures({"wecom_service_action": "skill_execute", "wecom_service_skill": "get-ai-cofounders",
        "wecom_service_confirmed": True, "wecom_service_parameters": {"q": "投资人"},
        "wecom_service_need": {"target": "出资合伙人", "title": "巴西电商"}})
    with patch("skillpacks.aihehuo.client.fetch") as fetch:
        with pytest.raises(ValueError, match="investor_search_parameters_required"):
            await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
    fetch.assert_not_called()


async def test_real_execution_emits_investor_review_and_host_owned_identity_label():
    from tests.unit_tests.group_agent_api.test_wecom_manual import fixtures
    from apps.group_agent_api.app.wecom_manual import execute_wecom_run
    req, session, state, emit = fixtures({"wecom_service_action": "skill_execute", "wecom_service_skill": "get-ai-cofounders",
        "wecom_service_confirmed": True, "wecom_service_parameters": {"investor_status": "pending", "q": "找投资人"}})
    state.planner_model.ainvoke.side_effect = [SimpleNamespace(content=json.dumps([review(1, "matched")], ensure_ascii=False)),
                                             SimpleNamespace(content="这是一位认证投资人，已经承诺出资。")]
    with patch("apps.group_agent_api.app.investor_client.fetch_investors", return_value=json.dumps({"success": True, "data": [card(1, "pending")], "meta": {"has_more": False}})):
        await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
    payload = emit.call_args.args[1]
    assert payload["recommendation"]["investment"]["status"] == "pending"
    assert payload["recommendation"]["matching_review"]["status"] == "needs_verification"
    assert "尚未认证" in payload["reply"]
    assert "已经承诺" not in payload["reply"]
    assert payload["search_scope"]["reviewed"] == 1


async def test_certified_pool_no_match_falls_back_to_pending_without_public_search():
    queries = []
    def fetch(parameters):
        queries.append(dict(parameters))
        status = parameters["investor_status"]
        rows = [card(1, "verified", "主动寻求天使投资人")] if status == "verified" else [card(2, "pending")]
        return json.dumps({"success": True, "data": rows, "meta": {"has_more": False}})
    model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
        SimpleNamespace(content=json.dumps([review(1, "rejected", "主动寻求天使投资人")], ensure_ascii=False)),
        SimpleNamespace(content=json.dumps([review(2, "matched")], ensure_ascii=False))]))
    async def reviewer(cards):
        return await review_candidates(model, "找投资人", cards)
    with patch("apps.group_agent_api.app.investor_client.fetch_investors", side_effect=fetch):
        selected, scope = await find_one(resource="users", parameters={"investor_status": "verified_or_pending"}, excluded=[], reviewer=reviewer)
    assert [q["investor_status"] for q in queries] == ["verified", "pending"]
    assert selected["id"] == "2" and selected["investment"]["status"] == "pending"
    assert selected["matching_review"]["status"] == "needs_verification"
    assert scope["rejected"] == 1


async def test_new_plan_parameters_are_negotiated_with_micro_and_feedback_is_used():
    from tests.unit_tests.group_agent_api.test_wecom_manual import fixtures
    from apps.group_agent_api.app.wecom_manual import execute_wecom_run
    for supported in [False, True]:
        metadata = {"wecom_service_action": "skill_plan", "wecom_service_skill": "get-ai-cofounders"}
        if supported:
            metadata["wecom_service_investor_search_supported"] = True
        req, session, state, emit = fixtures(metadata, '{"q":"找投资人"}')
        await execute_wecom_run(req=req, session=session, state=state, tid="t", emit_callback=emit)
        prompt = state.planner_model.ainvoke.call_args.args[0][0].content
        assert ('investor_status' in prompt) is supported
