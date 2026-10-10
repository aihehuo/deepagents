"""WeCom assessment must not inherit short Cafe Host model budgets."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from apps.group_agent_api.app.wecom_review_model import get_review_model, review_timeout, review_budget


def test_review_client_is_dedicated_cached_and_sdk_retries_disabled(monkeypatch):
    monkeypatch.delenv('WECOM_MATCH_REVIEW_TIMEOUT_S', raising=False)
    dedicated = object()
    state = SimpleNamespace(quality_model=object(), planner_model=object())
    with patch('apps.group_agent_api.agent_factory.model_builder.create_model', return_value=dedicated) as factory:
        assert get_review_model(state) is dedicated
        assert get_review_model(state) is dedicated
        factory.assert_called_once_with(log_prefix='[WecomMatchReview]', timeout_s=45.0, max_retries=0)


def test_review_budgets_are_bounded(monkeypatch):
    monkeypatch.setenv('WECOM_MATCH_REVIEW_TIMEOUT_S', '1000')
    monkeypatch.setenv('WECOM_MATCH_REVIEW_BUDGET_S', '1000')
    assert review_timeout() == 60
    assert review_budget() == 90


async def test_review_timeout_cancels_assessment_without_emitting_recommendation(caplog):
    from tests.unit_tests.group_agent_api.test_wecom_manual import fixtures
    from apps.group_agent_api.app.wecom_manual import execute_wecom_run
    req, session, state, emit = fixtures({'wecom_service_action': 'skill_execute', 'wecom_service_skill': 'get-ai-cofounders',
        'wecom_service_confirmed': True, 'wecom_service_parameters': {'investor_status': 'verified', 'q': '找出资方'}})
    cancelled = []
    async def slow(*_):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    async def search(**kwargs):
        await kwargs['reviewer']([{'id': '1', 'summary': 'PRIVATE_CANDIDATE_CONTENT'}])
    with patch('apps.group_agent_api.app.wecom_recommendation.find_one', side_effect=search), \
         patch('apps.group_agent_api.app.wecom_match_review.review_candidates', side_effect=slow), \
         patch('apps.group_agent_api.app.wecom_review_model.review_timeout', return_value=0.01):
        with pytest.raises(TimeoutError):
            await execute_wecom_run(req=req, session=session, state=state, tid='t', emit_callback=emit)
    assert cancelled == [True]
    emit.assert_not_awaited()
    state.planner_model.ainvoke.assert_not_awaited()
    assert 'wecom_match_review_failed' in caplog.text
    assert 'PRIVATE_CANDIDATE_CONTENT' not in caplog.text
