"""Autonomous model budgets must not reduce employee capabilities."""
from unittest.mock import Mock
import pytest
from langchain_core.messages import AIMessage
from apps.community_manager import agent


def test_employee_budget_unchanged() -> None:
    assert agent.execution_budget({'model_mode': 'staff'}) == (12, 48, 240_000)
    assert agent.execution_budget({}) == (12, 48, 240_000)


def test_autonomous_budget_and_no_extra_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    response = AIMessage(content='', tool_calls=[{'name': 'get_follow_ups', 'args': {}, 'id': 'read-1'}])
    model = Mock(); model.bind_tools.return_value = model; model.invoke.return_value = response
    monkeypatch.setattr(agent, 'client', lambda: model)
    monkeypatch.setattr(agent, 'invoke_tool', lambda *args: {'items': []})
    summary = Mock(side_effect=AssertionError('autonomous summary must not call a model'))
    monkeypatch.setattr(agent, 'partial_summary', summary)
    with pytest.raises(agent.BudgetExhausted):
        agent.execute({'model_mode': 'autonomous', 'text': '检查这名用户', 'trigger': {'kind': 'event', 'user_number': '123456'}}, Mock())
    assert model.invoke.call_count == 6
    summary.assert_not_called()


def test_budget_exhaustion_is_failed_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import worker
    calls = []
    def rpc(action: str, **kwargs: object) -> dict:
        calls.append((action, kwargs))
        return {'status': 'running', 'token': 'token', 'model_mode': 'autonomous'} if action == 'begin' else {}
    monkeypatch.setattr(worker, 'rpc', rpc)
    monkeypatch.setattr(worker, 'execute', Mock(side_effect=agent.BudgetExhausted('manager_budget_exhausted')))
    worker.run.run('budget-test')
    assert calls[-1][0] == 'result'
    assert calls[-1][1]['output']['error_code'] == 'manager_budget_exhausted'
