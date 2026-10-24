"""Autonomous model budgets must not reduce employee capabilities."""
from unittest.mock import Mock
import pytest
from langchain_core.messages import AIMessage
from apps.community_manager import agent


def test_employee_budget_unchanged() -> None:
    assert agent.execution_budget({'model_mode': 'staff'}) == (12, 48, 240_000)
    assert agent.execution_budget({}) == (12, 48, 240_000)



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


def configured_model(monkeypatch: pytest.MonkeyPatch, responses: list) -> Mock:
    model = Mock(); model.bind_tools.return_value = model; model.invoke.side_effect = responses
    monkeypatch.setattr(agent, 'client', lambda: model)
    return model


def event() -> dict:
    return {'model_mode': 'autonomous', 'trigger': {'kind': 'event', 'user_number': '123456'}}


def save_message(number: str = '123456') -> AIMessage:
    return AIMessage(content='', tool_calls=[{'id': 'save', 'name': 'save_follow_up', 'args': {
        'number': number, 'need_key': 'profile', 'focus': 'clarify', 'status': 'active',
        'reason': 'fictional', 'evidence': ['E1'],
        'next_review_at': '2026-10-25T10:00:00+08:00', 'expected_revision': 0}}])


def test_autonomous_reads_once_then_saves_and_notifies(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    def rpc(action: str, **kwargs: object) -> dict:
        seen.append(kwargs)
        return {'revision': 1, 'id': 1, 'allowed_evidence': [{'ref': {'source': 'profile', 'id': 1, 'field': 'bio'}, 'text': 'fictional'}]}
    model = configured_model(monkeypatch, [save_message(),
        AIMessage(content='', tool_calls=[{'id': 'notify', 'name': 'notify_staff', 'args': {'number': '123456', 'need_key': 'profile'}}]),
        AIMessage(content='已保存安排并通知负责人。')])
    text, _ = agent.execute(event(), rpc)
    assert model.invoke.call_count == 3
    assert [row['name'] for row in seen] == ['get_guidance_sources', 'resolve_owner', 'get_service_journey', 'save_follow_up', 'notify_staff']
    assert all(row['arguments']['number'] == '123456' for row in seen)
    assert {t['function']['name'] for t in model.bind_tools.call_args.args[0]} == {'save_follow_up', 'notify_staff', 'draft_guidance'}
    assert '已保存' in text


def test_budget_and_duplicate_write_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    model = configured_model(monkeypatch, [save_message() for _ in range(6)])
    rpc = Mock(return_value={'revision': 1, 'allowed_evidence': [{'ref': {'source': 'profile', 'id': 1, 'field': 'bio'}, 'text': 'fictional'}]})
    with pytest.raises(agent.BudgetExhausted): agent.execute(event(), rpc)
    assert model.invoke.call_count == 6
    assert [c.kwargs['name'] for c in rpc.call_args_list].count('save_follow_up') == 1


def test_paused_user_needs_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock(); monkeypatch.setattr(agent, 'client', client)
    text, _ = agent.execute(event(), Mock(return_value={'blocked_reason': 'paused'}))
    client.assert_not_called(); assert 'paused' in text


def test_cross_user_action_rejected_before_rpc(monkeypatch: pytest.MonkeyPatch) -> None:
    configured_model(monkeypatch, [save_message('999999')])
    rpc = Mock(return_value={})
    with pytest.raises(ValueError, match='autonomous_tool_scope_invalid'): agent.execute(event(), rpc)
    assert all(c.kwargs['name'] != 'save_follow_up' for c in rpc.call_args_list)


def test_rejected_write_cannot_be_success(monkeypatch: pytest.MonkeyPatch) -> None:
    configured_model(monkeypatch, [save_message(), save_message()])
    def rpc(action: str, **kwargs: object) -> dict:
        return {'tool_error': 'manager_follow_up_conflict'} if kwargs['name'] == 'save_follow_up' else {'allowed_evidence': [{'ref': {'source': 'profile', 'id': 1, 'field': 'bio'}, 'text': 'fictional'}]}
    with pytest.raises(ValueError, match='autonomous_action_rejected'): agent.execute(event(), rpc)


def test_compacted_evidence_preserves_refs_and_marks_truncation() -> None:
    rows = [{'ref': {'source': 'profile', 'id': n}, 'text': 'x' * 1000} for n in range(30)]
    result = agent.autonomous_sources({'allowed_evidence': rows})
    assert result['evidence_truncated']
    assert len(result['allowed_evidence']) == 24
    assert result['allowed_evidence'][0]['ref'] == rows[0]['ref']
    assert len(result['allowed_evidence'][0]['text']) == 400


def test_revision_is_backend_derived_and_refreshed_on_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    configured_model(monkeypatch, [save_message(), save_message(), AIMessage(content='已保存')])
    revisions = []; reads = []
    def rpc(action: str, **kwargs: object) -> dict:
        name = kwargs['name']
        if name == 'get_guidance_sources':return {'allowed_evidence': [{'ref': {'source': 'profile', 'id': 7, 'field': 'bio'}, 'text': 'fictional'}]}
        if name == 'resolve_owner':
            reads.append(name)
            return {'follow_ups': [{'need_key': 'profile', 'revision': 7 if len(reads) == 1 else 8}]}
        if name == 'save_follow_up':
            args = kwargs['arguments']; revisions.append(args['expected_revision'])
            assert args['evidence'] == [{'source': 'profile', 'id': 7, 'field': 'bio'}]
            return {'tool_error': 'manager_follow_up_conflict'} if len(revisions) == 1 else {'revision': 9}
        return {}
    agent.execute(event(), rpc)
    assert revisions == [7, 8]
    assert len(reads) == 2


def test_conflict_cannot_be_followed_by_success_text(monkeypatch: pytest.MonkeyPatch) -> None:
    configured_model(monkeypatch, [save_message(), AIMessage(content='已完成')])
    def rpc(action: str, **kwargs: object) -> dict:
        if kwargs['name'] == 'save_follow_up':return {'tool_error': 'manager_follow_up_conflict'}
        return {'allowed_evidence': [{'ref': {'source': 'profile', 'id': 1, 'field': 'bio'}, 'text': 'fictional'}]}
    with pytest.raises(ValueError, match='autonomous_action_unresolved'):
        agent.execute(event(), rpc)
