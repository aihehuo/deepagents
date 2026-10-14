"""Circle matcher contract and semantic-only boundary, without network."""
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from apps.group_agent_api.app import circle_matching as matching
from apps.group_agent_api.app.endpoints.ai_assistant import Input, execute


def candidate(uid=2):
    return {'user_id': uid, 'reason': '公开资料显示有相关经验', 'contributions': ['探索合作'],
            'steps': ['确认条件'], 'missing': ['合作意愿'], 'evidence_refs': [f'user:{uid}']}


def test_semantic_api_uses_phrase_and_internal_app_id(monkeypatch):
    seen = []
    monkeypatch.setattr(matching.client, 'fetch', lambda path, **kwargs: seen.append(path) or json.dumps({'data': [{'id': '888', 'user_id': '2', 'bio': '社群运营'}]}))
    assert matching.semantic_search('寻找具有创业者社群运营经验并愿意开展渠道合作的用户')[0]['user_id'] == 2
    assert 'q=' in seen[0] and 'keyword=' not in seen[0] and 'wechat_reachable=' not in seen[0]


def test_public_startup_number_is_never_treated_as_app_user_id(monkeypatch):
    monkeypatch.setattr(matching.client, 'fetch', lambda *a, **k: '{"data":[{"id":"888"}]}')
    with pytest.raises(ValueError, match='semantic_app_identity_required'):
        matching.semantic_search('寻找具有创业者社群运营经验并愿意开展渠道合作的用户')


@pytest.mark.parametrize('uid', [1, 999])
def test_model_cannot_invent_candidate_or_select_owner(uid):
    with pytest.raises(ValueError):
        matching.parse_candidates(json.dumps({'candidates': [candidate(uid)]}), [{'user_id': 2}], 1)


def test_wrong_evidence_and_duplicate_candidate_rejected():
    c = candidate(); c['evidence_refs'] = ['user:999']
    with pytest.raises(ValueError):
        matching.parse_candidates(json.dumps({'candidates': [c]}), [{'user_id': 2}], 1)
    with pytest.raises(ValueError):
        matching.parse_candidates(json.dumps({'candidates': [candidate(), candidate()]}), [{'user_id': 2}], 1)


@pytest.mark.asyncio
async def test_match_module_returns_candidates_without_model_search_tools(monkeypatch):
    from apps.group_agent_api.agent_factory import model_builder
    model = AsyncMock()
    model.ainvoke.side_effect = [AIMessage(content=json.dumps({'query': '寻找具有创业者社群运营经验并愿意开展渠道合作的用户'})),
                               AIMessage(content=json.dumps({'candidates': [candidate()]}))]
    monkeypatch.setattr(model_builder, 'create_model', lambda **k: model)
    monkeypatch.setattr(matching, 'semantic_search', lambda q: [{'user_id': 2, 'evidence_ref': 'user:2'}])
    p = Input(kind='match', user_id=1, circles=[{'id': 10, 'version': 1, 'owner_id': 1, 'kind': 'need', 'title': '社群渠道', 'description': '寻找创业者渠道'}])
    result = await execute(p)
    assert result['candidates'][0]['user_id'] == 2
    assert model.ainvoke.call_count == 2


@pytest.mark.asyncio
async def test_search_failure_is_not_successful_empty_result(monkeypatch):
    from apps.group_agent_api.agent_factory import model_builder
    model = AsyncMock(); model.ainvoke.return_value = AIMessage(content=json.dumps({'query': '寻找具有创业者社群运营经验并愿意开展渠道合作的用户'}))
    monkeypatch.setattr(model_builder, 'create_model', lambda **k: model)
    def fail(q):
        raise RuntimeError('semantic service unavailable')
    monkeypatch.setattr(matching, 'semantic_search', fail)
    with pytest.raises(RuntimeError):
        await matching.match_circle({'id': 10, 'owner_id': 1})
