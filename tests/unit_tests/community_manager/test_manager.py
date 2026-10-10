"""Manager isolation, identity, intent and fail-closed admission contracts."""
import json
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from apps.community_manager import protocol, worker, api
from apps.community_manager.celery_app import app as celery, QUEUE


@pytest.fixture(autouse=True)
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMMUNITY_MANAGER_SECRET", "s" * 32)


def test_signed_bytes_timestamp_and_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    body = b'{"run_id":"x"}'
    signed = {k.lower(): v for k, v in protocol.headers(body).items()}
    redis = Mock(); redis.set.return_value = True
    protocol.verify(body, signed, redis)
    with pytest.raises(protocol.ProtocolError):
        protocol.verify(body + b' ', signed, redis)
    redis.set.return_value = False
    with pytest.raises(protocol.ProtocolError, match="replayed"):
        protocol.verify(body, signed, redis)
    redis.set.return_value = True
    monkeypatch.setattr(protocol.time, "time", lambda: int(signed['x-cm-time']) + 100)
    with pytest.raises(protocol.ProtocolError, match="unauthorized"):
        protocol.verify(body, signed, redis)


def test_independent_queue_and_serializer() -> None:
    assert QUEUE == "community_manager_v1"
    assert celery.conf.task_default_queue != "group_agent"
    assert celery.conf.broker_transport_options['global_keyprefix'] == "community_manager:v1:"
    assert celery.conf.worker_prefetch_multiplier == 1
    assert celery.conf.accept_content == ['json']


def test_natural_message_reaches_agent_without_command_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock(side_effect=[{'token': 'token', 'text': '这周先关注技术合伙需求', 'trigger': {'kind': 'staff'}}, {}])
    monkeypatch.setattr(worker, 'rpc', call)
    execute = Mock(return_value=('我会先核对需求和已有服务，再调整建议。', 'manager-model'))
    monkeypatch.setattr(worker, 'execute', execute)
    worker.run.run('run')
    assert execute.call_args.args[0]['text'] == '这周先关注技术合伙需求'
    assert call.call_args.kwargs['output']['model_id'] == 'manager-model'


def test_duplicate_run_does_not_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock(return_value={"status": "running"})
    monkeypatch.setattr(worker, "rpc", call)
    execute = Mock(); monkeypatch.setattr(worker, "execute", execute)
    worker.run.run("id")
    execute.assert_not_called()


def test_failure_does_not_leak_provider_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock(side_effect=[{"status": "running", "token": "token", "text": "next"}, {}])
    monkeypatch.setattr(worker, "rpc", call)
    monkeypatch.setattr(worker, "execute", Mock(side_effect=RuntimeError("SECRET customer text")))
    worker.run.run("id")
    output = call.call_args.kwargs['output']
    assert "SECRET" not in json.dumps(output)
    assert output['error_code'] == 'manager_execution_failed'


def test_api_admission_does_not_accept_staff_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = Mock(); redis.set.return_value = True; redis.llen.return_value = 0
    monkeypatch.setattr(api, "redis_client", lambda: redis)
    enqueue = Mock(); monkeypatch.setattr(api.celery, "send_task", enqueue)
    client = TestClient(api.app)
    body = json.dumps({"run_id": "12345678-1234-1234-1234-123456789abc", "staff_id": "alice"}).encode()
    response = client.post('/manager/call_async', content=body, headers=protocol.headers(body))
    assert response.status_code == 422
    enqueue.assert_not_called()


def test_api_dedicated_queue_bounded_and_signed(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = Mock(); redis.set.return_value = True; redis.llen.return_value = 0
    monkeypatch.setattr(api, "redis_client", lambda: redis)
    enqueue = Mock(); monkeypatch.setattr(api.celery, "send_task", enqueue)
    client = TestClient(api.app)
    body = b'{"run_id":"12345678-1234-1234-1234-123456789abc"}'
    assert client.post('/manager/call_async', content=body, headers=protocol.headers(body)).status_code == 202
    assert enqueue.call_args.kwargs['queue'] == QUEUE
    redis.llen.return_value = 100
    assert client.post('/manager/call_async', content=body, headers=protocol.headers(body)).status_code == 429
    assert client.post('/manager/call_async', content=body, headers={'Content-Type': 'application/json'}).status_code == 401


def test_single_user_draft_preserves_versions_and_writes_only_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    sources = {'context': {'source_fingerprint': 'f', 'context_revision': 2},
               'strategy': {'revision': 3, 'current_version': {'id': 88}}, 'allowed_evidence': [{'ref': {'id': 1}}]}
    call = Mock(side_effect=[sources, {'draft': {'version': 4}}])
    monkeypatch.setattr(agent, 'draft', Mock(return_value=({'summary': {}, 'body': {}}, 'manager-model')))
    agent.invoke_tool(call, 'draft_guidance', {'number': '123'}, 'stable')
    args = call.call_args.kwargs
    assert args['name'] == 'save_guidance_draft'
    assert args['call_id'] == 'stable-draft'
    assert args['arguments']['output']['base_active_version_id'] == 88


def test_rpc_refuses_url_credentials_and_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COMMUNITY_MANAGER_AGENT_KEY', 'not-real')
    for url in ['https://secret@example.test', 'https://example.test?token=x', 'https://example.test/wrong']:
        monkeypatch.setenv('COMMUNITY_MANAGER_OPERATIONS_URL', url)
        with pytest.raises(protocol.ProtocolError, match='unconfigured'):
            protocol.rpc('tick')


def test_health_is_manager_only_and_status_path_is_signed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(api.app)
    assert client.get('/health').json()['service'] == 'community_manager'
    redis = Mock(); redis.set.return_value = True
    monkeypatch.setattr(api, 'redis_client', lambda: redis)
    monkeypatch.setattr(api, 'rpc', Mock(return_value={'status': 'completed'}))
    path = '/manager/runs/12345678-1234-1234-1234-123456789abc'
    assert client.get(path, headers=protocol.headers(path.encode())).status_code == 200
    assert client.get(path, headers=protocol.headers(b'/manager/runs/another')).status_code == 401


def test_model_unconfigured_and_invalid_json_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import model
    monkeypatch.delenv('COMMUNITY_MANAGER_MODEL', raising=False)
    with pytest.raises(ValueError, match='unconfigured'):
        model.draft({})
    for name in ['COMMUNITY_MANAGER_MODEL', 'COMMUNITY_MANAGER_MODEL_API_KEY', 'COMMUNITY_MANAGER_MODEL_BASE_URL']:
        monkeypatch.setenv(name, 'test')
    client = Mock(); client.invoke.return_value.content = '{"body":{},"summary":{},"approve":true}'
    monkeypatch.setattr(model, 'ChatOpenAI', Mock(return_value=client))
    with pytest.raises(ValueError, match='invalid'):
        model.draft({})


def test_staff_reply_is_limited_in_utf8_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock(side_effect=[{'token': 'token', 'text': 'next'}, {}])
    monkeypatch.setattr(worker, 'rpc', call)
    monkeypatch.setattr(worker, 'execute', Mock(return_value=('🧑' * 1000, 'deterministic')))
    worker.run.run('id')
    text = call.call_args.kwargs['output']['text']
    assert len(text.encode('utf-8')) <= 2000
    assert '�' not in text


def test_agent_only_exposes_fixed_business_tools_and_skill() -> None:
    from apps.community_manager.agent import SKILL
    from apps.community_manager.tool_schema import TOOLS
    names = {t['function']['name'] for t in TOOLS}
    assert {'search_users', 'get_user_history', 'save_plan', 'draft_guidance', 'notify_staff', 'get_memory'} <= names
    assert not {'shell', 'sql', 'approve', 'send_customer_message', 'claim'} & names
    assert '无需命令' in SKILL.read_text()


def test_unknown_tool_and_extra_staff_argument_refused_without_rpc() -> None:
    from apps.community_manager.agent import invoke_tool
    call = Mock()
    with pytest.raises(ValueError):
        invoke_tool(call, 'shell', {}, 'key')
    with pytest.raises(ValueError):
        invoke_tool(call, 'get_user', {'number': '123', 'staff_id': 'bob'}, 'key')
    call.assert_not_called()


def test_same_source_guidance_is_not_generated_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    model = Mock(); monkeypatch.setattr(agent, 'draft', model)
    call = Mock(return_value={'strategy': {}, 'current_source_draft': True})
    assert agent.invoke_tool(call, 'draft_guidance', {'number': '123'}, 'key')['drafted'] is False
    model.assert_not_called()
    assert call.call_count == 1


def test_natural_agent_tool_loop_uses_verified_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    from langchain_core.messages import AIMessage
    model = Mock()
    model.invoke.side_effect = [AIMessage(content='', tool_calls=[{'name': 'get_scope_metrics', 'args': {}, 'id': 'tool1', 'type': 'tool_call'}]),
                                AIMessage(content='当前授权范围有两位用户，我会先核实需求。')]
    monkeypatch.setattr(agent, 'client', lambda: Mock(bind_tools=Mock(return_value=model)))
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'configured-model')
    call = Mock(return_value={'user_count': 2})
    text, name = agent.execute({'text': '现在社群怎么样', 'trigger': {'kind': 'staff'}}, call)
    assert '两位用户' in text
    assert name == 'configured-model'
    assert call.call_args.kwargs['name'] == 'get_scope_metrics'
    assert model.invoke.call_count == 2


def test_unknown_write_aborts_instead_of_retrying_new_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    from langchain_core.messages import AIMessage
    model = Mock()
    model.invoke.return_value = AIMessage(content='', tool_calls=[{'name': 'save_memory', 'args': {'notes': 'later', 'expected_revision': 0, 'global': False}, 'id': 't1', 'type': 'tool_call'}])
    monkeypatch.setattr(agent, 'client', lambda: Mock(bind_tools=Mock(return_value=model)))
    call = Mock(side_effect=protocol.ProtocolError('uncertain_network'))
    with pytest.raises(protocol.ProtocolError):
        agent.execute({'text': '稍后跟进'}, call)
    assert call.call_count == 1


def test_scheduler_platform_failure_does_not_block_micro_events(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock(side_effect=[protocol.ProtocolError('platform_failed'), {}])
    monkeypatch.setattr(worker, 'rpc', call)
    worker.tick.run()
    assert [c.args[0] for c in call.call_args_list] == ['platform_drain', 'tick']


def test_upward_report_is_optional_and_uses_existing_assessment(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.group_agent_api.app.community_manager_report import build_report
    context = {'previous': None, 'sources': {'profile:1:goal': {'text': 'find a CTO'}}}
    assessment = {'needs': [{'id': 'need_tech', 'clarity': 'clear', 'evidence_keys': ['profile:1:goal']}]}
    monkeypatch.delenv('WECOM_SERVICE_MANAGER_REPORT_ENABLED', raising=False)
    assert build_report(assessment, context) is None
    monkeypatch.setenv('WECOM_SERVICE_MANAGER_REPORT_ENABLED', '1')
    report = build_report(assessment, context)
    assert report['type'] == 'new_clear_need'
    assert report['evidence_keys'] == ['profile:1:goal']
    context['previous'] = assessment
    assert build_report(assessment, context) is None


def test_upward_report_invalid_evidence_does_not_authorize_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.group_agent_api.app.community_manager_report import build_report
    monkeypatch.setenv('WECOM_SERVICE_MANAGER_REPORT_ENABLED', '1')
    assert build_report({'needs': [{'id': 'need_x', 'clarity': 'clear', 'evidence_keys': ['foreign']}]}, {'sources': {}, 'previous': None}) is None


def test_semantic_search_passes_only_numbers_through_scope_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager.agent import invoke_tool
    from apps.community_manager import public_users
    monkeypatch.setattr(public_users, 'candidate_numbers', lambda args: ['123', '456'])
    call = Mock(return_value={'users': [{'user_number': 123}]})
    result = invoke_tool(call, 'search_public_users', {'q': '希望找技术合伙人的创业者'}, 'stable')
    assert result['users'] == [{'user_number': 123}]
    call.assert_called_once_with('tool', name='filter_public_candidates', arguments={'numbers': ['123', '456']}, call_id='stable')


def test_public_search_has_fixed_url_no_auth_and_bounded_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import public_users
    response = Mock(); response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=None)
    response.read.return_value = b'{"data":[{"number":123,"phone":"not-exposed"}]}'
    opener = Mock(); opener.open.return_value = response
    monkeypatch.setattr(public_users, 'build_opener', lambda *args: opener)
    assert public_users.candidate_numbers({'keyword': 'AI'}) == ['123']
    request = opener.open.call_args.args[0]
    assert request.full_url.startswith('https://www.aihehuo.com/ai/users?')
    assert 'X-agent-key' not in request.headers
    with pytest.raises(ValueError):
        public_users.candidate_numbers({'q': 'AI', 'url': 'http://evil.test'})
