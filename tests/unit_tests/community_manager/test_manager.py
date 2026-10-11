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


def test_staff_reply_preserves_full_text_for_backend_multipart(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock(side_effect=[{'token': 'token', 'text': 'next'}, {}])
    monkeypatch.setattr(worker, 'rpc', call)
    monkeypatch.setattr(worker, 'execute', Mock(return_value=('🧑' * 1000, 'deterministic')))
    worker.run.run('id')
    text = call.call_args.kwargs['output']['text']
    assert text == '🧑' * 1000
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
    call.assert_called_once_with('tool', name='filter_public_candidates', arguments={'numbers': ['123', '456'], 'contact_scope': 'reachable'}, call_id='stable')


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


def test_failure_log_has_metadata_without_exception_payload(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    call = Mock(side_effect=[{"token": "token", "text": "private input"}, {}])
    monkeypatch.setattr(worker, "rpc", call)
    monkeypatch.setattr(worker, "execute", Mock(side_effect=RuntimeError("SECRET customer payload")))
    worker.run.run("opaque-run")
    assert "class=RuntimeError" in caplog.text
    assert "run_id=opaque-run" in caplog.text
    assert "SECRET" not in caplog.text
    assert "private input" not in caplog.text


def test_manager_does_not_copy_historical_failure_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    from langchain_core.messages import AIMessage
    model = Mock()
    model.invoke.side_effect = [AIMessage(content=agent.FAILURE_NOTICE),
        AIMessage(content='', tool_calls=[{'name': 'get_scope_metrics', 'args': {}, 'id': 't1', 'type': 'tool_call'}]),
        AIMessage(content='已核验当前范围。')]
    monkeypatch.setattr(agent, 'client', lambda: Mock(bind_tools=Mock(return_value=model)))
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'test')
    call = Mock(return_value={'user_count': 2})
    text, _ = agent.execute({'text': '请核验', 'history': [{'role': 'manager', 'text': agent.FAILURE_NOTICE}], 'global_write': True}, call)
    assert text == '已核验当前范围。'
    call.assert_called_once()
    first_messages = model.invoke.call_args_list[0].args[0]
    assert not any(isinstance(m, AIMessage) and m.content == agent.FAILURE_NOTICE for m in first_messages[:2])


def test_tool_budget_returns_verified_partial_summary_without_extra_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    from langchain_core.messages import AIMessage, ToolMessage
    model = Mock()
    model.invoke.return_value = AIMessage(content='', tool_calls=[
        {'name': 'get_scope_metrics', 'args': {}, 'id': 't' + str(i), 'type': 'tool_call'} for i in range(50)])
    plain = Mock()
    plain.bind_tools.return_value = model
    plain.invoke.return_value = AIMessage(content='已核验范围；逐人指导尚未生成。')
    monkeypatch.setattr(agent, 'client', lambda: plain)
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'test')
    call = Mock(return_value={'user_count': 2})
    text, _ = agent.execute({'text': '请分析十位'}, call)
    assert call.call_count == 48
    assert '已达到分析上限' in text
    assert '尚未生成' in text
    pending = [m for m in plain.invoke.call_args.args[0] if isinstance(m, ToolMessage) and 'not_executed_budget_limit' in m.content]
    assert len(pending) == 2


def test_step_budget_summarizes_without_running_more_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    from langchain_core.messages import AIMessage
    model = Mock()
    model.invoke.return_value = AIMessage(content='', tool_calls=[{'name': 'get_scope_metrics', 'args': {}, 'id': 't', 'type': 'tool_call'}])
    plain = Mock(); plain.bind_tools.return_value = model
    plain.invoke.return_value = AIMessage(content='已核验范围，剩余需求需继续核对。')
    monkeypatch.setattr(agent, 'client', lambda: plain)
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'test')
    call = Mock(return_value={'user_count': 2})
    text, _ = agent.execute({'text': '分析'}, call)
    assert call.call_count == 12
    assert model.invoke.call_count == 12
    assert '已达到分析上限' in text


def test_expired_budget_does_not_start_another_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    plain = Mock(); monkeypatch.setattr(agent, 'client', lambda: plain)
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'test')
    text, _ = agent.partial_summary([], '时间', agent.time.monotonic())
    plain.invoke.assert_not_called()
    assert '尚未完成' in text


def test_manager_extended_budget_does_not_change_customer_worker_limits() -> None:
    from apps.community_manager import agent
    assert (agent.MAX_TOOL_CALLS, agent.MAX_MODEL_STEPS, agent.MAX_SECONDS, agent.MAX_CONTEXT_BYTES) == (48, 12, 420, 240_000)
    assert worker.run.soft_time_limit == 500
    assert worker.run.time_limit == 540
    assert celery.conf.task_soft_time_limit == 220
    assert celery.conf.task_time_limit == 240


def test_ascii_reply_is_not_truncated_at_650_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock(side_effect=[{'token': 'token', 'text': 'next'}, {}])
    monkeypatch.setattr(worker, 'rpc', call)
    monkeypatch.setattr(worker, 'execute', Mock(return_value=('a' * 1200, 'test')))
    worker.run.run('id')
    assert call.call_args.kwargs['output']['text'] == 'a' * 1200


def test_candidate_tool_schema_defaults_and_outreach_instructions() -> None:
    from apps.community_manager.agent import SKILL
    from apps.community_manager.tool_schema import TOOLS
    for tool in TOOLS:
        if tool['function']['name'] in {'search_users', 'search_public_users'}:
            assert tool['function']['parameters']['properties']['contact_scope']['enum'] == ['reachable', 'unbound', 'all']
    assert '未加企微名单仅在客服询问时出现' in SKILL.read_text()


@pytest.mark.parametrize('mode, reachable', [('reachable', True), ('unbound', False)])
def test_public_query_reachable_filter_is_not_sent_for_outreach(monkeypatch: pytest.MonkeyPatch, mode: str, reachable: bool) -> None:
    from apps.community_manager import public_users
    from urllib.parse import parse_qs, urlparse
    response = Mock(); response.read.return_value = b'{"data":[{"number":123}]}'
    from contextlib import nullcontext
    opener = Mock(); opener.open.return_value = nullcontext(response)
    monkeypatch.setattr(public_users, 'build_opener', lambda *_: opener)
    assert public_users.candidate_numbers({'q': '技术合伙人', 'contact_scope': mode}) == ['123']
    query = parse_qs(urlparse(opener.open.call_args.args[0].full_url).query)
    assert (query.get('wechat_reachable_only') == ['true']) == reachable
    assert 'contact_scope' not in query
    assert query['per_page'] == ['20']


def test_public_search_forwards_valid_page_and_rejects_invalid_page(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import public_users
    from contextlib import nullcontext
    from urllib.parse import parse_qs, urlparse
    response = Mock(); response.read.return_value = b'{"data":[]}'
    opener = Mock(); opener.open.return_value = nullcontext(response)
    monkeypatch.setattr(public_users, 'build_opener', lambda *_: opener)
    assert public_users.candidate_numbers({'q': '合伙人', 'page': 2}) == []
    query = parse_qs(urlparse(opener.open.call_args.args[0].full_url).query)
    assert query['page'] == query['current_page'] == ['2']
    for page in [0, 11, True, '2']:
        with pytest.raises(ValueError):
            public_users.candidate_numbers({'q': '合伙人', 'page': page})


def test_oversized_reply_fails_explicitly_instead_of_silent_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock(side_effect=[{'token': 'token', 'text': 'next'}, {}])
    monkeypatch.setattr(worker, 'rpc', call)
    monkeypatch.setattr(worker, 'execute', Mock(return_value=('中' * 6000, 'test')))
    worker.run.run('id')
    assert call.call_args.kwargs['output']['error_code'] == 'manager_execution_failed'


def test_budget_process_reply_uses_verified_receipts_instead(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    from langchain_core.messages import AIMessage, ToolMessage
    plain = Mock(); plain.invoke.return_value = AIMessage(content='让我继续搜索，以确保找到10位最合适的用户：')
    monkeypatch.setattr(agent, 'client', lambda: plain)
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'test')
    receipts = [ToolMessage(content=json.dumps({'contact_scope': 'reachable', 'users': [
        {'user_number': 123, 'display_name': '已核验用户', 'bound': 'bound_snapshot'},
        {'user_number': 999, 'display_name': '冲突用户', 'bound': 'binding_conflict'}]}), tool_call_id='a')]
    text, _ = agent.partial_summary(receipts, '推理轮数', agent.time.monotonic() + 90)
    assert '123' in text and '999' not in text
    assert '让我继续搜索' not in text
    assert '尚未完成核验' in text


def test_ten_candidates_stop_repetitive_search_and_keep_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    from langchain_core.messages import AIMessage
    model = Mock(); model.invoke.return_value = AIMessage(content='继续搜索', tool_calls=[
        {'name': 'search_users', 'args': {}, 'id': 'search', 'type': 'tool_call'}])
    plain = Mock(); plain.bind_tools.return_value = model
    plain.invoke.return_value = AIMessage(content='让我继续搜索')
    monkeypatch.setattr(agent, 'client', lambda: plain)
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'test')
    call = Mock(return_value={'contact_scope': 'reachable', 'users': [
        {'user_number': 100+i, 'display_name': '用户', 'bound': 'bound_snapshot'} for i in range(10)]})
    text, _ = agent.execute({'text': '请列10位关注用户'}, call)
    assert call.call_count == 3
    assert all(str(100+i) in text for i in range(10))
    assert '让我继续搜索' not in text


@pytest.mark.parametrize('name', ['search_users', 'search_recent_users', 'filter_public_candidates', 'save_guidance_draft', 'notify_staff'])
def test_search_5xx_is_recoverable_but_uncertain_writes_abort(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    from urllib.error import HTTPError
    from apps.community_manager import protocol
    monkeypatch.setenv('COMMUNITY_MANAGER_OPERATIONS_URL', 'http://facade')
    monkeypatch.setenv('COMMUNITY_MANAGER_AGENT_KEY', 'test')
    monkeypatch.setenv('COMMUNITY_MANAGER_SECRET', 's' * 32)
    opener = Mock(); opener.open.side_effect = HTTPError('http://facade', 500, 'private error', {}, None)
    monkeypatch.setattr(protocol, 'build_opener', lambda *_: opener)
    if name in {'search_users', 'search_recent_users', 'filter_public_candidates'}:
        assert protocol.rpc('tool', name=name)['tool_error'] == 'candidate_search_unavailable'
    else:
        with pytest.raises(protocol.ProtocolError, match='manager_upstream_http_500'):
            protocol.rpc('tool', name=name)
    assert opener.open.call_count == 1


def test_outreach_search_converges_and_fallback_keeps_unbound_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import agent
    from langchain_core.messages import AIMessage
    model = Mock(); model.invoke.return_value = AIMessage(content='', tool_calls=[
        {'name': 'search_users', 'args': {'contact_scope': 'unbound'}, 'id': 'search', 'type': 'tool_call'}])
    plain = Mock(); plain.bind_tools.return_value = model
    plain.invoke.return_value = AIMessage(content='让我继续搜索')
    monkeypatch.setattr(agent, 'client', lambda: plain)
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'test')
    call = Mock(return_value={'contact_scope': 'unbound', 'users': [
        {'user_number': 200+i, 'display_name': '用户', 'bound': 'not_bound'} for i in range(10)]})
    text, _ = agent.execute({'text': '有哪些未添加企业微信但值得主动联系的用户'}, call)
    assert call.call_count == 3
    assert all(str(200+i) in text for i in range(10))
    assert '未加企微、主动联系候选' in text
    assert '已加企微候选' not in text
    assert '让我继续搜索' not in text


def test_recent_activity_search_is_explicit_db_tool_not_semantic_recall():
    from apps.community_manager.agent import invoke_tool, SKILL
    from apps.community_manager.tool_schema import TOOLS
    schema = next(t['function'] for t in TOOLS if t['function']['name'] == 'search_recent_users')
    assert schema['parameters']['properties']['sort']['enum'] == ['last_active_desc', 'registered_desc']
    assert schema['parameters']['properties']['contact_scope']['enum'] == ['reachable', 'unbound', 'all']
    calls = []
    def call(action, **kwargs):
        calls.append((action, kwargs))
        return {'users': [], 'sort_field': 'users.last_accessed_at'}
    args = {'city': '上海', 'sort': 'last_active_desc', 'contact_scope': 'unbound', 'limit': 5}
    assert invoke_tool(call, 'search_recent_users', args, 'recent')['sort_field'] == 'users.last_accessed_at'
    assert calls == [('tool', {'name': 'search_recent_users', 'arguments': args, 'call_id': 'recent'})]
    instructions = SKILL.read_text()
    assert '必须使用 search_recent_users' in instructions
    assert 'observed_at 是本次查询时间' in instructions


def ranking_receipt():
    return {'sort': 'last_active_desc', 'sort_field': 'users.last_accessed_at', 'contact_scope': 'unbound', 'city': '上海',
            'observed_at': '2026-10-11T04:20:00Z', 'users': [
                {'user_number': 123, 'display_name': '真实候选', 'bound': 'not_bound', 'last_active_at': '2026-10-11T04:10:18Z', 'profile': {}}]}


def test_ranking_without_current_search_cannot_repeat_history(monkeypatch):
    from langchain_core.messages import AIMessage
    from apps.community_manager import agent, ranking
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'test')
    model = Mock(); model.invoke.return_value = AIMessage(content='我已经搜索，旧用户最后活跃在九月。')
    monkeypatch.setattr(agent, 'client', lambda: Mock(bind_tools=Mock(return_value=model)))
    call = Mock()
    text, _ = agent.execute({'text': '上海未加企微用户按最近活跃倒序', 'history': [{'role': 'manager', 'text': '旧名单'}]}, call)
    assert text == ranking.FAILURE
    assert '九月' not in text
    assert model.invoke.call_count == 2
    call.assert_not_called()


def test_ranking_output_uses_only_current_receipt_dates(monkeypatch):
    from langchain_core.messages import AIMessage
    from apps.community_manager import agent
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'test')
    model = Mock(); model.invoke.side_effect = [
        AIMessage(content='', tool_calls=[{'name': 'search_recent_users', 'args': {'city': '上海', 'contact_scope': 'unbound', 'limit': 5}, 'id': 'search', 'type': 'tool_call'}]),
        AIMessage(content='我已搜索：假的用户最后活跃2026-09-22。')]
    bound = Mock(bind_tools=Mock(return_value=model))
    monkeypatch.setattr(agent, 'client', lambda: bound)
    call = Mock(return_value=ranking_receipt())
    text, _ = agent.execute({'text': '上海未加企微，按最近活跃时间倒序前五位'}, call)
    assert '真实候选' in text and '2026-10-11 12:10:18' in text
    assert '假的' not in text and '2026-09-22' not in text
    assert '实际返回1位' in text
    assert bound.bind_tools.call_args_list[-1].kwargs['tool_choice'] == 'search_recent_users'
    assert call.call_args.kwargs['name'] == 'search_recent_users'


@pytest.mark.parametrize('change', [{'city': '杭州'}, {'contact_scope': 'reachable'}, {'sort_field': 'users.updated_at'}, {'tool_error': 'candidate_search_unavailable'}])
def test_wrong_scope_or_failed_rank_receipt_is_rejected(change):
    from apps.community_manager import ranking
    receipt = ranking_receipt(); receipt.update(change)
    expected = ranking.intent({'text': '上海未加企微，按最近活跃时间倒序前五位'})
    assert not ranking.valid_receipt(receipt, {'limit': 5}, expected)


def test_rank_filter_adjustment_inherits_only_last_human_request():
    from apps.community_manager import ranking
    expected = ranking.intent({'text': '去掉上海的地域限制', 'history': [{'role': 'staff', 'text': '上海未加企微按最近活跃倒序'}, {'role': 'manager', 'text': '假的旧排名'}]})
    assert expected['sort'] == 'last_active_desc' and expected['no_city'] and expected['contact_scope'] == 'unbound'
    assert ranking.intent({'text': '给这位用户做指导'}) is None


def test_ranking_wrong_requested_count_and_unsorted_dates_rejected():
    from apps.community_manager import ranking
    expected = ranking.intent({'text': '上海未加企微按最近活跃倒序前五位'})
    assert expected['limit'] == 5
    assert not ranking.valid_receipt(ranking_receipt(), {'limit': 3}, expected)
    assert ranking.valid_receipt(ranking_receipt(), {'limit': 5}, expected)
    receipt = ranking_receipt()
    receipt['users'].append({**receipt['users'][0], 'user_number': 456, 'last_active_at': '2026-10-11T05:00:00Z'})
    assert not ranking.valid_receipt(receipt, {'limit': 5}, expected)
