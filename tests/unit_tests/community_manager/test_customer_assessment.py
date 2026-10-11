"""Background synthesis stays separate from replies, manager tools and notifications."""
import json
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from apps.community_manager import api, worker, protocol, customer_model
from apps.community_manager.celery_app import app as celery, QUEUE, CUSTOMER_QUEUE


@pytest.fixture(autouse=True)
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('COMMUNITY_MANAGER_SECRET', 's' * 32)
    for suffix in ('', '_API_KEY', '_BASE_URL'):
        monkeypatch.setenv('COMMUNITY_CUSTOMER_MODEL' + suffix, 'customer-test')


def sample() -> tuple[dict, dict]:
    """Return bounded host evidence and a valid synthesis envelope."""
    sources = {'profile:1:bio': {'role': 'profile_self_report', 'text': '创业者，能够提供开发能力', 'recorded_at': '2026-10-10T00:00:00Z'}}
    context = {'protocol': {'schema_version': 1, 'user_number': '123456', 'sources': sources, 'previous': None, 'observed_at': '2026-10-10T00:00:00Z'},
               'certifications': {'identity_verified': True, 'career_verified': False, 'records': []}, 'completeness': {'present': ['bio'], 'missing': ['headline', 'goal']}, 'service_facts': {'binding': {'status': 'not_bound'}, 'recent_service': [], 'introductions': []}}
    keys = ['profile:1:bio']
    body = {'journey': {'schema_version': 1, 'founder_profile': {'doing': '创业', 'background': '技术', 'offer': '开发能力', 'clarity': 'clear', 'missing': [], 'evidence_keys': keys},
                        'has_need': 'unknown', 'intent_evidence_keys': [], 'needs': [], 'needs_truncated': False, 'next_action': '确认意愿'},
            'evaluation': {'reason': '资料值得了解', 'next_action': '人工核实需求', 'evidence_keys': keys, 'escalation': None}}
    return context, body


def test_admission_only_enqueues_signed_opaque_run_on_customer_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = Mock(); redis.set.return_value = True; redis.llen.return_value = 0
    monkeypatch.setattr(api, 'redis_client', lambda: redis)
    enqueue = Mock(); monkeypatch.setattr(api.celery, 'send_task', enqueue)
    client = TestClient(api.app); body = b'{"run_id":"12345678-1234-1234-1234-123456789abc"}'
    assert client.post('/customer/assess_async', content=body, headers=protocol.headers(body)).status_code == 202
    assert enqueue.call_args.kwargs['queue'] == CUSTOMER_QUEUE
    assert enqueue.call_args.args[0] == 'community_customer.run'
    assert CUSTOMER_QUEUE != QUEUE
    assert celery.conf.task_routes['community_customer.*']['queue'] == CUSTOMER_QUEUE
    enqueue.reset_mock()
    assert client.post('/customer/assess_async', content=body, headers={'Content-Type': 'application/json'}).status_code == 401
    enqueue.assert_not_called()


def test_customer_model_has_no_manager_tool_bindings_and_distinct_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    context, body = sample()
    model = Mock(); model.invoke.return_value.content = json.dumps(body)
    factory = Mock(return_value=model); monkeypatch.setattr(customer_model, 'ChatOpenAI', factory)
    result, name = customer_model.assess(context)
    assert result['journey']['has_need'] == 'unknown'
    assert name == 'customer-test'
    assert factory.call_args.kwargs['max_retries'] == 0
    model.bind_tools.assert_not_called()
    assert '不能自行认证' in model.invoke.call_args.args[0][0][1]


@pytest.mark.parametrize('fault', ['foreign_evidence', 'extra_action', 'oversized_reason', 'false_escalation'])
def test_invalid_model_output_cannot_be_reported(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    context, body = sample()
    if fault == 'foreign_evidence': body['evaluation']['evidence_keys'] = ['other_user:1']
    elif fault == 'extra_action': body['send_customer_message'] = True
    elif fault == 'oversized_reason': body['evaluation']['reason'] = '汉' * 1000
    else: body['evaluation']['escalation'] = 'approve'
    model = Mock(); model.invoke.return_value.content = json.dumps(body)
    monkeypatch.setattr(customer_model, 'ChatOpenAI', Mock(return_value=model))
    with pytest.raises(ValueError): customer_model.assess(context)


def test_customer_failure_returns_fixed_failure_without_reply_or_manager_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    call = Mock(side_effect=[{'token': 't', 'trigger': {'kind': 'customer_assessment'}, 'customer_assessment': sample()[0]}, {}])
    monkeypatch.setattr(worker, 'rpc', call)
    execute = Mock(); monkeypatch.setattr(worker, 'execute', execute)
    monkeypatch.setattr(customer_model, 'assess', Mock(side_effect=RuntimeError('private secret')))
    worker.customer_run.run('id')
    assert call.call_args.kwargs['output'] == {'error_code': 'customer_assessment_failed'}
    execute.assert_not_called()


def test_duplicate_customer_delivery_does_not_run_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, 'rpc', Mock(return_value={'status': 'running'}))
    assess = Mock(); monkeypatch.setattr(customer_model, 'assess', assess)
    worker.customer_run.run('id'); assess.assert_not_called()


def test_customer_model_requires_own_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('COMMUNITY_CUSTOMER_MODEL', raising=False)
    monkeypatch.setenv('COMMUNITY_MANAGER_MODEL', 'manager-is-not-a-fallback')
    with pytest.raises(ValueError, match='customer_model_unconfigured'): customer_model.assess(sample()[0])


def test_customer_contract_does_not_import_customer_application() -> None:
    import subprocess
    import sys
    script = "import sys; import apps.community_manager.customer_model; assert 'apps.group_agent_api.app' not in sys.modules; assert 'apps.group_agent_api.agent_factory.agent' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_customer_failures_log_location_without_private_payload(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    call = Mock(side_effect=[{'token': 't', 'trigger': {'kind': 'customer_assessment'}, 'customer_assessment': sample()[0]}, {}])
    monkeypatch.setattr(worker, 'rpc', call)
    monkeypatch.setattr(customer_model, 'assess', Mock(side_effect=ValueError('SECRET private customer text')))
    worker.customer_run.run('opaque-run')
    assert 'customer_assessment_failed run_id=opaque-run class=ValueError' in caplog.text
    assert 'SECRET' not in caplog.text
    assert 'private customer text' not in caplog.text


def test_invalid_journey_gets_one_strict_correction(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import copy
    context, body = sample()
    bad = copy.deepcopy(body); del bad['journey']['founder_profile']['missing']
    model = Mock(); model.invoke.side_effect = [Mock(content=json.dumps(bad)), Mock(content=json.dumps(body))]
    monkeypatch.setattr(customer_model, 'ChatOpenAI', Mock(return_value=model))
    result, _ = customer_model.assess(context)
    assert result == body
    assert model.invoke.call_count == 2
    assert 'founder_profile_schema' in caplog.text
    assert 'profile:1:bio' not in caplog.text
    assert '只能使用sources内真实key' in model.invoke.call_args.args[0][-1][1]


def test_invalid_repair_is_rejected_without_third_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    context, bad = sample(); bad['journey']['founder_profile']['evidence_keys'] = ['foreign:1']
    model = Mock(); model.invoke.return_value.content = json.dumps(bad)
    monkeypatch.setattr(customer_model, 'ChatOpenAI', Mock(return_value=model))
    with pytest.raises(ValueError): customer_model.assess(context)
    assert model.invoke.call_count == 2


def test_provider_failure_is_not_a_format_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    model = Mock(); model.invoke.side_effect = RuntimeError('private provider response')
    monkeypatch.setattr(customer_model, 'ChatOpenAI', Mock(return_value=model))
    with pytest.raises(RuntimeError): customer_model.assess(sample()[0])
    assert model.invoke.call_count == 1
