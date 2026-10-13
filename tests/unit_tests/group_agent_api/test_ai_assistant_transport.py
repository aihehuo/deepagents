"""Assistant protocol isolation and reuse of the existing execution core."""
import asyncio
import json
from unittest.mock import AsyncMock
import pytest
from fastapi import HTTPException
from apps.group_agent_api.app.assistant_protocol import fingerprint
from apps.group_agent_api.app.models import AsyncCallRequest
from apps.group_agent_api.app.async_manager import execute_async_run_core
from apps.group_agent_api.agent_factory.integrations.principal import sign_principal


def request_data():
    """Build one strict frozen assistant request."""
    body = dict(request_schema_version=5, queue_schema_version=5, run_id='test-run', idempotency_key='test-key',
        user_id='12', unionid=None, group_id='global', conversation_id='ai_assistant_12', message='conversation',
        membership='unknown', group_token=None, user_token=None, callback_url='http://micro:3001/group_agent_callbacks/test-run',
        metadata={'source':'ai_assistant'}, rollout_context=None, run_match=False, run_invite=False, willing_to_at=False,
        assistant_input={'kind':'conversation','user_id':12,'messages':[],'circles':[]}, execution_mode='ai_assistant')
    body['request_fingerprint'] = fingerprint(body)
    return body


def test_assistant_roundtrip_and_legacy_fields_are_unchanged():
    body = request_data()
    assert AsyncCallRequest.model_validate(body).model_dump() == body
    headers = sign_principal(user_id='12', unionid=None, secret='synthetic', request_schema_version=5, request_body=json.dumps(body).encode())
    assert headers['X-GA-Request-Version'] == 'GA-REQUEST-V5'


@pytest.mark.parametrize('field,value', [('user_id','13'),('run_invite',True),('group_token','token'),('request_schema_version',1),('queue_schema_version',1)])
def test_assistant_refuses_identity_tools_and_downgrade(field,value):
    body = request_data(); body[field] = value
    with pytest.raises((HTTPException,ValueError)):
        AsyncCallRequest.model_validate(body)


def test_current_fingerprint_covers_complete_frozen_context():
    body = request_data(); body['assistant_input']['instruction'] = 'changed'
    with pytest.raises((HTTPException,ValueError)):
        AsyncCallRequest.model_validate(body)


def test_shared_core_executes_tool_free_assistant_before_legacy_graph(monkeypatch):
    from apps.group_agent_api.app.endpoints import ai_assistant
    execute = AsyncMock(return_value={'reply':'hello','profile':{},'proposals':[]})
    monkeypatch.setattr(ai_assistant,'execute',execute)
    callback=AsyncMock(); gates=[]
    asyncio.run(execute_async_run_core(req=AsyncCallRequest.model_validate(request_data()),session=None,state=None,tid='isolated',emit_callback=callback,side_effect_gate=gates.append))
    callback.assert_awaited_once_with('final', {'assistant_result':execute.return_value})
    assert gates == ['assistant_generate','assistant_result']


def test_shared_worker_restores_and_fences_assistant_identity():
    from apps.group_agent_worker.tasks import _rebuild_session
    from apps.group_agent_api.app.utils import thread_id
    body = request_data()
    envelope = {'request':body, 'thread_id':thread_id(user_id='12',group_id='global',conversation_id='ai_assistant_12'),
        'session':{'user_id':'12','unionid':None,'group_id':'global','membership_tier':'unknown','capabilities':{'can_search':False,'can_write_profile':False}}}
    restored, session, tid = _rebuild_session(envelope)
    assert restored.request_schema_version == 5
    assert not session.capabilities.can_search and not session.capabilities.can_write_profile
    envelope['session']['user_id'] = '13'
    with pytest.raises(ValueError):
        _rebuild_session(envelope)


def test_ruby_fingerprint_and_hmac_are_identical_to_python():
    import hashlib
    import hmac
    from pathlib import Path
    from apps.group_agent_api.agent_factory.integrations.principal import canonical_principal_payload, canonical_request_payload
    fixture = Path(__file__).parents[2] / 'fixtures' / 'ai_assistant' / 'v5-ruby.json'
    vector = json.loads(fixture.read_text())
    body = json.loads(vector['body']); headers = vector['headers']
    assert AsyncCallRequest.model_validate(body).model_dump() == body
    kwargs = dict(user_id=body['user_id'], unionid=None, user_token=None, group_token=None,ts=headers['X-GA-Ts'],nonce=headers['X-GA-Nonce'],method='POST',path='/call_async',version='GA-PRINCIPAL-V5')
    key = vector['secret'].encode()
    assert hmac.new(key,canonical_principal_payload(**kwargs).encode(),hashlib.sha256).hexdigest() == headers['X-GA-Signature']
    request = canonical_request_payload(method='POST',path='/call_async',body=vector['body'].encode(),ts=headers['X-GA-Ts'],nonce=headers['X-GA-Nonce'],request_schema_version=5)
    assert hmac.new(key,request.encode(),hashlib.sha256).hexdigest() == headers['X-GA-Request-Signature']


def test_v5_actual_principal_verifier_requires_body_signature_and_rejects_replay(monkeypatch):
    from starlette.requests import Request
    from apps.group_agent_api.agent_factory.integrations.principal import resolve_session_principal, clear_nonce_cache
    monkeypatch.setenv('GROUP_AGENT_INTEGRATION','http')
    monkeypatch.setenv('GROUP_AGENT_PRINCIPAL_HMAC_SECRET','synthetic-shared-key')
    clear_nonce_cache()
    body = json.dumps(request_data()).encode()
    headers = sign_principal(user_id='12',unionid=None,secret='synthetic-shared-key',method='POST',path='/call_async',request_schema_version=5,request_body=body)
    scope = dict(type='http',method='POST',path='/call_async',headers=[(k.lower().encode(),v.encode()) for k,v in headers.items()])
    request = Request(scope)
    with pytest.raises(HTTPException):
        resolve_session_principal(request,body_user_id='12',request_schema_version=5,request_body=body+b' ')
    clear_nonce_cache()
    principal = resolve_session_principal(request,body_user_id='12',request_schema_version=5,request_body=body)
    assert principal.user_id == '12' and principal.unionid is None
    with pytest.raises(HTTPException):
        resolve_session_principal(request,body_user_id='12',request_schema_version=5,request_body=body)
