from apps.community_manager import llm
"""No-network checks for background routing, tool parity and run-bound usage."""
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import APIStatusError

from apps.community_manager.llm import RoutedModel, MissingPrimary, usage_scope, validate_primary


class Model:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.calls: list = []
        self.bindings: list = []

    def bind_tools(self, tools: list, **kwargs: object) -> "Model":
        self.bindings.append((tools, kwargs))
        return self

    def invoke(self, messages: list, **kwargs: object) -> AIMessage:
        self.calls.append((messages, kwargs))
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def output() -> AIMessage:
    return AIMessage(content="private fixture text", response_metadata={"model_name":"auto","request_id":"fixture-request"},
                     usage_metadata={"input_tokens":10,"output_tokens":2,"total_tokens":12})


def routed(primary: Model | None, standby: Model) -> RoutedModel:
    return RoutedModel(standby, primary, node="manager_reasoning", standby_model="qwen-plus", timeout=30)


def test_primary_and_telemetry_are_separate_from_user_content() -> None:
    events = []
    def rpc(action: str, **values: object) -> dict:
        events.append((action, values))
        return {"recorded":True}
    primary, standby = Model(output()), Model()
    with usage_scope("run", "private-token", rpc=rpc):
        assert routed(primary, standby).invoke([("human","private prompt")]).content == "private fixture text"
    assert len(primary.calls) == 1 and not standby.calls
    assert [e[1]["usage"]["status"] for e in events] == ["started","completed"]
    assert events[1][1]["usage"]["total_tokens"] == 12
    assert "private prompt" not in repr(events) and "private fixture text" not in repr(events)
    assert primary.calls[0][1]["timeout"] <= 18
    assert all(e[0]=="usage" and e[1]["token"]=="private-token" for e in events)


def test_failure_preserves_both_attempts_and_tool_binding() -> None:
    error=APIStatusError("private error",response=httpx.Response(429,request=httpx.Request("POST","https://example.test")),body=None)
    primary,standby=Model(error),Model(output())
    events=[]
    with usage_scope("run","token",rpc=lambda action,**kw: events.append(kw["usage"]) or {}):
        model=routed(primary,standby).bind_tools([{"type":"function"}],tool_choice="search_recent_users")
        model.invoke([])
    assert primary.bindings == standby.bindings
    assert [e["status"] for e in events] == ["started","failed","started","completed"]
    assert events[0]["invocation_id"] == events[-1]["invocation_id"]
    assert events[1]["usage_status"] == "unknown" and "total_tokens" not in events[1]
    assert events[-1]["attempt"] == 2 and events[-1]["provider"] == "dashscope"


def test_missing_primary_uses_standby_and_local_failure_is_observed() -> None:
    events=[]
    with usage_scope("run","token",rpc=lambda action,**kw:events.append(kw["usage"]) or {}):
        routed(None,Model(output())).invoke([])
    assert events[1]["error_class"] == MissingPrimary.__name__
    assert events[-1]["attempt"] == 2


def test_metering_failure_never_replaces_result_or_leaks_error(caplog: pytest.LogCaptureFixture) -> None:
    def rpc(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("PRIVATE RESPONSE AND KEY")
    with usage_scope("run","token",rpc=rpc):
        assert routed(Model(output()),Model()).invoke([]).content == "private fixture text"
    assert "PRIVATE RESPONSE" not in caplog.text


def test_parsing_failure_is_not_a_supplier_fallback() -> None:
    standby=Model()
    with pytest.raises(ValueError):
        routed(Model(ValueError("parse")),standby).invoke([])
    assert not standby.calls


def test_each_repair_is_a_new_invocation_and_scopes_reset() -> None:
    events=[]
    model=routed(Model(output(),output(),output()),Model())
    with usage_scope("run","token",rpc=lambda action,**kw:events.append(kw["usage"]) or {}):
        model.invoke([])
        model.invoke([],usage_node="customer_assessment_repair")
    model.invoke([])
    completed=[e for e in events if e["status"]=="completed"]
    assert len(completed)==2 and completed[0]["invocation_id"]!=completed[1]["invocation_id"]
    assert completed[1]["node"]=="customer_assessment_repair"


def test_deployment_preflight_requires_primary_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMMUNITY_LLM_PRIMARY_API_KEY",raising=False)
    with pytest.raises(ValueError,match="community_primary_unconfigured"):validate_primary()
    monkeypatch.setenv("COMMUNITY_LLM_PRIMARY_API_KEY","fake")
    monkeypatch.setenv("COMMUNITY_LLM_PRIMARY_BASE_URL","http://example.test")
    with pytest.raises(ValueError,match="invalid_primary_endpoint"):validate_primary()
    monkeypatch.setenv("COMMUNITY_LLM_PRIMARY_BASE_URL","https://example.test/v1")
    validate_primary()


def test_worker_scopes_usage_and_preserves_business_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import worker
    calls=[]
    def rpc(action: str, **values: object) -> dict:
        calls.append((action,values))
        return {"token":"run-token","text":"fixture","trigger":{"kind":"staff"}} if action=="begin" else {"recorded":True}
    def execute(started: dict, call: object) -> tuple[str,str]:
        routed(Model(output()),Model()).invoke([])
        return "business reply","auto"
    monkeypatch.setattr(worker,"rpc",rpc)
    monkeypatch.setattr(worker,"execute",execute)
    worker.run.run("run-fixture")
    assert [name for name,_ in calls]==["begin","usage","usage","result"]
    assert calls[-1][1]["output"]["text"]=="business reply"
    assert "usage" not in calls[-1][1]["output"]
    assert all(values["token"]=="run-token" for name,values in calls if name=="usage")


def test_primary_preserves_json_contract_without_copying_standby_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import llm
    factory=Mock(return_value=Model(output()))
    monkeypatch.setattr(llm,"ChatOpenAI",factory)
    monkeypatch.setenv("COMMUNITY_LLM_PRIMARY_API_KEY","fake-primary-key")
    monkeypatch.setenv("COMMUNITY_LLM_PRIMARY_BASE_URL","https://example.test/v1")
    model=llm.routed(Model(),node="customer_assessment",model="qwen-plus",timeout=60,max_tokens=4000,response_format={"type":"json_object"})
    model.invoke([])
    kwargs=factory.call_args.kwargs
    assert kwargs["api_key"]=="fake-primary-key" and kwargs["model"]=="auto"
    assert kwargs["model_kwargs"]["response_format"]=={"type":"json_object"}
    assert kwargs["extra_body"]["enable_thinking"] is False
    assert kwargs["max_retries"]==0


def test_both_readiness_paths_reject_missing_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.community_manager import api
    from fastapi import HTTPException
    for prefix in ["COMMUNITY_MANAGER_MODEL","COMMUNITY_CUSTOMER_MODEL"]:
        for suffix in ["","_API_KEY","_BASE_URL"]:monkeypatch.setenv(prefix+suffix,"fake")
    monkeypatch.setenv("COMMUNITY_MANAGER_AGENT_KEY","fake-agent")
    monkeypatch.setenv("COMMUNITY_MANAGER_OPERATIONS_URL","https://example.test")
    monkeypatch.setenv("COMMUNITY_MANAGER_SECRET","s"*32)
    monkeypatch.delenv("COMMUNITY_LLM_PRIMARY_API_KEY",raising=False)
    monkeypatch.setattr(api,"redis_client",lambda:Mock())
    for ready in [api.ready,api.customer_ready]:
        with pytest.raises(HTTPException) as error:ready()
        assert error.value.status_code==503
    monkeypatch.setenv("COMMUNITY_LLM_PRIMARY_API_KEY","fake-primary")
    monkeypatch.setenv("COMMUNITY_LLM_PRIMARY_BASE_URL","https://example.test/v1")
    assert api.ready()["status"]=="ready"
    assert api.customer_ready()["status"]=="ready"


def test_manager_primary_gets_45_seconds_and_fallback_remaining(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(llm.time, 'monotonic', lambda: clock[0])
    class SlowPrimary(Model):
        def invoke(self, messages: list, **kwargs: object):
            self.calls.append((messages, kwargs)); clock[0] += 45
            raise TimeoutError('fixture')
    primary = SlowPrimary(); standby = Model(output())
    model = RoutedModel(standby, primary, node='manager_reasoning', standby_model='qwen-plus', timeout=75)
    model.invoke([])
    assert primary.calls[0][1]['timeout'] == 45
    assert standby.calls[0][1]['timeout'] == 30


def test_nested_task_deadline_caps_model_wait_and_resets(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(llm.time, 'monotonic', lambda: clock[0])
    primary = Model(output(), output())
    model = RoutedModel(Model(), primary, node='manager_guidance', standby_model='qwen-plus', timeout=90)
    with llm.model_deadline(10):
        model.invoke([])
        assert primary.calls[0][1]['timeout'] == 6
        clock[0] += 11
        import pytest
        with pytest.raises(TimeoutError):model.invoke([])
    model.invoke([])
    assert primary.calls[-1][1]['timeout'] == 45
