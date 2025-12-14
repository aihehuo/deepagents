from __future__ import annotations

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagents.middleware.routing import build_default_coder_routing_middleware


def test_coder_routing_middleware_injects_hint_for_html() -> None:
    model = FakeListChatModel(responses=["ok"])
    mw = build_default_coder_routing_middleware(coder_subagent_type="coder")

    req = ModelRequest(
        model=model,
        system_message=SystemMessage(content="BASE"),
        messages=[HumanMessage(content="Please write a small HTML snippet: <div>Hello</div>")],
    )

    captured: dict[str, str | None] = {"prompt": None}

    def handler(r: ModelRequest) -> ModelResponse:
        captured["prompt"] = r.system_prompt
        return ModelResponse(result=[AIMessage(content="done")])

    mw.wrap_model_call(req, handler)
    assert captured["prompt"] is not None
    assert 'subagent_type="coder"' in captured["prompt"]


def test_coder_routing_middleware_does_not_inject_for_non_code() -> None:
    model = FakeListChatModel(responses=["ok"])
    mw = build_default_coder_routing_middleware(coder_subagent_type="coder")

    req = ModelRequest(
        model=model,
        system_message=SystemMessage(content="BASE"),
        messages=[HumanMessage(content="Help me refine my elevator pitch for my startup.")],
    )

    captured: dict[str, str | None] = {"prompt": None}

    def handler(r: ModelRequest) -> ModelResponse:
        captured["prompt"] = r.system_prompt
        return ModelResponse(result=[AIMessage(content="done")])

    mw.wrap_model_call(req, handler)
    assert captured["prompt"] == "BASE"


