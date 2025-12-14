from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.routing import build_default_coder_routing_middleware
from deepagents.subagent_presets import build_coder_subagent_from_env
from tests.model_provider import create_test_model, load_test_model_config


def _called_task_with_subagent_type(messages, subagent_type: str) -> bool:
    """Return True if we see a `task(...)` tool call selecting the given subagent_type."""
    for message in messages or []:
        if getattr(message, "type", None) == "ai" and getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                if tool_call.get("name") != "task":
                    continue
                args = tool_call.get("args") or {}
                if args.get("subagent_type") == subagent_type:
                    return True
    return False


def _extract_model_name(model) -> str | None:
    """Best-effort extraction of model name across LangChain chat model implementations."""
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _last_ai_text(messages) -> str:
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "ai":
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _looks_like_full_html_document(text: str) -> bool:
    t = (text or "").lower()
    # Accept either raw HTML or fenced code blocks containing HTML.
    return ("<html" in t and "</html>" in t) or ("```html" in t and "<html" in t and "</html>" in t)


@pytest.mark.integration
def test_routing_delegates_html_to_coder_subagent_real_llm(tmp_path, monkeypatch, request) -> None:
    """Real-LLM integration test for routing -> coder delegation via the `task` tool.

    This intentionally asserts on the *tool call* (task + subagent_type="coder"), not on
    the exact content, to avoid brittleness across providers/models.
    """
    # Encourage deterministic tool choice when models support it.
    monkeypatch.setenv("MODEL_API_TEMPERATURE", "0")
    monkeypatch.setenv("CODER_MODEL_API_TEMPERATURE", "0")

    # This test is specifically meant to cover the subagent preset builder.
    # Require coder-specific model name; otherwise we'd just fall back to MODEL_NAME.
    if not (os.environ.get("CODER_MODEL_NAME") or ""):
        pytest.skip(
            "CODER_MODEL_NAME is not set; this test is meant to validate build_coder_subagent_from_env(). "
            "Set CODER_MODEL_NAME (and CODER_MODEL_API_KEY / CODER_MODEL_BASE_URL if needed) to run."
        )

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    model = create_test_model(cfg=cfg)

    coder_subagent = build_coder_subagent_from_env(tools=[], name="coder")
    if coder_subagent is None:
        pytest.skip(
            "Coder subagent preset is not configured. Set CODER_MODEL_API_KEY (and optionally CODER_MODEL_BASE_URL, "
            "CODER_MODEL_API_PROVIDER) plus CODER_MODEL_NAME to run."
        )

    # Sanity-check we actually used the coder-specific model name from env.
    expected_coder_model_name = (os.environ.get("CODER_MODEL_NAME") or "").strip()
    actual_coder_model_name = _extract_model_name(coder_subagent.get("model"))
    assert actual_coder_model_name == expected_coder_model_name

    agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=str(tmp_path)),
        tools=[],
        checkpointer=MemorySaver(),
        subagents=[coder_subagent],
        middleware=[build_default_coder_routing_middleware(coder_subagent_type="coder")],
        system_prompt=(
            "You are an orchestrator. Follow routing hints added by middleware. "
            "When a routing hint applies, use the `task` tool accordingly.\n\n"
            "Output rules:\n"
            "- If the user asks for HTML, your FINAL response MUST include a complete HTML document.\n"
            "- Prefer returning the HTML verbatim (do not summarize instead of returning code)."
        ),
    )

    config = {"configurable": {"thread_id": "test-routing-coder-html"}}

    user_request = (
        "Create a small but non-trivial responsive landing page in HTML (with embedded CSS + a tiny JS interaction). "
        "Make it look modern and include sections: hero, features, pricing, and FAQ. "
        "Use semantic HTML, and output the full HTML. "
        "Do not just describe it—produce the actual code. "
        "Return ONLY the HTML (either raw or in a single ```html fenced block)."
    )

    result = agent.invoke({"messages": [HumanMessage(content=user_request)]}, config)
    messages = result.get("messages", [])

    assert _called_task_with_subagent_type(messages, "coder"), (
        "Expected the orchestrator to delegate code/HTML work using the `task` tool with "
        'subagent_type="coder". If this fails consistently for a given provider, consider '
        "increasing the routing middleware strictness further or using a model with stronger tool-calling."
    )

    final_text = _last_ai_text(messages)
    should_print = (os.environ.get("BC_TEST_PRINT_HTML") or "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    )
    # If pytest is run with `-s`, capture mode is "no" and printing is useful for debugging.
    if request.config.getoption("capture") == "no":
        should_print = True

    if should_print:
        print("\n========== FINAL OUTPUT (HTML) ==========\n")
        print(final_text)
        print("\n========== END FINAL OUTPUT ==========\n")

    assert _looks_like_full_html_document(final_text), (
        "Expected the final assistant message to contain a complete HTML document (<html>..</html>). "
        "If your orchestrator is summarizing instead of returning code, tighten the system prompt further."
    )


