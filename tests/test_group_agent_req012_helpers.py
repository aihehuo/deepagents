"""REQ-012 helper tests — DEFAULT gate, NO network, NO real-LLM credentials.

These prove the budget guard, exception propagation, outcome classification,
the runner outcome/exit allow-matrix, and the model-mode fail-closed contract
WITHOUT the opt-in real scenario. They run on every plain `pytest` invocation
and must NOT be skipped alongside the real scenario (REQ-012-FIX2 §5 /
REQ-012-FIX3). No `real_llm` marker, no GROUP_AGENT_REAL_LLM_TEST gate.
"""

from __future__ import annotations

import pytest
from langchain_core.callbacks import CallbackManager

from apps.group_agent_api.agent_factory.model_builder import create_model
from tests.support.req012_llm_budget import (
    MAX_LLM_INVOCATIONS,
    GroupAgentIsolationError,
    LLMBudgetExceeded,
    LLMBudgetRecorder,
    OutcomeKind,
    classify_error_text,
    classify_exception,
    classify_exception_type,
    classify_http_response,
    decide_runner_result,
)


# ---------------------------------------------------------------------------
# §1: 13th invocation is blocked THROUGH a real CallbackManager (no network)
# ---------------------------------------------------------------------------

def test_budget_guard_propagates_through_real_callback_manager():
    """The recorder must have raise_error=True so the real CallbackManager
    re-raises LLMBudgetExceeded — proving the block happens on the actual
    callback path, not just a bare method call."""
    recorder = LLMBudgetRecorder()
    assert recorder.raise_error is True, (
        "raise_error must be True or CallbackManager swallows the guard exception"
    )

    cm = CallbackManager(handlers=[recorder])

    # First MAX_LLM_INVOCATIONS starts succeed through the manager.
    for i in range(MAX_LLM_INVOCATIONS):
        cm.on_chat_model_start({"id": ["ChatOpenAI"]}, [[]])
        assert recorder.llm_starts == i + 1

    # The (MAX+1)-th start must propagate LLMBudgetExceeded out of the manager.
    with pytest.raises(LLMBudgetExceeded):
        cm.on_chat_model_start({"id": ["ChatOpenAI"]}, [[]])

    assert recorder.llm_starts == MAX_LLM_INVOCATIONS + 1


def test_budget_guard_allows_exactly_max_invocations():
    recorder = LLMBudgetRecorder()
    cm = CallbackManager(handlers=[recorder])
    for _ in range(MAX_LLM_INVOCATIONS):
        cm.on_chat_model_start({"id": ["ChatOpenAI"]}, [[]])
    assert recorder.llm_starts == MAX_LLM_INVOCATIONS  # no raise at the boundary


def test_recorder_model_class_captured():
    recorder = LLMBudgetRecorder()
    cm = CallbackManager(handlers=[recorder])
    cm.on_chat_model_start({"id": ["langchain", "ChatOpenAI"]}, [[]])
    assert recorder.model_class == "ChatOpenAI"


# ---------------------------------------------------------------------------
# §4-support: delta helper
# ---------------------------------------------------------------------------

def test_delta_computes_per_round_increments():
    before = {"llm_starts": 2, "llm_ends": 2, "tool_calls": {"save_group_profile": 1}, "tokens": 100}
    after = {"llm_starts": 4, "llm_ends": 4, "tool_calls": {"save_group_profile": 2}, "tokens": 250}
    d = LLMBudgetRecorder.delta(before, after)
    assert d == {
        "llm_starts": 2,
        "llm_ends": 2,
        "tool_calls": {"save_group_profile": 1},
        "tokens": 150,
    }


def test_delta_omits_zero_tool_deltas():
    before = {"llm_starts": 0, "llm_ends": 0, "tool_calls": {"save_group_profile": 3}, "tokens": 0}
    after = {"llm_starts": 1, "llm_ends": 1, "tool_calls": {"save_group_profile": 3}, "tokens": 10}
    d = LLMBudgetRecorder.delta(before, after)
    assert d["tool_calls"] == {}  # unchanged tool count → omitted, no noise


# ---------------------------------------------------------------------------
# §2: explicit outcome enum classification
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# §1 (FIX3): classification keys on EXACT exception type, never message prose.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "type_name,expected",
    [
        ("LLMBudgetExceeded", OutcomeKind.BUDGET),
        ("GroupAgentIsolationError", OutcomeKind.ISOLATION),
        ("APITimeoutError", OutcomeKind.PROVIDER_NETWORK),
        ("RateLimitError", OutcomeKind.PROVIDER_NETWORK),
        ("APIConnectionError", OutcomeKind.PROVIDER_NETWORK),
        ("ConnectionError", OutcomeKind.PROVIDER_NETWORK),
        ("InternalServerError", OutcomeKind.PROVIDER_NETWORK),
        # Unknown / internal exception types → INTERNAL
        ("KeyError", OutcomeKind.INTERNAL),
        ("ValueError", OutcomeKind.INTERNAL),
        ("AssertionError", OutcomeKind.INTERNAL),
        ("AttributeError", OutcomeKind.INTERNAL),
        ("", OutcomeKind.INTERNAL),
    ],
)
def test_classify_exception_type_exact(type_name, expected):
    assert classify_exception_type(type_name) == expected


@pytest.mark.parametrize(
    "message",
    [
        "ValueError expected 500 members",       # "500" is prose, not a status
        "AssertionError connection field missing",  # "connection" is prose
        "KeyError timeout_policy",                # "timeout" is prose
    ],
)
def test_message_substrings_do_not_leak_into_provider_network(message):
    """FIX3 §1 counter-examples: business/internal errors whose MESSAGE happens
    to contain '500' / 'connection' / 'timeout' must NOT be classified as
    external — classification uses the exception TYPE only."""
    # The message string, if fed to the (exact-type) classifier, is not a known
    # type name, so it is INTERNAL — never PROVIDER_NETWORK.
    assert classify_error_text(message) == OutcomeKind.INTERNAL


def test_classify_live_exception_walks_cause_chain():
    try:
        try:
            raise ValueError("inner business bug")
        except ValueError as inner:
            raise GroupAgentIsolationError("guard tripped") from inner
    except GroupAgentIsolationError as exc:
        assert classify_exception(exc) == OutcomeKind.ISOLATION

    # A plain internal error with no provider cause stays INTERNAL.
    try:
        raise KeyError("user_id")
    except KeyError as exc:
        assert classify_exception(exc) == OutcomeKind.INTERNAL


def test_classify_error_text_is_exact_type_alias():
    assert classify_error_text("APITimeoutError") == OutcomeKind.PROVIDER_NETWORK
    assert classify_error_text("LLMBudgetExceeded") == OutcomeKind.BUDGET
    # A bare status number as prose is NOT external.
    assert classify_error_text("500") == OutcomeKind.INTERNAL


def test_outcome_kind_categories_are_mutually_exclusive():
    assert OutcomeKind.PROVIDER_NETWORK.is_blocked_external is True
    assert OutcomeKind.TIMEOUT.is_blocked_external is True
    assert OutcomeKind.BUDGET.is_failure is True
    assert OutcomeKind.ISOLATION.is_failure is True
    assert OutcomeKind.INTERNAL.is_failure is True
    # PASSED is neither
    assert OutcomeKind.PASSED.is_blocked_external is False
    assert OutcomeKind.PASSED.is_failure is False


# ---------------------------------------------------------------------------
# §2/§7: HTTP response classification uses status + type only, no body leak
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self._detail = detail

    def json(self):
        return {"detail": self._detail}


def test_classify_http_ok():
    kind, diag = classify_http_response(_FakeResponse(200, {}))
    assert kind == OutcomeKind.PASSED
    assert diag == "status=200"


def test_classify_http_provider_network_by_status():
    # 503 is a clean provider status → external regardless of type.
    resp = _FakeResponse(503, {"error_type": "APIStatusError", "error_message": "SECRET_BODY here"})
    kind, diag = classify_http_response(resp)
    assert kind == OutcomeKind.PROVIDER_NETWORK
    assert "SECRET_BODY" not in diag
    assert "status=503" in diag


def test_classify_http_502_internal_type_is_internal():
    # Chat wraps internal errors as 502; a KeyError at 502 is NOT provider fault.
    resp = _FakeResponse(502, {"error_type": "KeyError", "error_message": "user_id"})
    kind, diag = classify_http_response(resp)
    assert kind == OutcomeKind.INTERNAL
    assert "SECRET" not in diag  # message body never in diagnostic


def test_classify_http_502_provider_type_is_external():
    resp = _FakeResponse(502, {"error_type": "APITimeoutError", "error_message": "x"})
    kind, _ = classify_http_response(resp)
    assert kind == OutcomeKind.PROVIDER_NETWORK


def test_classify_http_500_internal_bug_counterexample():
    # FIX2/FIX3 counter-example: 500 + AttributeError must be INTERNAL.
    resp = _FakeResponse(500, {"error_type": "AttributeError", "error_message": "internal bug"})
    kind, diag = classify_http_response(resp)
    assert kind == OutcomeKind.INTERNAL
    assert "internal bug" not in diag


def test_classify_http_429_rate_limit_external():
    resp = _FakeResponse(429, {"error_type": "RateLimitError", "error_message": "quota"})
    kind, _ = classify_http_response(resp)
    assert kind == OutcomeKind.PROVIDER_NETWORK


# ---------------------------------------------------------------------------
# §2 (FIX3): Runner outcome/exit allow-matrix
# ---------------------------------------------------------------------------

def test_matrix_exit0_passed_ok():
    assert decide_runner_result(0, "PASSED") == "PASSED"


def test_matrix_exit0_nonpassed_is_internal():
    # Success exit but a non-PASSED outcome file → inconsistent → INTERNAL.
    assert decide_runner_result(0, "FAILED:BUDGET") == OutcomeKind.INTERNAL.value
    assert decide_runner_result(0, "BLOCKED_EXTERNAL:PROVIDER_NETWORK") == OutcomeKind.INTERNAL.value


def test_matrix_nonzero_passed_forbidden():
    # Residual/stale PASSED with a nonzero exit → INTERNAL, never PASSED.
    assert decide_runner_result(1, "PASSED") == OutcomeKind.INTERNAL.value


def test_matrix_nonzero_supported_failure_labels_kept():
    assert decide_runner_result(1, "FAILED:BUDGET").startswith("FAILED:BUDGET")
    assert decide_runner_result(1, "BLOCKED_EXTERNAL:PROVIDER_NETWORK R3 status=503").startswith(
        "BLOCKED_EXTERNAL:PROVIDER_NETWORK"
    )
    assert decide_runner_result(1, "BLOCKED_EXTERNAL:UNKNOWN_TIMEOUT") == "BLOCKED_EXTERNAL:UNKNOWN_TIMEOUT"


def test_matrix_unknown_outcome_is_internal():
    assert decide_runner_result(1, "SOMETHING_ELSE") == OutcomeKind.INTERNAL.value
    assert decide_runner_result(0, "definitely not a label") == OutcomeKind.INTERNAL.value


def test_matrix_empty_or_missing_outcome_is_internal():
    assert decide_runner_result(1, None) == OutcomeKind.INTERNAL.value
    assert decide_runner_result(1, "") == OutcomeKind.INTERNAL.value
    assert decide_runner_result(0, "") == OutcomeKind.INTERNAL.value


def test_matrix_multiline_outcome_is_internal():
    assert decide_runner_result(0, "PASSED\nPASSED") == OutcomeKind.INTERNAL.value
    assert decide_runner_result(1, "FAILED:BUDGET\nextra line") == OutcomeKind.INTERNAL.value


def test_matrix_oversized_outcome_is_internal():
    huge = "PASSED " + ("x" * 500)
    assert decide_runner_result(0, huge) == OutcomeKind.INTERNAL.value


# ---------------------------------------------------------------------------
# §4 (FIX3): model-mode fail-closed contract — no network, no real key
# ---------------------------------------------------------------------------

def _base_mode_env(monkeypatch):
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setenv("GROUP_AGENT_PROVIDER", "qwen")
    for k in ("GROUP_AGENT_MODEL", "GROUP_AGENT_BASE_URL", "DASHSCOPE_API_KEY", "QWEN_API_KEY"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("bad_mode", ["reall", "REAL ", "prod", "live", "on", "1", "yes"])
def test_unknown_model_mode_fails_closed(monkeypatch, bad_mode):
    _base_mode_env(monkeypatch)
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", bad_mode)
    with pytest.raises(RuntimeError):
        create_model()


@pytest.mark.parametrize("mode", ["real", ""])
@pytest.mark.parametrize(
    "present",
    [
        (),                                   # all three missing
        ("GROUP_AGENT_MODEL",),               # only model
        ("GROUP_AGENT_BASE_URL",),            # only base_url
        ("DASHSCOPE_API_KEY",),               # only key
        ("GROUP_AGENT_MODEL", "GROUP_AGENT_BASE_URL"),  # missing key
        ("GROUP_AGENT_MODEL", "DASHSCOPE_API_KEY"),   # missing base_url
        ("GROUP_AGENT_BASE_URL", "DASHSCOPE_API_KEY"),  # missing model
    ],
)
def test_qwen_real_or_empty_mode_missing_config_fails_closed(monkeypatch, mode, present):
    """qwen + real (and empty≡real) must fail-closed before any network when
    any of model / base_url / key is missing (FIX3 §4)."""
    _base_mode_env(monkeypatch)
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", mode)
    values = {
        "GROUP_AGENT_MODEL": "qwen-turbo",
        "GROUP_AGENT_BASE_URL": "https://example.invalid/v1",
        "DASHSCOPE_API_KEY": "test-key-not-real",
    }
    for k in present:
        monkeypatch.setenv(k, values[k])
    # Any missing config among the three → RuntimeError (all `present` subsets
    # here are strict subsets, so at least one is missing).
    with pytest.raises(RuntimeError):
        create_model()


@pytest.mark.parametrize("mode", ["real", ""])
def test_qwen_real_or_empty_mode_complete_config_builds(monkeypatch, mode):
    """With all three present, real/empty mode constructs a real ChatOpenAI
    (no network call is made at construction time)."""
    _base_mode_env(monkeypatch)
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", mode)
    monkeypatch.setenv("GROUP_AGENT_MODEL", "qwen-turbo")
    monkeypatch.setenv("GROUP_AGENT_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-not-real")
    model = create_model()
    assert model.__class__.__name__ == "ChatOpenAI"


def test_stub_mode_still_builds_stub(monkeypatch):
    _base_mode_env(monkeypatch)
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    model = create_model()
    assert model.__class__.__name__ == "StubGroupAgentChatModel"

