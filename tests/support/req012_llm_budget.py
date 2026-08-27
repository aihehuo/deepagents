"""REQ-012 shared support: LLM budget guard, evidence recorder, and outcome
classification.

Importable with no network and no real-LLM credentials.
Used by the brain-as-SUT real-LLM conversation test.
"""

from __future__ import annotations

import copy
import enum
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

MAX_LLM_INVOCATIONS = 12


class LLMBudgetExceeded(RuntimeError):
    """Raised at the (MAX_LLM_INVOCATIONS+1)-th real LLM invocation, from the
    callback handler's on_chat_model_start — i.e. BEFORE the provider network
    request is issued."""


class GroupAgentIsolationError(RuntimeError):
    """Raised by a test HTTP-client isolation guard when a forbidden external
    client (membership / match / callback) is invoked during the real-LLM
    scenario. A dedicated type lets classification key on the type, not prose."""


class OutcomeKind(str, enum.Enum):
    """Explicit, mutually-exclusive classification of a scenario outcome.

    REQ-012-FIX2 §2: distinguish provider/network vs internal error vs budget
    vs isolation vs unknown/timeout, instead of a single BLOCKED_EXTERNAL bucket.
    """

    PASSED = "PASSED"
    PROVIDER_NETWORK = "BLOCKED_EXTERNAL:PROVIDER_NETWORK"
    BUDGET = "FAILED:BUDGET"
    ISOLATION = "FAILED:ISOLATION"
    INTERNAL = "FAILED:INTERNAL"
    AUDIT_REDACTION = "FAILED:HUMAN_AUDIT_REDACTION"
    TIMEOUT = "BLOCKED_EXTERNAL:UNKNOWN_TIMEOUT"

    @property
    def is_blocked_external(self) -> bool:
        return self in {OutcomeKind.PROVIDER_NETWORK, OutcomeKind.TIMEOUT}

    @property
    def is_failure(self) -> bool:
        return self in {
            OutcomeKind.BUDGET,
            OutcomeKind.ISOLATION,
            OutcomeKind.INTERNAL,
            OutcomeKind.AUDIT_REDACTION,
        }


# --- Type-name / status whitelists (REQ-012-FIX3 §1) ---
#
# Classification keys on the EXACT exception type name or a structured HTTP
# status — never a substring match against an arbitrary error message. A
# free-form message like "ValueError expected 500 members" must classify as
# INTERNAL, not PROVIDER_NETWORK, because "500" appearing in prose is not an
# HTTP status.

# Exact exception type names that mean "external provider / network fault".
# lowercased for case-insensitive exact comparison.
_PROVIDER_NETWORK_TYPES = frozenset(
    t.lower()
    for t in (
        # openai / langchain-openai provider errors
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "APIError",
        "APIStatusError",
        "Timeout",
        "APIResponseValidationError",
        # httpx / requests / stdlib transport errors
        "ConnectTimeout",
        "ReadTimeout",
        "ConnectError",
        "ConnectionError",
        "ConnectionResetError",
        "ReadError",
        "PoolTimeout",
        "TimeoutError",
        "socket.timeout",
        "ProxyError",
        "SSLError",
    )
)

# HTTP status codes that ALONE indicate an external provider fault (not a
# code bug). 500/502 are intentionally EXCLUDED: the chat endpoint wraps any
# internal exception as HTTP 502, so those statuses are ambiguous and require a
# whitelisted provider/network error_type to count as external.
_PROVIDER_NETWORK_STATUSES = frozenset({408, 429, 503, 504})
# Ambiguous statuses: external only if error_type is a known provider type.
_AMBIGUOUS_STATUSES = frozenset({500, 502})

# Our own control exception type names.
_BUDGET_TYPES = frozenset({"llmbudgetexceeded"})
_ISOLATION_TYPES = frozenset({"groupagentisolationerror"})
_AUDIT_REDACTION_TYPES = frozenset({"humanauditredactionerror"})


def classify_exception_type(type_name: str) -> OutcomeKind:
    """Classify by an EXACT exception type name (case-insensitive).

    Never matches on message content — an unknown/internal exception type
    (ValueError, KeyError, AssertionError, AttributeError, …) is INTERNAL.
    """
    t = (type_name or "").strip().lower()
    if t in _BUDGET_TYPES:
        return OutcomeKind.BUDGET
    if t in _ISOLATION_TYPES:
        return OutcomeKind.ISOLATION
    if t in _AUDIT_REDACTION_TYPES:
        return OutcomeKind.AUDIT_REDACTION
    if t in _PROVIDER_NETWORK_TYPES:
        return OutcomeKind.PROVIDER_NETWORK
    return OutcomeKind.INTERNAL


def classify_exception(exc: BaseException) -> OutcomeKind:
    """Classify a live exception, walking the __cause__ chain.

    Uses only the exception TYPE (and its causes' types) — never str(exc).
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        kind = classify_exception_type(type(cur).__name__)
        if kind is not OutcomeKind.INTERNAL:
            return kind
        cur = cur.__cause__ or cur.__context__
    return OutcomeKind.INTERNAL


# Backwards-compatible name kept for existing call sites: now delegates to the
# exact-type classifier and accepts a bare type name (NOT a message string).
def classify_error_text(type_name: str) -> OutcomeKind:
    """DEPRECATED alias — pass an EXACT exception type name, not a message.

    Retained so existing helper tests keep a single entry point; internally
    this is exact-type classification, so message prose can never leak a
    misclassification (REQ-012-FIX3 §1).
    """
    return classify_exception_type(type_name)


def classify_http_response(response: Any) -> tuple[OutcomeKind, str]:
    """Classify an HTTP response into (OutcomeKind, safe_diagnostic).

    Provider/network is decided by the STATUS CODE (429/5xx), and our own
    guard exceptions by their exact error_type. Anything else non-2xx is a
    code failure (INTERNAL). The diagnostic contains only status + type name,
    never the error_message body.
    """
    status = getattr(response, "status_code", None)
    if status == 200:
        return OutcomeKind.PASSED, "status=200"
    detail: Any = {}
    try:
        detail = response.json().get("detail", {})
    except Exception:  # noqa: BLE001
        return OutcomeKind.INTERNAL, f"status={status}"
    if not isinstance(detail, dict):
        return OutcomeKind.INTERNAL, f"status={status}"

    error_type = str(detail.get("error_type", "") or "")

    # 1) Our own control exceptions surface via error_type regardless of status.
    typed = classify_exception_type(error_type)
    if typed in (OutcomeKind.BUDGET, OutcomeKind.ISOLATION):
        return typed, f"status={status} type={error_type}"

    # 2) Clean provider statuses (408/429/503/504) are external by status alone.
    if status in _PROVIDER_NETWORK_STATUSES:
        return OutcomeKind.PROVIDER_NETWORK, f"status={status} type={error_type}"

    # 3) Ambiguous statuses (500/502 — chat wraps internal errors as 502) are
    #    external ONLY when the error_type is a whitelisted provider/network
    #    type; otherwise they are internal code failures.
    if status in _AMBIGUOUS_STATUSES:
        if typed is OutcomeKind.PROVIDER_NETWORK:
            return OutcomeKind.PROVIDER_NETWORK, f"status={status} type={error_type}"
        return OutcomeKind.INTERNAL, f"status={status} type={error_type}"

    # 4) Everything else non-2xx is an internal code failure.
    return OutcomeKind.INTERNAL, f"status={status} type={error_type}"


def _usage_from_message(msg: Any) -> tuple[int, int, int]:
    """Return (input_tokens, output_tokens, total_tokens) from a LangChain message."""
    um = getattr(msg, "usage_metadata", None) or {}
    in_t = int(um.get("input_tokens", 0) or 0)
    out_t = int(um.get("output_tokens", 0) or 0)
    tot_t = int(um.get("total_tokens", 0) or 0)
    if not tot_t and not (in_t or out_t):
        meta = getattr(msg, "response_metadata", None) or {}
        tu = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(tu, dict):
            in_t = int(tu.get("prompt_tokens") or tu.get("input_tokens") or 0)
            out_t = int(tu.get("completion_tokens") or tu.get("output_tokens") or 0)
            tot_t = int(tu.get("total_tokens") or 0)
    if not tot_t:
        tot_t = in_t + out_t
    return in_t, out_t, tot_t


class LLMBudgetRecorder(BaseCallbackHandler):
    """Non-leaking LangChain callback handler.

    Records LLM call counts, tool-call names, and token usage numbers, and
    HARD-BLOCKS the (MAX_LLM_INVOCATIONS+1)-th invocation before the network.

    ``raise_error = True`` is REQUIRED: without it LangChain's CallbackManager
    swallows handler exceptions (default raise_error=False), so the budget
    guard would silently no-op. This is the core of REQ-012-FIX2 §1.
    """

    # Class attribute read by CallbackManager to decide whether to re-raise.
    raise_error = True

    def __init__(self) -> None:
        super().__init__()
        self.llm_starts = 0
        self.llm_ends = 0
        self.tool_call_counts: dict[str, int] = {}
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_class: str | None = None

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:  # noqa: ARG002
        self.llm_starts += 1
        if self.llm_starts > MAX_LLM_INVOCATIONS:
            raise LLMBudgetExceeded(
                f"LLM_BUDGET_EXCEEDED: invocation #{self.llm_starts} blocked "
                f"(max {MAX_LLM_INVOCATIONS})"
            )
        if self.model_class is None:
            ids = serialized.get("id", []) if isinstance(serialized, dict) else []
            self.model_class = ids[-1] if ids else "unknown"

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.llm_ends += 1
        try:
            for gen_list in getattr(response, "generations", []) or []:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is None:
                        continue
                    for tc in getattr(msg, "tool_calls", None) or []:
                        name = (
                            tc.get("name")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", None)
                        )
                        name = name or "unknown"
                        self.tool_call_counts[name] = self.tool_call_counts.get(name, 0) + 1
                    in_t, out_t, tot_t = _usage_from_message(msg)
                    self.input_tokens += in_t
                    self.output_tokens += out_t
                    self.total_tokens += tot_t or (in_t + out_t)
        except Exception:  # noqa: BLE001 — evidence must never break the test
            pass

    @property
    def llm_calls(self) -> int:
        return self.llm_ends

    def snapshot(self) -> dict[str, Any]:
        return {
            "llm_starts": self.llm_starts,
            "llm_ends": self.llm_ends,
            "tool_calls": copy.deepcopy(self.tool_call_counts),
            "tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @staticmethod
    def delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        tb = before.get("tool_calls", {})
        ta = after.get("tool_calls", {})
        tool_delta: dict[str, int] = {}
        for k in set(tb) | set(ta):
            diff = ta.get(k, 0) - tb.get(k, 0)
            if diff:
                tool_delta[k] = diff
        out: dict[str, Any] = {
            "llm_starts": after["llm_starts"] - before["llm_starts"],
            "llm_ends": after["llm_ends"] - before["llm_ends"],
            "tool_calls": tool_delta,
            "tokens": after["tokens"] - before["tokens"],
        }
        if "input_tokens" in after or "input_tokens" in before:
            out["input_tokens"] = int(after.get("input_tokens") or 0) - int(
                before.get("input_tokens") or 0
            )
            out["output_tokens"] = int(after.get("output_tokens") or 0) - int(
                before.get("output_tokens") or 0
            )
        return out

    def attach(self, model: Any) -> Any:
        self.model_class = model.__class__.__name__
        existing = list(getattr(model, "callbacks", None) or [])
        existing.append(self)
        model.callbacks = existing
        return model


# =====================================================================
# Runner outcome/exit allow-matrix (REQ-012-FIX3 §2)
# =====================================================================

# The exact set of outcome labels the test is allowed to emit.
_VALID_OUTCOME_LABELS = frozenset(k.value for k in OutcomeKind)
_MAX_OUTCOME_LEN = 200


def decide_runner_result(exit_code: int, outcome_text: str | None) -> str:
    """Reconcile pytest exit code with the outcome line the test wrote.

    Returns the authoritative single-line verdict string (an OutcomeKind
    value). Enforces (REQ-012-FIX3 §2):

    - exit == 0  → ONLY 'PASSED' is accepted; anything else → FAILED:INTERNAL.
    - exit != 0  → 'PASSED' is FORBIDDEN; only a supported FAILED:* /
      BLOCKED_EXTERNAL:* label is accepted; else → FAILED:INTERNAL.
    - unknown / empty / multiline / oversized outcome → FAILED:INTERNAL.

    The first token of the label is validated against the exact enum values so
    a residual, truncated, or injected outcome file cannot fake a pass.
    """
    internal = OutcomeKind.INTERNAL.value

    raw = outcome_text if outcome_text is not None else ""
    # Reject multiline / oversized payloads outright.
    if "\n" in raw.strip("\n"):
        return internal
    if len(raw) > _MAX_OUTCOME_LEN:
        return internal
    label = raw.strip()
    # The label may carry a trailing " <diagnostic>"; take the first token as
    # the canonical OutcomeKind value (which itself contains no spaces).
    canonical = label.split(" ", 1)[0] if label else ""

    if canonical not in _VALID_OUTCOME_LABELS:
        return internal

    passed = OutcomeKind.PASSED.value
    if exit_code == 0:
        # Success exit must be PASSED and nothing else.
        return passed if canonical == passed else internal

    # Non-zero exit: PASSED is impossible; must be a real failure/blocked label.
    if canonical == passed:
        return internal
    return label  # keep the diagnostic tail for supported non-pass labels
