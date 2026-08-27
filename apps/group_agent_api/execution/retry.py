"""Retry taxonomy and delay helpers (REQ-032)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from apps.group_agent_api.execution.models import RetryClass


# Stable error codes → retry class
_PERMANENT_CODES = frozenset(
    {
        "schema_invalid",
        "fingerprint_conflict",
        "binding_conflict",
        "idempotency_conflict",
        "run_binding_conflict",
        "payload_decrypt_failed",
        "security_guard_failed",
        "principal_rejected",
        "callback_url_rejected",
        "mouth_ingress_rejected",
    }
)
_POISON_CODES = frozenset(
    {
        "poison_payload",
        "unsupported_queue_schema",
        "deserialize_failed",
    }
)
_TRANSIENT_CODES = frozenset(
    {
        "llm_rate_limited",
        "llm_unavailable",
        "network_timeout",
        "redis_transient",
        "celery_transient",
        "callback_5xx",
        "provider_5xx",
        "queue_saturated_retry",
    }
)
_UNCERTAIN_CODES = frozenset(
    {
        "provider_timeout",
        "uncertain_completion",
        "callback_uncertain",
    }
)


def classify_error(error_code: str) -> RetryClass:
    """Map a stable error code to retry taxonomy."""
    code = (error_code or "").strip()
    if code in _POISON_CODES:
        return RetryClass.POISON
    if code in _PERMANENT_CODES:
        return RetryClass.PERMANENT
    if code in _UNCERTAIN_CODES:
        return RetryClass.UNCERTAIN
    if code in _TRANSIENT_CODES or code.startswith("transient_"):
        return RetryClass.TRANSIENT
    if code in {"worker_lost", "lease_expired"}:
        return RetryClass.WORKER_LOST
    # Unknown → treat as transient but still subject to budget.
    return RetryClass.TRANSIENT


@dataclass(frozen=True)
class RetryDecision:
    """Whether to retry and when."""

    should_retry: bool
    retry_class: RetryClass
    delay_s: float
    reason_code: str
    dead_letter: bool = False


def compute_retry_delay(
    *,
    attempt_count: int,
    base_s: float,
    max_s: float,
    full_jitter: bool = True,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with optional full jitter.

    delay = random_between(0, min(max_s, base_s * 2^(attempt-1)))
    when full_jitter is True; otherwise capped exponential without jitter.
    """
    exp = min(max_s, base_s * (2 ** max(0, attempt_count - 1)))
    if not full_jitter:
        return float(exp)
    gen = rng or random.Random()
    return float(gen.uniform(0.0, exp))


def decide_retry(
    *,
    error_code: str,
    attempt_count: int,
    max_attempts: int,
    base_s: float,
    max_s: float,
    full_jitter: bool = True,
    rng: random.Random | None = None,
) -> RetryDecision:
    """Decide retry / DLQ given taxonomy and attempt budget."""
    retry_class = classify_error(error_code)
    if retry_class in {RetryClass.PERMANENT, RetryClass.POISON}:
        return RetryDecision(
            should_retry=False,
            retry_class=retry_class,
            delay_s=0.0,
            reason_code=error_code or retry_class.value,
            dead_letter=True,
        )
    if attempt_count >= max_attempts:
        return RetryDecision(
            should_retry=False,
            retry_class=retry_class,
            delay_s=0.0,
            reason_code="max_attempts_exceeded",
            dead_letter=True,
        )
    delay = compute_retry_delay(
        attempt_count=attempt_count,
        base_s=base_s,
        max_s=max_s,
        full_jitter=full_jitter,
        rng=rng,
    )
    return RetryDecision(
        should_retry=True,
        retry_class=retry_class,
        delay_s=delay,
        reason_code=error_code or retry_class.value,
        dead_letter=False,
    )
