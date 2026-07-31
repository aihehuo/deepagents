"""Periodic recovery task for group_agent_worker (REQ-032-FIX2)."""

from __future__ import annotations

import logging

from celery import shared_task

from apps.group_agent_api.execution.broker import enqueue_delivery
from apps.group_agent_api.execution.recovery import run_recovery_once
from apps.group_agent_worker.runtime import get_worker_runtime

_logger = logging.getLogger(__name__)


@shared_task(name="group_agent.recovery_tick", ignore_result=True)
def recovery_tick() -> dict[str, int]:
    """Recovery pass with per-run delivery claim + quota reconcile (FIX2)."""
    rt = get_worker_runtime()
    store = rt["store"]
    cfg = rt["config"]
    bp = rt["backpressure"]

    def _enqueue(delivery):  # type: ignore[no-untyped-def]
        enqueue_delivery(cfg, delivery)

    report = run_recovery_once(store, cfg, _enqueue, backpressure=bp)
    _logger.info(
        "recovery_tick accepted=%d enqueue_failed=%d lease_expired=%d retry=%d busy=%d errors=%d",
        report.accepted_requeued,
        report.enqueue_failed_requeued,
        report.lease_expired,
        report.retry_promoted,
        report.skipped_busy,
        report.errors,
    )
    return {
        "accepted_requeued": report.accepted_requeued,
        "enqueue_failed_requeued": report.enqueue_failed_requeued,
        "lease_expired": report.lease_expired,
        "retry_promoted": report.retry_promoted,
        "skipped_busy": report.skipped_busy,
        "errors": report.errors,
    }
