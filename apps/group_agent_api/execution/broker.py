"""Celery broker enqueue helper shared by API admission and recovery (REQ-032)."""

from __future__ import annotations

import logging
from typing import Any

from apps.group_agent_api.execution.config import DurableQueueConfig
from apps.group_agent_api.execution.models import BrokerDeliveryRef

_logger = logging.getLogger("uvicorn.error")

_TASK_NAME = "group_agent.process_run"


def enqueue_delivery(config: DurableQueueConfig, delivery: BrokerDeliveryRef) -> None:
    """Publish minimal delivery ref to Celery broker (JSON only, no secrets).

    Raises:
        Exception: Propagated when broker does not accept the message.
    """
    # Lazy import so legacy mode never requires Celery at import time.
    from apps.group_agent_worker.celery_app import get_celery_app

    app = get_celery_app(config)
    # send_task returns AsyncResult; failure to connect raises.
    result = app.send_task(
        _TASK_NAME,
        args=[],
        kwargs={"delivery": delivery.to_dict()},
        queue=config.celery_queue,
        ignore_result=True,
    )
    # Touch .id to ensure broker handoff completed for Redis transport.
    _ = result.id
    _logger.info(
        "broker_enqueued run_id=%s delivery_id=%s",
        delivery.run_id,
        delivery.delivery_id,
    )


def broker_ready(config: DurableQueueConfig) -> dict[str, Any]:
    """Readiness probe for broker connectivity (no secrets in return value)."""
    try:
        from apps.group_agent_worker.celery_app import get_celery_app

        app = get_celery_app(config)
        conn = app.connection()
        conn.ensure_connection(max_retries=1)
        conn.release()
        return {"broker": "ok", "queue": config.celery_queue}
    except Exception:  # noqa: BLE001
        return {"broker": "unavailable", "queue": config.celery_queue}
