"""Celery application for Group Agent durable queue worker (REQ-032)."""

from __future__ import annotations

import os
from typing import Any

from celery import Celery

from apps.group_agent_api.execution.config import DurableQueueConfig, load_durable_queue_config

_APP: Celery | None = None


def build_celery_app(config: DurableQueueConfig, *, for_worker: bool = False) -> Celery:
    """Build a JSON-only Celery app for the given durable config."""
    app = Celery(
        "group_agent_worker",
        broker=config.redis_url,
    )
    conf: dict[str, Any] = {
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "task_ignore_result": True,
        "timezone": "UTC",
        "enable_utc": True,
        "worker_prefetch_multiplier": 1,
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
        "broker_connection_retry_on_startup": True,
        "task_soft_time_limit": int(config.soft_time_limit_s),
        "task_time_limit": int(config.hard_time_limit_s),
        "broker_transport_options": {
            "visibility_timeout": int(config.visibility_timeout_s),
        },
        "task_default_queue": config.celery_queue,
    }
    if for_worker:
        # Worker process loads task modules; API admission clients must not.
        conf["include"] = [
            "apps.group_agent_worker.tasks",
            "apps.group_agent_worker.recovery",
            "apps.group_agent_worker.poison",
        ]
        # FIX2: beat schedule only when this process is the dedicated beat/worker-with-beat.
        # Default worker replicas must set GROUP_AGENT_WORKER_BEAT=0 to avoid N× recovery.
        enable_beat = os.environ.get("GROUP_AGENT_WORKER_BEAT", "0").strip() == "1"
        if enable_beat:
            conf["beat_schedule"] = {
                "group-agent-recovery-tick": {
                    "task": "group_agent.recovery_tick",
                    "schedule": float(
                        os.environ.get("GROUP_AGENT_RECOVERY_INTERVAL_S", "15")
                    ),
                },
            }
    app.conf.update(conf)
    if app.conf.task_serializer != "json" or list(app.conf.accept_content) != ["json"]:
        raise RuntimeError("celery serializer must be json only")
    return app


def get_celery_app(
    config: DurableQueueConfig | None = None,
    *,
    for_worker: bool = False,
) -> Celery:
    """Return Celery app for admission client or worker process."""
    global _APP
    if config is not None and not for_worker:
        return build_celery_app(config, for_worker=False)
    if _APP is None:
        os.environ.setdefault("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
        loaded = config or load_durable_queue_config(require_enabled=True)
        assert loaded is not None
        _APP = build_celery_app(loaded, for_worker=True)
        # Ensure shared_task binds to this app.
        _APP.set_default()
    return _APP


class _AppProxy:
    """Defer config load until Celery CLI actually needs the app."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_celery_app(for_worker=True), name)

    def __dir__(self) -> list[str]:
        return dir(get_celery_app(for_worker=True))


# `celery -A apps.group_agent_worker.celery_app worker`
app = _AppProxy()
