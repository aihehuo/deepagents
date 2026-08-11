"""Celery app for wechat_greeter_worker (REQ-050).

模式：仿 apps/group_agent_worker/celery_app.py + apps/business_cofounder_worker/celery_app.py。
生产用 Redis broker；A 阶段冒烟用 CELERY_TASK_ALWAYS_EAGER=1 同步跑（同一进程内）。
"""

from __future__ import annotations

import os

from celery import Celery

# Broker URL. Test/冒烟用 memory:// 或 redis://localhost:6379/0（不连也行，eager mode 不真连）。
REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

app = Celery(
    "wechat_greeter_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Detect test/eager mode at import time (must be set before .delay() is called).
# In production, CELERY_TASK_ALWAYS_EAGER is unset/false → tasks go to Redis broker.
# In tests, monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "1") must run BEFORE this
# module is imported (e.g. inside the test function, before `from apps.wechat_greeter_api.main import app`).
_eager_mode = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "").strip().lower() in ("1", "true", "yes", "on")

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_ignore_result=True,
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_queue="wechat_greeter",
    task_always_eager=_eager_mode,
    task_eager_propagates=_eager_mode,
    include=["apps.wechat_greeter_worker.tasks"],
)

# Make this the default app for @shared_task registration
app.set_default()

# Import tasks to ensure they're registered
from apps.wechat_greeter_worker import tasks  # noqa: E402, F401
