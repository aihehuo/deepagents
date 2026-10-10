"""Independent manager queue, worker pool and scheduler; no customer task imports."""
import os
from celery import Celery

QUEUE = "community_manager_v1"
app = Celery("community_manager", broker=os.environ.get("COMMUNITY_MANAGER_REDIS_URL", "redis://localhost:6379/6"), include=["apps.community_manager.worker"])
app.conf.update(task_default_queue=QUEUE, task_routes={"community_manager.*": {"queue": QUEUE}},
    worker_prefetch_multiplier=1, task_acks_late=True, task_reject_on_worker_lost=False,
    task_ignore_result=True, broker_transport_options={"global_keyprefix": "community_manager:v1:", "visibility_timeout": 600},
    task_soft_time_limit=220, task_time_limit=240, task_serializer="json", accept_content=["json"],
    beat_schedule={"manager-check": {"task": "community_manager.tick", "schedule": 300.0, "options": {"expires": 300}}},
    broker_connection_retry_on_startup=True, worker_max_tasks_per_child=100)
