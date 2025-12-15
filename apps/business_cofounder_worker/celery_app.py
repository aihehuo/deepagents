"""Celery application for Business Co-Founder Worker.

Minimal implementation to test that worker threading works in production.
"""

from celery import Celery
import os

# Redis broker URL (can be overridden via environment)
REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Create Celery app
app = Celery(
    "business_cofounder_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,  # Use Redis as both broker and result backend
)

# Celery configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker configuration
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (memory management)
    # Task configuration
    task_acks_late=True,  # Acknowledge tasks after completion
    task_reject_on_worker_lost=True,
    # Include tasks module so tasks are registered
    include=["apps.business_cofounder_worker.tasks"],
)

# Import tasks to ensure they're registered
from apps.business_cofounder_worker import tasks  # noqa: F401, E402

