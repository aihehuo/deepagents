"""Broker DLQ inspect consumer for unmappable poison (REQ-032-FIX3)."""

from __future__ import annotations

import logging

from celery import shared_task

_logger = logging.getLogger(__name__)


@shared_task(name="group_agent.poison_inspect", ignore_result=True)
def poison_inspect(poison: dict) -> dict:
    """No-op consumer for broker DLQ inspect messages."""
    _logger.error(
        "poison_inspect poison_id=%s error_code=%s",
        (poison or {}).get("poison_id"),
        (poison or {}).get("error_code"),
    )
    return {"status": "logged"}
