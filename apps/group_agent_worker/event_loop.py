"""Process-scoped asyncio lifecycle for Celery deliveries."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")
_event_loop: asyncio.AbstractEventLoop | None = None
_event_loop_pid: int | None = None
_event_loop_lock = threading.Lock()


def persistent_worker_event_loop() -> asyncio.AbstractEventLoop:
    """Return one event loop for the lifetime of the current Celery child process.

    The worker runtime intentionally caches async HTTP/model clients. Creating a
    fresh loop with `asyncio.run` for every delivery closes the loop underneath
    those cached clients after the first task. A process-scoped loop keeps the
    clients and their connection pools on the loop that created them.
    """
    global _event_loop, _event_loop_pid  # noqa: PLW0603
    pid = os.getpid()
    with _event_loop_lock:
        if (
            _event_loop is None
            or _event_loop.is_closed()
            or _event_loop_pid != pid
        ):
            _event_loop = asyncio.new_event_loop()
            _event_loop_pid = pid
        asyncio.set_event_loop(_event_loop)
        return _event_loop


def run_on_worker_event_loop(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run one delivery on the process-scoped event loop."""
    return persistent_worker_event_loop().run_until_complete(coroutine)
