"""Regression tests for the Celery worker's async runtime lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from apps.group_agent_worker import event_loop as worker_event_loop


def test_worker_reuses_event_loop_for_cached_async_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequential deliveries must keep cached clients on their original loop."""
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(worker_event_loop, "_event_loop", loop)
    monkeypatch.setattr(
        worker_event_loop, "_event_loop_pid", worker_event_loop.os.getpid()
    )

    bound_loop = None

    async def cached_client_call() -> asyncio.AbstractEventLoop:
        nonlocal bound_loop
        running = asyncio.get_running_loop()
        if bound_loop is None:
            bound_loop = running
        if running is not bound_loop:
            raise RuntimeError("cached client moved to another event loop")
        return running

    try:
        first = worker_event_loop.run_on_worker_event_loop(cached_client_call())
        second = worker_event_loop.run_on_worker_event_loop(cached_client_call())
    finally:
        loop.close()

    assert first is loop
    assert second is loop
    assert first is second


def test_worker_replaces_inherited_event_loop_after_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forked Celery child must not use a loop created by its parent PID."""
    inherited = asyncio.new_event_loop()
    monkeypatch.setattr(worker_event_loop, "_event_loop", inherited)
    monkeypatch.setattr(
        worker_event_loop, "_event_loop_pid", worker_event_loop.os.getpid() - 1
    )

    child = worker_event_loop.persistent_worker_event_loop()
    try:
        assert child is not inherited
        assert not child.is_closed()
    finally:
        child.close()
        inherited.close()
