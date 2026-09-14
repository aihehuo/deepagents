"""Request-local receipts of actual public query results."""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from functools import wraps
from threading import Lock
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


class _Results:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self.lock = Lock()


_results: ContextVar[_Results | None] = ContextVar("aihehuo_public_results", default=None)


def record_result(resource: str, content: str) -> None:
    """Record tool output only when an application query scope is active.

    Args:
        resource: Resource actually requested by the tool.
        content: Exact output returned to the model.
    """
    results = _results.get()
    if results is not None:
        with results.lock:
            results.items.append((resource, content))


def current_results() -> list[tuple[str, str]]:
    """Return a snapshot of the active request's actual tool outputs."""
    results = _results.get()
    if results is None:
        return []
    with results.lock:
        return list(results.items)


def collect_public_evidence(function: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
    """Isolate public receipts for one async request, including tool threads.

    Args:
        function: Request execution function to wrap.

    Returns:
        Signature-preserving wrapper that releases receipts on every exit.
    """

    @wraps(function)
    async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        token = _results.set(_Results())
        try:
            return await function(*args, **kwargs)
        finally:
            _results.reset(token)

    return wrapped
