"""Wu Tanchang API FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from apps.wu_tanchang_api.app.endpoints import async_chat, chat, health, reset
from apps.wu_tanchang_api.app.models import (
    CallWuTanchangAsyncRequest,
    CallWuTanchangAsyncResponse,
    ChatRequest,
    ChatResponse,
    ResetRequest,
    ResetResponse,
)
from apps.wu_tanchang_api.app.startup import startup
from apps.wu_tanchang_api.app.state import AppState

_state: AppState | None = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(title="Wu Tanchang API", version="0.2.0")

    @app.on_event("startup")
    async def _startup() -> None:
        global _state
        state_ref: dict[str, AppState | None] = {}
        await startup(state_ref)
        _state = state_ref.get("state")

    @app.get("/health")
    async def health_endpoint() -> dict[str, str]:
        assert _state is not None
        return await health.health(_state)

    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(req: ChatRequest) -> ChatResponse:
        assert _state is not None
        return await chat.chat(req, _state)

    @app.post("/chat/stream")
    async def chat_stream_endpoint(req: ChatRequest) -> StreamingResponse:
        assert _state is not None
        return await chat.chat_stream(req, _state)

    @app.post("/call_async", response_model=CallWuTanchangAsyncResponse)
    async def call_async_endpoint(
        req: CallWuTanchangAsyncRequest,
    ) -> CallWuTanchangAsyncResponse:
        assert _state is not None
        return await async_chat.call_async(req, _state)

    @app.post("/reset", response_model=ResetResponse)
    async def reset_endpoint(req: ResetRequest) -> ResetResponse:
        assert _state is not None
        return await reset.reset(req, _state)

    return app


app = create_app()

__all__ = ["app", "_state"]
