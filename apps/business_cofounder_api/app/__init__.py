"""Main FastAPI application assembly."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from apps.business_cofounder_api.app.endpoints import canvas, chat, deep_agent, health, reset, simulated_user, state
from apps.business_cofounder_api.app.models import (
    CallDeepAgentAsyncRequest,
    CallDeepAgentAsyncResponse,
    CanvasResponse,
    ChatRequest,
    ChatResponse,
    KanbanRequest,
    ResetRequest,
    ResetResponse,
    SimulatedUserChatRequest,
    SimulatedUserChatResponse,
)
from apps.business_cofounder_api.app.startup import startup
from apps.business_cofounder_api.app.state import AppState

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")

# Global state (for backward compatibility)
_state: AppState | None = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app instance
    """
    app = FastAPI(title="Business Co-Founder Agent API", version="0.1.0")
    
    # Exception handler for validation errors
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Any:
        """Log validation errors for debugging."""
        body_bytes = await request.body()
        body_str = body_bytes.decode("utf-8", errors="ignore")[:1000]  # Truncate long bodies
        _logger.error(
            "=== VALIDATION ERROR === %s %s",
            request.method,
            request.url.path,
        )
        _logger.error("Validation errors: %s", exc.errors())
        _logger.error("Request body (first 1000 chars): %s", body_str)
        _logger.error("Request headers: %s", dict(request.headers))
        # Return the default FastAPI validation error response
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "body": body_str[:500]},
        )
    
    # Startup handler
    @app.on_event("startup")
    async def _startup() -> None:
        global _state
        state_ref: dict[str, Any] = {}
        await startup(state_ref)
        _state = state_ref.get("state")
    
    # Register endpoints
    # Health endpoint
    @app.get("/health")
    async def health_endpoint() -> dict[str, str]:
        assert _state is not None
        return await health.health(_state)
    
    # Canvas endpoint
    @app.post("/canvas", response_model=CanvasResponse)
    async def canvas_endpoint(req: KanbanRequest) -> CanvasResponse:
        assert _state is not None
        return await canvas.get_canvas(req, _state)
    
    # State endpoint
    @app.get("/state")
    async def state_endpoint(user_id: str, conversation_id: str = "default") -> dict[str, Any]:
        assert _state is not None
        return await state.get_state(user_id, conversation_id, _state)
    
    # Chat endpoints
    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(req: ChatRequest) -> ChatResponse:
        assert _state is not None
        return await chat.chat(req, _state)
    
    @app.post("/chat/stream")
    async def chat_stream_endpoint(req: ChatRequest) -> StreamingResponse:
        assert _state is not None
        return await chat.chat_stream(req, _state)
    
    # Reset endpoint
    @app.post("/reset", response_model=ResetResponse)
    async def reset_endpoint(req: ResetRequest) -> ResetResponse:
        assert _state is not None
        return await reset.reset(req, _state)
    
    # Deep agent async endpoint
    @app.post("/deep_agent/call_async", response_model=CallDeepAgentAsyncResponse)
    async def deep_agent_async_endpoint(req: CallDeepAgentAsyncRequest) -> CallDeepAgentAsyncResponse:
        assert _state is not None
        return await deep_agent.call_deep_agent_async(req, _state)
    
    # Simulated user agent endpoint
    @app.post("/simulated_user/chat", response_model=SimulatedUserChatResponse)
    async def simulated_user_chat_endpoint(req: SimulatedUserChatRequest) -> SimulatedUserChatResponse:
        assert _state is not None
        return await simulated_user.simulated_user_chat(req, _state)
    
    return app


# Create the app instance
app = create_app()

# Export for backward compatibility
__all__ = ["app", "_state"]
