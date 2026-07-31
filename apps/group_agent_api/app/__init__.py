"""group_agent_api FastAPI application (UC-34 REQ-004/005/006/007/009)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from apps.group_agent_api.app.endpoints import call_async, chat, health, invite, match, profile, reset
from apps.group_agent_api.app.models import (
    AsyncCallRequest,
    AsyncCallResponse,
    ChatRequest,
    ChatResponse,
    InviteRequest,
    InviteResponse,
    MatchRequest,
    MatchResponse,
    ProfileQueryResponse,
    ResetRequest,
    ResetResponse,
)
from apps.group_agent_api.app.startup import startup
from apps.group_agent_api.app.state import AppState

_state: AppState | None = None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global _state
    state_ref: dict[str, AppState | None] = {}
    await startup(state_ref)
    _state = state_ref.get("state")
    try:
        yield
    finally:
        if _state is not None:
            await _state.shutdown()


def create_app() -> FastAPI:
    app_inst = FastAPI(
        title="Group Agent API",
        version="0.4.1",
        description=(
            "UC-34 group agent · HMAC-signed principal (REQ-007 FIX3) + "
            "Micro WebSocket async callback integration (REQ-009 / RESP-009-FIX)"
        ),
        lifespan=lifespan,
    )

    @app_inst.get("/health")
    async def health_endpoint() -> dict[str, str]:
        assert _state is not None
        return await health.health(_state)

    @app_inst.get("/ready")
    async def ready_endpoint(response: Response) -> dict[str, object]:
        assert _state is not None
        return await health.readiness(_state, response)

    @app_inst.post("/call_async", response_model=AsyncCallResponse, status_code=202)
    async def call_async_endpoint(req: AsyncCallRequest, request: Request) -> AsyncCallResponse:
        assert _state is not None
        return await call_async.call_async(req, _state, request)

    @app_inst.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(req: ChatRequest, request: Request) -> ChatResponse:
        assert _state is not None
        return await chat.chat(req, _state, request)

    @app_inst.post("/match", response_model=MatchResponse)
    async def match_endpoint(req: MatchRequest, request: Request) -> MatchResponse:
        assert _state is not None
        return await match.match(req, _state, request)

    @app_inst.post("/invite", response_model=InviteResponse)
    async def invite_endpoint(req: InviteRequest, request: Request) -> InviteResponse:
        assert _state is not None
        return await invite.invite(req, _state, request)

    @app_inst.get("/profile", response_model=ProfileQueryResponse)
    async def profile_endpoint(
        request: Request,
        user_id: str,
        group_id: str,
        membership: str = "unknown",
    ) -> ProfileQueryResponse:
        assert _state is not None
        return await profile.get_profile(
            user_id=user_id,
            group_id=group_id,
            state=_state,
            request=request,
            membership=membership,
        )

    @app_inst.post("/reset", response_model=ResetResponse)
    async def reset_endpoint(req: ResetRequest, request: Request) -> ResetResponse:
        assert _state is not None
        return await reset.reset(req, _state, request)

    return app_inst


app = create_app()

__all__ = ["app", "_state"]
