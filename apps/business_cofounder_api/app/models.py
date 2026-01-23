"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Upstream server-provided user id")
    message: str = Field(..., description="User message")
    conversation_id: str = Field("default", description="Conversation id (defaults to 'default')")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata from upstream")
    expertise_type: str = Field("business_cofounder", description="Type of expertise to use for expert analysis (default: 'business_cofounder')")


class ChatResponse(BaseModel):
    user_id: str
    conversation_id: str
    thread_id: str
    reply: str


class KanbanRequest(BaseModel):
    user_id: str
    conversation_id: str = "default"


class CanvasResponse(BaseModel):
    """Response containing canvas data and expert guidance.
    
    The canvas is a domain-agnostic JSON structure defined by the expert's prompt.
    The backend treats it as an opaque blob - the frontend is responsible for
    interpreting and displaying it based on the expertise type.
    """
    
    user_id: str
    conversation_id: str
    thread_id: str
    canvas: dict[str, Any] | None = None
    expert_guidance: str | None = None
    current_round: int = 0
    last_sync_round: int = 0
    analysis_timestamp: str | None = None


class ResetRequest(BaseModel):
    user_id: str
    conversation_id: str = "default"


class ResetResponse(BaseModel):
    user_id: str
    conversation_id: str
    thread_id: str
    ok: bool


class CallDeepAgentAsyncRequest(BaseModel):
    user_id: str = Field(..., description="Upstream server-provided user id (accepts int or str, coerced to str)")
    message: str = Field(..., description="User message")
    conversation_id: str = Field("default", description="Conversation id (accepts int or str, coerced to str, defaults to 'default')")
    callback: str | None = Field(None, description="Callback URL to receive update messages (alias: callback_url)")
    callback_url: str | None = Field(None, alias="callback_url", description="Callback URL to receive update messages")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata from upstream")
    expertise_type: str = Field("business_cofounder", description="Type of expertise to use for expert analysis (default: 'business_cofounder')")
    
    @field_validator("user_id", mode="before")
    @classmethod
    def coerce_user_id_to_str(cls, v: Any) -> str:
        """Coerce user_id to string."""
        return str(v)
    
    @field_validator("conversation_id", mode="before")
    @classmethod
    def coerce_conversation_id_to_str(cls, v: Any) -> str:
        """Coerce conversation_id to string."""
        if v is None:
            return "default"
        return str(v)
    
    def model_post_init(self, __context: Any) -> None:
        """Ensure callback is set from callback_url if needed."""
        if not self.callback and self.callback_url:
            self.callback = self.callback_url
        if not self.callback:
            raise ValueError("Either 'callback' or 'callback_url' must be provided")


class CallDeepAgentAsyncResponse(BaseModel):
    success: bool
    session_id: str = Field(..., description="Session ID (same as thread_id)")
    message: str = Field(..., description="Success or error message")
