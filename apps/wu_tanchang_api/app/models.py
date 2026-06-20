"""Pydantic models for Wu Tanchang API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Upstream user id")
    message: str = Field(..., description="User message")
    conversation_id: str = Field("default", description="Conversation id")
    agent_name: str = Field("", description="Agent profile name; empty = default")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    user_id: str
    conversation_id: str
    thread_id: str
    reply: str


class CallWuTanchangAsyncRequest(BaseModel):
    user_id: str = Field(..., description="Upstream user id")
    message: str = Field(..., description="User message")
    conversation_id: str = Field("default", description="Conversation id")
    agent_name: str = Field("", description="Agent profile name; empty = default")
    callback: str | None = Field(
        None, description="Callback URL to receive stream events"
    )
    callback_url: str | None = Field(
        None, alias="callback_url", description="Callback URL to receive stream events"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Accept both callback and callback_url for parity with other agent APIs."""
        if not self.callback and self.callback_url:
            self.callback = self.callback_url
        if not self.callback:
            raise ValueError("Either 'callback' or 'callback_url' must be provided")


class CallWuTanchangAsyncResponse(BaseModel):
    success: bool
    session_id: str
    message: str


class ResetRequest(BaseModel):
    user_id: str
    conversation_id: str = "default"
    agent_name: str = Field("", description="Agent profile name")


class ResetResponse(BaseModel):
    user_id: str
    conversation_id: str
    thread_id: str
    ok: bool
