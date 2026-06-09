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


class ResetRequest(BaseModel):
    user_id: str
    conversation_id: str = "default"
    agent_name: str = Field("", description="Agent profile name")


class ResetResponse(BaseModel):
    user_id: str
    conversation_id: str
    thread_id: str
    ok: bool
