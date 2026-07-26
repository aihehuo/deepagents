"""Pydantic models for group_agent_api."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MembershipSignal = Literal["in_group", "not_in_group", "unknown"]
MatchStatus = Literal["matched", "weak", "empty", "skipped"]
DeliveryKind = Literal["directed", "undirected"]


class ChatRequest(BaseModel):
    user_id: str = Field(
        ...,
        description="Stub: body principal. HTTP: must match X-GA-User-Id (or omit conflict)",
    )
    group_id: str = Field(
        ...,
        description=(
            "Stub: path isolation id. HTTP: must match membership.event_id "
            "(trusted group id from micro)"
        ),
    )
    message: str = Field(..., description="User message")
    conversation_id: str = Field("default", description="Conversation id")
    membership: MembershipSignal = Field(
        "unknown",
        description="Stub-only membership; ignored when GROUP_AGENT_INTEGRATION=http",
    )
    unionid: str | None = Field(
        None,
        description="Stub only / must match X-GA-Unionid in HTTP mode",
    )
    group_token: str | None = Field(
        None, description="GroupAgent JWT (g); required for HTTP match/membership"
    )
    user_token: str | None = Field(
        None,
        description="Stub/HTTP: must match X-GA-User-Token when set; else AIHEHUO_API_KEY",
    )
    run_match: bool = Field(True)
    willing_to_at: bool = Field(True)
    run_invite: bool = Field(True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    user_id: str
    group_id: str
    conversation_id: str
    thread_id: str
    reply: str
    profile_persisted: bool = False
    profile_path: str | None = None
    assert_attempts: int = 0
    persist_alert: str | None = None
    capability: MembershipSignal = "unknown"
    capability_source: str | None = None
    match_status: MatchStatus = "skipped"
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    match_reason: str | None = None
    guard_blocked: bool = False
    guard_violations: list[str] = Field(default_factory=list)
    delivery_kind: DeliveryKind | None = None
    invite_text: str | None = None
    topic: str | None = None
    mentioned_user_ids: list[str] = Field(default_factory=list)
    invite_ok: bool | None = None
    invite_violations: list[str] = Field(default_factory=list)
    willing_to_at: bool = True


class MatchRequest(BaseModel):
    user_id: str
    group_id: str
    membership: MembershipSignal = "unknown"
    unionid: str | None = None
    group_token: str | None = None
    user_token: str | None = None
    query: str | None = None
    excluded_ids: list[str] = Field(default_factory=list)


class MatchResponse(BaseModel):
    user_id: str
    group_id: str
    capability: MembershipSignal
    capability_source: str | None = None
    match_status: MatchStatus
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    match_reason: str | None = None
    query: str = ""
    guard_blocked: bool = False
    guard_violations: list[str] = Field(default_factory=list)


class InviteRequest(BaseModel):
    user_id: str
    group_id: str
    membership: MembershipSignal = "in_group"
    unionid: str | None = None
    group_token: str | None = None
    user_token: str | None = None
    willing_to_at: bool = True
    query: str | None = None
    match_status: MatchStatus | None = None
    candidates: list[dict[str, Any]] | None = None
    use_llm_polish: bool | None = None


class InviteResponse(BaseModel):
    user_id: str
    group_id: str
    capability: MembershipSignal
    capability_source: str | None = None
    match_status: MatchStatus
    match_reason: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    delivery_kind: DeliveryKind
    invite_text: str
    topic: str
    mentioned_user_ids: list[str] = Field(default_factory=list)
    elements: dict[str, str] | None = None
    honest_note: str | None = None
    invite_ok: bool
    invite_violations: list[str] = Field(default_factory=list)
    invite_assert_attempts: int = 0
    guard_blocked: bool = False
    guard_violations: list[str] = Field(default_factory=list)
    willing_to_at: bool = True


class ProfileQueryResponse(BaseModel):
    user_id: str
    group_id: str
    exists: bool
    profile: dict[str, Any] | None = None
    path: str


class ResetRequest(BaseModel):
    user_id: str
    group_id: str
    conversation_id: str = "default"
    clear_profile: bool = False
    membership: MembershipSignal = "unknown"
    unionid: str | None = None
    group_token: str | None = None
    user_token: str | None = None


class ResetResponse(BaseModel):
    user_id: str
    group_id: str
    conversation_id: str
    thread_id: str
    ok: bool
    profile_cleared: bool = False


class AsyncCallRequest(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-]+$", description="Opaque run ID generated by Micro")
    idempotency_key: str = Field(..., min_length=1, max_length=128, description="Stable idempotency key for user message")
    user_id: str = Field(..., min_length=1, max_length=128, description="Trusted user ID")
    unionid: str = Field(..., min_length=1, max_length=128, description="Trusted session unionid")
    group_id: str = Field(..., min_length=1, max_length=128, description="Trusted group ID")
    conversation_id: str = Field("default", min_length=1, max_length=128, description="Micro owned conversation ID")
    message: str = Field(..., min_length=1, max_length=8192, description="User message text")
    group_token: str | None = Field(None, max_length=1024, description="Group capability token")
    user_token: str | None = Field(None, max_length=1024, description="User bearer token")
    callback_url: str = Field(..., min_length=1, max_length=1024, description="Micro callback URL")
    metadata: dict[str, Any] = Field(default_factory=dict)
    membership: MembershipSignal = Field("unknown")
    run_match: bool = Field(True)
    willing_to_at: bool = Field(True)
    run_invite: bool = Field(True)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("metadata must be a dictionary")
        if len(v) > 20:
            raise ValueError("metadata contains too many keys (max 20)")
        forbidden_keys = {"candidates", "candidate_pool", "mock_candidates", "override_group_id", "trusted_group_id"}
        for key, val in v.items():
            if key.lower() in forbidden_keys:
                raise ValueError(f"metadata key '{key}' is forbidden (candidate injection / group override strictly prohibited)")
            if not isinstance(key, str) or len(key) > 64:
                raise ValueError(f"metadata key '{key}' must be a string with max length 64")
            if isinstance(val, str) and len(val) > 1024:
                raise ValueError(f"metadata value for key '{key}' exceeds max length 1024")
            elif not isinstance(val, (int, float, bool, str, type(None))):
                raise ValueError(f"metadata value for key '{key}' must be a primitive scalar type")
        return v


class AsyncCallResponse(BaseModel):
    success: bool = True
    run_id: str
    session_id: str
    accepted: bool = True
    message: str = "accepted"


CallbackEventType = Literal["progress", "chunk", "final", "error", "heartbeat"]


class CallbackEnvelope(BaseModel):
    version: str = "GA-CALLBACK-V1"
    run_id: str
    idempotency_key: str
    seq: int
    event: CallbackEventType
    occurred_at: str
    user_id: str
    group_id: str
    conversation_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
