"""Pydantic models for group_agent_api."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MembershipSignal = Literal["in_group", "not_in_group", "unknown"]
MatchStatus = Literal["matched", "weak", "empty", "skipped"]
DeliveryKind = Literal["directed", "undirected"]


def validate_group_agent_metadata(v: dict[str, Any]) -> dict[str, Any]:
    """Shared metadata gate for ChatRequest + AsyncCallRequest (REQ-028)."""
    if not isinstance(v, dict):
        raise ValueError("metadata must be a dictionary")
    if len(v) > 20:
        raise ValueError("metadata contains too many keys (max 20)")
    forbidden_keys = {
        "candidates",
        "candidate_pool",
        "mock_candidates",
        "override_group_id",
        "trusted_group_id",
    }
    for key, val in v.items():
        if key.lower() in forbidden_keys:
            raise ValueError(
                f"metadata key '{key}' is forbidden "
                "(candidate injection / group override strictly prohibited)"
            )
        if not isinstance(key, str) or len(key) > 64:
            raise ValueError(
                f"metadata key '{key}' must be a string with max length 64"
            )
        if key == "prior_candidate_ids":
            _validate_prior_candidate_ids(val)
            continue
        if key == "revisit_hint":
            _validate_revisit_hint(val)
            continue
        if isinstance(val, str) and len(val) > 1024:
            raise ValueError(
                f"metadata value for key '{key}' exceeds max length 1024"
            )
        if not isinstance(val, (int, float, bool, str, type(None))):
            raise ValueError(
                f"metadata value for key '{key}' must be a primitive scalar type"
            )
    return v


def _validate_prior_candidate_ids(val: Any) -> None:
    if not isinstance(val, list):
        raise ValueError("metadata prior_candidate_ids must be a list")
    if len(val) > 100:
        raise ValueError("metadata prior_candidate_ids exceeds max length 100")
    for item in val:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError(
                "metadata prior_candidate_ids items must be string or int ids"
            )
        text = str(item).strip()
        if not text or len(text) > 64:
            raise ValueError(
                "metadata prior_candidate_ids items must be 1..64 chars"
            )


def _validate_revisit_hint(val: Any) -> None:
    if not isinstance(val, dict):
        raise ValueError("metadata revisit_hint must be an object")
    allowed = {"has_prior_invite", "candidate_names", "topic_summary"}
    unknown = set(val) - allowed
    if unknown:
        raise ValueError(
            f"metadata revisit_hint has unknown keys: {sorted(unknown)}"
        )
    if "has_prior_invite" in val and not isinstance(val["has_prior_invite"], bool):
        raise ValueError("metadata revisit_hint.has_prior_invite must be bool")
    if "candidate_names" in val:
        names = val["candidate_names"]
        if not isinstance(names, list):
            raise ValueError(
                "metadata revisit_hint.candidate_names must be a list"
            )
        if len(names) > 5:
            raise ValueError(
                "metadata revisit_hint.candidate_names exceeds max length 5"
            )
        for name in names:
            if not isinstance(name, str) or not name.strip() or len(name) > 64:
                raise ValueError(
                    "metadata revisit_hint.candidate_names items must be "
                    "non-empty strings ≤64 chars"
                )
    if "topic_summary" in val and val["topic_summary"] is not None:
        topic = val["topic_summary"]
        if not isinstance(topic, str) or len(topic) > 256:
            raise ValueError(
                "metadata revisit_hint.topic_summary must be a string ≤256 chars"
            )


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

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_group_agent_metadata(v)


class SearchLogEntry(BaseModel):
    search_id: str
    timestamp: str
    query: str = ""
    rank_query: str | None = None
    match_status: str = "skipped"
    match_reason: str | None = None
    candidate_count: int = 0
    candidate_names: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    user_id: str
    group_id: str
    conversation_id: str
    thread_id: str
    reply: str
    profile_persisted: bool = False
    profile_path: str | None = None
    profile_status: str = "failed"
    persistence_failure_reason: str | None = None
    assert_attempts: int = 0
    persist_alert: str | None = None
    capability: MembershipSignal = "unknown"
    capability_source: str | None = None
    match_status: MatchStatus = "skipped"
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    match_reason: str | None = None
    search_log: SearchLogEntry | None = None
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
    rank_query: str | None = None
    search_log: SearchLogEntry | None = None
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
    unionid: str | None = Field(None, max_length=128, description="Trusted session unionid")
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
    # REQ-032 durable admission (required when GROUP_AGENT_DURABLE_QUEUE_ENABLED=1)
    request_schema_version: int | None = Field(
        None,
        description="Micro fingerprint schema version; durable mode requires 1",
    )
    request_fingerprint: str | None = Field(
        None,
        min_length=64,
        max_length=64,
        description="Micro-authoritative SHA-256 hex fingerprint",
    )
    queue_schema_version: int | None = Field(
        None,
        description="Queue payload schema version; durable mode requires 1",
    )

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_group_agent_metadata(v)

    @field_validator("request_fingerprint")
    @classmethod
    def validate_fingerprint_exact(cls, v: str | None) -> str | None:
        """Reject non-exact fingerprints — no strip/lowercase repair (REQ-032-FIX1)."""
        if v is None:
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError("request_fingerprint must be exact 64-char lowercase sha256 hex")
        return v


class AsyncCallResponse(BaseModel):
    success: bool = True
    run_id: str
    session_id: str
    accepted: bool = True
    message: str = "accepted"
    # REQ-032 additive ACK fields (backward compatible)
    idempotency_key: str | None = None
    execution_status: str | None = None
    queue_schema_version: int | None = None


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
