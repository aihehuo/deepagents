"""Pydantic models for group_agent_api."""

from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from apps.group_agent_api.app.venue_protocol import (
    validate_v2_callback,
    validate_v2_request,
)

MembershipSignal = Literal["in_group", "not_in_group", "unknown"]
MatchStatus = Literal[
    "matched", "weak", "empty", "skipped", "rejected", "error"
]
DeliveryKind = Literal["directed", "undirected"]


def validate_group_agent_metadata(
    v: dict[str, Any], *, allow_referral_context: bool = False
) -> dict[str, Any]:
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
        "protocol_mode",
        "rollout_version",
        "contract_version",
        "rollout_context",
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
        if key == "prior_recommendation":
            if not allow_referral_context:
                raise ValueError("metadata prior_recommendation is reserved for trusted async calls")
            if not isinstance(val, dict):
                raise ValueError("metadata prior_recommendation must be a dictionary")
            continue
        if key in {"source_message_id", "source_message_text"}:
            if not allow_referral_context:
                raise ValueError(f"metadata {key} is reserved for trusted async calls")
            continue
        if key == "referral_context":
            if not allow_referral_context:
                raise ValueError("metadata referral_context is reserved for trusted async calls")
            _validate_referral_context(val)
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


def _validate_referral_context(val: Any) -> None:
    if not isinstance(val, dict):
        raise ValueError("metadata referral_context must be an object")
    allowed = {
        "referral_id",
        "applicant_id",
        "applicant_name",
        "applicant_doing",
        "applicant_need",
        "applicant_offer",
        "match_highlights",
        "status",
        "created_at",
        "intro_once",
    }
    unknown = set(val) - allowed
    if unknown:
        raise ValueError(
            f"metadata referral_context has unknown keys: {sorted(unknown)}"
        )
    for key in ("referral_id", "applicant_id"):
        item = val.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"metadata referral_context.{key} must be a positive int")
    limits = {
        "applicant_name": 64,
        "applicant_doing": 600,
        "applicant_need": 600,
        "applicant_offer": 600,
        "created_at": 64,
    }
    for key, limit in limits.items():
        item = val.get(key)
        if item is not None and (not isinstance(item, str) or len(item) > limit):
            raise ValueError(
                f"metadata referral_context.{key} must be a string ≤{limit} chars"
            )
    highlights = val.get("match_highlights", [])
    if not isinstance(highlights, list) or len(highlights) > 5:
        raise ValueError("metadata referral_context.match_highlights must be a list ≤5")
    if any(not isinstance(item, str) or len(item) > 160 for item in highlights):
        raise ValueError(
            "metadata referral_context.match_highlights items must be strings ≤160 chars"
        )
    if val.get("status") not in {"pending", "dispatched", "accepted"}:
        raise ValueError("metadata referral_context.status is invalid")
    if val.get("intro_once") is not True:
        raise ValueError("metadata referral_context.intro_once must be true")


def _validate_prior_candidate_ids(val: Any) -> None:
    if not isinstance(val, list):
        raise ValueError("metadata prior_candidate_ids must be a list")
    if len(val) > 500:
        raise ValueError("metadata prior_candidate_ids exceeds max length 500")
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


class RolloutContext(BaseModel):
    protocol_mode: Literal["legacy_v1", "grounded_v2"] = "legacy_v1"
    rollout_version: Literal["ga-v2-canary-v1"] = "ga-v2-canary-v1"


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
    rollout_context: RolloutContext | None = Field(
        None,
        description="Server-originated rollout protocol mode and version (REQ-XCUT-004)",
    )

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
    suggested_replies: list[dict[str, str]] = Field(default_factory=list)
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
    idempotency_key: str = Field(..., min_length=1, max_length=256, description="Stable idempotency key for user message")
    user_id: str = Field(..., min_length=1, max_length=128, description="Trusted user ID")
    unionid: str | None = Field(None, max_length=128, description="Trusted session unionid")
    group_id: str = Field(..., min_length=1, max_length=128, description="Trusted group ID")
    conversation_id: str = Field("default", min_length=1, max_length=128, description="Micro owned conversation ID")
    message: str | None = Field(..., min_length=0, max_length=8192, description="User text; null for v2 entry, empty for v3 entry")
    mode_contract_version: int | None = None
    execution_mode: Literal["cafe_host", "group_agent"] | None = None
    executor_policy_version: int | None = None
    profile_scope_id: str | None = None
    memory_scope_id: str | None = None
    episode_id: str | None = None
    input_kind: Literal["user_message", "venue_event"] | None = None
    event: dict[str, Any] | None = None
    venue_context: dict[str, Any] | None = None
    context_snapshot: dict[str, Any] | None = None
    access_contract_version: int | None = None
    credential_binding: dict[str, Any] | None = None
    chat_access: dict[str, Any] | None = None
    trusted_search: dict[str, Any] | None = None
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
    rollout_context: RolloutContext | None = Field(
        None,
        description="Server-originated rollout protocol mode and version (REQ-XCUT-004)",
    )

    @model_serializer(mode="wrap")
    def serialize_protocol(self, handler: Any) -> dict[str, Any]:
        """Keep old signed serialization byte-compatible with existing fields."""
        from apps.group_agent_api.app.execution_modes import MODE_FIELDS

        result = handler(self)
        if self.request_schema_version not in {3, 4}:
            for name in MODE_FIELDS:
                result.pop(name, None)
        if self.request_schema_version == 4:
            result.pop("group_token", None)
            result.pop("user_token", None)
            if result.get("trusted_search") is None:
                result.pop("trusted_search", None)
        else:
            for name in ("access_contract_version", "credential_binding", "chat_access", "trusted_search"):
                result.pop(name, None)
        return result

    @model_validator(mode="before")
    @classmethod
    def validate_input_protocol(cls, value: Any) -> Any:
        """Gate strict v2 before legacy coercion; never downgrade an event."""
        if not isinstance(value, dict):
            return value
        if value.get("request_schema_version") == 3:
            from apps.group_agent_api.app.execution_modes import validate_v3_request

            try:
                return validate_v3_request(value)
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "invalid execution mode request"}) from exc
        if value.get("request_schema_version") == 4:
            from apps.group_agent_api.app.execution_modes import validate_v4_request

            try:
                return validate_v4_request(value)
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "invalid Cafe Host v4 request"}) from exc
        from apps.group_agent_api.app.execution_modes import MODE_FIELDS
        if any(name in value for name in MODE_FIELDS):
            raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "execution fields require v3"})
        if value.get("request_schema_version") == 2:
            if type(value["request_schema_version"]) is not int or type(value.get("queue_schema_version")) is not int or value.get("queue_schema_version") != 2:
                raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "unsupported request/queue schema pair"})
            return validate_v2_request(value)
        if value.get("request_schema_version") not in (None, 1):
            raise HTTPException(status_code=400, detail={"error": "schema_invalid", "message": "unsupported request version"})
        if len(str(value.get("idempotency_key", ""))) > 128:
            raise ValueError("idempotency_key exceeds legacy maximum")
        if any(value.get(key) is not None for key in ("episode_id", "input_kind", "event", "venue_context", "context_snapshot")):
            raise ValueError("schema_invalid: typed input requires protocol v2")
        if value.get("message") is None or not str(value.get("message", "")).strip():
            raise ValueError("schema_invalid: user_message requires nonempty text")
        return value

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return validate_group_agent_metadata(v, allow_referral_context=True)

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
    execution_binding: dict[str, Any] | None = None
    success: bool = True
    run_id: str
    session_id: str
    accepted: bool = True
    message: str = "accepted"
    # REQ-032 additive ACK fields (backward compatible)
    idempotency_key: str | None = None
    execution_status: str | None = None
    queue_schema_version: int | None = None

    @model_serializer(mode="wrap")
    def serialize_ack(self, handler: Any) -> dict[str, Any]:
        """Preserve legacy ACK bytes without requiring newer `exclude_if`."""
        result = handler(self)
        if self.execution_binding is None:
            result.pop("execution_binding", None)
        return result


CallbackEventType = Literal[
    "progress", "chunk", "final", "error", "heartbeat", "search_log", "search_result"
]


class CallbackEnvelope(BaseModel):
    execution_binding: dict[str, Any] | None = None
    version: str | None = "GA-CALLBACK-V1"
    callback_version: int | None = None
    run_id: str
    idempotency_key: str | None = None
    seq: int
    event: CallbackEventType
    occurred_at: str
    user_id: str
    group_id: str
    conversation_id: str
    episode_id: str | None = None
    space_id: str | None = None
    host_agent_id: str | None = None
    memory_scope_id: str | None = None
    input_kind: Literal["user_message", "venue_event"] | None = None
    event_name: Literal["venue_entered"] | None = None
    visit_id: str | None = None
    deep_attempt_id: str | None = None
    deep_attempt_number: int | None = None
    routing_snapshot_hash: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def validate_callback_protocol(cls, value: Any) -> Any:
        """Preserve old callbacks while rejecting v2 coercion/unknown fields."""
        if isinstance(value, dict) and value.get("callback_version") is not None:
            if (
                value.get("callback_version") != 1
                or value.get("version") is not None
                or type(value.get("deep_attempt_id")) is not str
                or not value.get("deep_attempt_id")
                or type(value.get("deep_attempt_number")) is not int
                or value.get("deep_attempt_number", 0) < 1
                or value.get("input_kind") == "venue_event"
            ):
                raise ValueError("schema_invalid: strict callback envelope")
        if isinstance(value, dict) and value.get("execution_binding") is not None:
            binding = value["execution_binding"]
            fields = {"schema_version", "user_id", "execution_mode", "conversation_id", "episode_id", "run_id"}
            if type(binding) is not dict or set(binding) != fields or type(binding.get("schema_version")) is not int or binding["schema_version"] != 1 or binding.get("execution_mode") not in {"cafe_host", "group_agent"}:
                raise ValueError("binding_mismatch: invalid execution binding")
            if any(type(binding.get(key)) is not str or not binding[key] or binding[key] != value.get(key) for key in ("user_id", "conversation_id", "episode_id", "run_id")):
                raise ValueError("binding_mismatch: callback differs from execution binding")
            suffix = "cafe" if binding["execution_mode"] == "cafe_host" else "group"
            if binding["conversation_id"] != f"ga_mode_{suffix}_{binding['user_id']}" or value.get("group_id") != "global":
                raise ValueError("binding_mismatch: invalid mode conversation")
            return value
        if isinstance(value, dict) and value.get("input_kind") is not None:
            return validate_v2_callback(value)
        if isinstance(value, dict) and "idempotency_key" not in value:
            raise ValueError("legacy callback requires idempotency_key")
        return value

    @model_serializer(mode="wrap")
    def serialize_callback(self, handler: Any) -> dict[str, Any]:
        """Do not add v3 binding to old callback bytes."""
        result = handler(self)
        if self.execution_binding is None:
            result.pop("execution_binding", None)
        if self.version is None:
            result.pop("version", None)
        return result

    @model_validator(mode="after")
    def validate_venue_callback(self) -> CallbackEnvelope:
        """Reject unsupported event output before signing or delivery."""
        if self.input_kind is None:
            return self
        if not all((self.episode_id, self.space_id, self.host_agent_id, self.memory_scope_id)):
            raise ValueError("binding_mismatch: missing v2 callback identity")
        if self.space_id != "community_cafe" or self.memory_scope_id != "community_cafe_host" or self.host_agent_id not in {"community_barista", "community_bartender"}:
            raise ValueError("binding_mismatch: unsupported venue identity")
        if self.input_kind == "venue_event":
            if self.event_name != "venue_entered" or not re.fullmatch(r"venue_[A-Za-z0-9_-]{1,96}", self.visit_id or ""):
                raise ValueError("binding_mismatch: missing visit identity")
            if self.event == "chunk":
                raise ValueError("event_not_allowed")
            if self.event == "progress" and self.payload != {"phase": "preparing_greeting"}:
                raise ValueError("event_not_allowed")
            if self.event == "final":
                _GreetingPayload.model_validate(self.payload)
                if not self.payload["reply"].strip():
                    raise ValueError("greeting_schema_invalid: empty reply")
        elif self.event_name is not None or self.visit_id is not None:
            raise ValueError("binding_mismatch: user_message is not a visit")
        return self


class _SuggestedGreetingReply(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    label: str = Field(min_length=1, max_length=40)
    reply: str = Field(min_length=1, max_length=200)


class _GreetingPayload(BaseModel):
    model_config = {"extra": "forbid", "strict": True}
    result_kind: Literal["greeting"]
    reply: str = Field(min_length=1, max_length=1000)
    suggested_replies: list[_SuggestedGreetingReply] = Field(default_factory=list, max_length=3)
