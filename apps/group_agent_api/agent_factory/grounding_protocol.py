"""REQ-DA-066 / TSD-13 · Grounding Protocol Models & Fixtures (ga-grounding-v1).

Defines:
- ProfileClaimV2, MatchConstraintV1, GroupProfileV2
- CandidateFactV1, MatchEvidenceV1, MatchResultV2
- GroundedFinalV1, ReplyMode
- Canonical JSON digest calculation matching Micro / TSD-13 specs.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ClaimType(str, Enum):
    fact = "fact"
    goal = "goal"
    hypothesis = "hypothesis"
    estimate = "estimate"
    quoted = "quoted"
    inferred = "inferred"


class ConfidenceLevel(str, Enum):
    verified = "verified"
    user_stated = "user_stated"
    uncertain = "uncertain"


class QualityStatus(str, Enum):
    active = "active"
    quarantined = "quarantined"


class DisclosureLevelV2(str, Enum):
    confirmed_public = "confirmed_public"
    match_only = "match_only"
    inferred_unconfirmed = "inferred_unconfirmed"


class ReplyMode(str, Enum):
    dialogue = "dialogue"
    profile_confirmation = "profile_confirmation"
    recommendation = "recommendation"
    no_match = "no_match"
    action_status = "action_status"
    error = "error"


class DialogueKind(str, Enum):
    clarification_question = "clarification_question"
    general_help = "general_help"
    capability_boundary = "capability_boundary"


class ConstraintField(str, Enum):
    gender = "gender"
    city = "city"
    industry = "industry"
    role = "role"
    company_size = "company_size"
    experience_tags = "experience_tags"
    excluded_user_ids = "excluded_user_ids"


class ConstraintOperator(str, Enum):
    eq = "eq"
    in_ = "in"
    not_in = "not_in"
    range_ = "range"
    all_ = "all"
    any_ = "any"
    not_any = "not_any"


class ConstraintStrength(str, Enum):
    hard = "hard"
    soft = "soft"


class ProfileEvidenceV2(BaseModel):
    """Source evidence anchored to trusted conversation messages."""

    source_type: Literal["conversation_message", "authoritative_computation"] = "conversation_message"
    source_message_id: int | None = Field(default=None, description="Trusted message ID from context")
    evidence_text: str = Field(..., min_length=1, max_length=500, description="Exact substring in source message")
    evidence_digest: str | None = Field(default=None, description="SHA-256 digest of normalized evidence text")

    @field_validator("evidence_text")
    @classmethod
    def strip_evidence(cls, v: str) -> str:
        text = (v or "").strip()
        if not text:
            raise ValueError("evidence_text must be non-empty")
        return text


class ProfileClaimV2(BaseModel):
    """One profile dimension in group-profile-v2."""

    value: str = Field(..., min_length=1, max_length=600, description="Dimension statement")
    disclosure: DisclosureLevelV2 = DisclosureLevelV2.inferred_unconfirmed
    claim_type: ClaimType = ClaimType.fact
    confidence: ConfidenceLevel = ConfidenceLevel.user_stated
    quality_status: QualityStatus = QualityStatus.active
    evidence: list[ProfileEvidenceV2] = Field(default_factory=list)

    @field_validator("value")
    @classmethod
    def strip_value(cls, v: str) -> str:
        text = (v or "").strip()
        if not text:
            raise ValueError("profile claim value must be non-empty")
        return text


class ProfileClaimInput(BaseModel):
    """Input payload allowed from LLM tool call for save_group_profile."""

    value: str = Field(..., min_length=1, max_length=600)
    disclosure: str = "inferred_unconfirmed"
    claim_type: str = "fact"
    evidence_text: str | None = None


class MatchConstraintV1(BaseModel):
    """A structured matching constraint (TSD-13 §3.2)."""

    field: str
    operator: str
    values: list[Any] = Field(..., min_length=1, max_length=20)
    strength: ConstraintStrength = ConstraintStrength.hard
    source_message_id: int | None = None
    evidence_text: str | None = None

    @field_validator("field")
    @classmethod
    def validate_field(cls, v: str) -> str:
        v_clean = (v or "").strip()
        try:
            ConstraintField(v_clean)
        except ValueError:
            raise ValueError(f"unknown constraint field: '{v_clean}'")
        return v_clean

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        v_clean = (v or "").strip()
        valid_ops = {op.value for op in ConstraintOperator}
        if v_clean not in valid_ops:
            raise ValueError(f"unknown constraint operator: '{v_clean}'")
        return v_clean


class MatchConstraintInput(BaseModel):
    """Input constraint allowed from LLM tool call."""

    field: str
    operator: str
    values: list[Any] = Field(default_factory=list)
    strength: str = "hard"
    evidence_text: str | None = None


class GroupProfileV2(BaseModel):
    """Full group-profile-v2 model."""

    schema_version: int = 2
    user_id: str = Field(..., min_length=1)
    group_id: str = Field(..., min_length=1)
    doing: ProfileClaimV2
    need: ProfileClaimV2
    offer: ProfileClaimV2
    match_constraints: list[MatchConstraintV1] = Field(default_factory=list)
    updated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )

    def to_storage_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CandidateFactV1(BaseModel):
    """A grounded candidate fact (TSD-13 §4.3)."""

    field: Literal["doing", "need", "offer"]
    value: str = Field(..., min_length=1, max_length=200)
    disclosure: Literal["confirmed_public", "card_allowed", "match_only"]
    source_type: Literal["group_agent_profile"] = "group_agent_profile"
    source_ref: str = Field(..., min_length=1)
    source_version: int = Field(..., ge=1)
    source_group_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Exact Micro profile group_id used to verify this fact",
    )


class MatchEvidenceV1(BaseModel):
    """Auditable relation linking initiator and candidate facts."""

    initiator_field: Literal["doing", "need", "offer"]
    candidate_field: Literal["doing", "need", "offer"]
    relation: str = Field(default="need_matches_doing", description="e.g. need_matches_doing")
    summary: str | None = None


class CandidateConnectionV1(BaseModel):
    type: str = "admin_referral"
    available: bool = True


class CandidateV2(BaseModel):
    """A candidate validated against ga-match-v2 contract."""

    user_id: str = Field(..., pattern=r"^[1-9]\d*$")
    source_group_id: str = Field(..., min_length=1, max_length=128)
    display_name: str = Field(default="合伙人", max_length=64)
    is_masked: bool = True
    same_group: bool
    shared_group: dict[str, Any] | None = None
    wechat_reachable: Literal[True]
    match_score: float | None = None
    facts: list[CandidateFactV1] = Field(..., min_length=1)
    match_evidence: list[MatchEvidenceV1] = Field(..., min_length=1)
    connection: CandidateConnectionV1 = Field(default_factory=CandidateConnectionV1)
    # Forward-compatible copy fields
    invite_text: str | None = None
    match_highlights: list[str] | None = None
    forward_copy: str | None = None
    quick_connect_copy: str | None = None

    @model_validator(mode="after")
    def validate_fact_references(self) -> "CandidateV2":
        """Bind each opaque profile reference to candidate identity and version."""
        for fact in self.facts:
            expected_ref = f"profile:{self.user_id}:{fact.source_version}"
            if fact.source_ref != expected_ref:
                raise ValueError(
                    "candidate fact source_ref must equal "
                    "profile:<candidate_uid>:<source_version>"
                )
        return self


class MatchResultV2(BaseModel):
    """ga-match-v2 response contract."""

    contract_version: Literal["ga-match-v2"] = "ga-match-v2"
    status: Literal["matched", "empty", "failed"] = "empty"
    reason: str = "empty"
    source_scope: Literal["same_group", "cross_group", "mixed", "none"] = "none"
    candidates: list[CandidateV2] = Field(default_factory=list)


class ProfileSummaryBlock(BaseModel):
    schema_version: int = 1
    profile_version: int | None = None
    digest: str = ""
    source_group_id: str = "global"
    doing: dict[str, Any] | str | None = None
    need: dict[str, Any] | str | None = None
    offer: dict[str, Any] | str | None = None


class MatchSummaryBlock(BaseModel):
    contract_version: Literal["ga-match-v2"] = "ga-match-v2"
    status: str = "empty"
    reason_code: str = "empty"
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    candidate_count: int | None = None


class InviteBlock(BaseModel):
    status: Literal["not_available", "ready"] = "ready"
    delivery_kind: Literal["manual_copy", "admin_referral", "direct_intro"] | None = "manual_copy"
    text: str | None = None



class GroundingBlock(BaseModel):
    candidate_facts_digest: str = ""
    constraint_digest: str = ""


class GroundedFinalV1(BaseModel):
    """Typed final payload for ga-grounding-v1 (TSD-13 §5.2)."""

    protocol_version: Literal["ga-grounding-v1"] = "ga-grounding-v1"
    run_id: str
    reply_mode: ReplyMode
    dialogue_kind: DialogueKind | None = None
    dialogue_text: str | None = None
    candidate_count: int = 0
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    profile: ProfileSummaryBlock | None = None
    match: MatchSummaryBlock | None = None
    invite: InviteBlock | None = None
    grounding: GroundingBlock | None = None
    # Compatibility field for legacy consumers (Micro v1 adapter)
    reply: str | None = None


def extract_canonical_candidate_facts_string(candidates: list[dict[str, Any]]) -> str:
    """Micro-compatible candidate facts tuple serialization: uid|field|value|disclosure."""
    tuples: list[str] = []
    for c in candidates or []:
        uid = str(c.get("user_id") or "").strip()
        facts = c.get("facts") or []
        for f in facts:
            if isinstance(f, dict):
                f_field = str(f.get("field") or "").strip()
                f_val = str(f.get("value") or "").strip()
                f_disc = str(f.get("disclosure") or "").strip()
                tuples.append(f"{uid}|{f_field}|{f_val}|{f_disc}")
            elif hasattr(f, "field"):
                tuples.append(f"{uid}|{f.field}|{f.value}|{f.disclosure}")
    tuples.sort()
    return "\n".join(tuples)


def calculate_candidate_facts_digest(candidates: list[dict[str, Any]]) -> str:
    """Calculate sha256 candidate_facts_digest exactly matching Micro Validator."""
    canonical_str = extract_canonical_candidate_facts_string(candidates)
    hex_digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
    return f"sha256:{hex_digest}"


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize object to canonical, deterministically sorted compact UTF-8 JSON bytes."""
    if hasattr(obj, "model_dump"):
        data = obj.model_dump(mode="json", exclude_none=True)
    elif isinstance(obj, dict):
        data = {k: v for k, v in obj.items() if v is not None}
    else:
        data = obj

    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(obj: Any) -> str:
    """Calculate SHA-256 hex digest over canonical JSON bytes."""
    return f"sha256:{hashlib.sha256(canonical_json_bytes(obj)).hexdigest()}"



_NUMBER_PATTERN = re.compile(
    r"\d+(?:\.\d+)?%?"
    r"|\d+(?:万|千|百|亿|元|人|年|天|月|度|℃|kg|个|套|家|项|条|次)?"
)


def extract_numbers_and_units(text: str) -> set[str]:
    """Extract numeric literals, percentages, and basic quantified phrases from text."""
    if not text:
        return set()
    matches = _NUMBER_PATTERN.findall(text)
    return {m.strip() for m in matches if m.strip() and not m.strip().isalpha()}


def validate_profile_claim_grounding(
    claim: ProfileClaimV2,
    source_message_text: str | None,
) -> list[str]:
    """Local pre-check for claim grounding against trusted source message."""
    violations: list[str] = []
    if not claim.evidence:
        violations.append("missing_evidence")
        return violations

    msg_norm = " ".join((source_message_text or "").split())
    for ev in claim.evidence:
        ev_norm = " ".join(ev.evidence_text.split())
        if source_message_text is not None and ev_norm not in msg_norm:
            violations.append("evidence_not_in_source_message")

    # Numeric extraction check
    val_numbers = extract_numbers_and_units(claim.value)
    if val_numbers and source_message_text is not None:
        source_numbers = extract_numbers_and_units(source_message_text)
        missing_numbers = val_numbers - source_numbers
        if missing_numbers:
            violations.append(f"unsupported_numbers:{','.join(sorted(missing_numbers))}")

    # Fact claim vs speculative keywords check
    speculative_markers = ["可能", "预计", "设想", "目标", "还没", "未验证", "打算", "想做", "计划"]
    if claim.claim_type == ClaimType.fact:
        if any(marker in claim.value or any(marker in ev.evidence_text for ev in claim.evidence) for marker in speculative_markers):
            violations.append("fact_claim_contains_speculative_markers")

    return violations
