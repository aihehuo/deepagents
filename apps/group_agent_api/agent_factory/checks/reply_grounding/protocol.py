"""check.reply_grounding.v1 protocol models (TSD-14 §4.6.2)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Verdict(str, Enum):
    pass_ = "pass"
    fail = "fail"

    @classmethod
    def parse(cls, raw: Any) -> Verdict | None:
        text = str(raw or "").strip().lower()
        if text == "pass":
            return cls.pass_
        if text == "fail":
            return cls.fail
        return None


class RepairableBy(str, Enum):
    llm = "llm"
    orchestrator = "orchestrator"


GroundingCode = Literal[
    "unsupported_claim",
    "exaggeration",
    "unverified_action",
    "invented_candidate",
    "schema_invalid",
]

ALLOWED_CODES: frozenset[str] = frozenset(
    {
        "unsupported_claim",
        "exaggeration",
        "unverified_action",
        "invented_candidate",
        "schema_invalid",
    }
)


class FactItem(BaseModel):
    field: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=0, max_length=2000)


class MatchEvidenceItem(BaseModel):
    summary: str = Field(min_length=0, max_length=1000)


class CandidateGround(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    display_name: str = Field(default="", max_length=64)
    facts: list[FactItem] = Field(default_factory=list, max_length=32)
    match_evidence: list[MatchEvidenceItem] = Field(default_factory=list, max_length=16)


class GroundBlock(BaseModel):
    candidates: list[CandidateGround] = Field(default_factory=list, max_length=20)
    initiator_profile: dict[str, str] = Field(default_factory=dict)
    receipts: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    candidate_count: int = Field(default=0, ge=0, le=100)

    @field_validator("initiator_profile", mode="before")
    @classmethod
    def _coerce_profile(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        out: dict[str, str] = {}
        for key in ("doing", "need", "offer"):
            raw = value.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                out[key] = text[:2000]
        return out


class ReplyGroundingInput(BaseModel):
    """Inbound payload for check.reply_grounding.v1 — no chat history / PII supersets."""

    reply: str = Field(min_length=0, max_length=8000)
    reply_mode: str = Field(default="dialogue", max_length=64)
    ground: GroundBlock
    locale: str = Field(default="zh-CN", max_length=16)


class ReplyGroundingOutput(BaseModel):
    """Outbound payload. Illegal / missing fields must be coerced to fail closed."""

    verdict: Verdict
    codes: list[str] = Field(default_factory=list, max_length=8)
    spans: list[str] = Field(default_factory=list, max_length=8)
    repairable_by: RepairableBy = RepairableBy.llm
    message: str = Field(default="", max_length=500)
    layer: str = Field(default="l1", max_length=16)  # l0 | l1 | schema
    skipped: bool = False

    @field_validator("codes", mode="before")
    @classmethod
    def _filter_codes(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            code = str(item or "").strip()
            if code in ALLOWED_CODES and code not in out:
                out.append(code)
        return out[:8]

    @field_validator("spans", mode="before")
    @classmethod
    def _trim_spans(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text[:200])
        return out[:8]


def schema_invalid_output(*, message: str = "子代理输出无法解析") -> ReplyGroundingOutput:
    return ReplyGroundingOutput(
        verdict=Verdict.fail,
        codes=["schema_invalid"],
        spans=[],
        repairable_by=RepairableBy.llm,
        message=message[:500],
        layer="schema",
    )
