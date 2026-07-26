"""Structured group profile schema (FR-06 / AI-05 disclosure slots).

REQ-004: structured fields, not free markdown. Disclosure slots are recorded
for SAFE-01/02 forward-compat; this slice does not consume them for filtering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DisclosureLevel(str, Enum):
    """Field-level disclosure classification (AI-05)."""

    confirmed_public = "confirmed_public"
    match_only = "match_only"
    inferred_unconfirmed = "inferred_unconfirmed"


class ProfileField(BaseModel):
    """One profile dimension with disclosure metadata."""

    value: str = Field(..., min_length=1, description="Non-empty field content")
    disclosure: DisclosureLevel = DisclosureLevel.inferred_unconfirmed

    @field_validator("value")
    @classmethod
    def strip_nonempty(cls, v: str) -> str:
        text = (v or "").strip()
        if not text:
            raise ValueError("profile field value must be non-empty")
        return text


class GroupProfile(BaseModel):
    """Three-dimensional matchable profile scoped to user × group."""

    user_id: str = Field(..., min_length=1)
    group_id: str = Field(..., min_length=1)
    doing: ProfileField = Field(..., description="当前在做什么（创业意图/方向）")
    need: ProfileField = Field(..., description="缺什么（需求 gap）")
    offer: ProfileField = Field(..., description="能提供什么（资源/技能）")
    updated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )
    schema_version: int = 1

    def is_complete(self) -> bool:
        return bool(
            self.doing.value.strip()
            and self.need.value.strip()
            and self.offer.value.strip()
        )

    def to_storage_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def profile_from_flat(
    *,
    user_id: str,
    group_id: str,
    doing: str,
    need: str,
    offer: str,
    doing_disclosure: DisclosureLevel | str = DisclosureLevel.inferred_unconfirmed,
    need_disclosure: DisclosureLevel | str = DisclosureLevel.inferred_unconfirmed,
    offer_disclosure: DisclosureLevel | str = DisclosureLevel.inferred_unconfirmed,
) -> GroupProfile:
    """Build a GroupProfile from flat tool arguments."""
    return GroupProfile(
        user_id=user_id,
        group_id=group_id,
        doing=ProfileField(value=doing, disclosure=DisclosureLevel(doing_disclosure)),
        need=ProfileField(value=need, disclosure=DisclosureLevel(need_disclosure)),
        offer=ProfileField(value=offer, disclosure=DisclosureLevel(offer_disclosure)),
    )
