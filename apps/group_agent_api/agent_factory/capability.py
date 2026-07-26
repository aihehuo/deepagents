"""闸门 #4 · 能力分级契约（单一集中定义）。

PRC-01 §7.4 / §9.1 Q9：在群解锁人脉；不在群 / 判不出 → 仅通用对话。
本模块是唯一权威判定入口——禁在别处各判各的。
"""

from __future__ import annotations

from enum import Enum


class CapabilityTier(str, Enum):
    """能力档。人脉能力仅 in_group 解锁。"""

    in_group = "in_group"
    not_in_group = "not_in_group"
    unknown = "unknown"


def resolve_capability(membership: str | bool | None) -> CapabilityTier:
    """唯一权威：将 mock/软信号映射为能力档。

    Accepted inputs:
      - CapabilityTier / str: "in_group" | "not_in_group" | "unknown"
      - bool: True→in_group, False→not_in_group
      - None / "" / unrecognized → unknown（软失败，不硬拒）
    """
    if isinstance(membership, CapabilityTier):
        return membership
    if isinstance(membership, bool):
        return CapabilityTier.in_group if membership else CapabilityTier.not_in_group
    if membership is None:
        return CapabilityTier.unknown
    text = str(membership).strip().lower()
    if text in {"", "null", "none"}:
        return CapabilityTier.unknown
    if text in {"in_group", "in-group", "true", "1", "yes"}:
        return CapabilityTier.in_group
    if text in {"not_in_group", "not-in-group", "out_of_group", "false", "0", "no"}:
        return CapabilityTier.not_in_group
    if text in {"unknown", "undetermined"}:
        return CapabilityTier.unknown
    # Unrecognized soft signal → unknown (SAFE-07 soft-fail, never hard-deny)
    return CapabilityTier.unknown


def unlocks_network(tier: CapabilityTier) -> bool:
    """人脉能力（候选人 / 匹配 / @）是否解锁。"""
    return tier is CapabilityTier.in_group
