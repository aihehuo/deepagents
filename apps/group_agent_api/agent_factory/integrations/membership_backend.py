"""Membership resolve facade: stub membership field vs HTTP REQ-018."""

from __future__ import annotations

import os
from apps.group_agent_api.agent_factory.capability import (
    CapabilityTier,
    resolve_capability,
)
from apps.group_agent_api.agent_factory.integrations.config import integration_mode
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
    fetch_membership,
)


def resolve_session_capability(
    *,
    membership_override: str | None,
    unionid: str | None,
    group_token: str | None,
    force_mode: str | None = None,
    group_id: str | None = None,
    user_id: str | None = None,
) -> MembershipResult:
    """唯一入口：http 模式用 OAuth unionid + 群 JWT；stub 模式接入 Fixture 权威关系。

    http 模式**忽略**客户端 membership 覆盖（防注入提权）。

    REQ-032 / Micro REQ-028: full-network agent uses session bucket ``global`` and
    no longer forwards ``group_token``. Authenticated callers (unionid present)
    must still unlock matching — otherwise every turn collapses to tier=unknown
    and the profile-confirmation template overwrites the model reply.
    """
    mode = (force_mode or integration_mode()).strip().lower()
    if mode == "http":
        plain_gid = (group_id or "").strip()
        uid = (unionid or "").strip()
        if plain_gid in {"global", "admin"} and uid:
            reason = (
                "admin_session_authenticated"
                if plain_gid == "admin"
                else "global_session_authenticated"
            )
            source = "http_admin" if plain_gid == "admin" else "http_global"
            return MembershipResult(
                tier=CapabilityTier.in_group,
                event_id=None,
                reason=reason,
                source=source,
            )
        return fetch_membership(
            unionid=uid,
            group_token=group_token or "",
        )

    test_lvl = os.environ.get("GROUP_AGENT_TEST_LEVEL")
    if test_lvl:
        from apps.group_agent_api.fixtures.loader import load_fixture
        try:
            ds = load_fixture(test_lvl)
            if group_id and user_id:
                m_key = f"{group_id}:{user_id}"
                member = ds.members.get(m_key)
                if member and member.bound and member.membership == "in_group":
                    return MembershipResult(
                        tier=CapabilityTier.in_group,
                        reason="fixture_authoritative_in_group",
                        source="stub_fixture",
                    )
                return MembershipResult(
                    tier=CapabilityTier.not_in_group,
                    reason="fixture_authoritative_not_in_group",
                    source="stub_fixture",
                )
            return MembershipResult(
                tier=CapabilityTier.not_in_group,
                reason="fixture_missing_group_or_user",
                source="stub_fixture",
            )
        except Exception as err:
            return MembershipResult(
                tier=CapabilityTier.not_in_group,
                reason=f"fixture_error_{type(err).__name__}",
                source="stub_fixture",
            )

    tier = resolve_capability(membership_override)
    return MembershipResult(
        tier=tier,
        reason="stub_membership",
        source="stub",
    )
