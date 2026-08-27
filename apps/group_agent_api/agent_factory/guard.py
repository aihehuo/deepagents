"""后处理守卫：能力分级断言 + 越权断言 + 披露泄漏断言。

非 in_group → 零候选人 / 零匹配 / 零 @。
in_group：允许 new_api 标注的跨群候选人（``is_reachable is False``）
以及经其它共同群可触达的候选人（``is_reachable is True`` 且 source ≠ 入口群）；
未标注 reachability 的外群候选人仍拦截。
违反 → 拦截（清空人脉面）+ 告警，不静默放行。

YAML ``chk.capability_guard`` (via ``module_config.is_check_enabled``):
off = skip this guard entirely (passthrough reply/candidates).
**Risk when off:** cross-group / disclosure / non-in_group network leaks
are not intercepted — operators must treat off as debug-only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from apps.group_agent_api.agent_factory.capability import (
    CapabilityTier,
    unlocks_network,
)
from apps.group_agent_api.agent_factory.disclosure import (
    assert_visible_fields_public_only,
    public_match_basis,
    stable_candidate_user_id,
)
from apps.group_agent_api.agent_factory.contact_scrub import scrub_candidate_contacts
from apps.group_agent_api.agent_factory.match_stub import MAX_CANDIDATES

_logger = logging.getLogger("uvicorn.error")

CHECK_ID = "chk.capability_guard"

# Heuristic: @mention or explicit candidate list markers in prose
_AT_PATTERN = re.compile(r"@[^\s@，。,.！!？?\n]{1,32}")
_NETWORK_LEAK_MARKERS = re.compile(
    r"(候选人|匹配到|为你推荐|群里有人|推荐对象|值得认识的人|@)"
)


@dataclass
class GuardResult:
    ok: bool
    tier: CapabilityTier
    candidates: list[dict[str, Any]] = field(default_factory=list)
    reply: str = ""
    violations: list[str] = field(default_factory=list)
    blocked: bool = False


def extract_at_identities(text: str) -> list[str]:
    """Return every visible @ identity in source order, including duplicates."""
    return [match[1:] for match in _AT_PATTERN.findall(text or "")]


def alert_capability_violation(
    *,
    user_id: str,
    group_id: str,
    tier: CapabilityTier,
    violations: list[str],
) -> None:
    _logger.error(
        "ALERT action=capability_guard_blocked user_id=%s group_id=%s "
        "tier=%s violations=%s status=intercepted",
        user_id,
        group_id,
        tier.value,
        ",".join(violations) or "none",
    )


def _reply_has_network_leak(reply: str) -> list[str]:
    hits: list[str] = []
    if _AT_PATTERN.search(reply or ""):
        hits.append("reply_contains_at_mention")
    if _NETWORK_LEAK_MARKERS.search(reply or ""):
        hits.append("reply_contains_network_marker")
    return hits


def capability_guard_enabled(*, enabled: bool | None = None) -> bool:
    """Resolve YAML ``chk.capability_guard``; explicit ``enabled`` wins (tests).

    Missing key → **on** (fail-closed safety). Explicit ``false`` skips the guard.
    """
    if enabled is not None:
        return bool(enabled)
    from apps.group_agent_api.agent_factory.module_config import load_modules_config

    cfg = load_modules_config()
    if CHECK_ID not in cfg.checks:
        return True
    return cfg.is_check_enabled(CHECK_ID)


def enforce_capability_guard(
    *,
    tier: CapabilityTier,
    reply: str,
    candidates: list[dict[str, Any]],
    caller_group_id: str,
    user_id: str = "",
    enabled: bool | None = None,
) -> GuardResult:
    """Post-process assert. Returns sanitized reply/candidates; blocks on violation.

    When ``chk.capability_guard`` is off, returns a passthrough ``ok=True``
    result (no strip / no alert). See module docstring for off risk.
    """
    violations: list[str] = []
    safe_reply = reply or ""
    safe_candidates = list(candidates or [])

    if not capability_guard_enabled(enabled=enabled):
        _logger.info(
            "action=capability_guard_skipped check_id=%s user_id=%s group_id=%s "
            "tier=%s risk=network_leak_unintercepted",
            CHECK_ID,
            user_id,
            caller_group_id,
            tier.value,
        )
        return GuardResult(
            ok=True,
            tier=tier,
            candidates=safe_candidates,
            reply=safe_reply,
            violations=[],
            blocked=False,
        )

    if not unlocks_network(tier):
        # 非在群：任何人脉面都必须为空
        if safe_candidates:
            violations.append("candidates_present_without_in_group")
        violations.extend(_reply_has_network_leak(safe_reply))
        if violations:
            alert_capability_violation(
                user_id=user_id,
                group_id=caller_group_id,
                tier=tier,
                violations=violations,
            )
            # Intercept: strip network surface; soft-scrub @ from reply
            safe_candidates = []
            safe_reply = _AT_PATTERN.sub("[已拦截]", safe_reply)
            return GuardResult(
                ok=False,
                tier=tier,
                candidates=[],
                reply=safe_reply,
                violations=violations,
                blocked=True,
            )
        return GuardResult(
            ok=True,
            tier=tier,
            candidates=[],
            reply=safe_reply,
            violations=[],
            blocked=False,
        )

    # in_group: still enforce pool/disclosure/count/cross-group
    if len(safe_candidates) > MAX_CANDIDATES:
        violations.append(f"too_many_candidates:{len(safe_candidates)}")

    filtered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    accepted_ids: set[str] = set()
    for c in safe_candidates:
        user_id_value = stable_candidate_user_id(c)
        if user_id_value is None:
            violations.append("missing_candidate_id")
            continue
        duplicate_seen = user_id_value in seen_ids
        if duplicate_seen:
            violations.append(f"duplicate_candidate_id:{user_id_value}")
        seen_ids.add(user_id_value)
        src = str(c.get("source_group_id") or c.get("group_id") or "")
        is_reachable = c.get("is_reachable")
        # REQ-031 / REQ-053: foreign source is allowed when new_api marked
        # reachability explicitly (False = cross-group apply-join; True =
        # shared another group). Missing flag + foreign source stays blocked.
        if src and src != caller_group_id and is_reachable not in (True, False):
            violations.append(f"cross_group:{user_id_value}:{src}")
            continue
        leaks = assert_visible_fields_public_only(c)
        if leaks:
            violations.extend(
                f"disclosure_leak:{user_id_value}:{x}" for x in leaks
            )
            continue
        if not public_match_basis(c):
            violations.append(f"missing_public_match_basis:{user_id_value}")
            continue
        if user_id_value in accepted_ids:
            continue
        accepted_ids.add(user_id_value)
        filtered.append(scrub_candidate_contacts(c))
    filtered = filtered[:MAX_CANDIDATES]

    if violations:
        alert_capability_violation(
            user_id=user_id,
            group_id=caller_group_id,
            tier=tier,
            violations=violations,
        )
        return GuardResult(
            ok=False,
            tier=tier,
            candidates=filtered,  # only safe ones retained
            reply=safe_reply,
            violations=violations,
            blocked=True,
        )

    return GuardResult(
        ok=True,
        tier=tier,
        candidates=filtered,
        reply=safe_reply,
        violations=[],
        blocked=False,
    )
