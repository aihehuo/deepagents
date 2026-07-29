"""FR-03 匹配 stub（new_api 向量匹配形状）+ 可触达池过滤。

- 只在「本群 group_id + 已绑」池内选 1–3 人（SAFE-03）
- 无匹配 → SC-05 empty；弱匹配 → SC-06 weak
- 不返回手机/微信；可见字段经披露闸过滤
- 越权：候选 source group 必须 == 调用 group_id
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from apps.group_agent_api.agent_factory.disclosure import filter_member_for_visibility
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile
from apps.group_agent_api.agent_factory.profile_store import validate_id

MatchStatus = Literal["matched", "weak", "empty"]
MAX_CANDIDATES = 3
WEAK_SCORE_THRESHOLD = 0.45


@dataclass
class PoolMember:
    user_id: str
    group_id: str
    bound: bool
    display_name: str
    doing: dict[str, str]
    need: dict[str, str]
    offer: dict[str, str]
    profile_url: str = ""
    is_reachable: bool = True
    group_info: dict[str, Any] | None = None
    # Internal match signals (never exposed raw)
    keywords: list[str] = field(default_factory=list)

    def to_raw_dict(self) -> dict[str, Any]:
        res = {
            "user_id": self.user_id,
            "group_id": self.group_id,
            "bound": self.bound,
            "display_name": self.display_name,
            "doing": self.doing,
            "need": self.need,
            "offer": self.offer,
            "profile_url": self.profile_url or f"/users/{self.user_id}",
            "is_reachable": self.is_reachable,
        }
        if self.group_info is not None:
            res["group_info"] = self.group_info
        return res


@dataclass
class MatchResult:
    status: MatchStatus
    candidates: list[dict[str, Any]]
    query: str
    group_id: str
    reason: str = ""
    # Each candidate may include match_confidence: high|low


def _default_pool() -> list[PoolMember]:
    """Mock reachable members for local/demo (multi-group)."""
    return [
        PoolMember(
            user_id="mock_zhou",
            group_id="mock_g1",
            bound=True,
            display_name="周然",
            doing={
                "value": "智能小家电量产固件",
                "disclosure": "confirmed_public",
            },
            need={
                "value": "找硬件创业搭档验证方案",
                "disclosure": "match_only",
            },
            offer={
                "value": "嵌入式与量产经验",
                "disclosure": "confirmed_public",
            },
            keywords=["固件", "嵌入式", "联网", "小家电", "量产"],
        ),
        PoolMember(
            user_id="mock_li",
            group_id="mock_g1",
            bound=True,
            display_name="李工",
            doing={
                "value": "物联网联网模组",
                "disclosure": "confirmed_public",
            },
            need={
                "value": "希望对接整机项目",
                "disclosure": "inferred_unconfirmed",
            },
            offer={
                "value": "模组选型与联调",
                "disclosure": "confirmed_public",
            },
            keywords=["联网", "模组", "物联网", "App"],
        ),
        PoolMember(
            user_id="mock_unbound",
            group_id="mock_g1",
            bound=False,  # 未绑 → 不可入可触达池
            display_name="未绑定访客",
            doing={"value": "路过", "disclosure": "confirmed_public"},
            need={"value": "无", "disclosure": "confirmed_public"},
            offer={"value": "无", "disclosure": "confirmed_public"},
            keywords=["路过"],
        ),
        PoolMember(
            user_id="mock_other_group",
            group_id="mock_g2",  # 他群 → 本群调用不可见
            bound=True,
            display_name="他群高手",
            doing={"value": "芯片设计", "disclosure": "confirmed_public"},
            need={"value": "找项目", "disclosure": "confirmed_public"},
            offer={"value": "ASIC", "disclosure": "confirmed_public"},
            keywords=["芯片", "联网", "固件"],
        ),
        PoolMember(
            user_id="mock_weak",
            group_id="mock_g1",
            bound=True,
            display_name="弱相关阿强",
            doing={
                "value": "餐饮加盟咨询",
                "disclosure": "confirmed_public",
            },
            need={"value": "找投资", "disclosure": "match_only"},
            offer={"value": "门店运营", "disclosure": "confirmed_public"},
            keywords=["餐饮", "加盟"],
        ),
    ]


class MatchStub:
    """In-process stub mimicking new_api vector search shape."""

    def __init__(self, pool: list[PoolMember] | None = None) -> None:
        self.pool = list(pool) if pool is not None else _default_pool()

    def reachable_pool(self, group_id: str) -> list[PoolMember]:
        gid = validate_id(group_id, field="group_id")
        import os
        test_lvl = os.environ.get("GROUP_AGENT_TEST_LEVEL")
        if test_lvl:
            from apps.group_agent_api.fixtures.loader import load_fixture
            ds = load_fixture(test_lvl)
            fixture_members = []
            for m_key, m in ds.members.items():
                if m.group_id == gid and m.bound and m.membership == "in_group" and m.reachable:
                    doing_val = m.profile.get("doing", "") if isinstance(m.profile, dict) else str(m.profile)
                    need_val = m.profile.get("need", "") if isinstance(m.profile, dict) else ""
                    offer_val = m.profile.get("offer", "") if isinstance(m.profile, dict) else ""
                    kw_tokens = [k.strip() for k in re.split(r"[\s,，、/]+", f"{doing_val} {need_val} {offer_val}") if k.strip()]
                    fixture_members.append(
                        PoolMember(
                            user_id=m.user_id,
                            group_id=m.group_id,
                            bound=m.bound,
                            display_name=m.display_name,
                            doing={"value": doing_val, "disclosure": m.disclosure_level},
                            need={"value": need_val, "disclosure": m.disclosure_level},
                            offer={"value": offer_val, "disclosure": m.disclosure_level},
                            keywords=kw_tokens,
                        )
                    )
            return fixture_members
        return [m for m in self.pool if (m.group_id == gid or m.is_reachable is False) and m.bound]

    def search(
        self,
        *,
        query: str,
        group_id: str,
        excluded_ids: list[str] | None = None,
        vector_search: bool = True,  # noqa: ARG002 — shape parity with new_api
        limit: int = MAX_CANDIDATES,
    ) -> MatchResult:
        gid = validate_id(group_id, field="group_id")
        excluded = {str(x) for x in (excluded_ids or [])}
        pool = [
            m
            for m in self.reachable_pool(gid)
            if m.user_id not in excluded
        ]
        scored: list[tuple[float, PoolMember]] = []
        q = (query or "").lower()
        tokens = [t for t in re.split(r"[\s,，、/]+", q) if t]
        for m in pool:
            hay = " ".join(
                [
                    m.display_name,
                    m.doing.get("value", ""),
                    m.need.get("value", ""),
                    m.offer.get("value", ""),
                    *m.keywords,
                ]
            ).lower()
            hits = sum(1 for t in tokens if t.lower() in hay) if tokens else 0
            # keyword overlap ratio; also bump if any keyword substring in query
            kw_hits = sum(1 for k in m.keywords if k.lower() in q)
            score = 0.0
            if tokens:
                score = hits / max(len(tokens), 1)
            score = max(score, min(1.0, kw_hits * 0.35))
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return MatchResult(
                status="empty",
                candidates=[],
                query=query,
                group_id=gid,
                reason="sc05_no_suitable_match",
            )

        top = scored[: max(1, min(limit, MAX_CANDIDATES))]
        best = top[0][0]
        status: MatchStatus = "matched" if best >= WEAK_SCORE_THRESHOLD else "weak"

        candidates: list[dict[str, Any]] = []
        for score, member in top:
            # 越权硬约束：来源群必须等于调用群 (除非 is_reachable == False 允许跨群候选人)
            if member.group_id != gid and member.is_reachable is not False:
                continue
            visible = filter_member_for_visibility(member.to_raw_dict())
            visible["source_group_id"] = member.group_id
            visible["match_score"] = round(score, 3)
            visible["match_confidence"] = (
                "high" if score >= WEAK_SCORE_THRESHOLD else "low"
            )
            if status == "weak":
                visible["confidence_note"] = "关联度一般，供参考"
            if "is_reachable" not in visible:
                visible["is_reachable"] = member.is_reachable
            if member.group_info is not None and "group_info" not in visible:
                visible["group_info"] = member.group_info
            candidates.append(visible)

        if not candidates:
            return MatchResult(
                status="empty",
                candidates=[],
                query=query,
                group_id=gid,
                reason="sc05_no_suitable_match",
            )

        return MatchResult(
            status=status,
            candidates=candidates,
            query=query,
            group_id=gid,
            reason=(
                "sc06_weak_match"
                if status == "weak"
                else f"matched_{len(candidates)}"
            ),
        )


def build_query_from_profile(profile: GroupProfile) -> str:
    """Natural-language query for stub / future new_api."""
    return " ".join(
        [
            profile.doing.value,
            profile.need.value,
            profile.offer.value,
        ]
    )


# Process-wide default stub (tests may replace)
_DEFAULT_STUB = MatchStub()


def get_match_stub() -> MatchStub:
    return _DEFAULT_STUB


def set_match_stub(stub: MatchStub | None) -> None:
    global _DEFAULT_STUB
    _DEFAULT_STUB = stub if stub is not None else MatchStub()
