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
            same_group = member.group_id == gid
            visible.setdefault("same_group", same_group)
            visible.setdefault("wechat_reachable", True)
            visible.setdefault("app_registered", True)
            visible.setdefault("has_talked_with_agent", False)
            visible.setdefault("is_masked", not same_group)
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
    """Broad retrieval query (backward-compatible entrypoint).

    Fine-grained need details belong in ``build_rank_query_from_profile`` so
    hybrid_search stays recall-friendly while ranking stays precise.
    """
    return build_broad_query_from_profile(profile)


_NEED_HEAD_SPLIT = re.compile(r"[；;。．\n]|当前卡点|尤其是?|比如|例如|以及")
_QUERY_STOPWORDS = re.compile(
    r"(?:做面向|聚焦|建设|已有|试运行|需求|卡点|需要|想找|寻找|希望|最好|或有|的合伙人|的搭子|在拓展规划中|做过|团队|项目|工具|合伙人|帮我)"
)


def _compact(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", "", (text or "").strip())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


def _clean_query_keywords(text: str) -> str:
    cleaned = _QUERY_STOPWORDS.sub(" ", text or "")
    cleaned = re.sub(r"[^\w\u4e00-\u9fa5]+", " ", cleaned)
    tokens = [t.strip() for t in cleaned.split() if len(t.strip()) >= 2]
    return " ".join(tokens)


def build_broad_query_from_profile(profile: GroupProfile) -> str:
    """Short domain-level query for vector/keyword recall."""
    need_raw = str(getattr(profile.need, "value", "") or "").strip()
    doing_raw = str(getattr(profile.doing, "value", "") or "").strip()
    offer_raw = str(getattr(profile.offer, "value", "") or "").strip()

    need_head = _NEED_HEAD_SPLIT.split(need_raw, maxsplit=1)[0].strip() if need_raw else ""
    need_clean = _clean_query_keywords(need_head or need_raw)
    doing_clean = _clean_query_keywords(doing_raw)
    offer_clean = _clean_query_keywords(offer_raw)

    # Prioritize need keywords first for inverse matchmaking recall
    parts = [p for p in (need_clean, doing_clean, offer_clean) if p]
    if parts:
        res = " ".join(parts)
        if len(res) <= 60:
            return res
        return res[:60].strip()

    need = _compact(need_head or need_raw, 36)
    doing = _compact(doing_raw, 36)
    parts = [p for p in (need, doing) if p]
    return " ".join(parts) if parts else _compact(need_raw or doing_raw, 60)


def build_rank_query_from_profile(profile: GroupProfile) -> str:
    """Detailed need(+doing) text used only for post-retrieval ranking."""
    need = str(getattr(profile.need, "value", "") or "").strip()
    doing = str(getattr(profile.doing, "value", "") or "").strip()
    joined = " ".join(p for p in (need, doing) if p)
    return joined[:500]


def _char_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", (text or "").lower())
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def detail_overlap_score(rank_query: str, candidate_text: str) -> float:
    """Fraction of fine-need bigrams covered by candidate text (asymmetric).

    Coverage works better than Jaccard when ``rank_query`` is long and the
    candidate bio is short: we care how much of the user's need is reflected,
    not how much unrelated bio text dilutes the union.
    """
    left = _char_bigrams(rank_query)
    right = _char_bigrams(candidate_text)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def rerank_candidates_by_detail(
    candidates: list[dict[str, Any]],
    *,
    rank_query: str,
    hybrid_weight: float = 0.4,
) -> list[dict[str, Any]]:
    """Blend hybrid match_score with fine-need coverage; stable on ties.

    Default hybrid_weight=0.4 so a clearly-on-need candidate can outrank a
    higher hybrid score that only matched the broad domain.
    """
    rq = (rank_query or "").strip()
    if not rq or not candidates:
        return list(candidates)

    detail_w = max(0.0, min(1.0, 1.0 - hybrid_weight))
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, cand in enumerate(candidates):
        doing = cand.get("doing")
        if isinstance(doing, dict):
            body = str(doing.get("value") or "")
        else:
            body = str(doing or "")
        body = " ".join(
            [
                body,
                str(cand.get("display_name") or ""),
                str(cand.get("reason_summary") or ""),
            ]
        )
        try:
            hybrid = float(cand.get("match_score") or 0.0)
        except (TypeError, ValueError):
            hybrid = 0.0
        detail = detail_overlap_score(rq, body)
        blended = hybrid_weight * hybrid + detail_w * detail
        enriched = dict(cand)
        enriched["hybrid_score"] = round(hybrid, 3)
        enriched["match_score"] = round(blended, 3)
        enriched["rank_detail_score"] = round(detail, 3)
        scored.append((blended, idx, enriched))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in scored]


# Process-wide default stub (tests may replace)
_DEFAULT_STUB = MatchStub()



def get_match_stub() -> MatchStub:
    return _DEFAULT_STUB


def set_match_stub(stub: MatchStub | None) -> None:
    global _DEFAULT_STUB
    _DEFAULT_STUB = stub if stub is not None else MatchStub()
