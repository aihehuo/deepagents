"""Broad recall query + fine need re-ranking."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.match_stub import (
    build_broad_query_from_profile,
    build_rank_query_from_profile,
    detail_overlap_score,
    rerank_candidates_by_detail,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat


def test_broad_query_drops_fine_grained_need_tail() -> None:
    profile = profile_from_flat(
        user_id="1",
        group_id="763",
        doing="做面向小学生的AI数学辅导工具，核心功能为作业题讲解与错题原因分析",
        need=(
            "缺懂教研内容体系的合伙人，尤其有线下教培机构经验；"
            "当前卡点在错因的精细化分类（如概念不清的子类型）、"
            "针对性教学建议生成以及将错因直接转化为可执行的练习题设计"
        ),
        offer="具备NLP和LLM开发能力，已积累500名种子家长用户",
    )
    broad = build_broad_query_from_profile(profile)
    rank = build_rank_query_from_profile(profile)

    assert "变式题" not in broad
    assert "精细化分类" not in broad
    assert "小学" in broad or "数学" in broad or "教研" in broad
    assert "精细化分类" in rank
    assert len(broad) < len(rank)


def test_rerank_prefers_detail_overlap() -> None:
    rank_query = "小学数学教研 错因归因 教学建议 练习题设计"
    cands = [
        {
            "user_id": "1",
            "display_name": "A",
            "match_score": 0.9,
            "doing": {"value": "跨境电商供应链 SaaS", "disclosure": "confirmed_public"},
        },
        {
            "user_id": "2",
            "display_name": "B",
            "match_score": 0.5,
            "doing": {
                "value": "小学数学教研，做过错因归因和靶向练习设计",
                "disclosure": "confirmed_public",
            },
        },
    ]
    ranked = rerank_candidates_by_detail(cands, rank_query=rank_query)
    assert ranked[0]["user_id"] == "2"
    assert ranked[0]["rank_detail_score"] >= ranked[1]["rank_detail_score"]


def test_detail_overlap_score_monotonic() -> None:
    q = "小学数学教研错因"
    assert detail_overlap_score(q, "小学数学教研错因归因") > detail_overlap_score(
        q, "跨境物流仓配"
    )
