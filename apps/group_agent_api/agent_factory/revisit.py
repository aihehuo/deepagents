"""Revisit context from Micro-trusted call_async metadata (REQ-028 / TSD-03).

Only Micro Path A injection is authoritative. Browser metadata must not forge
these fields; AsyncCallRequest validates shape, and Micro strips client copies
before re-injecting server values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MAX_PRIOR_CANDIDATE_IDS = 100
MAX_CANDIDATE_NAME_LEN = 64
MAX_CANDIDATE_NAMES = 5
MAX_TOPIC_SUMMARY_LEN = 256
MAX_ID_LEN = 64


@dataclass(frozen=True)
class RevisitHint:
    has_prior_invite: bool = False
    candidate_names: tuple[str, ...] = ()
    topic_summary: str | None = None


def normalize_prior_candidate_ids(raw: Any) -> list[str]:
    """Normalize prior ids to unique non-empty strings (order preserved)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw[:MAX_PRIOR_CANDIDATE_IDS]:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            continue
        value = str(item).strip()
        if not value or len(value) > MAX_ID_LEN or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def parse_revisit_hint(raw: Any) -> RevisitHint:
    """Parse revisit_hint; unknown/malformed shapes fail closed to no-op."""
    if not isinstance(raw, dict):
        return RevisitHint()
    has_prior = bool(raw.get("has_prior_invite"))
    names_raw = raw.get("candidate_names") or []
    names: list[str] = []
    if isinstance(names_raw, list):
        for item in names_raw[:MAX_CANDIDATE_NAMES]:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if not name or len(name) > MAX_CANDIDATE_NAME_LEN:
                continue
            names.append(name)
    topic_raw = raw.get("topic_summary")
    topic: str | None = None
    if isinstance(topic_raw, str):
        topic = topic_raw.strip()[:MAX_TOPIC_SUMMARY_LEN] or None
    return RevisitHint(
        has_prior_invite=has_prior,
        candidate_names=tuple(names),
        topic_summary=topic,
    )


@dataclass(frozen=True)
class PriorCandidateFact:
    field: str
    value: str
    disclosure: str = "match_only"
    source_type: str = "group_agent_profile"


@dataclass(frozen=True)
class PriorCandidate:
    user_id: str
    display_name: str
    same_group: bool = False
    connection_status: str = "not_requested"
    facts: tuple[PriorCandidateFact, ...] = ()


@dataclass(frozen=True)
class PriorRecommendation:
    artifact_run_id: str | None = None
    artifact_digest: str | None = None
    candidates: tuple[PriorCandidate, ...] = ()


def parse_prior_recommendation(raw: Any) -> PriorRecommendation | None:
    """Parse trusted prior_recommendation injected by Micro (TSD-13 §5.4)."""
    if not isinstance(raw, dict):
        return None
    run_id = str(raw.get("artifact_run_id") or "").strip() or None
    digest = str(raw.get("artifact_digest") or "").strip() or None
    raw_cands = raw.get("candidates") or []
    cands: list[PriorCandidate] = []
    if isinstance(raw_cands, list):
        for c in raw_cands[:5]:
            if not isinstance(c, dict):
                continue
            uid = str(c.get("user_id") or "").strip()
            if not uid:
                continue
            dname = str(c.get("display_name") or "候选人").strip()
            same_g = bool(c.get("same_group"))
            conn_status = str(c.get("connection_status") or "not_requested").strip()
            raw_facts = c.get("facts") or []
            facts: list[PriorCandidateFact] = []
            if isinstance(raw_facts, list):
                for f in raw_facts[:12]:
                    if isinstance(f, dict):
                        f_field = str(f.get("field") or "").strip()
                        f_val = str(f.get("value") or "").strip()
                        if f_field and f_val:
                            facts.append(
                                PriorCandidateFact(
                                    field=f_field,
                                    value=f_val,
                                    disclosure=str(f.get("disclosure") or "match_only"),
                                    source_type=str(f.get("source_type") or "group_agent_profile"),
                                )
                            )
            cands.append(
                PriorCandidate(
                    user_id=uid,
                    display_name=dname,
                    same_group=same_g,
                    connection_status=conn_status,
                    facts=tuple(facts),
                )
            )
    return PriorRecommendation(
        artifact_run_id=run_id,
        artifact_digest=digest,
        candidates=tuple(cands),
    )


def parse_revisit_from_metadata(metadata: dict[str, Any] | None) -> tuple[list[str], RevisitHint]:
    meta = metadata or {}
    return (
        normalize_prior_candidate_ids(meta.get("prior_candidate_ids")),
        parse_revisit_hint(meta.get("revisit_hint")),
    )


def excluded_ids_for_match(
    user_id: str,
    metadata: dict[str, Any] | None = None,
    *,
    prior_candidate_ids: list[str] | None = None,
) -> list[str]:
    """Self + Micro prior ids. Exclude strategy (not demote) — matches new_api."""
    priors = (
        list(prior_candidate_ids)
        if prior_candidate_ids is not None
        else normalize_prior_candidate_ids((metadata or {}).get("prior_candidate_ids"))
    )
    out: list[str] = []
    seen: set[str] = set()
    for value in [str(user_id).strip(), *priors]:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def build_revisit_opener(hint: RevisitHint | None) -> str | None:
    """PRC-01 §9.2 first-sentence revisit branch; None when no prior invite."""
    if hint is None or not hint.has_prior_invite:
        return None
    if hint.candidate_names:
        who = "、".join(hint.candidate_names)
        base = f"上次我给你推荐过{who}"
    else:
        base = "上次我已经给你做过一轮推荐"
    if hint.topic_summary:
        base += f"（围绕「{hint.topic_summary}」）"
    return (
        f"{base}。对方有回音吗？"
        "若没有，要换人、换题，还是开新一轮？"
    )


_REMATCH_INTENT = re.compile(
    r"(换人|换一批|换题|再找|重新推荐|重新匹配|再匹配|另找|开新一轮.*找|"
    r"开始搜|开始匹配|开始找人|先匹配|继续匹配|帮我匹配|帮我找人|"
    r"\bgo\b|\brematch\b|\bmatch\s*(?:again|now)?\b)"
    r"|^(?:开始|开始吧)$",
    re.IGNORECASE,
)


def wants_rematch(message: str | None) -> bool:
    """True when the user explicitly asks to rematch / change candidates."""
    text = (message or "").strip()
    if not text:
        return False
    return bool(_REMATCH_INTENT.search(text))


def known_match_system_content(
    hint: RevisitHint | None,
    prior_rec: PriorRecommendation | None = None,
) -> str | None:
    """Dialogue reminder: candidates/invite already shown in the client UI.

    Match cards are produced after the LLM turn and pushed as separate WS
    kinds — they never enter LangGraph history. Without this injection the
    model falsely denies that recommendations were delivered.
    """
    if prior_rec and prior_rec.candidates:
        cand_lines = []
        for c in prior_rec.candidates:
            group_tag = "同群" if c.same_group else "异群"
            facts_desc = "；".join(f"{f.field}: {f.value}" for f in c.facts) if c.facts else "暂未提供其他公开事实"
            cand_lines.append(f"- 候选人【{c.display_name}】（{group_tag}，对接状态：{c.connection_status}）：{facts_desc}")

        lines = [
            "【系统已向用户界面交付的历史推荐事实——来自权威数据，严禁编造任何其他履历或数字】",
            *cand_lines,
            "规则：",
            "1. 用户追问候选人背景时：严格仅从上述已知事实回答；未知问题固定回答「现有资料未说明，可申请对接后向本人确认」。",
            "2. 用户问申请/对接进度时：仅如实说明上述对接状态，禁止声称「已发送/已通知管理员/对方已同意」。",
            "3. 用户说 go / 再匹配 / 换人 时：同意重新匹配。",
        ]
        return "\n".join(lines)

    if hint is None or not hint.has_prior_invite:
        return None
    if hint.candidate_names:
        who = "、".join(hint.candidate_names)
        people = f"已向用户展示候选人：{who}。"
    else:
        people = "已向用户展示过一轮推荐（候选人卡片 + 邀请词）。"
    lines = [
        "【系统已向用户界面交付的匹配结果——来自 Micro 权威 revisit_hint，不是模型猜测】",
        f"- {people}",
    ]
    if hint.topic_summary:
        lines.append(f"- 话题/邀请主题：「{hint.topic_summary}」。")
    lines.extend(
        [
            "规则：",
            "1. 用户问「怎么联系 / how to connect / 推荐了谁」时：必须承认已推荐，"
            "引导用户使用界面里的定向邀请词（复制到群里 @ 候选人），"
            "或按触达提示申请加入候选人所在群；禁止说「还没有推荐 / no recommendations」。",
            "2. 用户说 go / 再匹配 / 换人 时：同意重新匹配，不要假装上一轮没出结果。",
            "3. 不要编造新的候选人姓名或联系方式；以界面已展示的卡片为准。",
        ]
    )
    return "\n".join(lines)


def should_skip_auto_match(
    *,
    revisit_hint: RevisitHint | None,
    message: str | None,
) -> bool:
    """REQ-028: with prior invite, do not force rematch unless user asks."""
    if revisit_hint is None or not revisit_hint.has_prior_invite:
        return False
    return not wants_rematch(message)
