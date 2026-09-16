"""L0 deterministic short-circuits for reply grounding (TSD-14 §4.6.6)."""

from __future__ import annotations

import re

from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    RepairableBy,
    ReplyGroundingInput,
    ReplyGroundingOutput,
    Verdict,
)
from apps.group_agent_api.agent_factory.content_quality import (
    looks_like_invented_candidate_narrative,
    looks_like_unauthorized_action_claim,
)

# Concrete person / resume style when no adopted candidates this turn.
_NAMED_PERSON_STYLE = re.compile(
    r"(?:候选人|人选|伙伴|合伙人)[【\[][^】\]]+[】\]]|"
    r"找到(?:了一?)?(?:一位|一名|一个).{0,12}(?:人|人选|候选人|伙伴)|"
    r"匹配到(?:了一?)?(?:一位|一名|一个)|"
    r"(?:他|她)(?:曾|有|的).{0,20}(?:年|项目|经验|主导)"
)

_COUNT_CLAIM = re.compile(
    r"(?:找到|匹配(?:到)?|推荐)\s*(\d+)\s*位|"
    r"本群已有\s*(\d+)\s*位|"
    r"(\d+)\s*位值得"
)

_PUBLIC_GROUP_CLAIM = re.compile(r"(?:本群|当前群|群内).{0,12}(?:成员|人才|人选|匹配)|匹配到|@[^\s@，。,.！!？?]{1,32}")  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
_PUBLIC_GROUP_NEGATION = re.compile(r"(?:不证明|不代表|无法确认|并非|不是|不属于|没有|未完成|尚未|不能)[^。；，,但]{0,12}$")  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.


def _public_group_span(reply: str) -> str:
    for match in _PUBLIC_GROUP_CLAIM.finditer(reply):
        prefix = re.split(r"[。；;\n]", reply[: match.start()])[-1][-24:]  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
        if _PUBLIC_GROUP_NEGATION.search(prefix):
            continue
        return match.group(0)[:80]
    return ""


def run_l0(payload: ReplyGroundingInput) -> ReplyGroundingOutput | None:
    """Return a fail output when L0 can decide without LLM; else None."""
    reply = (payload.reply or "").strip()
    if not reply:
        return ReplyGroundingOutput(
            verdict=Verdict.pass_,
            codes=[],
            spans=[],
            repairable_by=RepairableBy.llm,
            message="",
            layer="l0",
        )

    count = int(payload.ground.candidate_count or 0)
    adopted = len(payload.ground.candidates or [])
    effective_count = min(count, adopted) if adopted else count
    public_people = [record for item in payload.ground.public_reads if item.resource in {"users", "group-members"} for record in item.records]
    # Names present in this-turn public_reads are supported without pasting
    # source URLs into the user-visible reply (URLs live in ground/debug only).
    supported_public_person = any(
        isinstance(record.get("name"), str) and bool(record["name"]) and record["name"] in reply
        for record in public_people
    )
    public_group_span = _public_group_span(reply)
    if payload.ground.public_reads and not adopted and public_group_span:
        return ReplyGroundingOutput(
            verdict=Verdict.fail,
            codes=["unsupported_claim"],
            spans=[public_group_span],
            repairable_by=RepairableBy.llm,
            message="公网资料不证明属于当前群、可 @ 或已完成群内匹配。只引用公开事实及来源。",
            layer="l0",
        )

    # Invented people when this turn adopted nobody.
    if effective_count <= 0 and not supported_public_person and (looks_like_invented_candidate_narrative(reply) or _NAMED_PERSON_STYLE.search(reply)):
        span = _first_span(reply, _NAMED_PERSON_STYLE) or "具体人选叙述"
        return ReplyGroundingOutput(
            verdict=Verdict.fail,
            codes=["invented_candidate"],
            spans=[span],
            repairable_by=RepairableBy.llm,
            message=("本轮无采用候选人，reply 却出现具体人选或履历叙述。删掉人选/履历，只保留澄清或空结果说明。"),
            layer="l0",
        )

    # Count mismatch vs adopted candidates.
    claimed = _claimed_count(reply)
    if claimed is not None and effective_count > 0 and claimed != effective_count:
        return ReplyGroundingOutput(
            verdict=Verdict.fail,
            codes=["unsupported_claim"],
            spans=[f"声称{claimed}位但本轮采用{effective_count}位"],
            repairable_by=RepairableBy.llm,
            message=(f"reply 声称 {claimed} 位人选，但本轮 ground.candidate_count={effective_count}。改成与采用人数一致，或去掉人数断言。"),
            layer="l0",
        )

    # Completion claims without receipts.
    if looks_like_unauthorized_action_claim(reply) and not payload.ground.receipts:
        return ReplyGroundingOutput(
            verdict=Verdict.fail,
            codes=["unverified_action"],
            spans=[_first_span(reply, None) or "已发送/已通知类完成态"],
            repairable_by=RepairableBy.llm,
            message=("reply 含完成态动作表述，但本轮 receipts 为空。改为建议用户手动复制发送，勿声称已发群或已通知管理员。"),
            layer="l0",
        )

    return None


def _claimed_count(reply: str) -> int | None:
    match = _COUNT_CLAIM.search(reply or "")
    if not match:
        return None
    for group in match.groups():
        if group is not None:
            try:
                return int(group)
            except ValueError:
                return None
    return None


def _first_span(reply: str, pattern: re.Pattern[str] | None) -> str:
    if pattern is not None:
        match = pattern.search(reply or "")
        if match:
            return match.group(0)[:80]
    # Fallback: first non-empty line snippet
    for line in (reply or "").splitlines():
        text = line.strip()
        if text:
            return text[:80]
    return ""
