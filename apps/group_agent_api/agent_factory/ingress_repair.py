"""Mouth ingress reject → brain repair helpers (BSD-01 P1 / TSD-14 §7.5)."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

_logger = logging.getLogger("uvicorn.error")

# First emit + one re-prompt (BSD §3.1.3 / §10 P1).
MOUTH_INGRESS_MAX_ATTEMPTS = 2

# reason_code → (class, repairable_by, hint)
_REJECT_META: dict[str, tuple[str, str, str]] = {
    "dialogue_mode_has_candidates": (
        "consistency",
        "orchestrator",
        "清空 candidates，保持 dialogue，不要带人上卡",
    ),
    "dialogue_mode_matched_status": (
        "consistency",
        "orchestrator",
        "match.status 改为 empty，并清空 candidates",
    ),
    "empty_status_has_candidates": (
        "consistency",
        "orchestrator",
        "清空 candidates，保持 empty/no_match",
    ),
    "candidate_count_mismatch": (
        "consistency",
        "orchestrator",
        "对齐 candidate_count 与 candidates 数组长度，或清空候选",
    ),
    "recommendation_candidates_empty": (
        "consistency",
        "orchestrator",
        "改为 no_match/dialogue，勿声称找到人",
    ),
    "matched_candidates_empty": (
        "consistency",
        "orchestrator",
        "改为 empty，勿保留 matched 空数组",
    ),
    "global_group_leak": (
        "leak",
        "orchestrator",
        "去掉 global 群引用或整条候选",
    ),
    "forbidden_fact_disclosure": (
        "leak",
        "orchestrator",
        "去掉禁披露 fact 或整条候选",
    ),
    "sensitive_contact_in_fact": (
        "leak",
        "orchestrator",
        "去掉含联系方式的 fact 或整条候选",
    ),
    "unverified_fact_source": (
        "truth",
        "orchestrator",
        "去掉无 GroupAgentProfile 的候选；可改为 no_match 文案",
    ),
    "fact_value_mismatch": (
        "truth",
        "orchestrator",
        "删除不符 facts 或整条候选，勿扩写履历",
    ),
    "fact_not_in_profile": (
        "truth",
        "orchestrator",
        "删除库中不存在的 fact 字段",
    ),
    "fact_version_mismatch": (
        "truth",
        "orchestrator",
        "对齐 source_version 或去掉该候选",
    ),
    "candidate_facts_empty": (
        "truth",
        "orchestrator",
        "recommendation 必须有权威 facts；否则改 no_match",
    ),
    "unverified_action_status": (
        "truth",
        "llm",
        "改口为 dialogue，禁止自称已发群/已通知",
    ),
    "protocol_mode_mismatch": ("consistency", "none", "协议模式不匹配，勿盲重放"),
    "protocol_crossover_rejected": ("consistency", "none", "协议交叉，勿盲重放"),
    "unknown_protocol_mode": ("consistency", "none", "未知协议模式"),
    "internal_error": ("transport", "none", "口内侧异常，勿盲重放同一稿"),
}


def reject_meta(reason_code: str) -> tuple[str, str, str]:
    code = (reason_code or "").strip()
    return _REJECT_META.get(code, ("truth", "llm", "按 reason_code 修订后再提，禁止假候选"))


def format_ingress_deny(
    *,
    reason_code: str,
    message: str = "",
    hint: str = "",
    repairable_by: str = "llm",
    reject_class: str = "truth",
    fields: list[str] | None = None,
    attempt: int = 1,
    max_attempts: int = MOUTH_INGRESS_MAX_ATTEMPTS,
) -> str:
    """Assemble ctx.ingress_reject block (agent-only; never user-visible)."""
    fields = fields or []
    fields_s = "; ".join(fields) if fields else "(none)"
    msg = (message or "").strip() or reason_code
    hint_s = (hint or "").strip() or "修订 payload 后以同一 seq 重提 final"
    return (
        f'<micro_ingress_reject attempt="{attempt}" max="{max_attempts}" '
        f'class="{reject_class}" code="{reason_code}" '
        f'repairable_by="{repairable_by}">\n'
        f"上一份 callback.final 未通过口检，未提交 seq、未写助手气泡。\n"
        f"原因: {msg}\n"
        f"fields: {fields_s}\n"
        f"可做: {hint_s}\n"
        f"不可做: 编造候选人履历；对用户展示 reason_code；声称已发群/已通知。\n"
        f"</micro_ingress_reject>"
    )


def peel_final_payload(payload: dict[str, Any], *, reason_code: str) -> dict[str, Any]:
    """Deterministic orchestrator peel for consistency/truth/leak rejects."""
    out = dict(payload)
    code = (reason_code or "").strip()

    def _clear_candidates() -> None:
        out["candidates"] = []
        out["match_status"] = "empty"
        out["invite_text"] = None
        out["delivery_kind"] = None
        out["invite_ok"] = False
        out["mentioned_user_ids"] = []
        out["at_users"] = []
        match = out.get("match")
        if isinstance(match, dict):
            match = dict(match)
            match["status"] = "empty"
            match["candidates"] = []
            match["candidate_count"] = 0
            out["match"] = match
        grounded = out.get("grounded_final")
        if isinstance(grounded, dict):
            grounded = dict(grounded)
            m2 = grounded.get("match")
            if isinstance(m2, dict):
                m2 = dict(m2)
                m2["status"] = "empty"
                m2["candidates"] = []
                m2["candidate_count"] = 0
                grounded["match"] = m2
            grounded["candidates"] = []
            out["grounded_final"] = grounded

    peelable = {
        "dialogue_mode_has_candidates",
        "dialogue_mode_matched_status",
        "empty_status_has_candidates",
        "candidate_count_mismatch",
        "recommendation_candidates_empty",
        "matched_candidates_empty",
        "global_group_leak",
        "forbidden_fact_disclosure",
        "sensitive_contact_in_fact",
        "unverified_fact_source",
        "fact_value_mismatch",
        "fact_not_in_profile",
        "fact_version_mismatch",
        "candidate_facts_empty",
        "fact_disclosure_mismatch",
        "legacy_fact_source",
        "missing_fact_source_version",
        "invalid_fact_source_version",
        "invalid_fact_source_type",
        "invalid_fact_source_group_id",
        "fact_source_not_active",
        "candidate_equals_current_user",
        "duplicate_candidate_user_id",
    }
    if code in peelable or code.startswith("fact_") or code.startswith("candidate_"):
        _clear_candidates()
        reply_mode = str(out.get("reply_mode") or "")
        if reply_mode in {"recommendation", "action_status"}:
            out["reply_mode"] = "no_match"
        if code.startswith("dialogue_"):
            out["reply_mode"] = "dialogue"

    if code == "unverified_action_status":
        out["reply_mode"] = "dialogue"
        out.pop("action_status", None)
        out.pop("referral_status", None)
        out.pop("action", None)

    return out


def build_abandon_final_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Safe last-chance draft: no candidates, dialogue-only; keep protocol keys."""
    out = peel_final_payload(payload, reason_code="recommendation_candidates_empty")
    keep = {
        k: payload.get(k)
        for k in (
            "protocol_version",
            "run_id",
            "protocol_mode",
            "rollout_version",
            "profile_persisted",
            "profile_status",
            "capability",
            "capability_source",
        )
        if k in payload
    }
    out.update(keep)
    out["reply_mode"] = "dialogue"
    out["dialogue_kind"] = "capability_boundary"
    out["dialogue_text"] = (
        "这次推荐结果没有通过事实校验，我没有展示未经确认的信息。"
        "你可以稍后补充需求再试。"
    )
    out["reply"] = out["dialogue_text"]
    out["text"] = out["dialogue_text"]
    out["message"] = out["dialogue_text"]
    out["candidates"] = []
    out["match_status"] = "empty"
    out["invite_text"] = None
    out["suggested_replies"] = []
    out.pop("match", None)
    out.pop("invite", None)
    out.pop("grounding", None)
    out.pop("grounded_final", None)
    return out


def apply_mouth_repair(
    payload: dict[str, Any],
    *,
    reject: Any,
    model: Any | None = None,
    attempt: int = 1,
) -> dict[str, Any]:
    """Peel + optional LLM rewrite of user-visible reply. Returns new payload."""
    reason = getattr(reject, "reason_code", None) or "mouth_rejected"
    repairable = getattr(reject, "repairable_by", None) or reject_meta(reason)[1]
    reject_class = getattr(reject, "reject_class", None) or reject_meta(reason)[0]
    message = getattr(reject, "message", None) or ""
    hint = getattr(reject, "hint", None) or reject_meta(reason)[2]
    fields = list(getattr(reject, "fields", None) or [])

    if repairable == "none":
        return build_abandon_final_payload(payload)

    repaired = peel_final_payload(payload, reason_code=reason)
    deny = format_ingress_deny(
        reason_code=reason,
        message=message,
        hint=hint,
        repairable_by=repairable,
        reject_class=reject_class,
        fields=fields,
        attempt=attempt,
    )

    previous = str(repaired.get("reply") or repaired.get("text") or "").strip()
    if model is not None and previous and repairable in {"llm", "orchestrator"}:
        rewritten = _llm_rewrite_reply(
            model=model,
            deny_block=deny,
            previous_reply=previous,
            payload=repaired,
        )
        if rewritten:
            repaired["reply"] = rewritten
            repaired["text"] = rewritten
            repaired["message"] = rewritten
            if isinstance(repaired.get("grounded_final"), dict):
                gf = dict(repaired["grounded_final"])
                # Prefer dialogue/no_match text after peel; drop typed free text conflicts
                if repaired.get("reply_mode") in {"dialogue", "no_match", "error"}:
                    gf.pop("match", None)
                    gf.pop("invite", None)
                repaired["grounded_final"] = gf

    _logger.info(
        "action=mouth_ingress_repair reason_code=%s repairable_by=%s attempt=%s",
        reason,
        repairable,
        attempt,
    )
    return repaired


def _llm_rewrite_reply(
    *,
    model: Any,
    deny_block: str,
    previous_reply: str,
    payload: dict[str, Any],
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    system = (
        "你是群智能体主对话模型的口检改口步骤。"
        "上一份 callback.final 被 Micro 口检拒绝，必须按 micro_ingress_reject 修改后"
        "只输出新的用户可见正文。\n"
        "硬约束：不得编造候选人履历/数字/项目；不得声称已发群/已通知；"
        "若 payload 已无候选人，只能写追问或「暂未展示人选」类表述；"
        "不要输出 JSON 或标签。"
    )
    safe_payload = {
        "reply_mode": payload.get("reply_mode"),
        "match_status": payload.get("match_status"),
        "candidate_count": len(payload.get("candidates") or []),
    }
    human = (
        f"{deny_block}\n\n"
        f"payload_summary: {json.dumps(safe_payload, ensure_ascii=False)}\n\n"
        f"previous_reply:\n{previous_reply}\n\n"
        "请只输出改口后的完整正文。"
    )
    try:
        msg = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("action=mouth_ingress_llm_repair_failed error=%s", exc)
        return ""
    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts).strip()
    return str(content).strip()


# Type alias for tests / callers
MouthRepairFn = Callable[[dict[str, Any], Any], dict[str, Any]]
