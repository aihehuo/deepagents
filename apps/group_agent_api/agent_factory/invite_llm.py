"""LLM polish for invite copy (REQ-007). Falls back to template on failure."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from apps.group_agent_api.agent_factory.disclosure import stable_user_id_value
from apps.group_agent_api.agent_factory.integrations.config import llm_polish_enabled
from apps.group_agent_api.agent_factory.invite_copy import (
    InviteResult,
    assert_directed_invite,
    assert_undirected_invite,
    generate_invite_copy,
)
from apps.group_agent_api.agent_factory.guard import extract_at_identities
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile

_logger = logging.getLogger("uvicorn.error")

_DIRECTED_POLISH_PROMPT = """你是群内智能体文案润色器。在保留硬约束的前提下，把「脚手架草稿」润色成更自然的第一人称中文群消息。

硬约束（违反即不合格）：
1. 必须保留五要素语义：我是谁+在做什么 / 已有资源 / 共同话题 / 为什么邀请这几位（含原有 @姓名）/ 低压力邀请
2. 保留全部 @姓名，不得增删换人；不得编造候选
3. 匹配理由必须含不确定性（如「不一定合适」「值得聊一次以确认」），禁止「结论：合适」
4. 可自然表达找合伙人或合伙意向（如有需要）
5. 只用对方已给出的公开 doing 信息，不要发明对方的 need/offer/资金等
6. 输出纯正文，不要 markdown 标题或解释

请只输出润色后的完整邀请词正文。
"""

_UNDIRECTED_POLISH_PROMPT = """你是群内智能体文案润色器。润色「不点名群话题」草稿。

硬约束：
1. 不要出现任何 @ 或点名对象
2. 保留：我是谁+在做什么、具体话题、开放邀请
3. 可自然表达交流或合伙意向
4. 禁止空泛「想认识一下/交流交流」
5. 输出纯正文

请只输出润色后的完整正文。
"""

_ELEMENT_PREFIXES = (
    "我在做的项目：",
    "我在做的项目",
    "我在做的具体项目",
    "我在做",
    "我能提供的资源或能力：",
    "我能提供的资源或能力",
    "我能提供的具体资源或能力",
    "目前手上有：",
    "目前手上有",
    "想聊聊：",
    "想聊聊",
)


def _extract_text_response(content: Any) -> str:
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


def _core_snippet(element: str) -> str:
    """Distinctive core of a scaffold element for presence checks in polished text."""
    text = (element or "").strip()
    for prefix in _ELEMENT_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return text


def assert_elements_present_in_text(
    *,
    text: str,
    elements: dict[str, str],
) -> list[str]:
    """Verify five-element semantics still appear *in the polished text* (not scaffold dict)."""
    violations: list[str] = []
    body = text or ""
    for key in ("who_doing", "resources", "topic", "low_pressure"):
        raw = (elements.get(key) or "").strip()
        if not raw:
            violations.append(f"missing_element:{key}")
            continue
        core = _core_snippet(raw)
        needle = core if len(core) >= 2 else raw
        if needle and needle not in body and raw not in body:
            violations.append(f"polished_missing_element:{key}")
    # why_invite: require uncertainty + at least one @ in polished text itself
    why = (elements.get("why_invite") or "").strip()
    if not why:
        violations.append("missing_element:why_invite")
    return violations


def assert_exact_polished_mentions(
    *,
    text: str,
    expected_user_ids: list[str],
) -> list[str]:
    """Require no missing, added, or duplicate @ identities after polish."""
    actual = extract_at_identities(text)
    violations: list[str] = []
    if any(stable_user_id_value(user_id) is None for user_id in expected_user_ids):
        violations.append("polished_invalid_expected_mention_id")
    if any(stable_user_id_value(user_id) is None for user_id in actual):
        violations.append("polished_invalid_actual_mention_id")
    if violations:
        return violations
    expected = list(expected_user_ids)
    if len(actual) != len(set(actual)):
        violations.append("polished_duplicate_mentions")
    if set(expected) - set(actual):
        violations.append("polished_missing_mentions")
    if set(actual) - set(expected):
        violations.append("polished_added_mentions")
    if len(actual) != len(expected) and not violations:
        violations.append("polished_mention_count_mismatch")
    return violations


def _polish_with_llm(*, kind: str, draft: str, model: Any | None) -> str | None:
    if model is None or not draft.strip():
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        system = (
            _DIRECTED_POLISH_PROMPT if kind == "directed" else _UNDIRECTED_POLISH_PROMPT
        )
        msg = model.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=f"草稿：\n{draft}"),
            ]
        )
        text = _extract_text_response(getattr(msg, "content", None))
        # Strip accidental fences
        text = re.sub(r"^```(?:\w+)?\n|\n```$", "", text).strip()
        return text or None
    except Exception as exc:  # noqa: BLE001
        _logger.warning("action=invite_llm_polish_failed error=%s", exc)
        return None


def generate_invite_with_optional_llm(
    *,
    profile: GroupProfile,
    candidates: list[dict[str, Any]],
    match_status: str,
    willing_to_at: bool,
    user_id: str = "",
    group_id: str = "",
    model: Any | None = None,
    use_llm: bool | None = None,
    _broken_first_draft: bool = False,
) -> InviteResult:
    """Template scaffold (+ assert) then optional LLM polish (+ re-assert on text)."""
    base = generate_invite_copy(
        profile=profile,
        candidates=candidates,
        match_status=match_status,
        willing_to_at=willing_to_at,
        user_id=user_id,
        group_id=group_id,
        _broken_first_draft=_broken_first_draft,
    )
    if not base.ok or not base.text:
        return base

    enabled = llm_polish_enabled() if use_llm is None else use_llm
    mode_stub = (os.environ.get("GROUP_AGENT_MODEL_MODE") == "stub") or (os.environ.get("GROUP_AGENT_INTEGRATION") == "stub")
    if not enabled or model is None or mode_stub or hasattr(model, "responses"):
        return base

    polished = _polish_with_llm(kind=base.kind, draft=base.text, model=model)
    if not polished:
        return base

    if base.kind == "directed":
        scaffold = dict(base.elements or {})
        # Gate: each scaffold element core must still appear in polished text.
        presence = assert_elements_present_in_text(text=polished, elements=scaffold)
        presence.extend(
            assert_exact_polished_mentions(
                text=polished,
                expected_user_ids=base.mentioned_user_ids,
            )
        )
        missing_keys = {
            v.split(":", 1)[1]
            for v in presence
            if v.startswith("polished_missing_element:")
            or v.startswith("missing_element:")
        }
        elements_for_assert = {
            "who_doing": "" if "who_doing" in missing_keys else scaffold.get("who_doing", ""),
            "resources": "" if "resources" in missing_keys else scaffold.get("resources", ""),
            "topic": "" if "topic" in missing_keys else scaffold.get("topic", ""),
            # Uncertainty / @ checked against full polished body
            "why_invite": "" if "why_invite" in missing_keys else polished,
            "low_pressure": ""
            if "low_pressure" in missing_keys
            else scaffold.get("low_pressure", ""),
        }
        violations = list(presence)
        for item in assert_directed_invite(
            text=polished,
            elements=elements_for_assert,
            candidates=candidates,
        ):
            if item not in violations:
                violations.append(item)
    else:
        violations = assert_undirected_invite(text=polished)

    if violations:
        _logger.warning(
            "action=invite_llm_polish_rejected violations=%s → template",
            ",".join(violations),
        )
        return base

    return InviteResult(
        kind=base.kind,
        text=polished,
        topic=base.topic,
        match_status=base.match_status,
        willing_to_at=base.willing_to_at,
        mentioned_user_ids=base.mentioned_user_ids,
        elements=base.elements,
        honest_note=base.honest_note,
        ok=True,
        violations=[],
        assert_attempts=base.assert_attempts,
    )
