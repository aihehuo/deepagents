"""L1 LLM semantic judge for reply grounding (no tools)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    ALLOWED_CODES,
    RepairableBy,
    ReplyGroundingInput,
    ReplyGroundingOutput,
    Verdict,
    schema_invalid_output,
)

_logger = logging.getLogger("uvicorn.error")

_JUDGE_SYSTEM_PROMPT = """你是群智能体「推荐文案事实对照」裁判。只根据给定 ground 判定 reply，不得用常识补履历。

判定：
- pass：reply 只复述或合理压缩 ground 中已有事实；闲聊/追问且无具体人选履历也可 pass
- fail：出现 ground 中不存在的经历/数字/项目/关系（unsupported_claim）；在已有事实上语义拔高（exaggeration）；无 receipts 却写完成态动作（unverified_action）；无候选却叙述具体人选（invented_candidate）

特别注意：
- 「高度契合 / 非常适合 / 完美匹配」等主观拔高，若 ground.match_evidence 未明确支持，判 exaggeration
- 不得因措辞通顺就放行无来源数字、全国性/上百所等扩大表述

只输出一个 JSON 对象，不要 markdown：
{"verdict":"pass"|"fail","codes":[string],"spans":[string],"repairable_by":"llm"|"orchestrator","message":"给主模型的短中文，禁止给用户"}

codes 仅允许：unsupported_claim, exaggeration, unverified_action, invented_candidate
spans 摘录 reply 中有问题的短句，最多 3 条。
pass 时 codes/spans 必须为空数组。
"""


def run_l1(*, payload: ReplyGroundingInput, model: Any | None) -> ReplyGroundingOutput:
    """Call the judge LLM. Fail closed on missing model / timeout / bad JSON."""
    if model is None:
        _logger.warning("action=reply_grounding_unavailable reason=no_model")
        return schema_invalid_output(message="语义对照模型不可用，fail closed")

    user_payload = _build_user_payload(payload)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        msg = model.invoke(
            [
                SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=user_payload),
            ]
        )
        text = _extract_text_response(getattr(msg, "content", None))
        parsed = parse_judge_json(text)
        if parsed is None:
            _logger.warning("action=reply_grounding_parse_failed")
            return schema_invalid_output(message="子代理输出无法解析")
        parsed.layer = "l1"
        return parsed
    except Exception as exc:  # noqa: BLE001
        _logger.warning("action=reply_grounding_llm_failed error=%s", exc)
        return schema_invalid_output(message=f"子代理调用失败: {type(exc).__name__}")


def parse_judge_json(text: str) -> ReplyGroundingOutput | None:
    """Parse judge JSON; return None on schema failure (caller fail-closes)."""
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        data = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    verdict = Verdict.parse(data.get("verdict"))
    if verdict is None:
        return None

    codes_raw = data.get("codes") or []
    if not isinstance(codes_raw, list):
        return None
    codes = [str(c).strip() for c in codes_raw if str(c).strip() in ALLOWED_CODES]
    # schema_invalid is implementation-only; judge must not emit it as soft pass.
    codes = [c for c in codes if c != "schema_invalid"][:8]

    if verdict == Verdict.pass_:
        codes = []
        spans: list[str] = []
    else:
        if not codes:
            codes = ["unsupported_claim"]
        spans_raw = data.get("spans") or []
        if not isinstance(spans_raw, list):
            spans_raw = []
        spans = [str(s).strip()[:200] for s in spans_raw if str(s).strip()][:8]

    repair_raw = str(data.get("repairable_by") or "llm").strip().lower()
    repairable = (
        RepairableBy.orchestrator
        if repair_raw == "orchestrator"
        else RepairableBy.llm
    )
    message = str(data.get("message") or "").strip()[:500]
    if verdict == Verdict.fail and not message:
        message = "相对本轮手 facts 存在无来源或夸大表述。"

    return ReplyGroundingOutput(
        verdict=verdict,
        codes=codes,
        spans=spans,
        repairable_by=repairable,
        message=message,
        layer="l1",
    )


def _build_user_payload(payload: ReplyGroundingInput) -> str:
    ground = payload.ground.model_dump(mode="json")
    return (
        f"reply_mode: {payload.reply_mode}\n"
        f"locale: {payload.locale}\n"
        f"ground: {json.dumps(ground, ensure_ascii=False)}\n"
        f"reply:\n{payload.reply}\n"
        "请输出 JSON。"
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
