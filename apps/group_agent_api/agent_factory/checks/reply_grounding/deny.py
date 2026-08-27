"""Assemble ctx.check_deny for mod.brain.reply_grounding (TSD-14 §4.6.4)."""

from __future__ import annotations

from apps.group_agent_api.agent_factory.checks.reply_grounding.ids import MODULE_ID
from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    ReplyGroundingOutput,
)


def format_check_deny(result: ReplyGroundingOutput) -> str:
    """System-block text for the main dialogue model. Never show to the end user."""
    codes = ",".join(result.codes) if result.codes else "schema_invalid"
    repairable = result.repairable_by.value
    spans = "；".join(result.spans) if result.spans else "（未标注具体片段）"
    reason = (result.message or "相对本轮手 facts 存在无来源或夸大表述。").strip()
    return (
        f'<check_deny module="{MODULE_ID}" codes="{codes}" '
        f'repairable_by="{repairable}">\n'
        "上一份 reply 未送出口。\n"
        f"spans: {spans}\n"
        f"原因: {reason}\n"
        "可做: 删掉 spans 所指表述，或只复述 facts 原文后再提交。\n"
        "不可做: 为通过检测而编造新经历；不可要求口改写句子。\n"
        "</check_deny>"
    )
