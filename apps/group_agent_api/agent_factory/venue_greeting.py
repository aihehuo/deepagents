"""Independent read-only greeting executor: no agent, tools, or checkpoint."""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from langchain_core.messages import SystemMessage

from apps.group_agent_api.agent_factory.contact_scrub import scrub_contact_text
from apps.group_agent_api.agent_factory.content_quality import looks_like_invented_candidate_narrative, looks_like_unauthorized_action_claim
from apps.group_agent_api.agent_factory.context.turns import venue_host_system_content
from apps.group_agent_api.agent_factory.host_context import HostContext, safe_context_text
from apps.group_agent_api.agent_factory.brain_repair.reject import MouthIngressRejected
from apps.group_agent_api.app.models import _GreetingPayload

GREETING_TASK = """可信到店事件：当前访客进入小馆，主动进行一次新问候，不把事件写成用户说了你好。
用中文，像专业的咖啡师/调酒师自然接待，1至3句，最多一个问题。结合授权的画像、近期对话或上次问候找一个轻巧关联；不要复述整份画像，不猜测新的进展。
has_previous_visit=false时不要说又见面、老朋友、上次、回来，不虚构熟悉；true时可以自然接续，但不要复用上次问候原句。
饮品仅虚拟推荐，不能假装用户已经点单、喝完或付费。不讲述其他客人的故事，不推荐具体人物，不声称执行任何动作。
只输出JSON对象：{\"result_kind\":\"greeting\",\"reply\":\"非空问候\",\"suggested_replies\":[{\"label\":\"短标签\",\"reply\":\"用户可主动发送的话\"}]}。
suggested_replies最多3个，标签≤40字，reply≤200字，问候≤1000字。不得输出工具调用或其他字段。"""
_COMPLETED = re.compile(r"(?:已|已经)(?:替你|帮你|为你)?(?:保存|更新画像|找到|匹配|发出|发送|联系|邀请|报名|扣款|付款)|(?:have|already)\s+(?:sent|saved|invited|contacted)|I(?:'ve| have)\s+(?:sent|saved|invited|contacted)", re.IGNORECASE)
_FAMILIAR = re.compile(r"又见|老朋友|上次|回来|还记得|welcome back|last time|again", re.IGNORECASE)
_PROGRESS = re.compile(r"融资(?:成功|完成)|(?:已经|已)(?:上线|完成开发|找到合伙人|拿到投资)|(?:项目|产品|公司|你)(?:已经|已)?(?:上线了|完成开发|拿到投资|获得融资|找到合伙人|合作成功)|funding secured|launched successfully", re.IGNORECASE)
_INTERNAL = re.compile(r"ga_global_|community_barista|community_bartender|community_cafe_host|(?:user_id|group_id|space_id|memory_scope_id|host_agent_id)\s*[:=]", re.IGNORECASE)


class VenueGreetingError(RuntimeError):
    """Stable redacted failure code, safe to pass through worker logging."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def validate_greeting_reply(payload: dict[str, Any], *, host: HostContext) -> dict[str, Any]:
    """Validate schema plus existing action/candidate/contact safety predicates.

    Args:
        payload: Parsed model JSON object.
        host: Authorized facts and previous-visit context.
    """
    parsed = _GreetingPayload.model_validate(payload).model_dump()
    if host.is_cafe_host:
        from apps.group_agent_api.agent_factory.cafe_output import extract_host_suggestions

        options = extract_host_suggestions("<suggested_replies>" + json.dumps(parsed["suggested_replies"], ensure_ascii=False) + "</suggested_replies>").suggestions
        parsed["suggested_replies"] = [{"label": item["label"], "reply": item["reply"]} for item in options]
    if not parsed["reply"].strip():
        raise ValueError("greeting_schema_invalid")
    facts = host.quoted_data(greeting=True)
    has_previous = bool(host.snapshot["recent_messages"] or host.snapshot["previous_greeting"] or host.snapshot["previous_visit_at"])
    for text in [parsed["reply"], *[item["label"] for item in parsed["suggested_replies"]], *[item["reply"] for item in parsed["suggested_replies"]]]:
        if looks_like_unauthorized_action_claim(text) or looks_like_invented_candidate_narrative(text) or _COMPLETED.search(text) or _INTERNAL.search(text):
            raise ValueError("security_guard_failed")
        if safe_context_text(text, limit=len(text)) != text or scrub_contact_text(text) != text.strip(" ，,;；"):
            raise ValueError("security_guard_failed")
    if not has_previous and _FAMILIAR.search(parsed["reply"]):
        raise ValueError("security_guard_failed")
    if any(match.group(0) not in facts for match in _PROGRESS.finditer(parsed["reply"])):
        raise ValueError("security_guard_failed")
    if host.snapshot["previous_greeting"] and parsed["reply"] == host.snapshot["previous_greeting"]["text"]:
        raise ValueError("greeting_schema_invalid")
    return parsed


async def execute_venue_greeting(*, host: HostContext, model: Any, emit_callback: Callable[[str, dict[str, Any]], Awaitable[bool]], side_effect_gate: Callable[[str], None] | None = None) -> None:
    """Generate a bounded greeting with an unbound chat model, not a deep agent.

    Args:
        host: Authorized read-only facts and frozen persona.
        model: Dedicated bare chat model; never the tool-bound member agent.
        emit_callback: Existing lease-fenced callback adapter.
        side_effect_gate: Attempt-ownership gate before model/callback work.
    """
    content = venue_host_system_content(host, greeting=True)
    if not host.is_cafe_host:
        content += "\n" + GREETING_TASK
    messages = [SystemMessage(content=content)]
    config = {"metadata": {"input_kind": "venue_event", "venue_mode": host.effective_mode, "venue_has_history": bool(host.snapshot["recent_messages"] or host.snapshot["previous_greeting"] or host.snapshot["previous_visit_at"])}}
    error = "greeting_schema_invalid"
    for attempt in range(2):
        if side_effect_gate is not None:
            side_effect_gate("venue_greeting_generate")
        try:
            output = await model.ainvoke(messages, config=config, tools=[], tool_choice="none", response_format={"type": "json_object"})
        except Exception:
            raise VenueGreetingError("llm_unavailable") from None
        try:
            extra = getattr(output, "additional_kwargs", {})
            if (getattr(output, "tool_calls", None) or getattr(output, "invalid_tool_calls", None)
                    or extra.get("tool_calls") or extra.get("function_call")):
                raise ValueError("security_guard_failed")
            if not isinstance(output.content, str) or len(output.content) > 10000:
                raise ValueError("greeting_schema_invalid")
            payload = validate_greeting_reply(json.loads(output.content), host=host)
        except (ValueError, TypeError, AttributeError) as exc:
            error = "security_guard_failed" if "security_guard_failed" in str(exc) else "greeting_schema_invalid"
            if attempt == 0:
                messages = [*messages, SystemMessage(content="上一输出格式或安全校验不合格。重新遵守原任务生成JSON，不复述不合格内容，不补充任何未经授权的事实或行动。")]
                continue
            raise VenueGreetingError(error) from None
        if side_effect_gate is not None:
            side_effect_gate("venue_greeting_final")
        try:
            await emit_callback("final", payload)
        except MouthIngressRejected as exc:
            # Lease adapter rewinds the rejected terminal seq. Regenerate from
            # trusted source only; do not echo mouth hints or rejected prose.
            if attempt == 0 and exc.repairable_by == "llm":
                messages = [*messages, SystemMessage(content="问候未通过服务端事实/隐私/格式校验。重新根据授权数据生成简短JSON问候，不做行动或进展断言，不复述不合格内容。")]
                continue
            raise VenueGreetingError("mouth_ingress_rejected") from None
        return
