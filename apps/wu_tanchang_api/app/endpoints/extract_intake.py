"""Extract intake dimensions from user utterances (LLM semantic)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from apps.wu_tanchang_api.agent_factory.model_builder import create_model
from apps.wu_tanchang_api.app.models import (
    ExtractIntakeRequest,
    ExtractIntakeResponse,
    IntakeDimension,
)

_logger = logging.getLogger("uvicorn.error")

DIMENSION_SPECS = [
    ("city", "所在城市", "计划开店或经营的城市/区域"),
    ("background", "创业背景", "个人或团队创业经验、过往行业背景"),
    ("track", "选择赛道", "意向/已选的实体店赛道、品类或业态"),
    ("stage", "当前阶段", "想法期 / 筹备中 / 已开店 / 瓶颈转型期 等阶段"),
    ("problem", "定义问题", "核心困惑、痛点、卡住的具体决策或难题"),
    ("goal", "聚焦目标", "阶段性经营目标、期待达到的突破或结果"),
    ("plan", "实施计划", "拟定的落地动作、推进步骤或行动计划"),
]

SYSTEM_PROMPT = """你是实体店商业面谈的信息抽取器。只根据用户说过的话做语义理解，不要编造。

任务：判断七维信息是否已被用户提及，并给出每维的短摘要（关键词级，尽量短）。

七维定义：
1. city（所在城市）：计划开店或经营的城市/商圈/区域
2. background（创业背景）：过往创业经历、行业背景或团队情况
3. track（选择赛道）：意向/已选的具体赛道、品类或业态
4. stage（当前阶段）：想法期 / 筹备中 / 已开店 / 拓展转型 等
5. problem（定义问题）：核心痛点、具体卡点或待决策难题
6. goal（聚焦目标）：想要达到的目标、业绩期望或核心诉求
7. plan（实施计划）：目前规划的推进动作、步骤或执行计划表

规则：
- covered=true 仅当用户话里确实表达了该维信息（可同义改写）
- summary 只写提炼后的关键词/短语，不要整句复述；未覆盖则为 null
- keywords 为该维相关词 0-4 个
- ready_for_prediagnosis=true 要求更严，须同时满足：
  1) 七维均 covered
  2) problem 不是空泛口号（如仅「想听听建议」「不知道怎么办」→ 仍应 false）
  3) 用户有效表述累计不少于约 4 段/轮（过短寒暄不算）
  4) 至少有一维带「原因/卡点/已尝试」类细节，足以写讨论提纲
- 不要根据助手的话推断；输入里若混入助手内容也请忽略，只信用户表述
"""


def _verify_agent_key(x_agent_key: str | None) -> None:
    expected = (
        os.environ.get("WU_CALLBACK_AGENT_KEY")
        or os.environ.get("WU_TANCHANG_CALLBACK_TOKEN")
        or ""
    ).strip()
    if not expected:
        # Dev/local without shared key: allow (same trust model as /call_async).
        return
    if not x_agent_key or x_agent_key.strip() != expected:
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})


def _collect_user_blob(req: ExtractIntakeRequest) -> str:
    parts: list[str] = []
    for t in req.user_texts or []:
        s = (t or "").strip()
        if s:
            parts.append(s)
    for m in req.messages or []:
        role = (m.role or "").strip().lower()
        if role and role != "user":
            continue
        s = (m.text or "").strip()
        if s:
            parts.append(s)
    # Deduplicate consecutive identical lines
    out: list[str] = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    return "\n".join(out).strip()


class _DimOut(BaseModel):
    key: str
    covered: bool = False
    summary: str | None = None
    keywords: list[str] = Field(default_factory=list)


class _ExtractOut(BaseModel):
    dimensions: list[_DimOut]
    ready_for_prediagnosis: bool = False


def _normalize(raw: _ExtractOut | dict[str, Any]) -> ExtractIntakeResponse:
    data = raw.model_dump() if hasattr(raw, "model_dump") else (
        raw.dict() if hasattr(raw, "dict") else dict(raw)
    )
    by_key = {
        str(d.get("key")): d
        for d in (data.get("dimensions") or [])
        if isinstance(d, dict)
    }
    dims: list[IntakeDimension] = []
    for key, label, _hint in DIMENSION_SPECS:
        d = by_key.get(key) or {}
        covered = bool(d.get("covered"))
        summary = d.get("summary")
        summary_s = str(summary).strip() if summary is not None else ""
        kws = d.get("keywords") or []
        if not isinstance(kws, list):
            kws = []
        dims.append(
            IntakeDimension(
                key=key,
                label=label,
                covered=covered,
                summary=summary_s[:48] if summary_s else None,
                keywords=[str(k).strip()[:24] for k in kws if str(k).strip()][:4],
            )
        )
    covered_count = sum(1 for d in dims if d.covered)
    ready = bool(data.get("ready_for_prediagnosis")) and covered_count >= 7
    return ExtractIntakeResponse(
        dimensions=dims,
        covered_count=covered_count,
        total=7,
        ready_for_prediagnosis=ready,
        source="llm",
    )


def _parse_json_fallback(content: str) -> dict[str, Any]:
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


async def extract_intake(
    req: ExtractIntakeRequest,
    x_agent_key: str | None = None,
) -> ExtractIntakeResponse:
    """Semantic intake extraction for wu-agent progress drawer."""
    _verify_agent_key(x_agent_key)

    blob = _collect_user_blob(req)
    if not blob:
        empty = [
            IntakeDimension(key=k, label=lab, covered=False, summary=None, keywords=[])
            for k, lab, _ in DIMENSION_SPECS
        ]
        return ExtractIntakeResponse(
            dimensions=empty,
            covered_count=0,
            total=7,
            ready_for_prediagnosis=False,
            source="llm",
        )

    provider = (os.environ.get("WU_API_MODEL_PROVIDER") or "qwen").strip() or "qwen"
    model = create_model(
        provider=provider,
        model_name_suffix="MAIN_AGENT_MODEL",
        log_prefix="[IntakeExtract]",
        max_tokens=800,
    )

    user_prompt = (
        "请抽取下列用户表述中的七维信息，输出结构化结果。\n\n"
        f"用户表述：\n{blob[:6000]}\n"
    )

    try:
        structured = model.with_structured_output(_ExtractOut)
        res = await structured.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        return _normalize(res)
    except Exception as e:
        _logger.warning("[IntakeExtract] structured_output failed: %s", e)

    # Fallback: plain JSON instruction
    try:
        raw = await model.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=user_prompt
                    + "\n只输出 JSON，字段 dimensions[{key,covered,summary,keywords}], "
                    "ready_for_prediagnosis。"
                ),
            ]
        )
        content = raw.content if hasattr(raw, "content") else str(raw)
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            )
        parsed = _parse_json_fallback(str(content))
        return _normalize(parsed)
    except Exception as e:
        _logger.exception("[IntakeExtract] failed: %s", e)
        raise HTTPException(
            status_code=502,
            detail={"error": "extract_failed", "message": str(e)[:200]},
        ) from e
