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
    ("stage", "当前阶段", "想法期 / 筹备中 / 已开店 等阶段标签"),
    ("category", "意向品类", "餐饮品类或业态，如火锅、烘焙、茶饮"),
    ("city", "城市区位", "计划开店的城市或区域"),
    ("budget", "预算区间", "投入/预算金额或区间"),
    ("challenge", "核心困惑", "用户最卡的点、担心或待解问题"),
]

SYSTEM_PROMPT = """你是餐饮创业面谈的信息抽取器。只根据用户说过的话做语义理解，不要编造。

任务：判断五维信息是否已被用户提及，并给出每维的短摘要（关键词级，尽量短）。

五维定义：
1. stage（当前阶段）：想法期 / 筹备中 / 已开店（或等价表述）
2. category（意向品类）：想做的餐饮品类/业态
3. city（城市区位）：城市或商圈
4. budget（预算区间）：资金/预算
5. challenge（核心困惑）：最卡的点、困惑、待决策问题

规则：
- covered=true 仅当用户话里确实表达了该维信息（可同义改写）
- summary 只写提炼后的关键词/短语，不要整句复述；未覆盖则为 null
- keywords 为该维相关词 0-4 个
- ready_for_prediagnosis=true 要求更严，须同时满足：
  1) 五维均 covered
  2) challenge 不是空泛口号（如仅「想听听建议」「不知道怎么办」→ 仍应 false）
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
    ready = bool(data.get("ready_for_prediagnosis")) and covered_count >= 5
    return ExtractIntakeResponse(
        dimensions=dims,
        covered_count=covered_count,
        total=5,
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
            total=5,
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
        "请抽取下列用户表述中的五维信息，输出结构化结果。\n\n"
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
