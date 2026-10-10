"""Dedicated bounded client for WeCom candidate assessment, separate from Cafe Host."""
from __future__ import annotations
import os
from typing import Any


def review_timeout() -> float:
    return min(60.0, max(10.0, float(os.getenv("WECOM_MATCH_REVIEW_TIMEOUT_S", "45"))))


def review_budget() -> float:
    return min(90.0, max(30.0, float(os.getenv("WECOM_MATCH_REVIEW_BUDGET_S", "75"))))


def get_review_model(state: Any) -> Any:
    existing = getattr(state, "wecom_review_model", None)
    if existing is not None:
        return existing
    from apps.group_agent_api.agent_factory.model_builder import create_model
    model = create_model(log_prefix="[WecomMatchReview]", timeout_s=review_timeout(), max_retries=0)
    from langchain_openai import ChatOpenAI
    if isinstance(model, ChatOpenAI):
        updates = {"max_tokens": 3000, "temperature": 0}
        if os.getenv("GROUP_AGENT_PROVIDER", "deepseek").strip().lower() in {"qwen", "dashscope"}:
            updates["extra_body"] = {**(model.extra_body or {}), "enable_thinking": False}
        model = model.model_copy(update=updates)
    state.wecom_review_model = model
    return model
