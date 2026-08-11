"""LLM client for wechat_greeter (REQ-050 / REQ-051 B 阶段).

3 模式：
  - "stub"       : A 阶段冒烟, 直接返回固定长文本 (用于测硬截断)
  - "deepseek"   : B 阶段默认, 走 init_chat_model + model_provider="deepseek"
  - "test_mock"  : 单测用, 由 monkeypatch 注入, 返回 WECHAT_GREETER_LLM_RAW env 值

Model 锁死: deepseek-v4-flash (老板 2026-08-11 拍板, 走 api.deepseek.com 兼容 OpenAI 协议)。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from wechat_greeter.config import model_mode

_logger = logging.getLogger(__name__)

# 老板 2026-08-11 拍板: deepseek-v4-flash 锁死, 兼容 OpenAI 协议, base_url=api.deepseek.com
DEFAULT_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _build_chat_model():
    """Build a real chat model via langchain init_chat_model. Only called in deepseek mode."""
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")  # fallback for compat
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set in env. B 阶段 (deepseek mode) requires this. "
            "Set it in deployment env (never commit real value)."
        )
    return init_chat_model(
        model=os.environ.get("WECHAT_GREETER_LLM_MODEL", DEFAULT_MODEL_NAME),
        model_provider="deepseek",
        api_key=api_key,
        base_url=os.environ.get("WECHAT_GREETER_LLM_BASE_URL", DEFAULT_BASE_URL),
        temperature=float(os.environ.get("WECHAT_GREETER_LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.environ.get("WECHAT_GREETER_LLM_MAX_TOKENS", "512")),
    )


def _build_system_prompt(*, identity_branch: str, user_message: str) -> str:
    """Render the v1 j2 system prompt with current identity branch + user message."""
    # Lazy import jinja2 (避免在 stub 模式下也强制要求 jinja2 安装, 虽然 A 阶段已 require)
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    env = Environment(
        loader=FileSystemLoader(prompt_dir),
        autoescape=select_autoescape(disabled_extensions=("j2",), default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("wechat_greeter_v1.j2")
    return tmpl.render(identity_branch=identity_branch, user_message=user_message)


def _determine_identity_branch(*, user_id: int) -> str:
    """Map user_id → 4 身份分支之一.

    A/B 阶段: 仅有 user_id 维度 (无 profile/invest 字段回查) — 真分支决策留 P2 接入 aihehuomicro profile_status 后再做.
    """
    if user_id <= 0:
        return "guest"
    # 简化: user_id > 0 都先归 registered_no_invest, 真 4 分支决策留 P2 接 profile_status 后做
    return "registered_no_invest"


def call_llm(*, user_message: str, user_id: int) -> str:
    """Call LLM and return raw reply text.

    3 模式分支:
      - stub:        返回固定长文本 (WECHAT_GREETER_LLM_STUB_RAW 覆盖, 否则默认长文本)
      - test_mock:   同 stub, 单测用 (默认就是 stub mode)
      - deepseek:    走 init_chat_model + 真实 deepseek API
    """
    mode = model_mode()

    if mode in ("stub", "test_mock"):
        # 1. 优先 env 覆盖 (test_06 单测用)
        env_raw = os.environ.get("WECHAT_GREETER_LLM_STUB_RAW")
        if env_raw is not None:
            return env_raw
        # 2. 默认长文本 (够长以触发 200 字硬截断, 用于冒烟)
        return (
            "你好!爱合伙是一个连接创业者和合伙人的平台。"
            "我们提供项目发布、合伙人匹配、社群交流等功能。"
            "这是一个较长的回复用于测试硬截断功能 — "
            "为了测试 200 字硬截断, 我们生成超过 200 字的文本。"
            "如果你看到这段话(超过 200 字), 说明硬截断逻辑生效, "
            "不会向用户输出超长内容。"
        )

    if mode == "deepseek":
        # 真实 LLM 调
        branch = _determine_identity_branch(user_id=user_id)
        system_prompt = _build_system_prompt(
            identity_branch=branch, user_message=user_message
        )
        model = _build_chat_model()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        resp = model.invoke(messages)
        # BaseMessage → str
        return resp.content if hasattr(resp, "content") else str(resp)

    raise ValueError(f"unknown WECHAT_GREETER_MODEL_MODE: {mode!r}")


def build_tools_for_agent(*, user_id: int) -> list[Any]:
    """Build 5 tools as langchain @tool-decorated callables.

    B 阶段: 真正把 5 工具转成 langchain BaseTool, 供 deepseek mode 走 init_agent.
    A 阶段 stub 模式: 返回原始 callables (与既有 process_greeting 兼容).
    """
    from apps.wechat_greeter_api.agent_factory import make_tools
    return make_tools(user_id=user_id)
