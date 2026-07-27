"""Group-context agent factory (REQ-004 slice 1: FR-02 + FR-06).

Independent of business_cofounder. Memory layer: FilesystemBackend + thread_id.
Business: dialogue → extract 3-dim profile → forced structured persist.
No matching / candidates / @ invite words in this slice.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from apps.group_agent_api.agent_factory.content_quality import (
    is_need_shaped_doing,
    is_preference_shaped_offer,
)
from apps.group_agent_api.agent_factory.integrations.config import integration_mode
from apps.group_agent_api.agent_factory.integrations.profile_client import (
    ProfileHttpError,
    persist_group_profile,
)
from apps.group_agent_api.agent_factory.model_builder import create_model
from apps.group_agent_api.agent_factory.profile_schema import (
    DisclosureLevel,
    profile_from_flat,
)
from apps.group_agent_api.agent_factory.profile_store import save_profile
from apps.group_agent_api.checkpointer import DiskBackedInMemorySaver
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.observability import UCObserver

_logger = logging.getLogger("uvicorn.error")

APP_NAME = "group_agent_api"
PROFILE_SUPERSEDED_RESULT_PREFIX = "ok: profile_superseded;"


class UC34Observer(UCObserver):
    """Observer for UC-34 group agent (slice 1: dialogue + profile persist)."""

    uc_name = "34_group_agent"


SYSTEM_PROMPT = """你是「群内智能体」对话助手（切片 2a：挖需求 + 画像落库 + 能力分级下的候选管线）。

## 目标（FR-02）
用最少对话补全可匹配画像三维：
1. **doing** — 当前在做什么（创业意图/方向）
2. **need** — 缺什么（需求 gap）
3. **offer** — 能提供什么（资源/技能）

## 对话规则
- 开口简洁，一次只聚焦缺的维度。
- 用户惜字（很短/敷衍）→ **最多再追问 1 次**，然后必须根据已有信息尽力生成三维画像并落库。
- 1–2 轮内应调用 `save_group_profile` 写出完整画像。
- 工具调用成功后，用具体的 doing / need / offer 简洁确认你理解了什么，并给出一个明确下一步；禁止只回复问候、致谢或「随时告诉我」。
- 你在正式匹配管线之前看不到候选结果，不要声称「不能推荐」「没有人选」或「已经找到人选」；最终匹配状态由系统在你回复后统一收口。

## 落库（FR-06 · 强制）
- 三维字段齐备后，**必须调用** `save_group_profile`（不要用 write_file 写自由 Markdown）。
- 后续消息若只是在重复/细化 need 或表达合作偏好，不得把 doing 改写成「找某类负责人/工程师」，也不得把 offer 改写成只有「合作方式可以谈/希望尽快启动」；应从对话历史重提仍有效的 doing/offer。若用户明确撤回资源，offer 写「暂无可提供资源」，不得沿用旧资源。
- 未确认的推断：disclosure 用 `inferred_unconfirmed`。
- 用户明确说可公开的：`confirmed_public`；仅用于匹配：`match_only`。

## 人脉与披露（SAFE-01/02 · prompt 闸）
- **不要自行编造/列举候选人、匹配结果或含 @ 的邀请词**——候选人由系统在能力解锁后附加；你只做对话与落库。
- 若谈及他人公开信息，**只用对方已确认可公开（confirmed_public）字段**；不得把 match_only / inferred_unconfirmed 说给当前用户听成「对方公开资料」。
- 匹配理由须是「值得一聊的理由 + 明确不确定性」，禁止「你们很合适」结论式断言（AI-03）。
- 本切片**不生成**共同话题长文案、不生成定向邀请词（那是切片 2b）。

## 红线
- user_id / group_id / membership 已由系统注入；不要编造其他用户身份或跨群人脉。
- 不得输出手机号、微信号等敏感联系方式。

## 语言
用中文，口语、短句。
"""

FORCE_SAVE_PROMPT = (
    "系统校验：本轮结束后该用户×群的结构化画像尚未落库。"
    "请立即根据对话内容调用 save_group_profile，补全 doing/need/offer 三维后保存。"
    "doing 必须是用户在推进的项目而不是正在找的人；offer 必须是实际资源/能力，"
    "不能只有合作偏好。未发生变化的维度从对话历史重提；"
    "用户撤回资源时明确写「暂无可提供资源」。"
    "不要再追问；不要推荐任何人；不要生成邀请词。"
)


def default_runtime_dir() -> Path:
    return Path.home() / ".deepagents" / APP_NAME


@tool(parse_docstring=True)
def save_group_profile(
    doing: str,
    need: str,
    offer: str,
    doing_disclosure: str = "inferred_unconfirmed",
    need_disclosure: str = "inferred_unconfirmed",
    offer_disclosure: str = "inferred_unconfirmed",
    *,
    config: RunnableConfig,
) -> str:
    """将用户×群的三维画像强制写入结构化 profile.json。三维齐备后必须调用。

    Args:
        doing: 当前在做什么（创业意图/方向）
        need: 缺什么（需求 gap）
        offer: 能提供什么（资源/技能）
        doing_disclosure: confirmed_public | match_only | inferred_unconfirmed
        need_disclosure: confirmed_public | match_only | inferred_unconfirmed
        offer_disclosure: confirmed_public | match_only | inferred_unconfirmed
    """
    metadata = config.get("metadata") or {}
    user_id = str(metadata.get("user_id") or "").strip()
    group_id = str(metadata.get("group_id") or "").strip()
    base_dir_raw = metadata.get("base_dir") or str(default_runtime_dir())
    base_dir = Path(str(base_dir_raw))
    run_id = str(metadata.get("run_id") or "").strip()

    if not user_id or not group_id:
        return "error: missing user_id or group_id in metadata"

    try:
        for raw in (doing_disclosure, need_disclosure, offer_disclosure):
            DisclosureLevel(raw)
        # REQ-014-FIX: reject semantic projections observably.  Never restore a
        # historical value here: this tool cannot prove whether the user kept
        # it or explicitly withdrew it during the current turn.
        semantic_errors: list[str] = []
        if is_need_shaped_doing(doing):
            semantic_errors.append("doing_describes_need")
        if is_preference_shaped_offer(offer):
            semantic_errors.append("offer_describes_preference")
        if semantic_errors:
            reason = ",".join(semantic_errors)
            UC34Observer.warn(
                f"action=save_group_profile_rejected user_id={user_id} "
                f"group_id={group_id} reason={reason} status=resubmit_required"
            )
            return f"error: semantic_projection:{reason}; resubmit_required"
        profile = profile_from_flat(
            user_id=user_id,
            group_id=group_id,
            doing=doing,
            need=need,
            offer=offer,
            doing_disclosure=doing_disclosure,
            need_disclosure=need_disclosure,
            offer_disclosure=offer_disclosure,
        )
        remote_ack: dict[str, Any] | None = None
        if integration_mode() == "http":
            remote_ack = persist_group_profile(profile=profile, run_id=run_id)
        if remote_ack is None or remote_ack["status"] != "stale_ignored":
            path = save_profile(base_dir, profile)
        else:
            path = None
    except ProfileHttpError as exc:
        UC34Observer.error(
            f"action=save_group_profile_remote_error user_id={user_id} "
            f"group_id={group_id} error={exc}"
        )
        return f"error: profile_database:{exc}"
    except Exception as exc:  # noqa: BLE001
        UC34Observer.error(
            f"action=save_group_profile_error user_id={user_id} "
            f"group_id={group_id} error={exc}"
        )
        return f"error: {exc}"

    UC34Observer.info(
        f"action=save_group_profile user_id={user_id} group_id={group_id} "
        f"path={path} status=success"
    )
    if remote_ack is not None:
        if remote_ack["status"] == "stale_ignored":
            return (
                f"{PROFILE_SUPERSEDED_RESULT_PREFIX} database kept a newer profile "
                f"(version={remote_ack['profile_version']}); local cache unchanged"
            )
        return (
            "ok: saved profile to database "
            f"(version={remote_ack['profile_version']}) and local cache"
        )
    return f"ok: saved profile to /users/{user_id}/groups/{group_id}/profile.json"


def create_agent(
    *,
    base_dir: Path | None = None,
    model: Any | None = None,
    checkpointer: Any | None = None,
) -> tuple[Any, Path]:
    """Create the group agent. Returns (agent, checkpoints_path)."""
    runtime = base_dir or default_runtime_dir()
    runtime.mkdir(parents=True, exist_ok=True)
    UCObserver.set_log_dir(runtime / "logs")

    ckpt_path = runtime / "checkpoints.pkl"
    ckpt = checkpointer or DiskBackedInMemorySaver(file_path=ckpt_path)
    backend = FilesystemBackend(root_dir=str(runtime), virtual_mode=True)
    llm = model or create_model()

    agent = create_deep_agent(
        model=llm,
        tools=[save_group_profile],
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        checkpointer=ckpt,
        # Slice 1: no subagents, no skills, no free-form memory middleware
    )
    _logger.info("group_agent_api ready runtime=%s", runtime)
    return agent, ckpt_path
