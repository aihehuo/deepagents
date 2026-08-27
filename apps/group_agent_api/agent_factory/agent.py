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

from apps.group_agent_api.agent_factory.admin_ops_tools import (
    ADMIN_OPS_TOOLS,
    ADMIN_SYSTEM_PROMPT,
)
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
from apps.group_agent_api.agent_factory.search_tool import search_candidates
from apps.group_agent_api.agent_factory.profile_schema import (
    DisclosureLevel,
    GroupProfile,
    ProfileField,
    profile_from_flat,
)
from apps.group_agent_api.agent_factory.profile_store import save_profile
from apps.group_agent_api.agent_factory.suggested_replies import (
    SUGGESTED_REPLIES_PROMPT,
)
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


SYSTEM_PROMPT = """你是「群内智能体」对话助手（挖需求 + 画像落库 + 能力分级下的候选管线）。

## 目标（FR-02 / REQ-029）
补全可匹配画像三维，并尽量挖到具体的 doing / need / offer「可匹配的最低充分」信息（不必完美）：
1. **doing** — 当前在做什么（创业意图/方向/场景）
2. **need** — 缺什么（需求 gap / 卡点）
3. **offer** — 能提供什么（资源/技能）

系统会在落库后由你决定是否开搜；开搜必须调用 `search_candidates` 工具，不要等系统在回复后再搜。

## 顾问式对话与情绪价值（核心规则）
- **先反馈/认同，再提问**：当用户分享项目或想法时，先用 1 句话给予真诚的共情、行业认同或正向价值反馈（如“这个方向痛点很明确！”、“这块市场很有潜力”），严禁直接抛出硬性问题。
- **拒绝“填表查户口”**：严禁连续抛出多个硬性维度追问（严禁在同一轮回复中同时问“你在做什么？缺什么？能提供什么？”）。一次只顺着用户的回答聊天，自然引导。
- 开口简洁，一次只聚焦缺的或过薄的维度。
- 可以尽早调用 `save_group_profile` 落库草稿；三维齐了也仍可继续追问具体场景与卡点。
- 用户惜字：再追问时要具体、可回答；不要编造空壳三维凑数。
- **方向切换 / 开新一轮**：若用户明确说换方向、另一条赛道、或「跟之前完全不一样」，必须按**本轮新说法**重写 doing/need/offer 并调用 `save_group_profile` 覆盖旧画像；禁止继续沿用上一轮项目。
- 用户回答「具体产品/场景」类追问时：必须把新细节**合并进 doing**（例如「做 AI 教育」+「AI 单词记忆线上产品」→ 更新为更具体的 doing）并再次 `save_group_profile`；禁止只口头确认却不落库。
- 同一轮内细化时：未撤回的 doing/offer 可从本轮已确认内容重提；不得把过期项目硬留着。
- **禁止说「请稍候 / 稍后返回匹配结果 / 系统正在筛选」**——搜人由你在本轮调用 `search_candidates` 完成，结果会出现在工具返回里。

## 搜人（必须用工具，禁止编排代搜）
- 用户要求「匹配 / 帮我搜 / 推荐 / 在群里找人 / 愿意 @」时：先按需 `save_group_profile`，然后**必须调用** `search_candidates`。
- `query` 必须由你根据本轮对话和已落库画像自己组织（关键词或短句），不能留空。
- 禁止不调用 `search_candidates` 就声称找到人、没找到人、或「系统会附加候选人」。
- 工具返回后，只能根据返回的 `candidates` 说话；`status=empty` 就说本轮没找到，不要编造人选。
- 同一轮最多搜一次（除非系统另行开启多级放宽说明）。

## 回复格式与微信口语化
- **严禁输出表单式标签**：在给用户的回复中，**绝对禁止**输出 `doing:`、`need:`、`offer:`、`- **doing**:`、`- **need**:`、`- **offer**:` 等调查问卷式的 Markdown 结构化标签或大段粗体列表。
- **字数控制与短句交流**：单次回复严格控制在 100 字以内，采用微信聊天式的自然短句与分段，保持轻松的口语化沟通。
- **工具调用确认**：工具调用成功后，用一句自然的口语确认你理解了什么并给出下一步，禁止格式化罗列三维清单。

## 落库（FR-06 · 强制）
- 三维字段齐备后，**必须调用** `save_group_profile`（不要用 write_file 写自由 Markdown）。
- **用户要求「匹配 / 帮我匹配 / 先匹配 / 选1 / 选2」时**：必须立即调用 `save_group_profile`（若画像有更新），并调用 `search_candidates`；**绝对禁止只在回复文本中口头说「已存入画像/正在匹配」而不触发工具**。
- 后续消息若只是在重复/细化 need 或表达合作偏好，不得把 doing 改写成「找某类负责人/工程师」，也不得把 offer 改写成只有「合作方式可以谈/希望尽快启动」。若用户明确撤回资源，offer 写「暂无可提供资源」，不得沿用旧资源。
- 未确认的推断：disclosure 用 `inferred_unconfirmed`。
- 用户明确说可公开的：`confirmed_public`；仅用于匹配：`match_only`。

## 人脉与披露（SAFE-01/02 · prompt 闸）
- **不要自行编造/列举候选人**。只能使用本轮 `search_candidates` 工具返回的 candidates。
- 用户问「有没有合适的人 / 详细说说他的背景」但本轮**没有**成功的 search 工具结果时：请调用 `search_candidates`，或说明还没搜；**禁止编造履历、年限、项目经验**。
- 若谈及他人公开信息，**只用对方已确认可公开（confirmed_public）字段**；不得把 match_only / inferred_unconfirmed 说给当前用户听成「对方公开资料」。
- 匹配理由须是「值得一聊的理由 + 明确不确定性」，禁止「你们很合适」结论式断言（AI-03）。
- 本切片**不生成**共同话题长文案、不生成定向邀请词（那是切片 2b）。

## 红线
- user_id / group_id / membership 已由系统注入；不要编造其他用户身份或跨群人脉。
- 若系统消息给出了已落库画像，必须当已知事实使用；用户问你是否知道他在做什么时禁止说「不知道」。
- 不得输出手机号、微信号等敏感联系方式。

## 语言
用中文，口语、短句，像自然的微信交流。
""" + SUGGESTED_REPLIES_PROMPT

FORCE_SAVE_PROMPT = (
    "系统校验：本轮结束后该用户×群在本 episode 尚无可用的结构化画像。"
    "请立即根据**本轮对话**调用 save_group_profile，补全 doing/need/offer 三维后保存。"
    "若用户已改方向，必须按新方向覆盖旧画像，禁止沿用上一轮项目。"
    "doing 必须是用户在推进的项目而不是正在找的人；offer 必须是实际资源/能力，"
    "不能只有合作偏好。用户撤回资源时明确写「暂无可提供资源」。"
    "不要再追问；不要推荐任何人；不要生成邀请词。"
)


def default_runtime_dir() -> Path:
    return Path.home() / ".deepagents" / APP_NAME


@tool(parse_docstring=True)
def save_group_profile(
    doing: str | dict[str, Any],
    need: str | dict[str, Any],
    offer: str | dict[str, Any],
    doing_disclosure: str = "inferred_unconfirmed",
    need_disclosure: str = "inferred_unconfirmed",
    offer_disclosure: str = "inferred_unconfirmed",
    match_constraints: list[dict[str, Any]] | None = None,
    *,
    config: RunnableConfig,
) -> str:
    """将用户×群的三维画像强制写入结构化 profile.json。三维齐备后必须调用。

    Args:
        doing: 当前在做什么（创业意图/方向，可为文本或带 value, claim_type, disclosure, evidence_text 的对象）
        need: 缺什么（需求 gap）
        offer: 能提供什么（资源/技能）
        doing_disclosure: confirmed_public | match_only | inferred_unconfirmed
        need_disclosure: confirmed_public | match_only | inferred_unconfirmed
        offer_disclosure: confirmed_public | match_only | inferred_unconfirmed
        match_constraints: 匹配硬/软约束列表，如 [{"field": "city", "operator": "in", "values": ["上海"], "strength": "hard"}]
    """
    metadata = config.get("metadata") or {}
    user_id = str(metadata.get("user_id") or "").strip()
    group_id = str(metadata.get("group_id") or "").strip()
    base_dir_raw = metadata.get("base_dir") or str(default_runtime_dir())
    base_dir = Path(str(base_dir_raw))
    run_id = str(metadata.get("run_id") or "").strip()
    source_message_id_trusted = metadata.get("source_message_id")
    source_message_text_trusted = metadata.get("source_message_text")

    if str(metadata.get("source") or "").strip() == "group_agent_admin_debug":
        return "error: admin_mode_read_only; do not save member profiles"

    if not user_id or not group_id:
        return "error: missing user_id or group_id in metadata"

    try:
        from apps.group_agent_api.agent_factory.grounding_protocol import (
            ClaimType,
            ConfidenceLevel,
            DisclosureLevelV2,
            GroupProfileV2,
            MatchConstraintV1,
            ProfileClaimV2,
            ProfileEvidenceV2,
            QualityStatus,
            canonical_sha256,
            validate_profile_claim_grounding,
        )

        def _unpack_dim(
            raw: str | dict[str, Any],
            default_disclosure: str,
        ) -> tuple[str, str, str, str | None]:
            if isinstance(raw, dict):
                val = str(raw.get("value") or "").strip()
                disc = str(raw.get("disclosure") or default_disclosure).strip()
                ctype = str(raw.get("claim_type") or "fact").strip()
                ev = raw.get("evidence_text")
                ev_str = str(ev).strip() if ev is not None else None
                return val, disc, ctype, ev_str
            return str(raw or "").strip(), str(default_disclosure).strip(), "fact", None

        doing_val, doing_disc, doing_ctype, doing_ev = _unpack_dim(doing, doing_disclosure)
        need_val, need_disc, need_ctype, need_ev = _unpack_dim(need, need_disclosure)
        offer_val, offer_disc, offer_ctype, offer_ev = _unpack_dim(offer, offer_disclosure)

        for raw_disc in (doing_disc, need_disc, offer_disc):
            DisclosureLevel(raw_disc)

        # REQ-014-FIX: reject semantic projections observably.
        semantic_errors: list[str] = []
        if is_need_shaped_doing(doing_val):
            semantic_errors.append("doing_describes_need")
        if is_preference_shaped_offer(offer_val):
            semantic_errors.append("offer_describes_preference")
        if semantic_errors:
            reason = ",".join(semantic_errors)
            UC34Observer.warn(
                f"action=save_group_profile_rejected user_id={user_id} "
                f"group_id={group_id} reason={reason} status=resubmit_required"
            )
            return f"error: semantic_projection:{reason}; resubmit_required"

        # TSD-13 / REQ-DA-066: match constraints validation
        parsed_constraints: list[MatchConstraintV1] = []
        if match_constraints and isinstance(match_constraints, list):
            for c_item in match_constraints:
                if isinstance(c_item, dict):
                    try:
                        parsed_c = MatchConstraintV1(
                            field=c_item.get("field", ""),
                            operator=c_item.get("operator", ""),
                            values=c_item.get("values", []),
                            strength=c_item.get("strength", "hard"),
                            source_message_id=int(source_message_id_trusted) if source_message_id_trusted else None,
                            evidence_text=c_item.get("evidence_text"),
                        )
                        parsed_constraints.append(parsed_c)
                    except Exception as c_exc:
                        return f"error: invalid_match_constraint:{c_exc}"

        # REQ-032-FIX3: atomic fencing commit at write boundary
        from apps.group_agent_api.execution.active_fence import (
            FenceRejectedError,
            assert_write_allowed,
            commit_profile_write_allowed,
            get_active_fence,
        )

        try:
            assert_write_allowed("save_group_profile")
            commit_profile_write_allowed(user_id=user_id, group_id=group_id)
        except FenceRejectedError as exc:
            UC34Observer.warn(
                f"action=save_group_profile_fence_rejected user_id={user_id} "
                f"group_id={group_id} reason={exc.code}"
            )
            return f"error: fence_rejected:{exc.code}"

        fence = get_active_fence()
        attempt_id = ""
        fencing_token = 0
        if fence is not None:
            attempt_id = fence.claim.attempt_id
            fencing_token = int(fence.claim.fencing_token)
        else:
            meta_attempt = str(metadata.get("attempt_id") or "").strip()
            meta_fencing = str(metadata.get("fencing_token") or "").strip()
            if meta_attempt and meta_fencing.isdigit():
                attempt_id = meta_attempt
                fencing_token = int(meta_fencing)

        # Build v2 claims
        def _make_v2_claim(
            val: str,
            disc: str,
            ctype_str: str,
            ev_text: str | None,
        ) -> ProfileClaimV2:
            try:
                c_type = ClaimType(ctype_str)
            except ValueError:
                c_type = ClaimType.fact
            try:
                d_level = DisclosureLevelV2(disc)
            except ValueError:
                d_level = DisclosureLevelV2.inferred_unconfirmed

            evidence_list: list[ProfileEvidenceV2] = []
            ev_source_text = ev_text or val
            if ev_source_text:
                ev_obj = ProfileEvidenceV2(
                    source_type="conversation_message",
                    source_message_id=int(source_message_id_trusted) if source_message_id_trusted else None,
                    evidence_text=ev_source_text,
                    evidence_digest=canonical_sha256(ev_source_text),
                )
                evidence_list.append(ev_obj)

            return ProfileClaimV2(
                value=val,
                disclosure=d_level,
                claim_type=c_type,
                confidence=ConfidenceLevel.user_stated,
                quality_status=QualityStatus.active,
                evidence=evidence_list,
            )

        doing_claim = _make_v2_claim(doing_val, doing_disc, doing_ctype, doing_ev)
        need_claim = _make_v2_claim(need_val, need_disc, need_ctype, need_ev)
        offer_claim = _make_v2_claim(offer_val, offer_disc, offer_ctype, offer_ev)

        # Grounding validation check
        if source_message_text_trusted:
            for dim_name, cl in [("doing", doing_claim), ("need", need_claim), ("offer", offer_claim)]:
                dim_violations = validate_profile_claim_grounding(cl, str(source_message_text_trusted))
                if dim_violations:
                    UC34Observer.warn(
                        f"action=profile_claim_grounding_warning dim={dim_name} violations={dim_violations}"
                    )

        profile = GroupProfile(
            user_id=user_id,
            group_id=group_id,
            doing=ProfileField(value=doing_val, disclosure=DisclosureLevel(doing_disc)),
            need=ProfileField(value=need_val, disclosure=DisclosureLevel(need_disc)),
            offer=ProfileField(value=offer_val, disclosure=DisclosureLevel(offer_disc)),
            match_constraints=[c.model_dump(mode="json") for c in parsed_constraints],
            schema_version=1,
        )

        remote_ack: dict[str, Any] | None = None
        if integration_mode() == "http":
            remote_ack = persist_group_profile(
                profile=profile,
                run_id=run_id,
                attempt_id=attempt_id or None,
                fencing_token=fencing_token or None,
            )
        if remote_ack is None or remote_ack["status"] not in {"stale_ignored", "fence_rejected"}:
            if remote_ack is not None:
                # The version is allocated by Micro.  Persist it alongside the
                # local cache so a later typed profile_confirmation can quote
                # the exact authoritative row version instead of guessing.
                profile.profile_version = remote_ack["profile_version"]
            try:
                commit_profile_write_allowed(user_id=user_id, group_id=group_id)
            except FenceRejectedError as exc:
                return f"error: fence_rejected:{exc.code}"
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
        try:
            from apps.group_agent_api.agent_factory.profile_quality import (
                bind_profile_to_episode,
            )

            bind_profile_to_episode(
                base_dir, user_id, group_id, metadata=metadata
            )
        except Exception:  # noqa: BLE001
            pass
        return (
            "ok: saved profile to database "
            f"(version={remote_ack['profile_version']}) and local cache"
        )
    try:
        from apps.group_agent_api.agent_factory.profile_quality import (
            bind_profile_to_episode,
        )

        bind_profile_to_episode(base_dir, user_id, group_id, metadata=metadata)
    except Exception:  # noqa: BLE001
        pass
    return f"ok: saved profile to /users/{user_id}/groups/{group_id}/profile.json"


def member_system_prompt() -> str:
    """Compose member system prompt; append search_relax addon when Module is on."""
    from apps.group_agent_api.agent_factory.search_relax import (
        search_relax_system_addon,
    )

    return SYSTEM_PROMPT + search_relax_system_addon()


def create_agent(
    *,
    base_dir: Path | None = None,
    model: Any | None = None,
    checkpointer: Any | None = None,
) -> tuple[Any, Any, Path]:
    """Create member + admin agents. Returns (member_agent, admin_agent, checkpoints_path)."""
    runtime = base_dir or default_runtime_dir()
    runtime.mkdir(parents=True, exist_ok=True)
    UCObserver.set_log_dir(runtime / "logs")

    ckpt_path = runtime / "checkpoints.pkl"
    ckpt = checkpointer or DiskBackedInMemorySaver(file_path=ckpt_path)
    backend = FilesystemBackend(root_dir=str(runtime), virtual_mode=True)
    llm = model or create_model()

    agent = create_deep_agent(
        model=llm,
        tools=[save_group_profile, search_candidates],
        system_prompt=member_system_prompt(),
        backend=backend,
        checkpointer=ckpt,
        # Slice 1: no subagents, no skills, no free-form memory middleware
    )
    admin_agent = create_deep_agent(
        model=llm,
        tools=list(ADMIN_OPS_TOOLS),
        system_prompt=ADMIN_SYSTEM_PROMPT,
        backend=backend,
        checkpointer=ckpt,
    )
    _logger.info("group_agent_api ready runtime=%s (member+admin agents)", runtime)
    return agent, admin_agent, ckpt_path
