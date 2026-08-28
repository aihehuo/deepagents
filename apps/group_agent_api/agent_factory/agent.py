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
from apps.group_agent_api.agent_factory.debug_trace import record_decision_point
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
from apps.group_agent_api.agent_factory.context import (
    FORCE_SAVE_PROMPT_TEXT,
    SYSTEM_FRAGMENT_IDS,
    build_system_prompt,
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


# Default-on assembly (all system fragments). Live YAML is applied in
# ``member_system_prompt`` via ``build_system_prompt()``.
SYSTEM_PROMPT = build_system_prompt(SYSTEM_FRAGMENT_IDS)

# Stable constant for force-save loops / tests; YAML gate via force_save_prompt().
FORCE_SAVE_PROMPT = FORCE_SAVE_PROMPT_TEXT


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
        match_constraints: 匹配约束列表。city/industry/「必须·不要」用 strength=hard；
            技术栈/长尾/「比如…」用 strength=soft（如 experience_tags）。
            例：field=city operator=in values=["上海"] strength=hard。
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

        if parsed_constraints:
            hard_constraints = [
                c.model_dump(mode="json")
                for c in parsed_constraints
                if getattr(c, "strength", "") == "hard"
            ]
            soft_constraints = [
                c.model_dump(mode="json")
                for c in parsed_constraints
                if getattr(c, "strength", "") != "hard"
            ]
            record_decision_point(
                phase="constraint_extraction",
                detail={
                    "source": "save_group_profile",
                    "hard_constraints": hard_constraints,
                    "soft_constraints": soft_constraints,
                    "total_count": len(parsed_constraints),
                },
                run_id=run_id,
                thread_id=str(metadata.get("thread_id") or ""),
            )

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
    """Compose member system prompt from YAML ctx.* + optional search_relax addon."""
    from apps.group_agent_api.agent_factory.search_relax import (
        search_relax_system_addon,
    )

    return build_system_prompt() + search_relax_system_addon()


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
