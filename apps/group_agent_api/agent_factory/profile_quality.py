"""Profile match-ready quality gate (REQ-029).

Layer 1: hard length thresholds (no LLM).
Layer 2: LLM semantic richness / completeness / focus (JSON ready + gaps).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apps.group_agent_api.agent_factory.content_quality import (
    is_need_shaped_doing,
    is_preference_shaped_offer,
)
from apps.group_agent_api.agent_factory.profile_schema import GroupProfile
from apps.group_agent_api.agent_factory.profile_store import validate_id

_logger = logging.getLogger("uvicorn.error")

MIN_DOING_CHARS = 5
MIN_NEED_CHARS = 6
MIN_OFFER_CHARS = 4
MAX_THIN_SKIPS_BEFORE_DEGRADED = 3

# Explicit start/match — overrides thin-gate + clarifying deferral.
# Bare「开始」must be whole-message (avoid「继续上次的方向」false positive).
# Also treat natural "do you have someone / I need a partner" as search intent
# so a complete first dump is not trapped in clarification (PRC-11 SC-1101).
_FORCE_MATCH_INTENT = re.compile(
    r"(?:先匹配|先搜一下|先搜索|先推荐|直接匹配|直接搜|不用再问|先找人|"
    r"开始匹配|开始找人|开始搜|帮我(?:在(?:群里|本群|全网))?匹配|帮我(?:在(?:群里|本群|全网))?找人|"
    r"立刻匹配|马上匹配|启动匹配|在(?:群里|本群|全网)匹配|"
    r"你这边有合适|有没有合适|有合适的(?:人|人选|伙伴|合伙人|搭子)?吗|"
    r"有合适的(?:人|人选|伙伴|合伙人|搭子)|"
    r"有没有人选|有人选吗|"
    r"帮我找(?:一?[个位名])?(?:人|伙伴|合伙人|合创|搭子|小伙伴)|"
    r"(?:想找|需要找|缺一个|找一个|找一位|找一名).{0,24}"
    r"(?:合伙人|合创|搭档|搭子|小伙伴))"
    r"|^(?:开始|开始吧)$"
)

# User is supplying doing/need/offer (or a direction switch), not a greeting.
_PROFILE_BEARING_MESSAGE = re.compile(
    r"(?:正在做|在做|我做|做的是|项目是|面向.{0,12}(?:学生|用户|客户)|"
    r"需要找|想找|缺(?:一个|一位)|能提供|有.{0,8}(?:能力|经验|原型|资源)|"
    r"合伙人|合创|搭子|教研|教培|社群运营|内容变现)"
)

# Model still digging — do not attach match cards on the same turn.
_CLARIFYING_REPLY = re.compile(
    r"(能说说|再确认|帮你挖|更具体一点|先聚焦|补一个|哪一类|"
    r"具体希望|请挑一个|先从第|我们先|对吗[？?]|"
    r"是学科老师|做过哪类|是否参与过|三个关键点|关键细节|"
    r"你希望优先看|优先看「|还是更倾向|更倾向「|"
    r"会影响匹配权重|对齐最相关|选一个你当前|"
    r"选一个方向|挑一个你当前)"
)

# A/B priority fork even when the model says「可直接匹配」in the same breath.
_PRIORITY_FORK = re.compile(
    r"(?:优先看|更倾向|更侧重).{0,40}还是|"
    r"还是.{0,40}(?:更倾向|更侧重|搭档|的人)"
)


_QUALITY_SYSTEM_PROMPT = """你是群内匹配前的画像质量裁判。根据用户已落库的三维字段，判断是否达到「可匹配的最低充分」——不是完美商业计划。

评判维度：
- 丰富度：场景/方向是否具体到能做有方向的匹配
- 完善度：doing / need / offer 是否各自成义
- 聚焦度：是否空泛到无法区分对接对象（如仅有「创业/交流/找资源」）

放行标准（偏松）：
- doing 能看出在做什么方向或场景
- need 能看出缺哪类帮助或卡点
- offer 能看出能拿出什么（可较粗）
- 不要因缺融资额、团队人数、精确职称等次要细节而判定不通过

仅在信息空泛到无法区分匹配方向时 ready=false。

只输出一个 JSON 对象，不要 markdown：
{"ready":bool,"score":0-100,"doing_ok":bool,"need_ok":bool,"offer_ok":bool,"reasons":[string],"gaps":[string]}
gaps 最多 2 条、口语、用于下一轮追问；禁止编造用户未提供的事实。
"""

GENERIC_GAP = "再具体一点你在做的事，以及你现在最卡的是人、渠道还是技术？"


@dataclass
class ProfileQuality:
    ready: bool
    score: int = 0
    gaps: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    layer_failed: str | None = None  # length | semantic | role | unavailable | None
    source: str = "rules"  # rules | llm | cache


@dataclass
class MatchGateDecision:
    allow_match: bool
    match_reason: str | None
    quality: ProfileQuality
    degraded: bool = False
    thin_skip_count: int = 0


def wants_force_match(message: str | None) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    return bool(_FORCE_MATCH_INTENT.search(text))


def looks_like_profile_bearing_message(message: str | None) -> bool:
    """True when the user turn looks like a profile dump or direction switch.

    Used to force-save a stale prior-episode profile so a new complete statement
    is not left un-bound (and therefore never match-gated).
    """
    text = (message or "").strip()
    if len(text) < 18:
        return False
    return bool(_PROFILE_BEARING_MESSAGE.search(text))


def should_defer_match_for_clarifying(
    *,
    reply: str | None,
    user_message: str | None,
    profile_ok: bool,
) -> bool:
    """Defer match only while still collecting a usable episode profile.

    If this episode already has a bound profile, or the user asked to find
    people, a follow-up question from the dialogue model must not block search.
    """
    if profile_ok or wants_force_match(user_message):
        return False
    return looks_like_clarifying_reply(reply)


def looks_like_clarifying_reply(reply: str | None) -> bool:
    """True when the model is still asking need-shaping questions.

    Used to defer match/invite so clarifying turns do not suddenly dump cards.
    """
    text = (reply or "").strip()
    if not text:
        return False
    # Explicit handoff to match — leave gate alone (user may still need to confirm).
    if re.search(r"(是否(?:现在)?启动匹配|是否现在输出匹配|帮你匹配|启动匹配[？?])", text):
        return True
    # A/B weight fork (「优先看 A 还是 B」) — defer even with a single 「？」.
    if _PRIORITY_FORK.search(text):
        return True
    qmarks = text.count("？") + text.count("?")
    if qmarks >= 2:
        return True
    if qmarks >= 1 and _CLARIFYING_REPLY.search(text):
        return True
    return False


def _field_value(profile: GroupProfile, name: str) -> str:
    field_obj = getattr(profile, name, None)
    return str(getattr(field_obj, "value", "") or "").strip()


def _length_for_gate(text: str) -> int:
    """Count chars for length gates; ignore whitespace so「做 AI 教育」≈「做AI教育」."""
    return len(re.sub(r"\s+", "", text or ""))


def profile_fingerprint(profile: GroupProfile) -> str:
    raw = "|".join(
        [
            _field_value(profile, "doing"),
            _field_value(profile, "need"),
            _field_value(profile, "offer"),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def assess_length_and_role(profile: GroupProfile) -> ProfileQuality:
    """Layer 1 + role misplacement hard gates (no LLM)."""
    doing = _field_value(profile, "doing")
    need = _field_value(profile, "need")
    offer = _field_value(profile, "offer")
    reasons: list[str] = []
    gaps: list[str] = []

    if _length_for_gate(doing) < MIN_DOING_CHARS:
        reasons.append("doing_too_short")
        gaps.append("你在做的具体产品或场景，再多说一两句？")
    if _length_for_gate(need) < MIN_NEED_CHARS:
        reasons.append("need_too_short")
        gaps.append("你现在最卡的是哪一类帮助（人/渠道/技术等）？")
    if _length_for_gate(offer) < MIN_OFFER_CHARS:
        reasons.append("offer_too_short")
        gaps.append("你这边能提供的具体资源或能力是什么？")

    if reasons:
        return ProfileQuality(
            ready=False,
            score=max(0, 40 - 10 * len(reasons)),
            gaps=gaps[:2],
            reasons=reasons,
            layer_failed="length",
            source="rules",
        )

    if is_need_shaped_doing(doing):
        return ProfileQuality(
            ready=False,
            score=30,
            gaps=["先说清你正在推进的项目本身，缺的人放到「需求」里。"],
            reasons=["doing_describes_need"],
            layer_failed="role",
            source="rules",
        )
    if is_preference_shaped_offer(offer):
        return ProfileQuality(
            ready=False,
            score=30,
            gaps=["你能提供的具体资源或能力是什么？（不只是合作方式偏好）"],
            reasons=["offer_describes_preference"],
            layer_failed="role",
            source="rules",
        )

    return ProfileQuality(ready=True, score=50, source="rules")


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


def _parse_quality_json(text: str) -> ProfileQuality | None:
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
    ready = bool(data.get("ready"))
    try:
        score = int(data.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))
    reasons = [str(x) for x in (data.get("reasons") or []) if str(x).strip()][:8]
    gaps = [str(x).strip() for x in (data.get("gaps") or []) if str(x).strip()][:2]
    if not ready and not gaps:
        gaps = [GENERIC_GAP]
    return ProfileQuality(
        ready=ready,
        score=score,
        gaps=gaps,
        reasons=reasons or (["semantic_thin"] if not ready else []),
        layer_failed=None if ready else "semantic",
        source="llm",
    )


def assess_with_llm(*, profile: GroupProfile, model: Any | None) -> ProfileQuality:
    """Layer 2 LLM semantic judge. Fail-closed on missing model / bad parse."""
    if model is None:
        _logger.warning("action=profile_quality_unavailable reason=no_model")
        return ProfileQuality(
            ready=False,
            score=0,
            gaps=[GENERIC_GAP],
            reasons=["quality_model_unavailable"],
            layer_failed="unavailable",
            source="rules",
        )
    doing = _field_value(profile, "doing")
    need = _field_value(profile, "need")
    offer = _field_value(profile, "offer")
    user_payload = (
        f"doing: {doing}\nneed: {need}\noffer: {offer}\n请输出 JSON。"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        msg = model.invoke(
            [
                SystemMessage(content=_QUALITY_SYSTEM_PROMPT),
                HumanMessage(content=user_payload),
            ]
        )
        text = _extract_text_response(getattr(msg, "content", None))
        parsed = _parse_quality_json(text)
        if parsed is None:
            _logger.warning("action=profile_quality_parse_failed")
            return ProfileQuality(
                ready=False,
                score=0,
                gaps=[GENERIC_GAP],
                reasons=["quality_parse_failed"],
                layer_failed="unavailable",
                source="llm",
            )
        return parsed
    except Exception as exc:  # noqa: BLE001
        _logger.warning("action=profile_quality_llm_failed error=%s", exc)
        return ProfileQuality(
            ready=False,
            score=0,
            gaps=[GENERIC_GAP],
            reasons=["quality_llm_failed"],
            layer_failed="unavailable",
            source="llm",
        )


def disk_match_gate_path(base_dir: Path, user_id: str, group_id: str) -> Path:
    uid = validate_id(user_id, field="user_id")
    gid = validate_id(group_id, field="group_id")
    root = base_dir.resolve()
    path = (root / "users" / uid / "groups" / gid / "match_gate.json").resolve()
    try:
        path.relative_to(root / "users")
    except ValueError as exc:
        raise ValueError(f"path escape blocked for match_gate {uid}/{gid}") from exc
    return path


def _load_gate(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_gate(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def episode_key_from_metadata(metadata: dict[str, Any] | None) -> str:
    meta = metadata or {}
    for key in ("episode_id", "episodeId"):
        val = str(meta.get(key) or "").strip()
        if val:
            return val[:128]
    return "_default"


def bind_profile_to_episode(
    base_dir: Path,
    user_id: str,
    group_id: str,
    *,
    metadata: dict[str, Any] | None,
) -> None:
    """Record which episode last successfully refreshed the profile."""
    path = disk_match_gate_path(base_dir, user_id, group_id)
    gate = _load_gate(path)
    gate["profile_bound_episode"] = episode_key_from_metadata(metadata)
    _save_gate(path, gate)


def profile_bound_to_episode(
    base_dir: Path,
    user_id: str,
    group_id: str,
    *,
    metadata: dict[str, Any] | None,
) -> bool:
    """True when persisted profile was last saved under this episode."""
    ep = episode_key_from_metadata(metadata)
    if ep == "_default":
        # No episode isolation available — allow reuse of existing profile.
        return True
    path = disk_match_gate_path(base_dir, user_id, group_id)
    gate = _load_gate(path)
    return str(gate.get("profile_bound_episode") or "") == ep


def assess_profile_match_ready(
    *,
    profile: GroupProfile,
    model: Any | None,
    base_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> ProfileQuality:
    """Full assess with Layer1 → Layer2, using fingerprint cache in match_gate.json."""
    pre = assess_length_and_role(profile)
    if not pre.ready:
        return pre

    fp = profile_fingerprint(profile)
    path = disk_match_gate_path(base_dir, profile.user_id, profile.group_id)
    gate = _load_gate(path)
    ep = episode_key_from_metadata(metadata)
    cached = gate.get("quality_cache") if isinstance(gate.get("quality_cache"), dict) else {}
    if (
        cached.get("fingerprint") == fp
        and cached.get("episode_id") == ep
        and isinstance(cached.get("ready"), bool)
    ):
        return ProfileQuality(
            ready=bool(cached["ready"]),
            score=int(cached.get("score") or 0),
            gaps=[str(x) for x in (cached.get("gaps") or []) if str(x).strip()][:2],
            reasons=[str(x) for x in (cached.get("reasons") or [])][:8],
            layer_failed=cached.get("layer_failed"),
            source="cache",
        )

    quality = assess_with_llm(profile=profile, model=model)
    gate["quality_cache"] = {
        "fingerprint": fp,
        "episode_id": ep,
        "ready": quality.ready,
        "score": quality.score,
        "gaps": quality.gaps,
        "reasons": quality.reasons,
        "layer_failed": quality.layer_failed,
    }
    _save_gate(path, gate)
    return quality


def decide_match_gate(
    *,
    profile: GroupProfile,
    model: Any | None,
    base_dir: Path,
    message: str | None,
    metadata: dict[str, Any] | None = None,
) -> MatchGateDecision:
    """Apply quality + thin-skip / force-match policy (1A + 2A)."""
    ep = episode_key_from_metadata(metadata)
    path = disk_match_gate_path(base_dir, profile.user_id, profile.group_id)
    gate = _load_gate(path)
    episodes = gate.get("episodes") if isinstance(gate.get("episodes"), dict) else {}
    ep_state = episodes.get(ep) if isinstance(episodes.get(ep), dict) else {}
    thin_skip_count = int(ep_state.get("thin_skip_count") or 0)

    force = wants_force_match(message)
    quality = assess_profile_match_ready(
        profile=profile,
        model=model,
        base_dir=base_dir,
        metadata=metadata,
    )

    gate = _load_gate(path)
    episodes = gate.get("episodes") if isinstance(gate.get("episodes"), dict) else {}
    ep_state = dict(episodes.get(ep) if isinstance(episodes.get(ep), dict) else {})

    if quality.ready:
        ep_state["thin_skip_count"] = 0
        episodes[ep] = ep_state
        gate["episodes"] = episodes
        _save_gate(path, gate)
        return MatchGateDecision(
            allow_match=True,
            match_reason=None,
            quality=quality,
            degraded=False,
            thin_skip_count=0,
        )

    # Force-match / 3-skip degrade override semantic thin AND judge unavailable.
    if force or thin_skip_count >= MAX_THIN_SKIPS_BEFORE_DEGRADED:
        ep_state["thin_skip_count"] = thin_skip_count
        ep_state["degraded"] = True
        episodes[ep] = ep_state
        gate["episodes"] = episodes
        _save_gate(path, gate)
        return MatchGateDecision(
            allow_match=True,
            match_reason="profile_thin_degraded",
            quality=quality,
            degraded=True,
            thin_skip_count=thin_skip_count,
        )

    if quality.layer_failed == "unavailable":
        return MatchGateDecision(
            allow_match=False,
            match_reason="profile_quality_unavailable",
            quality=quality,
            degraded=False,
            thin_skip_count=thin_skip_count,
        )

    thin_skip_count += 1
    ep_state["thin_skip_count"] = thin_skip_count
    episodes[ep] = ep_state
    gate["episodes"] = episodes
    _save_gate(path, gate)
    return MatchGateDecision(
        allow_match=False,
        match_reason="profile_too_thin",
        quality=quality,
        degraded=False,
        thin_skip_count=thin_skip_count,
    )
