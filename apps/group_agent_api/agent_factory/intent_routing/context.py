"""Trusted L1 intent context recovered from completed checkpoint tool results."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .canonical import canonical_hash
from .constraint_lexicon import canonicalize_constraint_token
from .schema import (
    ClassifierContextV1,
    IntentConstraintsV1,
    IntentDecisionV1,
    IntentKind,
    PlannerContextV2,
    PlannerProfileV2,
    PlannerTurnV2,
    ResourceQueriesV2,
)
from .signals import normalize_text

_MAX_RECENT_USER_TURNS = 5
_MAX_RECENT_TURN_CHARS = 240
_MAX_CONSTRAINT_TOKEN_CHARS = 64
_MAX_EPISODE_TURNS = 40
_MAX_TURN_CHARS = 1000
_MAX_EPISODE_CHARS = 12_000
_MAX_PROFILE_FIELD_CHARS = 500
_MAX_MATCH_CONSTRAINT_LINES = 32
_PROFILE_FIELD_TO_CONSTRAINT = {
    "city": "location",
    "location": "location",
    "industry": "industry",
    "role": "role",
    "experience_tags": "capability",
    "capability": "capability",
}
_POSITIVE_PROFILE_OPERATORS = frozenset({"eq", "in", "all", "any"})
_NEGATIVE_PROFILE_OPERATORS = frozenset({"not_in", "not_any"})

_RESOURCE_INTENT: dict[str, IntentKind] = {
    "users": "people_search",
    "ideas": "opportunity_search",
    "projects": "project_partnership_search",
    "groups": "community_search",
    "group-members": "community_search",
    "weekly": "startup_digest",
    "blog": "startup_digest",
    "faqs": "platform_help",
}
_ELLIPTICAL = ("的呢", "换成", "方向", "不要", "改成", "那", "合适")
_SERVER_CONTEXT_KEY = "group_agent_completed_intent_context_v1"
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_INTENT_PLAN = {
    "people_search": "people-search-v1",
    "opportunity_search": "opportunity-search-v1",
    "project_partnership_search": "project-partnership-search-v1",
    "community_search": "community-search-v1",
    "startup_digest": "startup-digest-v1",
    "platform_help": "platform-help-v1",
}


@dataclass(frozen=True)
class IntentContextScope:
    """Scope already enforced by the checkpoint thread identifier."""

    thread_id: str
    user_id: str
    execution_mode: str
    episode_id: str | None
    run_status: Literal["completed", "failed", "cancelled"] = "completed"

    def __post_init__(self) -> None:
        """Reject unscoped or cross-mode context at the trust boundary."""
        if not self.thread_id or not self.user_id:
            msg = "intent context requires thread_id and user_id"
            raise ValueError(msg)
        if self.execution_mode not in {"group_agent", "cafe_host"}:
            msg = "intent context execution_mode is invalid"
            raise ValueError(msg)


@dataclass(frozen=True)
class TrustedIntentContext:
    """Last completed business tool route and normalized arguments."""

    intent: IntentKind
    constraints: IntentConstraintsV1
    scope: IntentContextScope
    tool_call_id: str


def completed_intent_context_message(
    *,
    decision: IntentDecisionV1,
    scope: IntentContextScope,
    routing_snapshot_hash: str,
    plan_id: str,
    receipt_ids: list[str],
    user_message: str,
) -> SystemMessage:
    """Create a server-role marker after QueryPlan success and before composition."""
    if decision.action != "execute" or not receipt_ids:
        raise ValueError("completed intent context requires successful receipts")
    body = {
        "version": 1,
        "intent": decision.primary_intent,
        "constraints": decision.constraints.model_dump(mode="json"),
        "thread_id": scope.thread_id,
        "user_id": scope.user_id,
        "execution_mode": scope.execution_mode,
        "episode_id": scope.episode_id,
        "routing_snapshot_hash": routing_snapshot_hash,
        "plan_id": plan_id,
        "receipt_ids": receipt_ids[:20],
        "user_message_hash": canonical_hash({"message": user_message}),
    }
    marker = {**body, "context_hash": canonical_hash(body)}
    return SystemMessage(
        content="[SERVER-COMPLETED-INTENT-CONTEXT]",
        additional_kwargs={_SERVER_CONTEXT_KEY: marker},
    )


def is_elliptical_followup(message: str) -> bool:
    """Return whether a short message can inherit an earlier business object."""
    text = normalize_text(message)
    return len(text) <= 80 and any(marker in text for marker in _ELLIPTICAL)


def _successful_tool_ids(messages: list[Any]) -> set[str]:
    successful_results: dict[str, int] = {}
    completed_at = {
        index
        for index, message in enumerate(messages)
        if isinstance(message, AIMessage)
        and bool(str(message.content or "").strip())
        and not (getattr(message, "tool_calls", None) or [])
    }
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content or "").strip()
        if (
            not content
            or content.startswith("error:")
            or any(
                marker in content.lower()
                for marker in ("timed_out", "rejected", "cancelled", "superseded")
            )
        ):
            continue
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict) and parsed.get("status") in {
            "failed",
            "error",
            "timed_out",
            "rejected",
        }:
            continue
        tool_call_id = str(getattr(message, "tool_call_id", "") or "")
        if tool_call_id:
            successful_results[tool_call_id] = index
    return {
        tool_call_id
        for tool_call_id, result_index in successful_results.items()
        if any(final_index > result_index for final_index in completed_at)
    }


def _constraints_from_args(args: dict[str, Any]) -> IntentConstraintsV1:
    def values(name: str) -> tuple[str, ...]:
        value = args.get(name)
        if isinstance(value, str) and value.strip():
            return (value.strip()[:64],)
        if isinstance(value, list):
            return tuple(
                str(item).strip()[:64] for item in value[:10] if str(item).strip()
            )
        return ()

    time_range = args.get("time_range")
    return IntentConstraintsV1(
        location=values("location"),
        industry=values("industry"),
        role=values("role"),
        capability=values("capability"),
        time_range=str(time_range).strip()[:64] if time_range else None,
    )


def _completed_server_context(
    messages: list[Any], *, scope: IntentContextScope
) -> TrustedIntentContext | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, SystemMessage):
            continue
        marker = message.additional_kwargs.get(_SERVER_CONTEXT_KEY)
        if not isinstance(marker, dict):
            continue
        expected_keys = {
            "version",
            "intent",
            "constraints",
            "thread_id",
            "user_id",
            "execution_mode",
            "episode_id",
            "routing_snapshot_hash",
            "plan_id",
            "receipt_ids",
            "user_message_hash",
            "context_hash",
        }
        if set(marker) != expected_keys or marker.get("version") != 1:
            continue
        body = {key: value for key, value in marker.items() if key != "context_hash"}
        try:
            valid_hash = marker.get("context_hash") == canonical_hash(body)
        except Exception:  # noqa: BLE001 - malformed checkpoint marker is ignored.
            valid_hash = False
        if not valid_hash:
            continue
        human_seen = False
        completed_same_turn = False
        for following in messages[index + 1 :]:
            if (
                isinstance(following, SystemMessage)
                and _SERVER_CONTEXT_KEY in following.additional_kwargs
            ):
                break
            if isinstance(following, HumanMessage):
                if human_seen:
                    break
                human_seen = True
                if marker.get("user_message_hash") != canonical_hash(
                    {"message": str(following.content or "")}
                ):
                    break
                continue
            if (
                human_seen
                and isinstance(following, AIMessage)
                and bool(str(following.content or "").strip())
                and not (getattr(following, "tool_calls", None) or [])
            ):
                completed_same_turn = True
                break
        if not completed_same_turn:
            continue
        if (
            marker.get("thread_id") != scope.thread_id
            or marker.get("user_id") != scope.user_id
            or marker.get("execution_mode") != scope.execution_mode
            or marker.get("episode_id") != scope.episode_id
        ):
            continue
        intent = marker.get("intent")
        receipt_ids = marker.get("receipt_ids")
        if (
            intent not in set(_RESOURCE_INTENT.values())
            or not isinstance(receipt_ids, list)
            or not receipt_ids
        ):
            continue
        if (
            marker.get("plan_id") != _INTENT_PLAN.get(intent)
            or not isinstance(marker.get("routing_snapshot_hash"), str)
            or _HASH.fullmatch(marker["routing_snapshot_hash"]) is None
            or len(receipt_ids) > 20
            or any(
                not isinstance(receipt_id, str) or _HASH.fullmatch(receipt_id) is None
                for receipt_id in receipt_ids
            )
        ):
            continue
        try:
            constraints = IntentConstraintsV1.model_validate(marker.get("constraints"))
        except Exception:  # noqa: BLE001 - malformed checkpoint marker is ignored.
            continue
        return TrustedIntentContext(
            intent=intent,
            constraints=constraints,
            scope=scope,
            tool_call_id=f"server:{marker['context_hash']}",
        )
    return None


def extract_completed_context(
    messages: list[Any],
    *,
    scope: IntentContextScope,
) -> TrustedIntentContext | None:
    """Recover only a completed tool call from the already scoped checkpoint.

    Args:
        messages: Messages loaded from the exact current checkpoint thread.
        scope: Trusted user/mode/episode provenance of that checkpoint.

    Returns:
        The newest completed supported business route, or `None`.
    """
    if scope.run_status != "completed":
        return None
    server_context = _completed_server_context(messages, scope=scope)
    if server_context is not None:
        return server_context
    successful = _successful_tool_ids(messages)
    if not successful:
        return None
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in reversed(list(getattr(message, "tool_calls", None) or [])):
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            if call_id not in successful:
                continue
            name = str(call.get("name") or "")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if name == "search_candidates":
                intent: IntentKind | None = "people_search"
            elif name == "query_aihehuo_public_data":
                intent = _RESOURCE_INTENT.get(str(args.get("resource") or ""))
            else:
                intent = None
            if intent is not None:
                return TrustedIntentContext(
                    intent=intent,
                    constraints=_constraints_from_args(args),
                    scope=scope,
                    tool_call_id=call_id,
                )
    return None


def merge_followup_constraints(
    previous: IntentConstraintsV1,
    current: IntentConstraintsV1,
    message: str,
) -> tuple[IntentConstraintsV1, tuple[str, ...]]:
    """Apply frozen replace/delete semantics for an elliptical follow-up."""
    text = normalize_text(message)
    locations = current.location or previous.location
    roles = current.role or previous.role
    industries = current.industry or previous.industry
    capabilities = current.capability or previous.capability
    negations: list[str] = []
    if "不要" in text:
        for location in previous.location:
            if location.lower() in text:
                locations = tuple(value for value in locations if value != location)
                negations.append(f"location:{location}")
    return (
        IntentConstraintsV1(
            location=locations,
            industry=industries,
            role=roles,
            capability=capabilities,
            time_range=current.time_range or previous.time_range,
        ),
        tuple(negations),
    )


def _clip_constraint_tokens(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    clipped: list[str] = []
    for value in values:
        text = str(value or "").strip()[:_MAX_CONSTRAINT_TOKEN_CHARS]
        if text and text not in clipped:
            clipped.append(text)
        if len(clipped) >= 10:
            break
    return tuple(clipped)


def bound_intent_constraints(
    constraints: IntentConstraintsV1 | None,
) -> IntentConstraintsV1:
    """Clip every constraint token to a fixed length and item count."""
    if constraints is None:
        return IntentConstraintsV1()
    time_range = constraints.time_range
    if time_range is not None:
        time_range = str(time_range).strip()[:_MAX_CONSTRAINT_TOKEN_CHARS] or None
    return IntentConstraintsV1(
        location=_clip_constraint_tokens(
            tuple(
                canonicalize_constraint_token("location", value)
                for value in constraints.location
            )
        ),
        industry=_clip_constraint_tokens(
            tuple(
                canonicalize_constraint_token("industry", value)
                for value in constraints.industry
            )
        ),
        role=_clip_constraint_tokens(
            tuple(
                canonicalize_constraint_token("role", value)
                for value in constraints.role
            )
        ),
        capability=_clip_constraint_tokens(constraints.capability),
        time_range=time_range,
    )


def intent_constraints_from_profile(profile: Any | None) -> IntentConstraintsV1 | None:
    """Return only positive profile constraints; see ``profile_routing_hints``."""
    constraints, _negations = profile_routing_hints(profile)
    return constraints


def profile_routing_hints(
    profile: Any | None,
) -> tuple[IntentConstraintsV1 | None, tuple[str, ...]]:
    """Convert profile match_constraints into positive constraints and exclusions.

    Positive operators project into forward constraints. Negative operators such
    as ``not_in`` / ``not_any`` become ``kind:value`` negations and never invert
    into positive search objects. Values are canonicalized through the shared
    constraint lexicon when possible.
    """
    if profile is None:
        return None, ()
    buckets: dict[str, list[str]] = {
        "location": [],
        "industry": [],
        "role": [],
        "capability": [],
    }
    negations: list[str] = []
    raw_items = getattr(profile, "match_constraints", None) or []
    if not isinstance(raw_items, list):
        return None, ()
    for item in raw_items:
        if isinstance(item, dict):
            field = str(item.get("field") or "").strip()
            operator = str(item.get("operator") or "").strip()
            values = item.get("values") or []
        else:
            field = str(getattr(item, "field", "") or "").strip()
            operator = str(getattr(item, "operator", "") or "").strip()
            values = getattr(item, "values", None) or []
        target = _PROFILE_FIELD_TO_CONSTRAINT.get(field)
        if target is None or not isinstance(values, list):
            continue
        if operator in _POSITIVE_PROFILE_OPERATORS:
            for value in values:
                text = canonicalize_constraint_token(
                    target, str(value or "").strip()[:_MAX_CONSTRAINT_TOKEN_CHARS]
                )
                if text and text not in buckets[target]:
                    buckets[target].append(text)
                if len(buckets[target]) >= 10:
                    break
            continue
        if operator in _NEGATIVE_PROFILE_OPERATORS:
            for value in values:
                text = canonicalize_constraint_token(
                    target, str(value or "").strip()[:_MAX_CONSTRAINT_TOKEN_CHARS]
                )
                key = f"{target}:{text}"
                if text and key not in negations:
                    negations.append(key)
                if len(negations) >= 16:
                    break
    constraints = IntentConstraintsV1(
        location=tuple(buckets["location"]),
        industry=tuple(buckets["industry"]),
        role=tuple(buckets["role"]),
        capability=tuple(buckets["capability"]),
    )
    if not (
        constraints.location
        or constraints.industry
        or constraints.role
        or constraints.capability
    ):
        return None, tuple(negations)[:16]
    return constraints, tuple(negations)[:16]


def _bounded_recent_user_turns(messages: list[Any] | None) -> tuple[str, ...]:
    turns: list[str] = []
    for message in messages or []:
        if not isinstance(message, HumanMessage):
            continue
        text = str(message.content or "").strip()
        if not text:
            continue
        turns.append(text[:_MAX_RECENT_TURN_CHARS])
    return tuple(turns[-_MAX_RECENT_USER_TURNS:])


def build_classifier_context(
    *,
    trusted: TrustedIntentContext | None,
    prior_messages: list[Any] | None,
    profile: IntentConstraintsV1 | None,
    current_negations: tuple[str, ...],
) -> ClassifierContextV1:
    """Assemble the closed L1 context object with strict field and length limits."""
    profile_fields = bound_intent_constraints(profile)
    prior_constraints = bound_intent_constraints(
        trusted.constraints if trusted is not None else None
    )
    return ClassifierContextV1(
        prior_intent=trusted.intent if trusted is not None else None,
        prior_constraints=prior_constraints,
        profile_location=profile_fields.location,
        profile_industry=profile_fields.industry,
        profile_role=profile_fields.role,
        profile_capability=profile_fields.capability,
        recent_user_turns=_bounded_recent_user_turns(prior_messages),
        current_negations=tuple(
            str(item).strip()[:_MAX_CONSTRAINT_TOKEN_CHARS]
            for item in current_negations
            if str(item).strip()
        )[:16],
    )


def merge_classifier_constraints(
    previous: IntentConstraintsV1 | None,
    current: IntentConstraintsV1,
) -> IntentConstraintsV1:
    """Prefer current non-empty fields, otherwise keep trusted prior constraints."""
    if previous is None:
        return bound_intent_constraints(current)
    previous = bound_intent_constraints(previous)
    current = bound_intent_constraints(current)
    return IntentConstraintsV1(
        location=current.location or previous.location,
        industry=current.industry or previous.industry,
        role=current.role or previous.role,
        capability=current.capability or previous.capability,
        time_range=current.time_range or previous.time_range,
    )


def has_structured_search_object(constraints: IntentConstraintsV1) -> bool:
    """Return whether constraints contain a usable who/what search object.

    Location and time alone are insufficient; they only modify who/what.
    """
    return bool(constraints.industry or constraints.role or constraints.capability)


def _profile_field_text(profile: Any | None, name: str) -> str | None:
    if profile is None:
        return None
    raw = getattr(profile, name, None)
    if raw is None:
        return None
    if hasattr(raw, "value"):
        raw = raw.value
    text = str(raw or "").strip()
    if not text:
        return None
    return text[:_MAX_PROFILE_FIELD_CHARS]


def _match_constraint_lines(profile: Any | None) -> tuple[str, ...]:
    if profile is None:
        return ()
    raw_items = getattr(profile, "match_constraints", None) or []
    if not isinstance(raw_items, list):
        return ()
    lines: list[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            field = str(item.get("field") or "").strip()
            operator = str(item.get("operator") or "").strip()
            values = item.get("values") or []
        else:
            field = str(getattr(item, "field", "") or "").strip()
            operator = str(getattr(item, "operator", "") or "").strip()
            values = getattr(item, "values", None) or []
        if not field or not operator or not isinstance(values, list):
            continue
        rendered_values = [
            str(value).strip()[:_MAX_CONSTRAINT_TOKEN_CHARS]
            for value in values[:10]
            if str(value).strip()
        ]
        if not rendered_values:
            continue
        line = f"{field}:{operator}:{','.join(rendered_values)}"[:160]
        if line and line not in lines:
            lines.append(line)
        if len(lines) >= _MAX_MATCH_CONSTRAINT_LINES:
            break
    return tuple(lines)


def planner_profile_from_store(profile: Any | None) -> PlannerProfileV2:
    """Project persistent profile free text and match constraints into planner data."""
    return PlannerProfileV2(
        doing=_profile_field_text(profile, "doing"),
        need=_profile_field_text(profile, "need"),
        offer=_profile_field_text(profile, "offer"),
        match_constraints=_match_constraint_lines(profile),
    )


def _episode_turns(
    messages: list[Any] | None,
) -> tuple[list[PlannerTurnV2], bool]:
    """Build episode turns; flag when any message exceeds the hard char bound."""
    turns: list[PlannerTurnV2] = []
    truncated = False
    for message in messages or []:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            if getattr(message, "tool_calls", None):
                continue
            role = "assistant"
        else:
            continue
        text = str(message.content or "").strip()
        if not text:
            continue
        if len(text) > _MAX_TURN_CHARS:
            truncated = True
            text = text[:_MAX_TURN_CHARS]
        turns.append(PlannerTurnV2(role=role, content=text))
    return turns, truncated


def _compress_episode_turns(
    turns: list[PlannerTurnV2],
) -> tuple[tuple[PlannerTurnV2, ...], bool, str | None]:
    """Keep recent turns when they fit; otherwise fail closed without silent loss.

    Lossy middle-history compression recreated the Phase-2 failure mode
    (missing industry/object terms). When the retained episode exceeds the
    bounded window, mark an explicit overflow reason so the resolver clarifies
    instead of searching on incomplete context.
    """
    if len(turns) <= _MAX_EPISODE_TURNS and sum(len(item.content) for item in turns) <= (
        _MAX_EPISODE_CHARS
    ):
        return tuple(turns), False, None
    retained: list[PlannerTurnV2] = []
    used = 0
    for turn in reversed(turns):
        cost = len(turn.content)
        if retained and (
            len(retained) >= _MAX_EPISODE_TURNS or used + cost > _MAX_EPISODE_CHARS
        ):
            break
        retained.append(turn)
        used += cost
    retained.reverse()
    return tuple(retained), True, "episode_context_overflow_fail_closed"


def queries_from_constraints(constraints: IntentConstraintsV1) -> ResourceQueriesV2:
    """Best-effort prior query hint from completed-context constraints (data only)."""
    parts = [
        *constraints.location,
        *constraints.industry,
        *constraints.role,
        *constraints.capability,
    ]
    text = " ".join(item.strip() for item in parts if item.strip())[:160]
    if not text:
        return ResourceQueriesV2()
    return ResourceQueriesV2(users=text)


def build_planner_context(
    *,
    message: str,
    prior_messages: list[Any] | None,
    profile: Any | None,
    trusted: TrustedIntentContext | None,
) -> PlannerContextV2:
    """Assemble the Phase 2 planner context with full retained episode history."""
    current = (message or "").strip()
    current_truncated = len(current) > 2000
    current = (current[:2000] if current else "(empty)")
    turns_list, turn_truncated = _episode_turns(prior_messages)
    if current_truncated or turn_truncated:
        prior_queries = ResourceQueriesV2()
        prior_intent = None
        if trusted is not None:
            prior_intent = trusted.intent
            prior_queries = queries_from_constraints(trusted.constraints)
        return PlannerContextV2(
            current_message=current or "(empty)",
            episode_turns=tuple(turns_list[-_MAX_EPISODE_TURNS:]),
            profile=planner_profile_from_store(profile),
            prior_completed_intent=prior_intent,
            prior_completed_queries=prior_queries,
            context_compressed=True,
            compression_reason="episode_turn_truncated_fail_closed",
        )
    turns, compressed, reason = _compress_episode_turns(turns_list)
    prior_queries = ResourceQueriesV2()
    prior_intent = None
    if trusted is not None:
        prior_intent = trusted.intent
        prior_queries = queries_from_constraints(trusted.constraints)
    return PlannerContextV2(
        current_message=current or "(empty)",
        episode_turns=turns,
        profile=planner_profile_from_store(profile),
        prior_completed_intent=prior_intent,
        prior_completed_queries=prior_queries,
        context_compressed=compressed,
        compression_reason=reason,
    )
