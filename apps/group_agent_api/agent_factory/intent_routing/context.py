"""Trusted L1 intent context recovered from completed checkpoint tool results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage

from .schema import IntentConstraintsV1, IntentKind
from .signals import normalize_text

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
