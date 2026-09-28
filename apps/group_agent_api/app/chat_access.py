"""Run-scoped Cafe Host v4 access policy and defense-in-depth guards."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Literal

from apps.group_agent_api.app.execution_modes import (
    PAID_SEARCH_INTENTS,
    validate_v4_request,
)

AccessCode = Literal[
    "chat_session_required",
    "search_confirmation_required",
    "confirmed_search_intent_mismatch",
    "search_capability_unavailable",
    "pending_search_unavailable",
]

_INTENT_RESOURCES = {
    "people_search": frozenset({"users", "group-members"}),
    "opportunity_search": frozenset({"ideas", "blog", "weekly"}),
    "project_partnership_search": frozenset({"projects"}),
}
# Resources a confirmed Cafe Host plan may actually call. Conditional steps stay
# inside the frozen registry plan; blog/weekly remain denied.
_AUTHORIZED_RESOURCES = {
    "people_search": frozenset({"users", "group-members"}),
    "opportunity_search": frozenset({"ideas", "projects"}),
    "project_partnership_search": frozenset({"projects", "ideas"}),
}
_SEARCH_CLAIM = re.compile(
    r"(?:正在|已经|刚刚|后台|马上).{0,8}(?:搜索|查询|查找)|(?:搜索|查询).{0,8}(?:完成|结果)|(?:找到了|查到了)",
    re.IGNORECASE,
)


class ChatAccessRejected(RuntimeError):
    """Stable denial raised before a paid search side effect can occur."""


@dataclass(frozen=True)
class ChatAccessContext:
    """Trusted access snapshot frozen into one accepted v4 run."""

    mode: Literal["ordinary", "active"]
    policy_version: str
    revision: int
    drink_session_id: int | None
    pending_search_ref: str | None
    search_confirmation_id: str | None
    confirmed_search_intent: str | None
    allowed_search_intents: tuple[str, ...]
    restricted_search_intents: tuple[str, ...]
    user_token_sha256: str
    group_token_sha256: str
    user_id: str

    @classmethod
    def from_request(cls, req: Any) -> ChatAccessContext | None:
        """Construct policy only from a fully validated v4 request.

        Args:
            req: Parsed `AsyncCallRequest` or compatible frozen request.

        Returns:
            A trusted context for v4, otherwise `None` for legacy versions.
        """
        if getattr(req, "request_schema_version", None) != 4:
            return None
        body = validate_v4_request(req.model_dump())
        access = body["chat_access"]
        binding = body["credential_binding"]
        return cls(
            mode=access["mode"],
            policy_version=access["policy_version"],
            revision=access["revision"],
            drink_session_id=access["drink_session_id"],
            pending_search_ref=access["pending_search_ref"],
            search_confirmation_id=access["search_confirmation_id"],
            confirmed_search_intent=access["confirmed_search_intent"],
            allowed_search_intents=tuple(access["allowed_search_intents"]),
            restricted_search_intents=tuple(access["restricted_search_intents"]),
            user_token_sha256=binding["user_token_sha256"],
            group_token_sha256=binding["group_token_sha256"],
            user_id=str(body["user_id"]),
        )

    def control_for_intent(self, intent: str) -> AccessCode | None:
        """Return the deterministic control outcome for one paid intent.

        Args:
            intent: Server-resolved intent, never a model-selected permission.

        Returns:
            A typed denial/unavailable code, or `None` for non-paid intents.
        """
        if intent not in PAID_SEARCH_INTENTS:
            return None
        if self.mode == "ordinary":
            return "chat_session_required"
        if self.search_confirmation_id is None or self.confirmed_search_intent is None:
            return "search_confirmation_required"
        if self.confirmed_search_intent != intent:
            return "confirmed_search_intent_mismatch"
        from apps.group_agent_api.agent_factory.intent_routing.registry import (
            registry_entry,
        )
        from apps.group_agent_api.app.cafe_host_search import (
            search_execution_enabled,
            search_user_allowed,
        )

        if (
            search_execution_enabled()
            and search_user_allowed(self.user_id)
            and registry_entry(intent) is not None
        ):
            return None
        return "search_capability_unavailable"

    def paid_execution_open(self, intent: str) -> bool:
        """Return whether this frozen run may execute one paid intent now."""
        return intent in PAID_SEARCH_INTENTS and self.control_for_intent(intent) is None

    def assert_resource_allowed(self, *, intent: str, resource: str) -> None:
        """Fail closed at the scoped wrapper even if registry policy was bypassed.

        Args:
            intent: Frozen plan intent.
            resource: Requested public-data resource.

        Raises:
            ChatAccessRejected: The run has no matching paid authorization.
        """
        if self.paid_execution_open(intent) and resource in _AUTHORIZED_RESOURCES.get(
            intent, frozenset()
        ):
            from apps.group_agent_api.app.cafe_host_search import (
                current_bound_credential_digest,
            )

            if current_bound_credential_digest() != self.user_token_sha256:
                raise ChatAccessRejected("search_capability_unavailable")
            return
        if intent in PAID_SEARCH_INTENTS or any(
            resource in resources for resources in _INTENT_RESOURCES.values()
        ):
            code = self.control_for_intent(intent) or "chat_session_required"
            raise ChatAccessRejected(code)

    def guard_final(
        self,
        *,
        reply: str,
        receipt_intents: tuple[str, ...] = (),
        has_paid_results: bool = False,
        projection_digest: str | None = None,
        bound_projection_digest: str | None = None,
        card_identities: tuple[str, ...] = (),
        projection_identities: tuple[str, ...] = (),
    ) -> None:
        """Reject fabricated execution claims or mixed result projections.

        Args:
            reply: Candidate user-visible text.
            receipt_intents: Verified receipt intents for this exact run.
            has_paid_results: Whether cards/candidates would be projected.
            projection_digest: Digest about to be delivered.
            bound_projection_digest: Digest of the frozen current-run projection.
            card_identities: Identities about to be delivered.
            projection_identities: Identities frozen in that projection.

        Raises:
            ChatAccessRejected: If no valid current-run receipt supports output.
        """
        authorized = (
            self.confirmed_search_intent is not None
            and self.paid_execution_open(self.confirmed_search_intent)
            and self.confirmed_search_intent in receipt_intents
        )
        if (
            projection_digest != bound_projection_digest
            or card_identities != projection_identities
        ):
            raise ChatAccessRejected("search_receipt_required")
        if (has_paid_results or _SEARCH_CLAIM.search(reply)) and not authorized:
            raise ChatAccessRejected("search_receipt_required")


_ACTIVE_ACCESS: ContextVar[ChatAccessContext | None] = ContextVar(
    "cafe_host_chat_access",
    default=None,
)


def current_chat_access() -> ChatAccessContext | None:
    """Return access for the current run, never checkpoint or metadata state."""
    return _ACTIVE_ACCESS.get()


@contextmanager
def active_chat_access(access: ChatAccessContext | None) -> Iterator[None]:
    """Scope frozen access to one worker execution and restore it afterward.

    Args:
        access: Trusted v4 access, or `None` for legacy runs.
    """
    token = _ACTIVE_ACCESS.set(access)
    try:
        yield
    finally:
        _ACTIVE_ACCESS.reset(token)
