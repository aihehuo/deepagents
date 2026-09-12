"""Request-scoped café action boundaries, independent of client metadata."""

from __future__ import annotations


class HostActionRejected(RuntimeError):
    """Reject an action forbidden by the authenticated café host policy."""


def is_readonly_host() -> bool:
    """Return whether this execution has the authenticated new host policy."""
    # Lazy import avoids the HostContext -> profile_store import cycle.
    from apps.group_agent_api.agent_factory.host_context import current_host_context

    host = current_host_context()
    return host is not None and host.is_cafe_host


def assert_host_action_allowed(action: str) -> None:
    """Reject business tools and profile writes for the active new host.

    Args:
        action: Internal action name, never a permission from the user/model.
    """
    if is_readonly_host():
        raise HostActionRejected(f"cafe_host_action_not_allowed:{action}")
