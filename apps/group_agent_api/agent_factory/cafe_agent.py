"""Tool-free café graph sharing the existing authorized conversation saver."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain.agents import create_agent
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from apps.group_agent_api.agent_factory.context.venue import VenueHostMiddleware
from apps.group_agent_api.agent_factory.host_policy import HostActionRejected, is_readonly_host


class CafeHostMiddleware(VenueHostMiddleware):
    """Remove tools and fail closed if a provider still requests an action."""

    def prepare(self, request: ModelRequest) -> ModelRequest:
        """Use only authenticated host context and the independent role prompt.

        Args:
            request: Current model invocation, not a persisted instruction.
        """
        if not is_readonly_host():
            raise HostActionRejected("cafe_host_context_required")
        from apps.group_agent_api.execution.active_fence import assert_write_allowed

        assert_write_allowed("host_model")
        return super().prepare(request).override(tools=[], tool_choice="none")

    def _validate(self, response: ModelResponse) -> ModelResponse:
        from apps.group_agent_api.execution.active_fence import assert_write_allowed

        assert_write_allowed("host_checkpoint")
        for message in response.result:
            extra = getattr(message, "additional_kwargs", {}) or {}
            if getattr(message, "tool_calls", None) or getattr(message, "invalid_tool_calls", None) or extra.get("tool_calls") or extra.get("function_call"):
                raise HostActionRejected("cafe_host_tool_call_rejected")
        return response

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        """Validate a synchronous tool-free model response.

        Args:
            request: Current model invocation.
            handler: Underlying model execution.
        """
        return self._validate(handler(self.prepare(request)))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        """Validate an asynchronous tool-free model response.

        Args:
            request: Current model invocation.
            handler: Underlying model execution.
        """
        return self._validate(await handler(self.prepare(request)))


def create_cafe_agent(*, model: Any, checkpointer: Any) -> Any:
    """Create a café graph without Deep Agent filesystem/subagent/todo tools.

    Args:
        model: Existing bare provider model; not the legacy compiled agent.
        checkpointer: Same saver as the original authorized conversation graph.
    """
    if model is None or checkpointer is None:
        raise HostActionRejected("cafe_host_executor_unavailable")
    return create_agent(model=model, tools=[], checkpointer=checkpointer, middleware=[CafeHostMiddleware()])
