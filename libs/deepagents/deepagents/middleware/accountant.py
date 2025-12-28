"""Middleware for tracking tool calls and token usage.

This middleware:
1. Counts the number of tool calls and enforces an upper limit (default: 25)
2. Tracks input and output tokens from model calls
"""

from collections.abc import Awaitable, Callable
from typing import Annotated, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command


def _tool_call_count_reducer(left: int | None, right: int | None) -> int:
    """Reducer for tool_call_count that handles concurrent updates.
    
    When multiple tool calls happen in parallel, each sees the same initial state
    and tries to increment. The reducer needs to count how many increments occurred.
    
    Strategy: Each tool call sets the count to an increment value (1), and the reducer
    sums all increments to get the total count.
    
    Args:
        left: Existing count (may be None during initialization).
        right: Increment value (typically 1 per tool call).
        
    Returns:
        Sum of left and right (left is the running total, right is the increment).
    """
    if left is None:
        return right if right is not None else 0
    if right is None:
        return left
    # Sum increments: left is running total, right is the increment from this tool call
    return left + right


def _token_count_reducer(left: int | None, right: int | None) -> int:
    """Reducer for token counts that sums concurrent updates.
    
    When multiple model calls happen, each adds tokens. This reducer sums them.
    
    Args:
        left: Existing token count (may be None during initialization).
        right: New token count to add.
        
    Returns:
        Sum of left and right.
    """
    if left is None:
        return right if right is not None else 0
    if right is None:
        return left
    return left + right


# Note: We can't use Annotated with a function that takes parameters directly.
# Instead, we'll create the reducer dynamically in __init__ and store it.
# For now, we'll use a simple reducer and handle the limit in the middleware logic.
def _tool_call_count_reducer(left: int | None, right: int | None) -> int:
    """Reducer for tool_call_count that handles concurrent updates.
    
    When multiple tool calls happen in parallel, each sees the same initial state
    and tries to increment. The reducer sums all increments.
    
    Note: The limit is enforced in wrap_tool_call, but parallel calls can still
    exceed it in a single batch. The reducer just sums increments.
    
    Args:
        left: Existing count (may be None during initialization).
        right: Increment value (typically 1 per tool call).
        
    Returns:
        Sum of left and right (left is running total, right is the increment).
    """
    if left is None:
        return right if right is not None else 0
    if right is None:
        return left
    # Sum increments: left is running total, right is the increment from this tool call
    return left + right


class AccountantState(AgentState):
    """State for tracking tool calls and token usage."""

    tool_call_count: Annotated[NotRequired[int], _tool_call_count_reducer]
    """Current count of tool calls made in this conversation."""

    total_input_tokens: Annotated[NotRequired[int], _token_count_reducer]
    """Total input tokens used across all model calls."""

    total_output_tokens: Annotated[NotRequired[int], _token_count_reducer]
    """Total output tokens used across all model calls."""


class AccountantStateUpdate(TypedDict):
    """A state update for the accountant middleware."""

    tool_call_count: NotRequired[int]
    """Current count of tool calls."""

    total_input_tokens: NotRequired[int]
    """Total input tokens."""

    total_output_tokens: NotRequired[int]
    """Total output tokens."""


class ToolCallLimitExceeded(Exception):
    """Raised when the tool call limit is exceeded."""

    def __init__(self, current_count: int, limit: int) -> None:
        """Initialize the exception.

        Args:
            current_count: Current number of tool calls.
            limit: The maximum allowed tool calls.
        """
        self.current_count = current_count
        self.limit = limit
        super().__init__(
            f"Tool call limit exceeded: {current_count} >= {limit}. "
            f"Maximum allowed tool calls is {limit}."
        )


class AccountantMiddleware(AgentMiddleware):
    """Middleware for tracking tool calls and token usage.

    This middleware:
    1. Counts the number of tool calls and enforces an upper limit (default: 25)
    2. Tracks input and output tokens from model calls

    Example:
        ```python
        from deepagents.middleware.accountant import AccountantMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            model="anthropic:claude-sonnet-4-20250514",
            middleware=[AccountantMiddleware(max_tool_calls=25)],
        )
        ```
    """

    state_schema = AccountantState

    def __init__(self, *, max_tool_calls: int = 25) -> None:
        """Initialize the AccountantMiddleware.

        Args:
            max_tool_calls: Maximum number of tool calls allowed. Defaults to 25.
        """
        self.max_tool_calls = max_tool_calls

    def before_agent(
        self,
        state: AccountantState,
        runtime: Runtime,
    ) -> AccountantStateUpdate | None:
        """Initialize accounting state if not present.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with accounting fields initialized to 0 if not present.
        """
        updates = {}
        if "tool_call_count" not in state:
            updates["tool_call_count"] = 0
        if "total_input_tokens" not in state:
            updates["total_input_tokens"] = 0
        if "total_output_tokens" not in state:
            updates["total_output_tokens"] = 0

        return AccountantStateUpdate(**updates) if updates else None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Count tool calls and enforce limit before execution.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The tool result from the handler wrapped in a Command with state updates,
            or a ToolMessage with an error if limit exceeded (allows LLM to handle gracefully).
        """
        # Get current state
        state = cast(AccountantState, request.runtime.state if hasattr(request.runtime, "state") else {})
        current_count = state.get("tool_call_count", 0)

        # Check if limit would be exceeded
        if current_count >= self.max_tool_calls:
            tool_name = request.tool_call.get("name", "unknown")
            tool_call_id = request.tool_call.get("id", "")
            error_message = (
                f"Tool call limit exceeded: {current_count} >= {self.max_tool_calls}. "
                f"Maximum allowed tool calls is {self.max_tool_calls}. "
                f"Cannot execute tool '{tool_name}'. "
                f"Please provide a final response to the user without making additional tool calls."
            )
            # Return an error ToolMessage instead of raising an exception
            # This allows the LLM to see the error and respond appropriately
            return ToolMessage(
                content=error_message,
                tool_call_id=tool_call_id,
            )

        # Execute the tool call
        result = handler(request)

        # Increment count after successful execution
        # Use increment value (1) instead of absolute value for reducer to sum correctly
        increment = 1

        # If result is already a Command, merge our state update
        if isinstance(result, Command):
            # Merge state updates, preserving existing messages
            existing_update = dict(result.update or {})
            # Set increment value (reducer will sum with existing count)
            existing_update["tool_call_count"] = increment
            return Command(update=existing_update)
        else:
            # Wrap ToolMessage in Command with state update
            return Command(
                update={
                    "tool_call_count": increment,  # Increment value, not absolute
                    "messages": [result] if isinstance(result, ToolMessage) else [],
                }
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async) Count tool calls and enforce limit before execution.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The tool result from the handler wrapped in a Command with state updates,
            or a ToolMessage with an error if limit exceeded (allows LLM to handle gracefully).
        """
        # Get current state
        state = cast(AccountantState, request.runtime.state if hasattr(request.runtime, "state") else {})
        current_count = state.get("tool_call_count", 0)

        # Check if limit would be exceeded
        if current_count >= self.max_tool_calls:
            tool_name = request.tool_call.get("name", "unknown")
            tool_call_id = request.tool_call.get("id", "")
            error_message = (
                f"Tool call limit exceeded: {current_count} >= {self.max_tool_calls}. "
                f"Maximum allowed tool calls is {self.max_tool_calls}. "
                f"Cannot execute tool '{tool_name}'. "
                f"Please provide a final response to the user without making additional tool calls."
            )
            # Return an error ToolMessage instead of raising an exception
            # This allows the LLM to see the error and respond appropriately
            return ToolMessage(
                content=error_message,
                tool_call_id=tool_call_id,
            )

        # Execute the tool call
        result = await handler(request)

        # Increment count after successful execution
        # Use increment value (1) instead of absolute value for reducer to sum correctly
        increment = 1

        # If result is already a Command, merge our state update
        if isinstance(result, Command):
            # Merge state updates, preserving existing messages
            existing_update = dict(result.update or {})
            # Set increment value (reducer will sum with existing count)
            existing_update["tool_call_count"] = increment
            return Command(update=existing_update)
        else:
            # Wrap ToolMessage in Command with state update
            return Command(
                update={
                    "tool_call_count": increment,  # Increment value, not absolute
                    "messages": [result] if isinstance(result, ToolMessage) else [],
                }
            )

    def _extract_tokens_from_response(self, response: ModelResponse) -> tuple[int, int]:
        """Extract input and output tokens from a model response.

        Args:
            response: The model response.

        Returns:
            Tuple of (input_tokens, output_tokens).
        """
        input_tokens = 0
        output_tokens = 0

        # ModelResponse might have messages in different attributes
        # Try both 'messages' and 'result' attributes
        messages = None
        if hasattr(response, "messages"):
            messages = response.messages
        elif hasattr(response, "result"):
            result = response.result
            if isinstance(result, list):
                messages = result
            elif hasattr(result, "messages"):
                messages = result.messages

        # First, try to extract from messages (most reliable for LangChain)
        if messages:
            for message in messages:
                # Look for AI messages which contain token usage
                if hasattr(message, "type") and getattr(message, "type", None) == "ai":
                    # Try usage_metadata from message
                    usage_metadata = getattr(message, "usage_metadata", None)
                    if usage_metadata:
                        if isinstance(usage_metadata, dict):
                            msg_input = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                            msg_output = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                        else:
                            msg_input = getattr(usage_metadata, "input_tokens", None) or getattr(usage_metadata, "prompt_tokens", None) or 0
                            msg_output = getattr(usage_metadata, "output_tokens", None) or getattr(usage_metadata, "completion_tokens", None) or 0
                        
                        if msg_input or msg_output:
                            input_tokens += msg_input
                            output_tokens += msg_output
                            continue  # Found tokens, no need to check response_metadata

                    # Try response_metadata from message
                    response_metadata = getattr(message, "response_metadata", None)
                    if response_metadata:
                        if isinstance(response_metadata, dict):
                            msg_input = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                            msg_output = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                        else:
                            msg_input = getattr(response_metadata, "input_tokens", None) or getattr(response_metadata, "prompt_tokens", None) or 0
                            msg_output = getattr(response_metadata, "output_tokens", None) or getattr(response_metadata, "completion_tokens", None) or 0
                        
                        if msg_input or msg_output:
                            input_tokens += msg_input
                            output_tokens += msg_output

        # Fallback: Try to get usage_metadata from response directly
        if not input_tokens and not output_tokens:
            usage_metadata = getattr(response, "usage_metadata", None)
            if usage_metadata:
                if isinstance(usage_metadata, dict):
                    input_tokens = usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0
                    output_tokens = usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
                else:
                    input_tokens = getattr(usage_metadata, "input_tokens", None) or getattr(usage_metadata, "prompt_tokens", None) or 0
                    output_tokens = getattr(usage_metadata, "output_tokens", None) or getattr(usage_metadata, "completion_tokens", None) or 0

        # Fallback: Try response_metadata if usage_metadata didn't work
        if not input_tokens and not output_tokens:
            response_metadata = getattr(response, "response_metadata", None)
            if response_metadata:
                if isinstance(response_metadata, dict):
                    input_tokens = response_metadata.get("input_tokens") or response_metadata.get("prompt_tokens") or 0
                    output_tokens = response_metadata.get("output_tokens") or response_metadata.get("completion_tokens") or 0
                else:
                    input_tokens = getattr(response_metadata, "input_tokens", None) or getattr(response_metadata, "prompt_tokens", None) or 0
                    output_tokens = getattr(response_metadata, "output_tokens", None) or getattr(response_metadata, "completion_tokens", None) or 0

        return (input_tokens, output_tokens)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Track token usage from model responses.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Execute the model call
        response = handler(request)

        # Extract token usage from response
        input_tokens, output_tokens = self._extract_tokens_from_response(response)

        # Debug: Print token extraction results (can be removed later)
        if input_tokens == 0 and output_tokens == 0:
            # Try to inspect response structure
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Token extraction returned 0. Response type: {type(response)}")
            logger.debug(f"Response attributes: {dir(response)}")
            if hasattr(response, "result"):
                logger.debug(f"Response.result type: {type(response.result)}")
                if isinstance(response.result, list) and response.result:
                    msg = response.result[0]
                    logger.debug(f"First message type: {type(msg)}")
                    logger.debug(f"First message attributes: {dir(msg)}")
                    if hasattr(msg, "usage_metadata"):
                        logger.debug(f"Message usage_metadata: {msg.usage_metadata}")
                    if hasattr(msg, "response_metadata"):
                        logger.debug(f"Message response_metadata: {msg.response_metadata}")

        # Update state with token counts
        # Note: We can't return a Command from wrap_model_call, so we update state directly
        # The state should be mutable and updates will be persisted
        # Use runtime.state if available (like in wrap_tool_call), otherwise fall back to request.state
        state = cast(
            AccountantState,
            request.runtime.state if hasattr(request, "runtime") and hasattr(request.runtime, "state") else request.state
        )
        current_input = state.get("total_input_tokens", 0) or 0
        current_output = state.get("total_output_tokens", 0) or 0

        # Add to existing counts (reducer handles concurrent updates, but we're sequential here)
        new_input = current_input + input_tokens
        new_output = current_output + output_tokens

        # Update state directly (state is typically mutable)
        if isinstance(state, dict):
            state["total_input_tokens"] = new_input
            state["total_output_tokens"] = new_output

        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Track token usage from model responses.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Execute the model call
        response = await handler(request)

        # Extract token usage from response
        input_tokens, output_tokens = self._extract_tokens_from_response(response)

        # Update state with token counts
        # Note: We can't return a Command from awrap_model_call, so we update state directly
        # The state should be mutable and updates will be persisted
        # Use runtime.state if available (like in wrap_tool_call), otherwise fall back to request.state
        state = cast(
            AccountantState,
            request.runtime.state if hasattr(request, "runtime") and hasattr(request.runtime, "state") else request.state
        )
        current_input = state.get("total_input_tokens", 0) or 0
        current_output = state.get("total_output_tokens", 0) or 0

        # Add to existing counts (reducer handles concurrent updates, but we're sequential here)
        new_input = current_input + input_tokens
        new_output = current_output + output_tokens

        # Update state directly (state is typically mutable)
        if isinstance(state, dict):
            state["total_input_tokens"] = new_input
            state["total_output_tokens"] = new_output

        return response

