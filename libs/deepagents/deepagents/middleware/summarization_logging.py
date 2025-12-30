"""Logging wrapper for SummarizationMiddleware to track when summarization is triggered."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse

# Use a logger that will be visible in application logs
# Prefer uvicorn.error if it has handlers (API context), otherwise use module name
_uvicorn_logger = logging.getLogger("uvicorn.error")
_logger = _uvicorn_logger if _uvicorn_logger.handlers else logging.getLogger(__name__)
# Ensure the logger is at least at INFO level
if _logger.level == logging.NOTSET:
    _logger.setLevel(logging.INFO)


class LoggingSummarizationMiddleware(AgentMiddleware):
    """Wrapper around SummarizationMiddleware that adds logging when summarization is triggered.
    
    This middleware wraps SummarizationMiddleware and logs:
    - When summarization is triggered
    - Token counts before and after summarization
    - Number of messages before and after summarization
    - The trigger configuration being used
    """

    def __init__(self, summarization_middleware: SummarizationMiddleware):
        """Initialize the logging wrapper.
        
        Args:
            summarization_middleware: The SummarizationMiddleware instance to wrap
        """
        self.summarization_middleware = summarization_middleware
        # Extract trigger and keep config for logging
        self.trigger = getattr(summarization_middleware, "trigger", None)
        self.keep = getattr(summarization_middleware, "keep", None)
        
        # Check if wrapped middleware supports async
        self._has_async_support = hasattr(summarization_middleware, "awrap_model_call")
        
        init_msg = (
            f"[LoggingSummarizationMiddleware] Initialized with trigger={self.trigger}, "
            f"keep={self.keep}, async_support={self._has_async_support}"
        )
        _logger.info(init_msg)
        # Also print to ensure visibility during initialization
        print(init_msg)
        
        if not self._has_async_support:
            warning_msg = (
                "[LoggingSummarizationMiddleware] Wrapped SummarizationMiddleware doesn't support async - "
                "summarization will be skipped in async contexts"
            )
            _logger.warning(warning_msg)
            print(f"WARNING: {warning_msg}")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Wrap model call to log summarization events.
        
        Args:
            request: The model request
            handler: The handler function
            
        Returns:
            The model response
        """
        # Get state and messages before summarization middleware runs
        state = cast("AgentState", request.state)
        messages_before = list(state.get("messages", []))
        num_messages_before = len(messages_before)
        
        # Estimate token count before (rough estimate: 4 chars per token)
        # This is approximate but gives us a sense of context size
        total_chars_before = sum(
            len(str(msg.content)) if hasattr(msg, "content") else 0
            for msg in messages_before
        )
        estimated_tokens_before = total_chars_before // 4
        
        # Also check messages in the request itself (SummarizationMiddleware might modify these)
        request_messages_before = list(request.messages) if hasattr(request, "messages") and request.messages else []
        num_request_messages_before = len(request_messages_before)
        
        # Log before summarization middleware
        _logger.info(
            "[SummarizationMiddleware] Before processing: state_messages=%d, request_messages=%d, estimated_tokens=%d, trigger=%s, keep=%s",
            num_messages_before,
            num_request_messages_before,
            estimated_tokens_before,
            self.trigger,
            self.keep,
        )
        
        # Call the wrapped summarization middleware
        response = self.summarization_middleware.wrap_model_call(request, handler)
        
        # Check if summarization occurred by comparing message counts
        # Note: SummarizationMiddleware modifies the request, so we check the request after
        request_messages_after = list(request.messages) if hasattr(request, "messages") and request.messages else []
        num_request_messages_after = len(request_messages_after)
        
        # Estimate tokens after
        total_chars_after = sum(
            len(str(msg.content)) if hasattr(msg, "content") else 0
            for msg in request_messages_after
        )
        estimated_tokens_after = total_chars_after // 4
        
        # Detect if summarization occurred
        summarization_occurred = (
            num_request_messages_before > num_request_messages_after
            or estimated_tokens_before > estimated_tokens_after * 1.2  # Significant reduction
        )
        
        if summarization_occurred:
            _logger.warning(
                "[SummarizationMiddleware] ⚠️  SUMMARIZATION TRIGGERED! "
                "messages: %d -> %d, estimated_tokens: %d -> %d (reduction: %d tokens, %.1f%%)",
                num_request_messages_before,
                num_request_messages_after,
                estimated_tokens_before,
                estimated_tokens_after,
                estimated_tokens_before - estimated_tokens_after,
                ((estimated_tokens_before - estimated_tokens_after) / estimated_tokens_before * 100) if estimated_tokens_before > 0 else 0,
            )
        else:
            _logger.info(
                "[SummarizationMiddleware] After processing: messages=%d, estimated_tokens=%d (no summarization)",
                num_request_messages_after,
                estimated_tokens_after,
            )
        
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call.
        
        Args:
            request: The model request
            handler: The async handler function
            
        Returns:
            The model response
        """
        # Get state and messages before summarization middleware runs
        state = cast("AgentState", request.state)
        messages_before = list(state.get("messages", []))
        num_messages_before = len(messages_before)
        
        # Estimate token count before (rough estimate: 4 chars per token)
        total_chars_before = sum(
            len(str(msg.content)) if hasattr(msg, "content") else 0
            for msg in messages_before
        )
        estimated_tokens_before = total_chars_before // 4
        
        # Also check messages in the request itself
        request_messages_before = list(request.messages) if hasattr(request, "messages") and request.messages else []
        num_request_messages_before = len(request_messages_before)
        
        # Log before summarization middleware
        _logger.info(
            "[SummarizationMiddleware] Before processing (async): state_messages=%d, request_messages=%d, estimated_tokens=%d, trigger=%s, keep=%s",
            num_messages_before,
            num_request_messages_before,
            estimated_tokens_before,
            self.trigger,
            self.keep,
        )
        
        # Call the wrapped summarization middleware
        # Use the cached async support flag for efficiency
        if self._has_async_support:
            response = await self.summarization_middleware.awrap_model_call(request, handler)
        else:
            # If async method doesn't exist, SummarizationMiddleware doesn't support async
            # In this case, we'll log a warning and pass through to handler directly
            # (summarization won't occur in async context, but at least we won't crash)
            _logger.warning(
                "[SummarizationMiddleware] Wrapped SummarizationMiddleware doesn't support async - "
                "summarization will be skipped in async context. Passing through to handler."
            )
            response = await handler(request)
        
        # Check if summarization occurred by comparing message counts
        request_messages_after = list(request.messages) if hasattr(request, "messages") and request.messages else []
        num_request_messages_after = len(request_messages_after)
        
        # Estimate tokens after
        total_chars_after = sum(
            len(str(msg.content)) if hasattr(msg, "content") else 0
            for msg in request_messages_after
        )
        estimated_tokens_after = total_chars_after // 4
        
        # Detect if summarization occurred
        summarization_occurred = (
            num_request_messages_before > num_request_messages_after
            or estimated_tokens_before > estimated_tokens_after * 1.2  # Significant reduction
        )
        
        if summarization_occurred:
            _logger.warning(
                "[SummarizationMiddleware] ⚠️  SUMMARIZATION TRIGGERED! (async) "
                "messages: %d -> %d, estimated_tokens: %d -> %d (reduction: %d tokens, %.1f%%)",
                num_request_messages_before,
                num_request_messages_after,
                estimated_tokens_before,
                estimated_tokens_after,
                estimated_tokens_before - estimated_tokens_after,
                ((estimated_tokens_before - estimated_tokens_after) / estimated_tokens_before * 100) if estimated_tokens_before > 0 else 0,
            )
        else:
            _logger.info(
                "[SummarizationMiddleware] After processing (async): messages=%d, estimated_tokens=%d (no summarization)",
                num_request_messages_after,
                estimated_tokens_after,
            )
        
        return response

