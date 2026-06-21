"""Middleware for logging the complete prompt stack right before LLM invocation.

This middleware should be added last in the middleware chain to capture
the final state of all prompts after all other middleware has processed them.
"""

import logging
from collections.abc import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


class PromptLoggingMiddleware(AgentMiddleware):
    """Middleware that logs the complete prompt stack right before LLM invocation.

    This middleware should be added last in the middleware chain to ensure
    it captures the final state of all prompts after all other middleware
    has processed them.
    """

    state_schema = AgentState

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Log the complete prompt stack before calling the LLM.

        Args:
            request: The model request being processed
            handler: The handler function to call with the modified request

        Returns:
            The model response from the handler
        """
        self._log_prompt_stack(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Log the complete prompt stack before calling the LLM.

        Args:
            request: The model request being processed
            handler: The handler function to call with the modified request

        Returns:
            The model response from the handler
        """
        self._log_prompt_stack(request)
        return await handler(request)

    def _log_prompt_stack(self, request: ModelRequest) -> None:  # noqa: PLR0912, PLR0915
        """Log the complete prompt stack including system prompt and messages.

        Args:
            request: The model request containing all prompt information
        """
        _logger.debug("=" * 100)
        _logger.debug("[PromptLoggingMiddleware] ===== COMPLETE PROMPT STACK (RIGHT BEFORE LLM CALL) =====")
        _logger.debug("=" * 100)

        # Log system prompt with detailed verification
        if hasattr(request, "system_prompt"):
            system_prompt = request.system_prompt
            if system_prompt:
                _logger.debug("[PromptLoggingMiddleware] SYSTEM PROMPT (VERIFIED - EXISTS):")
                _logger.debug("-" * 100)
                _logger.debug("%s", system_prompt)
                _logger.debug("-" * 100)
                _logger.debug("[PromptLoggingMiddleware] System prompt length: %d characters", len(system_prompt))

                # Check if it contains expert guidance markers
                if "CRITICAL INSTRUCTION" in system_prompt or "PRIMARY DIRECTIVE" in system_prompt:
                    _logger.debug("[PromptLoggingMiddleware] ✓ EXPERT GUIDANCE DETECTED in system prompt!")
                elif "Strategic Guidance" in system_prompt:
                    _logger.debug("[PromptLoggingMiddleware] ⚠ Strategic Guidance section found but may not be strong enough")
                else:
                    _logger.warning("[PromptLoggingMiddleware] ⚠ NO EXPERT GUIDANCE MARKERS FOUND in system prompt!")
            else:
                _logger.warning("[PromptLoggingMiddleware] SYSTEM PROMPT: (EMPTY/NONE) - This is a problem!")
        else:
            _logger.error("[PromptLoggingMiddleware] SYSTEM PROMPT: (attribute not found on request!)")

        # Also check for system_message as alternative
        if hasattr(request, "system_message"):
            system_message = request.system_message
            if system_message:
                content = getattr(system_message, "content", str(system_message))
                _logger.debug("[PromptLoggingMiddleware] SYSTEM MESSAGE (alternative):")
                _logger.debug("-" * 100)
                _logger.debug("%s", content[:500] if len(str(content)) > 500 else content)  # noqa: PLR2004
                _logger.debug("-" * 100)

        _logger.debug("")

        # Log messages
        if hasattr(request, "messages") and request.messages:
            _logger.debug("[PromptLoggingMiddleware] MESSAGES (%d total):", len(request.messages))
            _logger.debug("-" * 100)
            for i, msg in enumerate(request.messages, 1):
                msg_type = getattr(msg, "type", type(msg).__name__)
                content = getattr(msg, "content", str(msg))
                if isinstance(content, list):
                    # Handle structured content (e.g., tool calls)
                    content_str = f"[Structured content with {len(content)} items]"
                    for item in content:
                        if hasattr(item, "type"):
                            content_str += f"\n  - {item.type}: {str(item)[:200]}"
                        else:
                            content_str += f"\n  - {str(item)[:200]}"
                else:
                    content_str = str(content)

                # Truncate very long messages for readability
                if len(content_str) > 1000:  # noqa: PLR2004
                    content_preview = content_str[:1000] + f"\n... [truncated, total length: {len(content_str)} characters]"
                else:
                    content_preview = content_str

                _logger.debug("[PromptLoggingMiddleware] Message %d (%s):", i, msg_type)
                _logger.debug("%s", content_preview)
                _logger.debug("")
        else:
            _logger.debug("[PromptLoggingMiddleware] MESSAGES: (none or not accessible)")

        _logger.debug("")

        # Log tools if available
        if hasattr(request, "tools") and request.tools:
            tool_names = []
            for tool in request.tools:
                if hasattr(tool, "name"):
                    tool_names.append(tool.name)
                elif isinstance(tool, dict) and "name" in tool:
                    tool_names.append(tool["name"])
                else:
                    tool_names.append(str(type(tool).__name__))
            _logger.debug("[PromptLoggingMiddleware] AVAILABLE TOOLS (%d total): %s", len(request.tools), ", ".join(tool_names))
        else:
            _logger.debug("[PromptLoggingMiddleware] AVAILABLE TOOLS: (none)")

        _logger.debug("")

        # Log model information if available
        if hasattr(request, "model") and request.model:
            model_info = str(request.model)
            if hasattr(request.model, "model_name"):
                model_info = f"{request.model.model_name} ({model_info})"
            _logger.debug("[PromptLoggingMiddleware] MODEL: %s", model_info)
        else:
            _logger.debug("[PromptLoggingMiddleware] MODEL: (not accessible from request)")

        _logger.debug("=" * 100)
        _logger.debug("[PromptLoggingMiddleware] ===== END OF PROMPT STACK =====")
        _logger.debug("=" * 100)
        _logger.debug("")
