"""Async-compatible summarization middleware to replace langchain's SummarizationMiddleware.

This middleware automatically summarizes conversation history when token limits are approached,
preventing context length errors in async contexts.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# Use a logger that will be visible in application logs
_uvicorn_logger = logging.getLogger("uvicorn.error")
_logger = _uvicorn_logger if _uvicorn_logger.handlers else logging.getLogger(__name__)
# Ensure the logger is at least at INFO level
if _logger.level == logging.NOTSET:
    _logger.setLevel(logging.INFO)


class AsyncSummarizationMiddleware(AgentMiddleware):
    """Async-compatible middleware that summarizes conversation history when token limits are approached.
    
    This is a drop-in replacement for langchain's SummarizationMiddleware that supports async contexts.
    
    Features:
    - Supports both sync and async model calls
    - Accurate token counting using model's tokenizer
    - Configurable trigger thresholds (fraction or absolute tokens)
    - Configurable keep strategies (fraction or message count)
    - Graceful error handling
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        trigger: tuple[str, float | int] = ("fraction", 0.85),
        keep: tuple[str, float | int] = ("fraction", 0.10),
        trim_tokens_to_summarize: int | None = None,
    ):
        """Initialize the async summarization middleware.
        
        Args:
            model: The model to use for summarization (typically same as main agent)
            trigger: When to trigger summarization:
                - ("fraction", 0.85): Trigger at 85% of max_input_tokens
                - ("tokens", 100000): Trigger at absolute token count
            keep: Which messages to keep (recent ones):
                - ("fraction", 0.10): Keep last 10% of messages
                - ("messages", 6): Keep last N messages
            trim_tokens_to_summarize: Optional max tokens for summary prompt (None = no limit)
        """
        self.model = model
        self.trigger = trigger
        self.keep = keep
        self.trim_tokens_to_summarize = trim_tokens_to_summarize
        
        # Validate trigger and keep configurations
        trigger_type, trigger_value = trigger
        if trigger_type not in ("fraction", "tokens"):
            raise ValueError(f"Invalid trigger type: {trigger_type}. Must be 'fraction' or 'tokens'")
        if trigger_type == "fraction" and not (0 < trigger_value <= 1):
            raise ValueError(f"Fraction trigger must be between 0 and 1, got {trigger_value}")
        if trigger_type == "tokens" and trigger_value <= 0:
            raise ValueError(f"Token trigger must be positive, got {trigger_value}")
            
        keep_type, keep_value = keep
        if keep_type not in ("fraction", "messages"):
            raise ValueError(f"Invalid keep type: {keep_type}. Must be 'fraction' or 'messages'")
        if keep_type == "fraction" and not (0 < keep_value <= 1):
            raise ValueError(f"Fraction keep must be between 0 and 1, got {keep_value}")
        if keep_type == "messages" and keep_value < 0:
            raise ValueError(f"Message keep count must be non-negative, got {keep_value}")
        
        _logger.info(
            "[AsyncSummarizationMiddleware] Initialized with trigger=%s, keep=%s, trim_tokens_to_summarize=%s",
            trigger,
            keep,
            trim_tokens_to_summarize,
        )
    
    def _count_tokens(self, messages: Sequence[BaseMessage], system_prompt: str | None = None) -> tuple[int, bool]:
        """Count tokens in messages and system prompt using the model's tokenizer.
        
        Args:
            messages: List of messages to count
            system_prompt: Optional system prompt to include in count
            
        Returns:
            Tuple of (estimated token count, using_fallback)
            where using_fallback is True if we used fallback estimation
        """
        use_fallback = False
        try:
            # Build messages list for token counting
            counting_messages: list[BaseMessage] = []
            
            # Add system prompt as SystemMessage if provided
            if system_prompt:
                counting_messages.append(SystemMessage(content=system_prompt))
            
            # Add all messages
            counting_messages.extend(messages)
            
            # Use model's token counting method if available
            if hasattr(self.model, "get_num_tokens_from_messages"):
                try:
                    count = self.model.get_num_tokens_from_messages(counting_messages)
                    return count, False
                except Exception as e:
                    # Method exists but failed - use fallback
                    use_fallback = True
                    _logger.warning(
                        "[AsyncSummarizationMiddleware] Token counting method failed, using fallback estimation: %s",
                        e,
                    )
            
            # Try tiktoken if available (for OpenAI-compatible models)
            if not use_fallback:
                try:
                    import tiktoken
                    
                    # Try to determine encoding from model name
                    model_name = getattr(self.model, "model_name", None) or getattr(self.model, "model", None) or ""
                    encoding_name = "cl100k_base"  # Default for GPT-4, GPT-3.5
                    
                    # Map common model names to encodings
                    if "gpt-4" in str(model_name).lower() or "gpt-3.5" in str(model_name).lower():
                        encoding_name = "cl100k_base"
                    elif "gpt-3" in str(model_name).lower():
                        encoding_name = "p50k_base"
                    elif "qwen" in str(model_name).lower():
                        # Qwen models typically use similar tokenization to GPT
                        encoding_name = "cl100k_base"
                    
                    try:
                        encoding = tiktoken.get_encoding(encoding_name)
                    except Exception:
                        # Fallback to cl100k_base if specific encoding not found
                        encoding = tiktoken.get_encoding("cl100k_base")
                    
                    # Count tokens for all message content
                    total_tokens = 0
                    for msg in counting_messages:
                        content = str(msg.content) if hasattr(msg, "content") and msg.content else ""
                        if content:
                            # Add overhead for message formatting (role, etc.) - roughly 4 tokens per message
                            total_tokens += len(encoding.encode(content)) + 4
                    
                    _logger.debug(
                        "[AsyncSummarizationMiddleware] Token count using tiktoken: %d",
                        total_tokens,
                    )
                    return total_tokens, False
                except ImportError:
                    # tiktoken not available
                    pass
                except Exception as e:
                    _logger.debug(
                        "[AsyncSummarizationMiddleware] tiktoken counting failed, using character fallback: %s",
                        e,
                    )
            
            # Fallback: estimate based on character count
            # Use more conservative ratio: 2.5 chars per token (accounts for Chinese chars, formatting, etc.)
            # This is more conservative than 4 chars/token to avoid underestimating
            total_chars = sum(
                len(str(msg.content)) if hasattr(msg, "content") and msg.content else 0
                for msg in counting_messages
            )
            if system_prompt:
                total_chars += len(system_prompt)
            
            # More conservative estimation: 2.5 chars per token
            # Also add overhead for message formatting (roughly 10 tokens per message for role, formatting, etc.)
            estimated_tokens = int(total_chars / 2.5) + (len(counting_messages) * 10)
            
            if use_fallback:
                _logger.warning(
                    "[AsyncSummarizationMiddleware] Using fallback token estimation: %d chars -> ~%d tokens (conservative estimate)",
                    total_chars,
                    estimated_tokens,
                )
            
            return estimated_tokens, use_fallback
            
        except Exception as e:
            # Final fallback to character-based estimation if everything fails
            _logger.error(
                "[AsyncSummarizationMiddleware] Token counting completely failed, using emergency fallback: %s",
                e,
            )
            total_chars = sum(
                len(str(msg.content)) if hasattr(msg, "content") and msg.content else 0
                for msg in messages
            )
            if system_prompt:
                total_chars += len(system_prompt)
            # Very conservative: 2 chars per token + message overhead
            return int(total_chars / 2) + (len(messages) * 10), True
    
    def _should_trigger(self, token_count: int, model: BaseChatModel, message_count: int = 0, using_fallback: bool = False) -> bool:
        """Check if summarization should be triggered based on token count.
        
        Args:
            token_count: Current token count
            model: The model (to check max_input_tokens from profile)
            message_count: Number of messages (for safety checks)
            using_fallback: Whether we're using fallback token estimation
            
        Returns:
            True if summarization should be triggered
        """
        trigger_type, trigger_value = self.trigger
        
        if trigger_type == "tokens":
            # Absolute token threshold
            return token_count >= trigger_value
        
        elif trigger_type == "fraction":
            # Fraction of max_input_tokens
            if (
                model.profile is not None
                and isinstance(model.profile, dict)
                and "max_input_tokens" in model.profile
                and isinstance(model.profile["max_input_tokens"], int)
            ):
                max_input_tokens = model.profile["max_input_tokens"]
                threshold = int(max_input_tokens * trigger_value)
                
                # If using fallback estimation and we have many messages, be more aggressive
                # This is a safety mechanism to prevent context length errors when token counting is inaccurate
                if using_fallback and message_count > 20:
                    # Lower the threshold by 10% when using fallback with many messages
                    safety_threshold = int(threshold * 0.9)
                    _logger.warning(
                        "[AsyncSummarizationMiddleware] Using fallback estimation with %d messages - "
                        "applying safety margin: threshold %d -> %d",
                        message_count,
                        threshold,
                        safety_threshold,
                    )
                    return token_count >= safety_threshold
                
                return token_count >= threshold
            else:
                # No max_input_tokens in profile, can't use fraction trigger
                _logger.warning(
                    "[AsyncSummarizationMiddleware] Fraction trigger specified but model.profile.max_input_tokens not found. "
                    "Skipping summarization check."
                )
                return False
        
        return False
    
    def _split_messages(
        self, messages: Sequence[BaseMessage]
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """Split messages into those to keep (recent) and those to summarize (old).
        
        Args:
            messages: All messages in the conversation
            
        Returns:
            Tuple of (messages_to_summarize, messages_to_keep)
        """
        if not messages:
            return [], []
        
        keep_type, keep_value = self.keep
        
        if keep_type == "messages":
            # Keep last N messages
            keep_count = int(keep_value)
            if keep_count >= len(messages):
                # Keep all messages, nothing to summarize
                return [], list(messages)
            split_idx = len(messages) - keep_count
            return list(messages[:split_idx]), list(messages[split_idx:])
        
        elif keep_type == "fraction":
            # Keep last fraction of messages
            keep_count = max(1, int(len(messages) * keep_value))
            if keep_count >= len(messages):
                # Keep all messages, nothing to summarize
                return [], list(messages)
            split_idx = len(messages) - keep_count
            return list(messages[:split_idx]), list(messages[split_idx:])
        
        # Should not reach here due to validation in __init__, but handle gracefully
        return [], list(messages)
    
    async def _create_summary(self, messages_to_summarize: list[BaseMessage]) -> str:
        """Create a summary of old messages using the model.
        
        Args:
            messages_to_summarize: Messages to summarize
            
        Returns:
            Summary text
        """
        if not messages_to_summarize:
            return ""
        
        # Build summary prompt
        summary_prompt = """Please provide a concise summary of the following conversation history. 
Focus on key points, decisions, and context that would be important for continuing the conversation.
Do not include every detail, but preserve important information that might be referenced later.

Conversation history to summarize:
"""
        
        # Format messages for summary
        message_texts = []
        for msg in messages_to_summarize:
            msg_type = msg.__class__.__name__
            content = str(msg.content) if hasattr(msg, "content") and msg.content else ""
            if content:
                message_texts.append(f"{msg_type}: {content}")
        
        full_prompt = summary_prompt + "\n\n".join(message_texts)
        
        # Trim if needed
        if self.trim_tokens_to_summarize:
            # Rough estimate: 4 chars per token
            max_chars = self.trim_tokens_to_summarize * 4
            if len(full_prompt) > max_chars:
                _logger.warning(
                    "[AsyncSummarizationMiddleware] Trimming summary prompt from %d to %d chars",
                    len(full_prompt),
                    max_chars,
                )
                full_prompt = full_prompt[:max_chars] + "\n\n[Content truncated...]"
        
        # Call model to generate summary
        try:
            summary_messages = [SystemMessage(content=full_prompt)]
            response = await self.model.ainvoke(summary_messages)
            
            # Extract summary from response
            if hasattr(response, "content"):
                summary = str(response.content)
            else:
                summary = str(response)
            
            _logger.info(
                "[AsyncSummarizationMiddleware] Generated summary (%d chars) from %d messages",
                len(summary),
                len(messages_to_summarize),
            )
            
            # Print the summary content for debugging
            _logger.warning(
                "[AsyncSummarizationMiddleware] ========== SUMMARY CONTENT ==========\n%s\n==========================================",
                summary[:2000] if len(summary) > 2000 else summary,  # Limit to 2000 chars for readability
            )
            
            return summary
            
        except Exception as e:
            _logger.error(
                "[AsyncSummarizationMiddleware] Failed to generate summary: %s: %s",
                type(e).__name__,
                str(e),
            )
            raise
    
    async def _summarize_messages(
        self, messages: Sequence[BaseMessage], system_prompt: str | None
    ) -> tuple[list[BaseMessage], int, int]:
        """Summarize old messages and return new message list with summary.
        
        Args:
            messages: All messages in conversation
            system_prompt: Current system prompt
            
        Returns:
            Tuple of (new_messages, tokens_before, tokens_after)
        """
        # Count tokens before
        tokens_before, _ = self._count_tokens(messages, system_prompt)
        
        # Split messages
        messages_to_summarize, messages_to_keep = self._split_messages(messages)
        
        if not messages_to_summarize:
            # Nothing to summarize
            return list(messages), tokens_before, tokens_before
        
        # Generate summary
        try:
            summary_text = await self._create_summary(messages_to_summarize)
            
            # Create summary message (use AIMessage to indicate it's a summary from the assistant)
            # This matches the pattern used by langchain's SummarizationMiddleware
            summary_message = AIMessage(
                content=f"[Previous conversation summary]\n{summary_text}"
            )
            
            # Build new message list: [summary, ...recent_messages]
            new_messages = [summary_message] + messages_to_keep
            
            # Count tokens after
            tokens_after, _ = self._count_tokens(new_messages, system_prompt)
            
            _logger.warning(
                "[AsyncSummarizationMiddleware] ⚠️  SUMMARIZATION TRIGGERED! "
                "messages: %d -> %d, tokens: %d -> %d (reduction: %d tokens, %.1f%%)",
                len(messages),
                len(new_messages),
                tokens_before,
                tokens_after,
                tokens_before - tokens_after,
                ((tokens_before - tokens_after) / tokens_before * 100) if tokens_before > 0 else 0,
            )
            
            return new_messages, tokens_before, tokens_after
            
        except Exception as e:
            _logger.error(
                "[AsyncSummarizationMiddleware] Summarization failed, keeping original messages: %s: %s",
                type(e).__name__,
                str(e),
            )
            # On failure, return original messages
            return list(messages), tokens_before, tokens_before
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async entry point - check token count and summarize if needed.
        
        Args:
            request: The model request
            handler: The async handler function
            
        Returns:
            The model response
        """
        # Get messages from request
        messages = list(request.messages) if hasattr(request, "messages") and request.messages else []
        system_prompt = request.system_prompt if hasattr(request, "system_prompt") else None
        
        # Log all messages at the start of each model call
        _logger.warning(
            "[AsyncSummarizationMiddleware] ========== ALL MESSAGES AT START OF MODEL CALL =========="
        )
        _logger.warning(
            "[AsyncSummarizationMiddleware] Total messages: %d",
            len(messages),
        )
        for i, msg in enumerate(messages):
            msg_type = msg.__class__.__name__
            content_preview = ""
            if hasattr(msg, "content") and msg.content:
                content_str = str(msg.content)
                # Show first 200 chars of each message
                content_preview = content_str[:200] + ("..." if len(content_str) > 200 else "")
            _logger.warning(
                "[AsyncSummarizationMiddleware] Message[%d]: %s - %s",
                i,
                msg_type,
                content_preview,
            )
        _logger.warning(
            "[AsyncSummarizationMiddleware] ========================================================="
        )
        
        # Also check state messages
        try:
            state = request.state
            if isinstance(state, dict) and "messages" in state:
                state_messages = state.get("messages", [])
                _logger.warning(
                    "[AsyncSummarizationMiddleware] State messages count: %d (request messages: %d)",
                    len(state_messages) if state_messages else 0,
                    len(messages),
                )
                if len(state_messages) != len(messages):
                    _logger.error(
                        "[AsyncSummarizationMiddleware] ⚠️ MISMATCH: State has %d messages but request has %d messages!",
                        len(state_messages) if state_messages else 0,
                        len(messages),
                    )
        except Exception as e:
            _logger.debug(
                "[AsyncSummarizationMiddleware] Could not check state messages: %s",
                e,
            )
        
        if not messages:
            # No messages, nothing to summarize
            return await handler(request)
        
        # Prevent re-summarization: if the first message is already a summary, skip summarization
        # This prevents infinite loops where we keep summarizing the same messages
        if messages and isinstance(messages[0], AIMessage):
            first_content = str(messages[0].content) if hasattr(messages[0], "content") else ""
            if first_content.startswith("[Previous conversation summary]"):
                _logger.debug(
                    "[AsyncSummarizationMiddleware] First message is already a summary, skipping summarization to prevent loops"
                )
                return await handler(request)
        
        # Count tokens
        token_count, using_fallback = self._count_tokens(messages, system_prompt)
        
        # Log token count for debugging
        _logger.info(
            "[AsyncSummarizationMiddleware] Token count: %d (messages: %d, system_prompt: %s, fallback: %s)",
            token_count,
            len(messages),
            "yes" if system_prompt else "no",
            using_fallback,
        )
        
        # Check if we should trigger summarization
        should_trigger = self._should_trigger(token_count, self.model, len(messages), using_fallback)
        
        # Log trigger decision
        trigger_type, trigger_value = self.trigger
        if trigger_type == "fraction":
            if (
                self.model.profile is not None
                and isinstance(self.model.profile, dict)
                and "max_input_tokens" in self.model.profile
            ):
                max_input = self.model.profile["max_input_tokens"]
                threshold = int(max_input * trigger_value)
                _logger.info(
                    "[AsyncSummarizationMiddleware] Trigger check: %d >= %d (%.1f%% of %d)? %s",
                    token_count,
                    threshold,
                    trigger_value * 100,
                    max_input,
                    should_trigger,
                )
            else:
                _logger.warning(
                    "[AsyncSummarizationMiddleware] Fraction trigger but no max_input_tokens in profile"
                )
        else:
            _logger.info(
                "[AsyncSummarizationMiddleware] Trigger check: %d >= %d? %s",
                token_count,
                trigger_value,
                should_trigger,
            )
        
        if should_trigger:
            _logger.info(
                "[AsyncSummarizationMiddleware] Token count %d exceeds threshold, triggering summarization",
                token_count,
            )
            
            # Summarize messages
            new_messages, tokens_before, tokens_after = await self._summarize_messages(
                messages, system_prompt
            )
            
            # Log all messages after summarization
            _logger.warning(
                "[AsyncSummarizationMiddleware] ========== ALL MESSAGES AFTER SUMMARIZATION =========="
            )
            _logger.warning(
                "[AsyncSummarizationMiddleware] Total messages: %d (was %d)",
                len(new_messages),
                len(messages),
            )
            for i, msg in enumerate(new_messages):
                msg_type = msg.__class__.__name__
                content_preview = ""
                if hasattr(msg, "content") and msg.content:
                    content_str = str(msg.content)
                    # Show first 200 chars of each message
                    content_preview = content_str[:200] + ("..." if len(content_str) > 200 else "")
                _logger.warning(
                    "[AsyncSummarizationMiddleware] Message[%d]: %s - %s",
                    i,
                    msg_type,
                    content_preview,
                )
            _logger.warning(
                "[AsyncSummarizationMiddleware] ======================================================"
            )
            
            # Update request with summarized messages
            request = request.override(messages=new_messages)
            
            # CRITICAL: Also update state messages so the summarization persists across calls
            # The request.override() only affects the current request, but state is what persists
            # We need to update state so the next call uses the summarized messages
            try:
                state = request.state
                if isinstance(state, dict):
                    # Update state messages to match the summarized version
                    # This ensures the next call uses the summarized messages, not the original ones
                    old_state_count = len(state.get("messages", [])) if state.get("messages") else 0
                    state["messages"] = new_messages
                    _logger.warning(
                        "[AsyncSummarizationMiddleware] ✅ Updated state messages: %d -> %d (persisted for next call)",
                        old_state_count,
                        len(new_messages),
                    )
                    # Verify the update
                    verify_state_messages = state.get("messages", [])
                    _logger.warning(
                        "[AsyncSummarizationMiddleware] ✅ Verification: state['messages'] now has %d messages",
                        len(verify_state_messages) if verify_state_messages else 0,
                    )
                else:
                    _logger.warning(
                        "[AsyncSummarizationMiddleware] State is not a dict, cannot update messages directly. "
                        "State type: %s",
                        type(state).__name__,
                    )
            except Exception as e:
                _logger.error(
                    "[AsyncSummarizationMiddleware] Failed to update state messages: %s: %s",
                    type(e).__name__,
                    str(e),
                )
                # Continue anyway - the request is updated, even if state update failed
        else:
            _logger.debug(
                "[AsyncSummarizationMiddleware] Token count %d below threshold, no summarization needed",
                token_count,
            )
        
        # Pass to handler
        return await handler(request)
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync entry point - delegates to async version.
        
        Args:
            request: The model request
            handler: The sync handler function
            
        Returns:
            The model response
        """
        import asyncio
        
        # Create async handler wrapper for sync handler
        async def async_handler(req: ModelRequest) -> ModelResponse:
            # Handler is sync, so we call it directly (it will block, but that's expected in sync context)
            return handler(req)
        
        # Run async version
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - this shouldn't happen for sync wrap_model_call
                # but if it does, we need to handle it differently
                _logger.warning(
                    "[AsyncSummarizationMiddleware] wrap_model_call called from async context. "
                    "This may cause issues. The framework should use awrap_model_call instead."
                )
                # Can't use asyncio.run() when loop is running, so we'll skip summarization
                # This is a fallback - ideally the framework should call awrap_model_call
                return handler(request)
            except RuntimeError:
                # No running loop, we can use asyncio.run
                return asyncio.run(self.awrap_model_call(request, async_handler))
        except Exception as e:
            # If anything goes wrong, fall back to passing through unchanged
            _logger.error(
                "[AsyncSummarizationMiddleware] Error in wrap_model_call, passing through: %s: %s",
                type(e).__name__,
                str(e),
            )
            return handler(request)

