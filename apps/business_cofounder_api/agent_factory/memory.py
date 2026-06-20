"""Memory middleware for user and conversation-level memory management."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypedDict, NotRequired, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langgraph.runtime import Runtime

from apps.business_cofounder_api.agent_factory.utils import ensure_memory_directories_exist

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


class ApiMemoryState(AgentState):
    """State for API memory middleware."""
    
    user_id: NotRequired[str]
    """User identifier for memory paths."""
    
    conversation_id: NotRequired[str]
    """Conversation identifier for memory paths."""


def build_memory_documentation(user_id: str | None, conversation_id: str | None) -> str:
    """Build memory structure documentation for the system prompt.
    
    Args:
        user_id: User identifier (optional)
        conversation_id: Conversation identifier (optional)
        
    Returns:
        Memory documentation string to include in system prompt
    """
    if not user_id:
        # No user context, return minimal documentation
        return ""
    
    # Build virtual paths (relative to base_dir, with leading /)
    user_memory_path = f"/users/{user_id}/agent.md"
    conversation_memory_path = f"/users/{user_id}/conversations/{conversation_id}/agent.md" if conversation_id else None
    
    memory_docs = f"""
## Long-term Memory (CRITICAL - MUST USE)

Your memory is stored in files on the filesystem and **persists across all sessions**. This is how you remember user preferences, business ideas, and important context. **Memory writing is expected behavior when appropriate.**

### When to Write to Memory

**Use your judgment to determine when information should be written to memory. Consider writing to memory when:**

**User Memory (`{user_memory_path}`) - Write when:**
- User provides feedback on your work, style, or behavior
- User expresses preferences about how you should communicate or respond
- User describes how you should behave or what your role should be
- User gives corrections or guidance that should apply to future conversations
- Patterns emerge in user preferences that would be useful to remember

**Conversation Memory (`{conversation_memory_path}`) - Write when:**
- User shares a business idea or describes their startup concept
- Business idea develops, pivots, or is refined
- Important decisions are made about the business
- User provides context, background, or market information relevant to the business idea
- Milestones are reached or progress is made on the business idea

**Key Principle**: If information would be valuable to remember for future interactions (user preferences) or for continuing this conversation (business context), write it to memory. Use your judgment to determine what's significant enough to remember.

### Memory Structure

Your memory is organized in two tiers:

**User Memory**: `{user_memory_path}`
- Stores user preferences, communication style, general behavior patterns
- **Shared across ALL conversations for this user**
- **CRITICAL**: Update this when user provides feedback, changes preferences, or describes how you should behave

"""
    
    if conversation_memory_path:
        memory_docs += f"""**Conversation Memory**: `{conversation_memory_path}`
- Stores business idea context, progress, decisions, and milestones for THIS conversation
- **Isolated per conversation** - each conversation has its own memory
- **CRITICAL**: Update this as the business idea develops, when milestones are reached, or when important decisions are made

"""
    
    memory_docs += f"""### Memory-First Protocol (CRITICAL - FOLLOW THIS)

**At conversation start (MUST DO FIRST):**
1. Check if user memory exists: `ls '/users/{user_id}'`
2. If user memory exists, read it: `read_file '{user_memory_path}'`
"""
    
    if conversation_memory_path:
        memory_docs += f"""3. Check if conversation memory exists: `ls '/users/{user_id}/conversations/{conversation_id}'`
4. If conversation memory exists, read it: `read_file '{conversation_memory_path}'`
"""
    
    memory_docs += f"""
### Memory Writing Workflow

**For each user message, consider:**

1. **Read existing memory first** (if available):
   - Check if user memory exists: `ls '/users/{user_id}'`
   - If it exists, read it: `read_file '{user_memory_path}'`
   - If conversation memory exists, read it: `read_file '{conversation_memory_path}'`
   - Use this context to inform your response

2. **Evaluate if memory writing is needed**:
   - Does the user message contain information that should be remembered?
   - Is this a preference, feedback, or important context?
   - Would this information be useful in future interactions?
   - **If yes, you must write to the appropriate memory file using `write_file` or `edit_file` BEFORE responding to the user**

3. **Then respond to the user** with your answer or action

**Critical**: When you identify information worth remembering, you must actually call the `write_file` or `edit_file` tool. Simply acknowledging the information in your response is not sufficient - you need to persist it to memory files.
"""
    
    memory_docs += f"""
### Memory Writing Examples

**Example 1: User Feedback**
```
User: "I've noticed your responses are quite long. I prefer shorter, more concise answers that get to the point quickly."
Your Actions:
1. Recognize this as user feedback about communication style that should be remembered
2. IMMEDIATELY call the write_file tool: write_file('{user_memory_path}', '# User Preferences\\n\\nUser prefers shorter, more concise responses. Keep answers brief and to the point.')
3. Wait for the tool call to complete
4. Then respond to the user acknowledging the feedback

Note: You MUST call the write_file tool - simply acknowledging the feedback in your response is not enough. The memory file must be created/updated.
```

**Example 2: Business Idea**
```
User: "I have an idea for a food delivery app that connects local restaurants with customers in underserved neighborhoods."
Your Actions:
1. Recognize this as a business idea that should be remembered
2. IMMEDIATELY call the write_file tool: write_file('{conversation_memory_path}', '# Business Idea\\n\\nFood delivery app connecting local restaurants with customers in underserved neighborhoods. [Capture additional details from conversation]')
3. Wait for the tool call to complete
4. Then respond to the user

Note: You MUST call the write_file tool to persist the business idea to memory.
```

**Example 3: User Preference**
```
User: "I find it easier to process information when it's in bullet points rather than long paragraphs. Please use that format going forward."
Your Actions:
1. Recognize this as a formatting preference that should be remembered
2. IMMEDIATELY call the edit_file tool: edit_file('{user_memory_path}', '[old content]', '[old content]\\n\\nUser prefers bullet points over paragraph format. Use lists when presenting information.')
3. Wait for the tool call to complete
4. Then respond to the user

Note: You MUST call the edit_file tool - simply saying you'll remember is not enough.
```
"""
    
    memory_docs += f"""
### Memory Writing Decision Process

**When processing a user message, think through:**

1. **Does this contain user preferences or feedback?**
   - User expresses how they want you to communicate or behave
   - User gives feedback on your work or style
   - User describes preferences that should apply to future conversations
   - **If yes**: Write to `'{user_memory_path}'` using `write_file` or `edit_file` to remember this for future conversations

2. **Does this contain business idea information?**
   - User shares a business idea or startup concept
   - User describes progress on their business idea
   - User makes decisions about their business
   - **If yes**: Write to `'{conversation_memory_path}'` using `write_file` or `edit_file` to track the business idea development

3. **Does this contain important context?**
   - User provides background information relevant to the conversation
   - User shares information that would help in future interactions
   - **If yes**: Write to the appropriate memory file

**Action Required**: When you identify information that should be remembered (preferences, feedback, business ideas, important context), you need to actually call the `write_file` or `edit_file` tool on the appropriate memory path. Simply acknowledging the information is not enough - you must write it to memory so it persists for future use.
"""

    memory_docs += f"""
### How to Access Memory (Exact Tool Calls)

**Reading Memory:**
```
read_file '{user_memory_path}'
"""
    
    if conversation_memory_path:
        memory_docs += f"""read_file '{conversation_memory_path}'
"""
    
    memory_docs += f"""ls '/users/{user_id}'
"""
    
    if conversation_memory_path:
        memory_docs += f"""ls '/users/{user_id}/conversations/{conversation_id}'
```
"""
    
    memory_docs += f"""
**Writing Memory (Use these exact paths):**
```
# Create or update user memory
write_file '{user_memory_path}' '# User Preferences\\n\\n[content here]'

# Or edit existing user memory
edit_file '{user_memory_path}' [old_string] [new_string]
"""
    
    if conversation_memory_path:
        memory_docs += f"""
# Create or update conversation memory
write_file '{conversation_memory_path}' '# Business Idea: [title]\\n\\n[content here]'

# Or edit existing conversation memory
edit_file '{conversation_memory_path}' [old_string] [new_string]
```
"""
    
    memory_docs += f"""
### Example Memory File Content

**User Memory (`{user_memory_path}`) should contain:**
- Communication style preferences (tone, format, length)
- Coding preferences (style, conventions, tools)
- Workflow preferences (how you should approach tasks)
- Feedback patterns (what to avoid, what to emphasize)
- General behavior guidelines
"""
    
    if conversation_memory_path:
        memory_docs += f"""
**Conversation Memory (`{conversation_memory_path}`) should contain:**
- Business idea description and current state
- Key decisions and rationale
- Milestones reached and progress made
- Important context (market, competition, user needs)
- Next steps and planned actions
"""
    
    memory_docs += f"""
### CRITICAL Reminders

- **Memory files persist across ALL sessions** - What you write now will be available in future conversations
- **Always use absolute virtual paths** starting with `/` (e.g., `'{user_memory_path}'`)
- **If a memory file doesn't exist, create it** with `write_file` - don't wait for it to exist
- **Memory writing is expected behavior** - Not writing memory when you should is a failure
- **IMMEDIATELY means IMMEDIATELY** - Don't delay memory updates, do them as soon as the trigger occurs
- **User memory takes precedence** over global agent.md for user-specific context
"""
    
    if conversation_memory_path:
        memory_docs += f"""
- **Conversation memory is isolated** - Each conversation has its own memory file
"""
    
    memory_docs += """
**Remember: Your memory is how you learn and improve. Writing to memory is not optional - it is how you become better over time.**

"""
    
    return memory_docs


class ApiMemoryMiddleware(AgentMiddleware):
    """Middleware that injects user/conversation memory paths into system prompt dynamically.
    
    This middleware reads user_id and conversation_id from request metadata and injects
    memory documentation into the system prompt so the agent knows where to find/update
    user-level and conversation-level memory.
    """
    
    state_schema = ApiMemoryState
    
    def __init__(self, base_dir: Path):
        """Initialize the API memory middleware.

        Args:
            base_dir: Base directory for the API (~/.deepagents/business_cofounder_api)
        """
        self.base_dir = base_dir
        # Store user_id and conversation_id per thread to avoid relying on state timing
        self._thread_context: dict[str, dict[str, str | None]] = {}
        # Test memory file write/read during initialization to verify filesystem permissions
        self._test_memory_file_access()
    
    def before_agent(
        self,
        state: ApiMemoryState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Extract user_id and conversation_id from runtime config metadata.

        Also ensures memory directories exist for the user and conversation.
        Stores the context in the middleware instance for use in wrap_model_call.

        Args:
            state: Current agent state
            runtime: Runtime context with config

        Returns:
            Updated state with user_id and conversation_id if available
        """
        from typing import Any
        
        updates: dict[str, Any] = {}

        # Extract metadata from config
        # Try runtime.config first, then fallback to langgraph.config.get_config()
        config = {}
        if hasattr(runtime, "config") and runtime.config:
            config = runtime.config
        else:
            # Fallback: try to get config from LangGraph's context
            try:
                from langgraph.config import get_config
                config = get_config()
            except Exception:
                pass

        metadata = config.get("metadata", {}) if isinstance(config, dict) else {}
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        thread_id = configurable.get("thread_id", "") if isinstance(configurable, dict) else ""

        user_id = metadata.get("user_id")
        conversation_id = None

        if user_id and isinstance(user_id, str):
            updates["user_id"] = user_id
            # Extract conversation_id from thread_id if available
            # thread_id format: "bc::{user_id}::{conversation_id}"
            if thread_id.startswith("bc::") and "::" in thread_id[4:]:
                parts = thread_id.split("::")
                if len(parts) >= 3:
                    conversation_id = parts[2]
                    updates["conversation_id"] = conversation_id
                    print(f"[ApiMemoryMiddleware.before_agent] Extracted conversation_id from thread_id: {conversation_id}")

            # Store in middleware instance for use in wrap_model_call
            # This ensures we have the context even if state isn't updated yet
            self._thread_context[thread_id] = {
                "user_id": user_id,
                "conversation_id": conversation_id,
            }

            # Ensure memory directories exist
            ensure_memory_directories_exist(self.base_dir, user_id, conversation_id)
        else:
            print(f"[ApiMemoryMiddleware.before_agent] No user_id found or user_id is not a string. user_id={user_id}, type={type(user_id)}")

        return updates if updates else None
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject memory documentation into system prompt based on user/conversation context.

        Args:
            request: The model request being processed
            handler: The handler function to call with the modified request

        Returns:
            The model response from the handler
        """
        # Try to get user_id and conversation_id from state first
        state = cast("ApiMemoryState", request.state)
        user_id = state.get("user_id")
        conversation_id = state.get("conversation_id")
        
        # Fallback: try to get from middleware instance context (set in before_agent)
        # This handles cases where state hasn't been updated yet
        if not user_id and self._thread_context:
            # Use the first available context (for single-threaded tests)
            if len(self._thread_context) == 1:
                context = next(iter(self._thread_context.values()))
                user_id = context.get("user_id")
                conversation_id = context.get("conversation_id")
                print(f"[ApiMemoryMiddleware.wrap_model_call] Using fallback context: user_id={user_id}, conversation_id={conversation_id}")
        
        # If still no user_id, try to extract from any available source
        # This is a last resort for cases where before_agent didn't run or config wasn't available
        if not user_id:
            print(f"[ApiMemoryMiddleware.wrap_model_call] Still no user_id after all checks. Attempting to extract from state messages or other sources...")
            # Could try to extract from messages or other state fields if needed
        
        # Build memory documentation if we have user context
        memory_docs = build_memory_documentation(user_id, conversation_id)
        
        if memory_docs:
            # Append memory documentation to system prompt
            if request.system_prompt:
                system_prompt = request.system_prompt + memory_docs
            else:
                system_prompt = memory_docs
            
            _logger.info("System prompt with memory docs injected (user_id=%s, conversation_id=%s)", user_id, conversation_id)

            return handler(request.override(system_prompt=system_prompt))
        
        # No user context, pass through unchanged
        return handler(request)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        # Try to get user_id and conversation_id from state first
        state = cast("ApiMemoryState", request.state)
        user_id = state.get("user_id")
        conversation_id = state.get("conversation_id")
        
        # Fallback: try to get from middleware instance context (set in before_agent)
        # This handles cases where state hasn't been updated yet
        if not user_id and hasattr(request, "config"):
            config = getattr(request, "config", {})
            configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
            thread_id = configurable.get("thread_id", "")
            if thread_id in self._thread_context:
                context = self._thread_context[thread_id]
                user_id = context.get("user_id")
                conversation_id = context.get("conversation_id")
        
        # Build memory documentation if we have user context
        memory_docs = build_memory_documentation(user_id, conversation_id)
        
        if memory_docs:
            # Append memory documentation to system prompt
            if request.system_prompt:
                system_prompt = request.system_prompt + memory_docs
            else:
                system_prompt = memory_docs
            
            _logger.info("System prompt with memory docs injected (user_id=%s, conversation_id=%s)", user_id, conversation_id)

            return await handler(request.override(system_prompt=system_prompt))
        
        # No user context, pass through unchanged
        return await handler(request)
    
    def _test_memory_file_access(self) -> None:
        """Test write/read functionality for memory files during initialization.
        
        This writes a temporary test memory file, reads it back, and verifies
        that filesystem write permissions work correctly for memory files.
        """
        try:
            # Create a test user directory
            test_user_id = "__init_test__"
            test_user_dir = self.base_dir / "users" / test_user_id
            test_user_dir.mkdir(parents=True, exist_ok=True)
            
            # Test file path
            test_memory_file = test_user_dir / "agent.md"
            
            # Generate test content with timestamp
            from datetime import datetime
            timestamp = datetime.utcnow().isoformat() + "Z"
            test_content = f"""# Memory File Access Test

This is a test file created during ApiMemoryMiddleware initialization.

Timestamp: {timestamp}

This file verifies that:
- Memory directories can be created
- Memory files can be written
- Memory files can be read
- Filesystem permissions are correct

This file will be automatically deleted after the test.
"""
            
            _logger.info("[ApiMemoryMiddleware] Testing memory file write/read access...")
            _logger.info("  Test memory file: %s", test_memory_file)
            
            # Test write
            try:
                test_memory_file.write_text(test_content, encoding="utf-8")
                _logger.info("[ApiMemoryMiddleware] ✓ Write test passed")
            except Exception as write_err:
                _logger.error(
                    "[ApiMemoryMiddleware] ❌ WRITE TEST FAILED: %s: %s",
                    type(write_err).__name__,
                    str(write_err),
                )
                return
            
            # Test read
            try:
                read_content = test_memory_file.read_text(encoding="utf-8")
                if read_content == test_content:
                    _logger.info("[ApiMemoryMiddleware] ✓ Read test passed")
                    _logger.info("  Read content verified (%d bytes)", len(read_content))
                    _logger.info("[ApiMemoryMiddleware] ✓ All memory file access tests passed")
                else:
                    _logger.warning(
                        "[ApiMemoryMiddleware] ⚠️  READ TEST WARNING: Content mismatch"
                    )
                    _logger.warning("  Expected length: %d bytes", len(test_content))
                    _logger.warning("  Actual length: %d bytes", len(read_content))
            except Exception as read_err:
                _logger.error(
                    "[ApiMemoryMiddleware] ❌ READ TEST FAILED: %s: %s",
                    type(read_err).__name__,
                    str(read_err),
                )
                return
            
            # Clean up test file and directory
            try:
                test_memory_file.unlink(missing_ok=True)
                # Only remove directory if it's empty (don't remove if user has other files)
                try:
                    test_user_dir.rmdir()
                    _logger.info("[ApiMemoryMiddleware] ✓ Cleanup test passed")
                except OSError:
                    # Directory not empty or other error - that's fine, leave it
                    pass
            except Exception as cleanup_err:
                _logger.warning(
                    "[ApiMemoryMiddleware] ⚠️  Cleanup warning (non-fatal): %s: %s",
                    type(cleanup_err).__name__,
                    str(cleanup_err),
                )
                
        except Exception as e:
            _logger.error(
                "[ApiMemoryMiddleware] ❌ MEMORY FILE TEST ERROR: %s: %s",
                type(e).__name__,
                str(e),
            )
            import traceback
            _logger.debug("[ApiMemoryMiddleware] Traceback:\n%s", traceback.format_exc())
