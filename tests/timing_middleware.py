"""Timing middleware to track execution time for each step in the agent workflow."""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langgraph.runtime import Runtime
from langchain_core.messages import ToolMessage
from langgraph.types import Command


class TimingMiddleware(AgentMiddleware):
    """Middleware that tracks and prints execution time for each workflow step.
    
    This middleware instruments:
    - before_agent: Time to initialize before agent execution
    - wrap_model_call: Time for each LLM call
    - wrap_tool_call: Time for each tool execution
    - after_tool: Time after tool execution (if implemented)
    
    Example:
        ```python
        from tests.timing_middleware import TimingMiddleware
        from langchain.agents import create_agent
        
        agent = create_agent(
            model="anthropic:claude-sonnet-4-20250514",
            middleware=[TimingMiddleware()],
        )
        ```
    """
    
    def __init__(self, verbose: bool = True) -> None:
        """Initialize the TimingMiddleware.
        
        Args:
            verbose: If True, print timing information. Defaults to True.
        """
        super().__init__()
        self.verbose = verbose
        self.timings: list[dict[str, Any]] = []
        self.model_call_count = 0
        self.tool_call_count = 0
        self.start_time: float | None = None
        self.total_time: float | None = None
    
    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Track time before agent execution starts."""
        if self.verbose:
            print("\n" + "="*80)
            print("⏱️  TIMING MIDDLEWARE: Starting agent execution")
            print("="*80)
        
        self.start_time = time.time()
        self.timings = []
        self.model_call_count = 0
        self.tool_call_count = 0
        
        step_start = time.time()
        # Call next handler if needed (no-op for now)
        step_end = time.time()
        step_duration = step_end - step_start
        
        if self.verbose:
            print(f"  ⏱️  before_agent: {step_duration*1000:.2f}ms")
        
        self.timings.append({
            "step": "before_agent",
            "duration": step_duration,
            "timestamp": step_start,
        })
        
        return None
    
    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Track time after agent execution completes."""
        step_start = time.time()
        step_end = time.time()
        step_duration = step_end - step_start
        
        if self.verbose and self.start_time:
            elapsed = time.time() - self.start_time
            print(f"\n  ⏱️  after_agent: {step_duration*1000:.2f}ms (total elapsed: {elapsed:.2f}s)")
        
        self.timings.append({
            "step": "after_agent",
            "duration": step_duration,
            "timestamp": step_start,
        })
        
        return None
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Track time for each LLM model call."""
        self.model_call_count += 1
        call_num = self.model_call_count
        
        step_start = time.time()
        
        # Check todos in state before model call
        todos_before = []
        try:
            if hasattr(request, 'state') and request.state:
                todos_before = request.state.get("todos", [])
        except (AttributeError, TypeError):
            # State might not be available or accessible
            pass
        
        if self.verbose:
            print(f"\n  🔄 Model Call #{call_num} starting...")
            # Show tool names if any
            if request.tools:
                tool_names = [getattr(t, 'name', str(t)) for t in request.tools[:5]]
                if len(request.tools) > 5:
                    tool_names.append(f"... ({len(request.tools)} total)")
                print(f"     Available tools: {', '.join(tool_names)}")
            
            # Show message history size and context information
            try:
                if hasattr(request, 'messages') and request.messages:
                    msg_count = len(request.messages)
                    total_chars = sum(len(str(msg.content)) for msg in request.messages if hasattr(msg, 'content'))
                    print(f"     📨 Message history: {msg_count} messages, ~{total_chars:,} characters")
                    
                    # Show last message preview if it's a user or AI message
                    if msg_count > 0:
                        last_msg = request.messages[-1]
                        if hasattr(last_msg, 'content'):
                            last_content = str(last_msg.content)
                            if len(last_content) > 200:
                                print(f"     📝 Last message preview: {last_content[:200]}... ({len(last_content):,} chars)")
                            else:
                                print(f"     📝 Last message: {last_content}")
                elif hasattr(request, 'state') and request.state:
                    # Try to get messages from state
                    state_messages = request.state.get("messages", [])
                    if state_messages:
                        msg_count = len(state_messages)
                        total_chars = sum(len(str(msg.content)) for msg in state_messages if hasattr(msg, 'content'))
                        print(f"     📨 Message history (from state): {msg_count} messages, ~{total_chars:,} characters")
            except Exception as e:
                print(f"     ⚠️  Could not analyze message history: {e}")
            
            # Show system prompt size if available
            try:
                if hasattr(request, 'system_prompt') and request.system_prompt:
                    sys_prompt_len = len(str(request.system_prompt))
                    print(f"     📋 System prompt: ~{sys_prompt_len:,} characters")
            except Exception:
                pass
            
            # Show current todos (complete list)
            if todos_before:
                print(f"     📋 Current todos: {len(todos_before)} items")
                for i, todo in enumerate(todos_before, 1):
                    status_emoji = "✅" if todo.get("status") == "completed" else "⏳" if todo.get("status") == "in_progress" else "📋"
                    content = todo.get("content", "N/A")
                    print(f"        {i}. {status_emoji} [{todo.get('status', 'pending')}] {content}")
            else:
                print(f"     📋 No todos in state yet")
        
        # Execute the actual model call
        response = handler(request)
        
        step_end = time.time()
        step_duration = step_end - step_start
        
        # Check todos after model call (from response state if available)
        todos_after = todos_before
        try:
            if hasattr(response, 'state') and response.state:
                todos_after = response.state.get("todos", todos_before)
        except (AttributeError, TypeError):
            # Response state might not be available
            pass
        
        if self.verbose:
            print(f"  ⏱️  Model Call #{call_num}: {step_duration:.2f}s ({step_duration*1000:.2f}ms)")
            # Show response details
            if hasattr(response, 'messages') and response.messages:
                last_msg = response.messages[-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    tool_call_names = [tc.get('name', 'unknown') for tc in last_msg.tool_calls]
                    print(f"     → Tool calls: {', '.join(tool_call_names)}")
                elif hasattr(last_msg, 'content'):
                    content = str(last_msg.content)
                    if len(content) > 300:
                        print(f"     → Response: {content[:300]}... ({len(content):,} chars total)")
                    else:
                        print(f"     → Response: {content}")
            
            # Warn if this call took a very long time
            if step_duration > 30:
                print(f"     ⚠️  WARNING: This model call took {step_duration:.1f}s - this is unusually long!")
            elif step_duration > 60:
                print(f"     🚨 CRITICAL: This model call took {step_duration:.1f}s - possible timeout or stuck generation!")
            
            # Show todo changes - print complete list if changed
            if len(todos_after) != len(todos_before):
                print(f"     📋 Todos changed: {len(todos_before)} → {len(todos_after)} items")
                print(f"     📋 Updated todo list:")
                for i, todo in enumerate(todos_after, 1):
                    status_emoji = "✅" if todo.get("status") == "completed" else "⏳" if todo.get("status") == "in_progress" else "📋"
                    content = todo.get("content", "N/A")
                    print(f"        {i}. {status_emoji} [{todo.get('status', 'pending')}] {content}")
            elif todos_after:
                # Check if any todos changed status
                status_changes = []
                for i, (before, after) in enumerate(zip(todos_before, todos_after)):
                    if before.get("status") != after.get("status"):
                        status_changes.append(f"Todo {i+1}: {before.get('status')} → {after.get('status')}")
                if status_changes:
                    print(f"     📋 Todo status changes: {', '.join(status_changes)}")
                    # Also show the complete updated list
                    print(f"     📋 Complete todo list after changes:")
                    for i, todo in enumerate(todos_after, 1):
                        status_emoji = "✅" if todo.get("status") == "completed" else "⏳" if todo.get("status") == "in_progress" else "📋"
                        content = todo.get("content", "N/A")
                        print(f"        {i}. {status_emoji} [{todo.get('status', 'pending')}] {content}")
        
        self.timings.append({
            "step": f"model_call_{call_num}",
            "duration": step_duration,
            "timestamp": step_start,
            "tool_calls": len(getattr(response.messages[-1], 'tool_calls', [])) if hasattr(response, 'messages') and response.messages else 0,
            "todos_before": len(todos_before),
            "todos_after": len(todos_after),
        })
        
        return response
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        self.model_call_count += 1
        call_num = self.model_call_count
        
        step_start = time.time()
        
        # Check todos in state before model call
        todos_before = []
        try:
            if hasattr(request, 'state') and request.state:
                todos_before = request.state.get("todos", [])
        except (AttributeError, TypeError):
            # State might not be available or accessible
            pass
        
        if self.verbose:
            print(f"\n  🔄 Model Call #{call_num} starting (async)...")
            if request.tools:
                tool_names = [getattr(t, 'name', str(t)) for t in request.tools[:5]]
                if len(request.tools) > 5:
                    tool_names.append(f"... ({len(request.tools)} total)")
                print(f"     Available tools: {', '.join(tool_names)}")
            
            # Show current todos (complete list)
            if todos_before:
                print(f"     📋 Current todos: {len(todos_before)} items")
                for i, todo in enumerate(todos_before, 1):
                    status_emoji = "✅" if todo.get("status") == "completed" else "⏳" if todo.get("status") == "in_progress" else "📋"
                    content = todo.get("content", "N/A")
                    print(f"        {i}. {status_emoji} [{todo.get('status', 'pending')}] {content}")
            else:
                print(f"     📋 No todos in state yet")
        
        response = await handler(request)
        
        step_end = time.time()
        step_duration = step_end - step_start
        
        # Check todos after model call (from response state if available)
        todos_after = todos_before
        try:
            if hasattr(response, 'state') and response.state:
                todos_after = response.state.get("todos", todos_before)
        except (AttributeError, TypeError):
            # Response state might not be available
            pass
        
        if self.verbose:
            print(f"  ⏱️  Model Call #{call_num}: {step_duration:.2f}s ({step_duration*1000:.2f}ms)")
            # Show response details
            if hasattr(response, 'messages') and response.messages:
                last_msg = response.messages[-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    tool_call_names = [tc.get('name', 'unknown') for tc in last_msg.tool_calls]
                    print(f"     → Tool calls: {', '.join(tool_call_names)}")
                elif hasattr(last_msg, 'content'):
                    content = str(last_msg.content)
                    if len(content) > 300:
                        print(f"     → Response: {content[:300]}... ({len(content):,} chars total)")
                    else:
                        print(f"     → Response: {content}")
            
            # Warn if this call took a very long time
            if step_duration > 30:
                print(f"     ⚠️  WARNING: This model call took {step_duration:.1f}s - this is unusually long!")
            elif step_duration > 60:
                print(f"     🚨 CRITICAL: This model call took {step_duration:.1f}s - possible timeout or stuck generation!")
            
            # Show todo changes - print complete list if changed
            if len(todos_after) != len(todos_before):
                print(f"     📋 Todos changed: {len(todos_before)} → {len(todos_after)} items")
                print(f"     📋 Updated todo list:")
                for i, todo in enumerate(todos_after, 1):
                    status_emoji = "✅" if todo.get("status") == "completed" else "⏳" if todo.get("status") == "in_progress" else "📋"
                    content = todo.get("content", "N/A")
                    print(f"        {i}. {status_emoji} [{todo.get('status', 'pending')}] {content}")
            elif todos_after:
                # Check if any todos changed status
                status_changes = []
                for i, (before, after) in enumerate(zip(todos_before, todos_after)):
                    if before.get("status") != after.get("status"):
                        status_changes.append(f"Todo {i+1}: {before.get('status')} → {after.get('status')}")
                if status_changes:
                    print(f"     📋 Todo status changes: {', '.join(status_changes)}")
                    # Also show the complete updated list
                    print(f"     📋 Complete todo list after changes:")
                    for i, todo in enumerate(todos_after, 1):
                        status_emoji = "✅" if todo.get("status") == "completed" else "⏳" if todo.get("status") == "in_progress" else "📋"
                        content = todo.get("content", "N/A")
                        print(f"        {i}. {status_emoji} [{todo.get('status', 'pending')}] {content}")
        
        self.timings.append({
            "step": f"model_call_{call_num}",
            "duration": step_duration,
            "timestamp": step_start,
            "tool_calls": len(getattr(response.messages[-1], 'tool_calls', [])) if hasattr(response, 'messages') and response.messages else 0,
            "todos_before": len(todos_before),
            "todos_after": len(todos_after),
        })
        
        return response
    
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Track time for each tool execution."""
        self.tool_call_count += 1
        call_num = self.tool_call_count
        tool_name = request.tool_call.get("name", "unknown")
        
        step_start = time.time()
        
        if self.verbose:
            print(f"\n  🔧 Tool Call #{call_num}: {tool_name} starting...")
            # Show tool arguments (more detailed for write_file)
            if "args" in request.tool_call:
                args = request.tool_call["args"]
                if tool_name == "write_file":
                    # Show detailed info for write_file
                    file_path = args.get("file_path", args.get("path", "N/A"))
                    content = args.get("content", "")
                    content_len = len(content) if content else 0
                    print(f"     📝 Writing to: {file_path}")
                    print(f"     📏 Content size: {content_len:,} characters")
                    if content_len > 0:
                        # Show first and last 100 chars
                        preview = content[:100] if len(content) > 100 else content
                        print(f"     📄 Content preview (first 100 chars): {preview}...")
                        if content_len > 200:
                            preview_end = content[-100:] if len(content) > 100 else ""
                            print(f"     📄 Content preview (last 100 chars): ...{preview_end}")
                else:
                    # For other tools, show truncated args
                    args_str = str(args)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    print(f"     Args: {args_str}")
        
        # Execute the actual tool call
        try:
            result = handler(request)
        except Exception as e:
            step_end = time.time()
            step_duration = step_end - step_start
            if self.verbose:
                print(f"  ❌ Tool Call #{call_num} ({tool_name}) FAILED after {step_duration:.2f}s")
                print(f"     Error: {type(e).__name__}: {str(e)}")
            self.timings.append({
                "step": f"tool_call_{call_num}",
                "tool_name": tool_name,
                "duration": step_duration,
                "timestamp": step_start,
                "error": str(e),
            })
            raise
        
        step_end = time.time()
        step_duration = step_end - step_start
        
        if self.verbose:
            print(f"  ⏱️  Tool Call #{call_num} ({tool_name}): {step_duration:.2f}s ({step_duration*1000:.2f}ms)")
            # Show detailed result analysis
            if isinstance(result, ToolMessage):
                content = str(result.content)
                content_len = len(content)
                if tool_name == "write_file":
                    print(f"     ✅ Write result: {content[:500] if content_len > 500 else content}")
                    if content_len > 500:
                        print(f"     ... (truncated, full result is {content_len} characters)")
                elif tool_name == "read_file":
                    # Analyze read_file results for search data
                    print(f"     📄 Read result: {content_len:,} characters")
                    if content_len > 1000:
                        print(f"     → Result preview (first 200 chars): {content[:200]}...")
                    # Check if this looks like JSON search results
                    if content.strip().startswith("{") or content.strip().startswith("["):
                        try:
                            import json
                            data = json.loads(content)
                            if isinstance(data, dict):
                                # Check for search result structure
                                if "hits" in data or "results" in data or "users" in data:
                                    hits = data.get("hits", data.get("results", data.get("users", [])))
                                    if isinstance(hits, list):
                                        print(f"     🔍 Search results detected: {len(hits)} users found")
                                        if "total" in data:
                                            print(f"     📊 Total matches: {data.get('total')}")
                                        # Show size of each hit
                                        if hits:
                                            avg_hit_size = sum(len(str(hit)) for hit in hits) / len(hits)
                                            print(f"     📏 Average user data size: ~{avg_hit_size:.0f} characters per user")
                                            total_hits_size = sum(len(str(hit)) for hit in hits)
                                            print(f"     📏 Total users data size: {total_hits_size:,} characters")
                        except (json.JSONDecodeError, ValueError):
                            pass
                elif tool_name == "execute":
                    # Analyze execute results - might be search script output
                    print(f"     ⚡ Execute result: {content_len:,} characters")
                    if content_len > 500:
                        print(f"     → Result preview (first 300 chars): {content[:300]}...")
                        print(f"     → Result preview (last 200 chars): ...{content[-200:]}")
                    # Check if this looks like JSON search results
                    if content.strip().startswith("{") or content.strip().startswith("["):
                        try:
                            import json
                            data = json.loads(content)
                            if isinstance(data, dict):
                                # Check for search result structure
                                if "hits" in data or "results" in data or "users" in data:
                                    hits = data.get("hits", data.get("results", data.get("users", [])))
                                    if isinstance(hits, list):
                                        print(f"     🔍 Search results detected: {len(hits)} users found")
                                        if "total" in data:
                                            print(f"     📊 Total matches: {data.get('total')}")
                                        # Show size of each hit
                                        if hits:
                                            avg_hit_size = sum(len(str(hit)) for hit in hits) / len(hits)
                                            print(f"     📏 Average user data size: ~{avg_hit_size:.0f} characters per user")
                                            total_hits_size = sum(len(str(hit)) for hit in hits)
                                            print(f"     📏 Total users data size: {total_hits_size:,} characters")
                                            # Warn if very large
                                            if total_hits_size > 50000:
                                                print(f"     ⚠️  WARNING: Very large search results ({total_hits_size:,} chars) - may cause context overflow!")
                        except (json.JSONDecodeError, ValueError):
                            pass
                elif content_len > 1000:
                    print(f"     → Result size: {content_len:,} characters")
                    # Show first 200 chars of result
                    print(f"     → Result preview: {content[:200]}...")
        
        self.timings.append({
            "step": f"tool_call_{call_num}",
            "tool_name": tool_name,
            "duration": step_duration,
            "timestamp": step_start,
        })
        
        return result
    
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call."""
        self.tool_call_count += 1
        call_num = self.tool_call_count
        tool_name = request.tool_call.get("name", "unknown")
        
        step_start = time.time()
        
        if self.verbose:
            print(f"\n  🔧 Tool Call #{call_num}: {tool_name} starting (async)...")
            # Show tool arguments (more detailed for write_file)
            if "args" in request.tool_call:
                args = request.tool_call["args"]
                if tool_name == "write_file":
                    # Show detailed info for write_file
                    file_path = args.get("file_path", args.get("path", "N/A"))
                    content = args.get("content", "")
                    content_len = len(content) if content else 0
                    print(f"     📝 Writing to: {file_path}")
                    print(f"     📏 Content size: {content_len:,} characters")
                    if content_len > 0:
                        # Show first and last 100 chars
                        preview = content[:100] if len(content) > 100 else content
                        print(f"     📄 Content preview (first 100 chars): {preview}...")
                        if content_len > 200:
                            preview_end = content[-100:] if len(content) > 100 else ""
                            print(f"     📄 Content preview (last 100 chars): ...{preview_end}")
                else:
                    # For other tools, show truncated args
                    args_str = str(args)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    print(f"     Args: {args_str}")
        
        # Execute the actual tool call
        try:
            result = await handler(request)
        except Exception as e:
            step_end = time.time()
            step_duration = step_end - step_start
            if self.verbose:
                print(f"  ❌ Tool Call #{call_num} ({tool_name}) FAILED after {step_duration:.2f}s")
                print(f"     Error: {type(e).__name__}: {str(e)}")
            self.timings.append({
                "step": f"tool_call_{call_num}",
                "tool_name": tool_name,
                "duration": step_duration,
                "timestamp": step_start,
                "error": str(e),
            })
            raise
        
        step_end = time.time()
        step_duration = step_end - step_start
        
        if self.verbose:
            print(f"  ⏱️  Tool Call #{call_num} ({tool_name}): {step_duration:.2f}s ({step_duration*1000:.2f}ms)")
            # Show detailed result analysis
            if isinstance(result, ToolMessage):
                content = str(result.content)
                content_len = len(content)
                if tool_name == "write_file":
                    print(f"     ✅ Write result: {content[:500] if content_len > 500 else content}")
                    if content_len > 500:
                        print(f"     ... (truncated, full result is {content_len} characters)")
                elif tool_name == "read_file":
                    # Analyze read_file results for search data
                    print(f"     📄 Read result: {content_len:,} characters")
                    if content_len > 1000:
                        print(f"     → Result preview (first 200 chars): {content[:200]}...")
                    # Check if this looks like JSON search results
                    if content.strip().startswith("{") or content.strip().startswith("["):
                        try:
                            import json
                            data = json.loads(content)
                            if isinstance(data, dict):
                                # Check for search result structure
                                if "hits" in data or "results" in data or "users" in data:
                                    hits = data.get("hits", data.get("results", data.get("users", [])))
                                    if isinstance(hits, list):
                                        print(f"     🔍 Search results detected: {len(hits)} users found")
                                        if "total" in data:
                                            print(f"     📊 Total matches: {data.get('total')}")
                                        # Show size of each hit
                                        if hits:
                                            avg_hit_size = sum(len(str(hit)) for hit in hits) / len(hits)
                                            print(f"     📏 Average user data size: ~{avg_hit_size:.0f} characters per user")
                                            total_hits_size = sum(len(str(hit)) for hit in hits)
                                            print(f"     📏 Total users data size: {total_hits_size:,} characters")
                        except (json.JSONDecodeError, ValueError):
                            pass
                elif tool_name == "execute":
                    # Analyze execute results - might be search script output
                    print(f"     ⚡ Execute result: {content_len:,} characters")
                    if content_len > 500:
                        print(f"     → Result preview (first 300 chars): {content[:300]}...")
                        print(f"     → Result preview (last 200 chars): ...{content[-200:]}")
                    # Check if this looks like JSON search results
                    if content.strip().startswith("{") or content.strip().startswith("["):
                        try:
                            import json
                            data = json.loads(content)
                            if isinstance(data, dict):
                                # Check for search result structure
                                if "hits" in data or "results" in data or "users" in data:
                                    hits = data.get("hits", data.get("results", data.get("users", [])))
                                    if isinstance(hits, list):
                                        print(f"     🔍 Search results detected: {len(hits)} users found")
                                        if "total" in data:
                                            print(f"     📊 Total matches: {data.get('total')}")
                                        # Show size of each hit
                                        if hits:
                                            avg_hit_size = sum(len(str(hit)) for hit in hits) / len(hits)
                                            print(f"     📏 Average user data size: ~{avg_hit_size:.0f} characters per user")
                                            total_hits_size = sum(len(str(hit)) for hit in hits)
                                            print(f"     📏 Total users data size: {total_hits_size:,} characters")
                                            # Warn if very large
                                            if total_hits_size > 50000:
                                                print(f"     ⚠️  WARNING: Very large search results ({total_hits_size:,} chars) - may cause context overflow!")
                        except (json.JSONDecodeError, ValueError):
                            pass
                elif content_len > 1000:
                    print(f"     → Result size: {content_len:,} characters")
                    # Show first 200 chars of result
                    print(f"     → Result preview: {content[:200]}...")
        
        self.timings.append({
            "step": f"tool_call_{call_num}",
            "tool_name": tool_name,
            "duration": step_duration,
            "timestamp": step_start,
        })
        
        return result
    
    def print_summary(self) -> None:
        """Print a summary of all timings."""
        if not self.verbose or not self.start_time:
            return
        
        if self.total_time is None:
            self.total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("⏱️  TIMING SUMMARY")
        print("="*80)
        
        # Group timings by type
        model_calls = [t for t in self.timings if t["step"].startswith("model_call")]
        tool_calls = [t for t in self.timings if t["step"].startswith("tool_call")]
        other_steps = [t for t in self.timings if not t["step"].startswith(("model_call", "tool_call"))]
        
        total_model_time = sum(t["duration"] for t in model_calls)
        total_tool_time = sum(t["duration"] for t in tool_calls)
        total_other_time = sum(t["duration"] for t in other_steps)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total execution time: {self.total_time:.2f}s ({self.total_time*1000:.2f}ms)")
        print(f"  Model calls: {len(model_calls)} ({total_model_time:.2f}s total, {total_model_time/len(model_calls)*1000:.2f}ms avg)" if model_calls else "  Model calls: 0")
        print(f"  Tool calls: {len(tool_calls)} ({total_tool_time:.2f}s total, {total_tool_time/len(tool_calls)*1000:.2f}ms avg)" if tool_calls else "  Tool calls: 0")
        
        if model_calls:
            print(f"\n🔄 Model Call Breakdown:")
            for i, timing in enumerate(model_calls, 1):
                tool_call_count = timing.get("tool_calls", 0)
                print(f"  {i}. {timing['step']}: {timing['duration']:.2f}s ({timing['duration']*1000:.2f}ms)" + 
                      (f" → {tool_call_count} tool calls" if tool_call_count > 0 else ""))
        
        if tool_calls:
            print(f"\n🔧 Tool Call Breakdown:")
            # Group by tool name
            tool_groups: dict[str, list[float]] = {}
            write_file_calls = []
            for timing in tool_calls:
                tool_name = timing.get("tool_name", "unknown")
                if tool_name not in tool_groups:
                    tool_groups[tool_name] = []
                tool_groups[tool_name].append(timing["duration"])
                # Track write_file calls specifically
                if tool_name == "write_file":
                    write_file_calls.append(timing)
            
            for tool_name, durations in sorted(tool_groups.items(), key=lambda x: sum(x[1]), reverse=True):
                total = sum(durations)
                avg = total / len(durations)
                print(f"  {tool_name}: {len(durations)} calls, {total:.2f}s total, {avg*1000:.2f}ms avg")
                # Show slowest calls
                if len(durations) > 1:
                    slowest = max(durations)
                    if slowest > avg * 2:  # If slowest is more than 2x average
                        print(f"    ⚠️  Slowest call: {slowest:.2f}s")
            
            # Special section for write_file operations
            if write_file_calls:
                print(f"\n📝 Write File Operations Analysis:")
                print(f"  Total write_file calls: {len(write_file_calls)}")
                for i, call in enumerate(write_file_calls, 1):
                    step = call.get("step", "unknown")
                    duration = call.get("duration", 0)
                    error = call.get("error")
                    if error:
                        print(f"  {i}. {step}: {duration:.2f}s - ❌ ERROR: {error}")
                    else:
                        print(f"  {i}. {step}: {duration:.2f}s - ✅ Success")
                if len(write_file_calls) > 1:
                    print(f"  ⚠️  WARNING: Multiple write_file calls detected - agent may be retrying or stuck in a loop")
        
        # Show time breakdown
        print(f"\n⏱️  Time Breakdown:")
        print(f"  Model calls: {total_model_time:.2f}s ({total_model_time/self.total_time*100:.1f}%)" if self.total_time > 0 else "  Model calls: 0s")
        print(f"  Tool calls: {total_tool_time:.2f}s ({total_tool_time/self.total_time*100:.1f}%)" if self.total_time > 0 else "  Tool calls: 0s")
        print(f"  Other: {total_other_time:.2f}s ({total_other_time/self.total_time*100:.1f}%)" if self.total_time > 0 else "  Other: 0s")
        print(f"  Overhead/Unaccounted: {self.total_time - total_model_time - total_tool_time - total_other_time:.2f}s")
        
        # Detect potential infinite loops or stuck behavior
        print(f"\n🔍 Behavior Analysis:")
        if len(model_calls) > 20:
            print(f"  ⚠️  WARNING: High number of model calls ({len(model_calls)}) - agent may be stuck in a loop")
        
        # Check for repeated todo updates
        write_todos_count = sum(1 for t in tool_calls if t.get("tool_name") == "write_todos")
        if write_todos_count > 3:
            print(f"  ⚠️  WARNING: write_todos called {write_todos_count} times - agent may be repeatedly updating todos instead of executing them")
        
        # Check for excessive filesystem exploration
        filesystem_tools = ["ls", "glob", "read_file"]
        filesystem_count = sum(1 for t in tool_calls if t.get("tool_name") in filesystem_tools)
        if filesystem_count > 10:
            print(f"  ⚠️  WARNING: Excessive filesystem exploration ({filesystem_count} calls) - agent may be stuck looking for files")
        
        # Check if agent is making progress (completing different tasks)
        unique_tools = len(set(t.get("tool_name", "unknown") for t in tool_calls))
        if len(tool_calls) > 10 and unique_tools < 5:
            print(f"  ⚠️  WARNING: Agent is repeating the same tools ({unique_tools} unique tools out of {len(tool_calls)} calls) - may be stuck")
        
        # Check for actual skill usage (look for skill-related tool calls)
        skill_indicators = ["aihehuo", "search", "member"]
        skill_usage = any(
            any(indicator in str(t.get("tool_name", "")).lower() for indicator in skill_indicators)
            for t in tool_calls
        )
        if not skill_usage and len(tool_calls) > 5:
            print(f"  ⚠️  WARNING: No skill usage detected after {len(tool_calls)} tool calls - agent may not be executing the requested task")
        
        print("="*80 + "\n")

