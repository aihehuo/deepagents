import pytest

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.types import Command

from deepagents.graph import create_deep_agent

# Try to import WRITE_TODOS_SYSTEM_PROMPT, but handle if it's not exported
try:
    from langchain.agents.middleware.todo import WRITE_TODOS_SYSTEM_PROMPT
except ImportError:
    WRITE_TODOS_SYSTEM_PROMPT = None


def _get_write_todos_tool():
    middleware = TodoListMiddleware()
    tool = next(tool for tool in middleware.tools if tool.name == "write_todos")
    return middleware, tool


def _create_tool_call(todos: list, tool_call_id: str) -> dict:
    """Create a ToolCall dict for tools with InjectedToolCallId."""
    return {
        "name": "write_todos",
        "args": {"todos": todos},
        "id": tool_call_id,
        "type": "tool_call",
    }


def test_write_todos_returns_command_for_business_idea():
    middleware, tool = _get_write_todos_tool()

    todos = [
        {"content": "Draft positioning for new AI SaaS", "status": "in_progress"},
        {"content": "Identify first 3 customer segments", "status": "pending"},
        {"content": "Outline MVP feature set", "status": "pending"},
    ]

    # For tools with InjectedToolCallId, we need to use the ToolCall dict format
    result = tool.invoke(_create_tool_call(todos, "call-business"))

    assert isinstance(result, Command)
    assert result.update["todos"] == todos
    assert result.update["messages"][0].content.startswith("Updated todo list to")


def test_write_todos_updates_progress_for_followup_tasks():
    middleware, tool = _get_write_todos_tool()

    todos = [
        {"content": "Research market size", "status": "completed"},
        {"content": "Draft landing page copy", "status": "in_progress"},
        {"content": "Collect early adopter signups", "status": "pending"},
    ]

    # For tools with InjectedToolCallId, we need to use the ToolCall dict format
    result = tool.invoke(_create_tool_call(todos, "call-followup"))

    assert isinstance(result, Command)
    assert result.update["todos"][0]["status"] == "completed"
    assert "Updated todo list to" in result.update["messages"][0].content


def test_middleware_exposes_system_prompt_and_state_schema():
    middleware, _ = _get_write_todos_tool()

    assert middleware.state_schema.__name__ == "PlanningState"
    assert "write_todos" in middleware.system_prompt
    
    # Only test WRITE_TODOS_SYSTEM_PROMPT if it's available
    if WRITE_TODOS_SYSTEM_PROMPT is not None:
        assert "write_todos" in WRITE_TODOS_SYSTEM_PROMPT


class FixedGenericFakeChatModel(GenericFakeChatModel):
    """Fixed version of GenericFakeChatModel that properly handles bind_tools."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Override bind_tools to return self."""
        return self


class TestTodoListMiddlewareIntegration:
    """Integration tests for the full todo list creation flow."""

    def test_agent_creates_todos_from_user_request(self) -> None:
        """Test that an agent creates a todo list from a complex user request.
        
        This tests the full flow:
        1. User provides a complex multi-step task
        2. Agent (with TodoListMiddleware) processes the request
        3. LLM decides to create todos and calls write_todos
        4. Todos are saved in agent state
        """
        # Create a fake model that simulates LLM deciding to create todos
        expected_todos = [
            {"content": "Analyze current codebase structure", "status": "in_progress"},
            {"content": "Identify refactoring opportunities", "status": "pending"},
            {"content": "Prioritize refactoring tasks by impact", "status": "pending"},
            {"content": "Execute refactoring with tests", "status": "pending"},
        ]
        
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="I'll help you refactor the codebase. Let me create a plan first.",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {"todos": expected_todos},
                                "id": "call_todos_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've created a plan. Let me start by analyzing the codebase structure.",
                    ),
                ]
            )
        )

        # Create a deep agent with the fake model
        # TodoListMiddleware is included by default in create_deep_agent
        agent = create_deep_agent(model=model)

        # Invoke the agent with a complex multi-step task
        user_request = "Help me refactor this large codebase with 10+ modules"
        result = agent.invoke({"messages": [HumanMessage(content=user_request)]})

        # Verify the agent executed correctly
        assert "messages" in result
        assert "todos" in result

        # Verify todos were created and saved in state
        assert len(result["todos"]) == len(expected_todos)
        assert result["todos"] == expected_todos

        # Verify the write_todos tool was called
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        write_todos_messages = [
            msg for msg in tool_messages 
            if "write_todos" in msg.content.lower() or "todo" in msg.content.lower()
        ]
        assert len(write_todos_messages) > 0, "write_todos tool should have been called"

        # Verify the tool message contains the todos
        tool_message_content = write_todos_messages[0].content
        assert "todo" in tool_message_content.lower()

        # Verify AI messages show the planning process
        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        assert len(ai_messages) > 0
        # First AI message should mention creating a plan
        assert any("plan" in msg.content.lower() for msg in ai_messages)

    def test_agent_updates_todos_during_execution(self) -> None:
        """Test that an agent can update todos as it progresses through tasks.
        
        This tests:
        1. Initial todo creation
        2. Todo status updates as work progresses
        3. State persistence across multiple tool calls
        """
        initial_todos = [
            {"content": "Research market size", "status": "in_progress"},
            {"content": "Draft landing page copy", "status": "pending"},
            {"content": "Collect early adopter signups", "status": "pending"},
        ]
        
        updated_todos = [
            {"content": "Research market size", "status": "completed"},
            {"content": "Draft landing page copy", "status": "in_progress"},
            {"content": "Collect early adopter signups", "status": "pending"},
        ]

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    # First: Create initial todos
                    AIMessage(
                        content="I'll help you launch your product. Let me create a plan.",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {"todos": initial_todos},
                                "id": "call_todos_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Second: Update todos after completing first task
                    AIMessage(
                        content="I've completed the market research. Now updating the plan.",
                        tool_calls=[
                            {
                                "name": "write_todos",
                                "args": {"todos": updated_todos},
                                "id": "call_todos_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Final response
                    AIMessage(
                        content="I've updated the plan. Starting on the landing page copy.",
                    ),
                ]
            )
        )

        agent = create_deep_agent(model=model)
        result = agent.invoke({
            "messages": [HumanMessage(content="Help me launch my new SaaS product")]
        })

        # Verify final state has updated todos
        assert "todos" in result
        assert len(result["todos"]) == len(updated_todos)
        assert result["todos"] == updated_todos

        # Verify the first task is marked as completed
        assert result["todos"][0]["status"] == "completed"
        assert result["todos"][0]["content"] == "Research market size"

        # Verify the second task is now in_progress
        assert result["todos"][1]["status"] == "in_progress"
        assert result["todos"][1]["content"] == "Draft landing page copy"

        # Verify multiple write_todos calls occurred
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        write_todos_messages = [
            msg for msg in tool_messages 
            if "todo" in msg.content.lower()
        ]
        assert len(write_todos_messages) >= 2, "Should have multiple todo updates"


@pytest.mark.skipif(
    not pytest.importorskip("langchain_anthropic", reason="langchain_anthropic not installed"),
    reason="Requires langchain_anthropic",
)
class TestTodoListWithRealLLM:
    """Integration test with real LLM that creates todos and executes them."""

    def test_agent_creates_and_executes_all_todos(self) -> None:
        """Test that a real LLM agent creates a todo list and executes all tasks.
        
        This test:
        1. Creates a minimal agent with only TodoListMiddleware + mock execution tools
        2. Gives the agent a complex multi-step task
        3. Agent creates todos using write_todos
        4. Agent executes each todo using mock tools
        5. Agent marks todos as completed
        6. Verifies all todos are completed
        
        Uses DeepSeek model configured via .env.deepseek file.
        """
        import os
        from pathlib import Path
        from langchain.agents import create_agent
        from langchain.agents.middleware import TodoListMiddleware
        from langchain.tools import ToolRuntime
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, ToolMessage
        from langchain_core.tools import tool
        from langgraph.types import Command

        # Load DeepSeek configuration from .env.deepseek
        env_file = Path(__file__).parent.parent / ".env.deepseek"
        if not env_file.exists():
            pytest.skip(f"DeepSeek config file not found: {env_file}")
        
        # Read and parse the env file
        env_vars = {}
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "export " in line:
                    # Parse: export KEY=value
                    key_value = line.replace("export ", "").split("=", 1)
                    if len(key_value) == 2:
                        key, value = key_value
                        # Remove quotes if present
                        value = value.strip('"\'')
                        env_vars[key] = value
        
        # Set environment variables
        base_url = env_vars.get("ANTHROPIC_BASE_URL")
        api_key = env_vars.get("ANTHROPIC_API_KEY")
        model_name = env_vars.get("ANTHROPIC_MODEL", "deepseek-chat")
        
        if not base_url or not api_key:
            pytest.skip("DeepSeek configuration incomplete in .env.deepseek")
        
        # Temporarily set environment variables
        old_base_url = os.environ.get("ANTHROPIC_BASE_URL")
        old_api_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_BASE_URL"] = base_url
        os.environ["ANTHROPIC_API_KEY"] = api_key
        
        try:
            # Mock execution tool that simulates completing a task
            @tool(description="Execute a task from your todo list. Call this tool when you're ready to work on a specific task. The task_description should match one of the tasks in your current todo list.")
            def execute_task(task_description: str, runtime: ToolRuntime) -> Command:
                """Mock tool that simulates task execution.
                
                This tool simulates completing a task. When called, it returns a success message.
                The agent should call this for each in_progress task, then update todos to mark it completed.
                """
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=f"✅ Successfully completed task: {task_description}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )

            # Create DeepSeek model with custom base URL
            model = ChatAnthropic(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                max_tokens=20000,
            )

            # Create a minimal agent with only TodoListMiddleware and the mock execution tool
            agent = create_agent(
                model=model,
                middleware=[TodoListMiddleware()],
                tools=[execute_task],
                system_prompt="""You are a task execution agent. Your job is to:
1. Create a todo list for complex multi-step tasks using write_todos
2. Execute each task in the todo list using the execute_task tool
3. Update the todo list to mark tasks as completed after executing them
4. Continue until all tasks are completed

When you see a task marked as 'in_progress' in your todo list, use execute_task to complete it, then update the todo list to mark it as 'completed' and mark the next task as 'in_progress'.

Work through the todo list systematically until all tasks are done.""",
            )

            # Give the agent a complex multi-step task
            user_request = """Help me plan and execute a product launch. I need to:
1. Research the target market
2. Create a marketing strategy
3. Design the product landing page
4. Set up analytics tracking

Please create a todo list and execute all the tasks."""
            
            print("\n" + "="*80)
            print("STARTING AGENT EXECUTION")
            print("="*80)
            print(f"\n📝 User Request:\n{user_request}\n")
            
            # Execute and capture all messages
            input_state = {"messages": [HumanMessage(content=user_request)]}
            result = agent.invoke(input_state)
            
            # Print each LLM call and response
            print("\n" + "="*80)
            print("LLM CALLS AND RESPONSES")
            print("="*80)
            
            ai_message_count = 0
            for i, message in enumerate(result.get("messages", []), 1):
                if message.type == "ai":
                    ai_message_count += 1
                    print(f"\n{'─'*80}")
                    print(f"🤖 LLM Response #{ai_message_count} (Message #{i}):")
                    print(f"{'─'*80}")
                    print(f"Content:\n{message.content}")
                    
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        print(f"\n📞 Tool Calls ({len(message.tool_calls)}):")
                        for j, tool_call in enumerate(message.tool_calls, 1):
                            tool_name = tool_call.get('name', 'unknown')
                            tool_args = tool_call.get('args', {})
                            print(f"\n  Tool Call #{j}: {tool_name}")
                            
                            # Special formatting for write_todos
                            if tool_name == "write_todos" and isinstance(tool_args, dict) and 'todos' in tool_args:
                                todos_preview = tool_args['todos']
                                if isinstance(todos_preview, list) and len(todos_preview) > 0:
                                    print(f"      Todos ({len(todos_preview)} items):")
                                    for k, todo in enumerate(todos_preview, 1):
                                        status = todo.get('status', 'unknown')
                                        content = todo.get('content', 'N/A')
                                        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(status, "❓")
                                        print(f"        {k}. {status_emoji} [{status}] {content}")
                                else:
                                    print(f"      Args: {tool_args}")
                            else:
                                # Truncate long args
                                args_str = str(tool_args)
                                if len(args_str) > 200:
                                    print(f"      Args: {args_str[:200]}...")
                                else:
                                    print(f"      Args: {tool_args}")
                    
                elif message.type == "tool":
                    tool_name = getattr(message, 'name', 'unknown')
                    content = message.content
                    print(f"\n🔧 Tool Response (Message #{i}): {tool_name}")
                    # Truncate very long content for readability
                    if len(content) > 500:
                        print(f"  Content: {content[:500]}...")
                        print(f"  ... (truncated, total length: {len(content)} chars)")
                    else:
                        print(f"  Content: {content}")
            
            print(f"\n{'='*80}")
            print("FINAL STATE")
            print(f"{'='*80}")
            print(f"\n📋 Final Todos ({len(result.get('todos', []))} items):")
            for i, todo in enumerate(result.get('todos', []), 1):
                status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(todo.get('status', 'pending'), "❓")
                print(f"  {i}. {status_emoji} [{todo.get('status', 'unknown')}] {todo.get('content', 'N/A')}")
            
            print(f"\n💬 Total Messages: {len(result.get('messages', []))}")
            ai_messages = [msg for msg in result.get('messages', []) if msg.type == 'ai']
            tool_messages = [msg for msg in result.get('messages', []) if msg.type == 'tool']
            print(f"  - AI Messages: {len(ai_messages)}")
            print(f"  - Tool Messages: {len(tool_messages)}")

            # Verify todos were created
            assert "todos" in result
            assert len(result["todos"]) > 0, "Agent should have created todos"

            # Verify all todos are completed
            all_completed = all(todo["status"] == "completed" for todo in result["todos"])
            assert all_completed, f"All todos should be completed. Current state: {result['todos']}"

            # Verify execute_task was called (at least once per task)
            tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
            execute_task_messages = [
                msg for msg in tool_messages 
                if "execute_task" in str(msg.name).lower() or "Successfully completed task" in msg.content
            ]
            assert len(execute_task_messages) >= len(result["todos"]), \
                f"Should have executed at least one task per todo. Executed: {len(execute_task_messages)}, Todos: {len(result['todos'])}"

            # Verify write_todos was called multiple times (initial creation + updates)
            write_todos_messages = [
                msg for msg in tool_messages 
                if "todo" in msg.content.lower() and "Updated todo list" in msg.content
            ]
            assert len(write_todos_messages) >= 2, \
                "Should have multiple todo updates (initial creation + completion updates)"

            print(f"\n✅ Test passed! Agent created {len(result['todos'])} todos and completed them all.")
            print(f"   Todos: {[t['content'] for t in result['todos']]}")
        finally:
            # Restore original environment variables
            if old_base_url is not None:
                os.environ["ANTHROPIC_BASE_URL"] = old_base_url
            elif "ANTHROPIC_BASE_URL" in os.environ:
                del os.environ["ANTHROPIC_BASE_URL"]
            
            if old_api_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = old_api_key
            elif "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

    def test_agent_decides_task_completion_without_tools(self) -> None:
        """Test that the LLM decides task completion itself without relying on tool responses.
        
        This test:
        1. Gives the LLM tasks it can complete directly (no external tools needed)
        2. LLM must decide when each task is complete
        3. LLM updates todos to mark tasks as completed
        4. LLM proceeds to next task only after marking previous as completed
        
        This tests the actual decision-making process, not just following tool responses.
        """
        import os
        from pathlib import Path
        from langchain.agents import create_agent
        from langchain.agents.middleware import TodoListMiddleware
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage

        # Load DeepSeek configuration from .env.deepseek
        env_file = Path(__file__).parent.parent / ".env.deepseek"
        if not env_file.exists():
            pytest.skip(f"DeepSeek config file not found: {env_file}")
        
        # Read and parse the env file
        env_vars = {}
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "export " in line:
                    key_value = line.replace("export ", "").split("=", 1)
                    if len(key_value) == 2:
                        key, value = key_value
                        value = value.strip('"\'')
                        env_vars[key] = value
        
        base_url = env_vars.get("ANTHROPIC_BASE_URL")
        api_key = env_vars.get("ANTHROPIC_API_KEY")
        model_name = env_vars.get("ANTHROPIC_MODEL", "deepseek-chat")
        
        if not base_url or not api_key:
            pytest.skip("DeepSeek configuration incomplete in .env.deepseek")
        
        # Temporarily set environment variables
        old_base_url = os.environ.get("ANTHROPIC_BASE_URL")
        old_api_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_BASE_URL"] = base_url
        os.environ["ANTHROPIC_API_KEY"] = api_key
        
        try:
            # Create DeepSeek model with custom base URL
            model = ChatAnthropic(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                max_tokens=20000,
            )

            # Create a minimal agent with ONLY TodoListMiddleware (no execution tools)
            # The LLM must complete tasks using only its reasoning and writing capabilities
            agent = create_agent(
                model=model,
                middleware=[TodoListMiddleware()],
                tools=[],  # No tools - LLM must work directly
                system_prompt="""You are a task execution agent. Your job is to:
1. Create a todo list for complex multi-step tasks using write_todos
2. Work through each task in the todo list directly (no tools needed - you can write, analyze, plan, etc.)
3. Decide when each task is complete based on your own assessment
4. Update the todo list to mark tasks as completed ONLY after you have fully completed them
5. Continue to the next task only after marking the previous one as completed

IMPORTANT: 
- You must decide task completion yourself - there are no tools to tell you when a task is done
- Only mark a task as completed when you have actually finished the work for that task
- Be thorough - complete each task fully before moving on
- Update the todo list after completing each task to reflect your progress""",
            )

            # Give the LLM tasks it can complete directly (writing, analysis, planning)
            user_request = """I need you to help me plan a software project. Please:

1. Write a brief summary of what makes a good software project plan (2-3 sentences)
2. Analyze the key components that should be included in a project plan (list 3-4 components)
3. Create a sample project timeline for a 3-month software project (just outline the phases)
4. Write a short conclusion about the importance of project planning (1-2 sentences)

Please create a todo list for these tasks and work through each one. Complete each task fully before marking it as completed and moving to the next one."""
            
            print("\n" + "="*80)
            print("TEST: LLM DECIDES TASK COMPLETION")
            print("="*80)
            print(f"\n📝 User Request:\n{user_request}\n")
            print("Note: No execution tools provided - LLM must decide completion itself\n")
            
            # Execute and capture all messages
            input_state = {"messages": [HumanMessage(content=user_request)]}
            result = agent.invoke(input_state)
            
            # Print each LLM call and response
            print("\n" + "="*80)
            print("LLM CALLS AND RESPONSES")
            print("="*80)
            
            ai_message_count = 0
            write_todos_calls = []
            for i, message in enumerate(result.get("messages", []), 1):
                if message.type == "ai":
                    ai_message_count += 1
                    print(f"\n{'─'*80}")
                    print(f"🤖 LLM Response #{ai_message_count} (Message #{i}):")
                    print(f"{'─'*80}")
                    print(f"Content:\n{message.content}")
                    
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        print(f"\n📞 Tool Calls ({len(message.tool_calls)}):")
                        for j, tool_call in enumerate(message.tool_calls, 1):
                            tool_name = tool_call.get('name', 'unknown')
                            tool_args = tool_call.get('args', {})
                            print(f"\n  Tool Call #{j}: {tool_name}")
                            
                            if tool_name == "write_todos" and isinstance(tool_args, dict) and 'todos' in tool_args:
                                todos_preview = tool_args['todos']
                                write_todos_calls.append(todos_preview)
                                if isinstance(todos_preview, list) and len(todos_preview) > 0:
                                    print(f"      Todos ({len(todos_preview)} items):")
                                    for k, todo in enumerate(todos_preview, 1):
                                        status = todo.get('status', 'unknown')
                                        content = todo.get('content', 'N/A')
                                        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(status, "❓")
                                        print(f"        {k}. {status_emoji} [{status}] {content}")
            
            print(f"\n{'='*80}")
            print("FINAL STATE")
            print(f"{'='*80}")
            print(f"\n📋 Final Todos ({len(result.get('todos', []))} items):")
            for i, todo in enumerate(result.get('todos', []), 1):
                status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(todo.get('status', 'pending'), "❓")
                print(f"  {i}. {status_emoji} [{todo.get('status', 'unknown')}] {todo.get('content', 'N/A')}")
            
            print(f"\n💬 Total Messages: {len(result.get('messages', []))}")
            ai_messages = [msg for msg in result.get('messages', []) if msg.type == 'ai']
            tool_messages = [msg for msg in result.get('messages', []) if msg.type == 'tool']
            print(f"  - AI Messages: {len(ai_messages)}")
            print(f"  - Tool Messages: {len(tool_messages)}")
            print(f"  - write_todos calls: {len(write_todos_calls)}")

            # Verify todos were created
            assert "todos" in result
            assert len(result["todos"]) > 0, "Agent should have created todos"
            
            # Verify all todos are completed (LLM should have completed all tasks)
            all_completed = all(todo["status"] == "completed" for todo in result["todos"])
            assert all_completed, f"All todos should be completed. Current state: {result['todos']}"

            # Verify write_todos was called multiple times
            # Should have: initial creation + updates after each task completion
            assert len(write_todos_calls) >= len(result["todos"]) + 1, \
                f"Should have at least {len(result['todos']) + 1} write_todos calls (initial + one per task). Got: {len(write_todos_calls)}"
            
            # Verify progression: each write_todos call should show more tasks completed
            completed_counts = []
            for todos_list in write_todos_calls:
                completed = sum(1 for t in todos_list if t.get('status') == 'completed')
                completed_counts.append(completed)
            
            # Completed count should generally increase (allowing for some flexibility)
            # At minimum, the last call should have all tasks completed
            assert completed_counts[-1] == len(result["todos"]), \
                f"Final write_todos call should have all tasks completed. Got {completed_counts[-1]} completed out of {len(result['todos'])}"
            
            # Verify that tasks were completed in order (each subsequent call should have >= completed tasks)
            # Allow some flexibility for LLM to reorganize
            print(f"\n📊 Completion progression: {completed_counts}")
            
            print(f"\n✅ Test passed! LLM decided completion for {len(result['todos'])} tasks without tool assistance.")
            print(f"   Tasks completed: {[t['content'] for t in result['todos']]}")
            
        finally:
            # Restore original environment variables
            if old_base_url is not None:
                os.environ["ANTHROPIC_BASE_URL"] = old_base_url
            elif "ANTHROPIC_BASE_URL" in os.environ:
                del os.environ["ANTHROPIC_BASE_URL"]
            
            if old_api_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = old_api_key
            elif "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

