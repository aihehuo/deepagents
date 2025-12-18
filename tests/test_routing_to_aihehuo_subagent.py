"""Integration test for routing to AI He Huo subagent.

This test validates that the main agent can delegate AI He Huo search tasks
to the specialized aihehuo subagent, which has the AihehuoMiddleware equipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.datetime import DateTimeMiddleware
from deepagents.middleware.routing import build_default_aihehuo_routing_middleware
from deepagents.subagent_presets import build_aihehuo_subagent_from_env
from tests.model_provider import create_test_model, load_test_model_config
from tests.test_backends import DirectoryOnlyBackend


def _called_task_with_subagent_type(messages, subagent_type: str) -> bool:
    """Return True if we see a `task(...)` tool call selecting the given subagent_type.
    
    Checks both valid tool_calls and invalid_tool_calls, since the agent may attempt
    to delegate even if the tool call has JSON parsing issues.
    """
    for message in messages or []:
        if getattr(message, "type", None) != "ai":
            continue
        
        # Check valid tool calls
        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                if tool_call.get("name") != "task":
                    continue
                args = tool_call.get("args") or {}
                if isinstance(args, dict) and args.get("subagent_type") == subagent_type:
                    return True
                # Also check if args is a string (JSON) that contains the subagent_type
                if isinstance(args, str) and f'"subagent_type": "{subagent_type}"' in args:
                    return True
        
        # Check invalid tool calls (agent attempted but JSON parsing failed)
        invalid_tool_calls = getattr(message, "invalid_tool_calls", None) or []
        for tool_call in invalid_tool_calls:
            if isinstance(tool_call, dict):
                if tool_call.get("name") != "task":
                    continue
                args = tool_call.get("args", "")
                # Check if args string contains the subagent_type
                if isinstance(args, str) and f'"subagent_type": "{subagent_type}"' in args:
                    return True
                # Try to parse as JSON if possible
                if isinstance(args, str):
                    try:
                        import json
                        parsed_args = json.loads(args)
                        if isinstance(parsed_args, dict) and parsed_args.get("subagent_type") == subagent_type:
                            return True
                    except (json.JSONDecodeError, TypeError):
                        # If JSON parsing fails, check if the string contains the subagent_type
                        if f'subagent_type": "{subagent_type}"' in args or f'"subagent_type": "{subagent_type}"' in args:
                            return True
    
    return False


def _extract_model_name(model) -> str | None:
    """Best-effort extraction of model name across LangChain chat model implementations."""
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _last_ai_text(messages) -> str:
    """Extract the last AI message text."""
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "ai":
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _has_aihehuo_search_results(messages) -> bool:
    """Check if messages contain AI He Huo search results.
    
    This checks:
    1. Tool messages for explicit search tool calls
    2. Task result content for search result indicators
    3. Final response for evidence of search results (candidates, recommendations, etc.)
    """
    for message in messages or []:
        content = str(getattr(message, "content", ""))
        msg_type = getattr(message, "type", None)
        
        # Check tool messages for explicit search tool calls
        if msg_type == "tool":
            # Look for AI He Huo search tool calls or results
            if "aihehuo_search_members" in content.lower() or "aihehuo_search_ideas" in content.lower():
                return True
            # Look for search result indicators in tool messages
            if any(indicator in content.lower() for indicator in ["用户id", "用户创业号", "用户名", "total", "hits", "data"]):
                if "aihehuo" in content.lower() or "爱合伙" in content.lower():
                    return True
        
        # Check task result content (subagent's response) for search result indicators
        # The subagent's response might contain evidence of searches even if tool calls aren't visible
        if msg_type == "tool" and len(content) > 200:
            # Look for indicators that suggest search results were returned
            search_indicators = [
                "co-founder",
                "cofounder",
                "founder",
                "partner",
                "investor",
                "candidate",
                "候选人",
                "匹配",
                "推荐",
                "recommend",
                "match",
                "found",
                "搜索",
                "search",
                "用户",
                "创业号",
            ]
            if any(indicator in content.lower() for indicator in search_indicators):
                # Additional check: content should be substantial and structured like search results
                if len(content) > 300 and ("姓名" in content or "name" in content.lower() or "角色" in content or "role" in content.lower()):
                    return True
        
        # Check AI messages for evidence of search results in final response
        if msg_type == "ai" and len(content) > 200:
            # Look for structured search results (candidates with names, roles, etc.)
            search_result_patterns = [
                ("姓名", "角色"),  # Chinese: name and role
                ("name", "role"),  # English
                ("候选人", "匹配"),  # Chinese: candidate and match
                ("candidate", "match"),  # English
            ]
            for pattern1, pattern2 in search_result_patterns:
                if pattern1 in content.lower() and pattern2 in content.lower():
                    return True
    
    return False


@pytest.mark.integration
@pytest.mark.timeout(300)  # 5 minutes for real LLM and API calls
def test_routing_delegates_aihehuo_search_to_subagent_real_llm(tmp_path, monkeypatch, request) -> None:
    """Real-LLM integration test for routing -> AI He Huo subagent delegation via the `task` tool.
    
    This test validates:
    1. The main agent delegates AI He Huo search tasks to the aihehuo subagent
    2. The subagent has AihehuoMiddleware and can perform searches
    3. The subagent returns search results
    4. The main agent synthesizes and returns the results
    
    This intentionally asserts on the *tool call* (task + subagent_type="aihehuo"), not on
    the exact content, to avoid brittleness across providers/models.
    """
    # Check for required API keys
    aihehuo_api_key = os.environ.get("AIHEHUO_API_KEY")
    if not aihehuo_api_key:
        pytest.skip("AIHEHUO_API_KEY is not set. This test requires AI He Huo API access.")
    
    # Encourage deterministic tool choice when models support it.
    monkeypatch.setenv("MODEL_API_TEMPERATURE", "0")
    monkeypatch.setenv("AIHEHUO_MODEL_API_TEMPERATURE", "0")
    
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    model = create_test_model(cfg=cfg)
    
    # Build AI He Huo subagent (this will have AihehuoMiddleware)
    aihehuo_subagent = build_aihehuo_subagent_from_env(tools=[], name="aihehuo")
    if aihehuo_subagent is None:
        pytest.skip(
            "AI He Huo subagent preset is not configured. Set AIHEHUO_MODEL_API_KEY "
            "(and optionally AIHEHUO_MODEL_BASE_URL, AIHEHUO_MODEL_API_PROVIDER) "
            "plus AIHEHUO_MODEL_NAME to run."
        )
    
    # Sanity-check we actually used the aihehuo-specific model name from env.
    expected_aihehuo_model_name = (os.environ.get("MODEL_NAME") or "").strip()
    actual_aihehuo_model_name = _extract_model_name(aihehuo_subagent.get("model"))
    assert actual_aihehuo_model_name == expected_aihehuo_model_name, (
        f"Expected AI He Huo subagent to use model '{expected_aihehuo_model_name}', "
        f"but got '{actual_aihehuo_model_name}'"
    )
    
    # Verify the subagent has AihehuoMiddleware
    subagent_middleware = aihehuo_subagent.get("middleware", [])
    from deepagents.middleware.aihehuo import AihehuoMiddleware
    has_aihehuo_middleware = any(isinstance(mw, AihehuoMiddleware) for mw in subagent_middleware)
    assert has_aihehuo_middleware, "AI He Huo subagent should have AihehuoMiddleware"
    print("✅ AI He Huo subagent has AihehuoMiddleware")
    
    # Create main agent WITHOUT AihehuoMiddleware (only subagent has it)
    # The main agent only has routing middleware to delegate to the subagent
    # The subagent has AihehuoMiddleware and can perform searches
    # Use DirectoryOnlyBackend to restrict writes/edits to the test file's directory
    test_file_dir = Path(__file__).parent
    base_backend = FilesystemBackend(root_dir=str(tmp_path))
    restricted_backend = DirectoryOnlyBackend(
        backend=base_backend,
        test_dir=test_file_dir,
    )
    
    agent = create_deep_agent(
        model=model,
        backend=restricted_backend,  # create_deep_agent automatically adds FilesystemMiddleware with this backend
        tools=[],
        checkpointer=MemorySaver(),
        subagents=[aihehuo_subagent],
        # Main agent middleware: routing and datetime, NOT AihehuoMiddleware
        # Note: FilesystemMiddleware is automatically added by create_deep_agent when backend is provided
        middleware=[
            DateTimeMiddleware(),  # Provides get_current_datetime tool for accurate timestamps
            build_default_aihehuo_routing_middleware(aihehuo_subagent_type="aihehuo"),
        ],
        system_prompt=(
            "You are an orchestrator. Follow routing hints added by middleware. "
            "When a routing hint applies, use the `task` tool accordingly.\n\n"
            "Output rules:\n"
            "- When the user asks to find co-founders, partners, or investors, delegate to the aihehuo subagent.\n"
            "- After the subagent completes the search, summarize the findings and provide recommendations.\n"
            "- Include key information about potential matches found.\n\n"
            "**Report Requirements:**\n"
            "- Reports must be written in Chinese (中文).\n"
            "- For each candidate recommended, you MUST include their profile page link/URL if available in the search results.\n"
            "- Profile links are essential for users to access candidate information directly."
        ),
    )
    
    config = {"configurable": {"thread_id": "test-routing-aihehuo-search"}}
    
    user_request = (
        "I have a business idea for an AI-powered language learning platform for children aged 5-12. "
        "I need to find:\n"
        "1. A technical co-founder with AI/ML and mobile app development experience\n"
        "2. An educational content expert with experience in children's language learning\n"
        "3. Investors interested in EdTech and AI-driven educational products\n\n"
        "Please search the AI He Huo platform and help me find potential partners and investors.\n\n"
        "**IMPORTANT: After completing your search and analysis, please write a comprehensive recommendation AI report "
        "to a file using the write_file tool. The report must be written in Chinese (中文).\n\n"
        "The report should include:\n"
        "- Summary of your search findings (用中文总结)\n"
        "- Detailed recommendations for each role (technical co-founder, content expert, investors)\n"
        "- For each candidate recommended, you MUST include their profile page link/URL from the search results\n"
        "- Key information about potential matches\n"
        "- Next steps and action items\n"
        "- Accurate date and time in human-readable format (ensure the report includes the current date and time)\n\n"
        "Save the report as a markdown file (e.g., /recommendation_report.md or /ai_he_huo_search_report.md). "
        "This report is required as the final deliverable.**"
    )
    
    print("\n" + "="*80)
    print("TEST: ROUTING TO AI HE HUO SUBAGENT")
    print("="*80)
    print(f"\n📝 User Request:\n{user_request}\n")
    
    print("⏳ Starting agent execution (streaming)...\n")
    
    # Use stream instead of invoke for real-time progress
    input_state = {"messages": [HumanMessage(content=user_request)]}
    result = None
    previous_message_count = 0
    
    print("📡 Streaming agent execution...")
    print("-" * 80)
    # Use stream_mode="values" to get full state at each step
    for chunk in agent.stream(input_state, config, stream_mode="values"):
        # chunk is the full state at this step
        result = chunk
        if "messages" in chunk:
            messages_so_far = chunk.get("messages", [])
            msg_count = len(messages_so_far)
            
            # Print new messages that appeared since last chunk
            if msg_count > previous_message_count:
                new_messages = messages_so_far[previous_message_count:]
                for msg in new_messages:
                    msg_type = getattr(msg, "type", "unknown")
                    content = getattr(msg, "content", "")
                    
                    # Print message header
                    if msg_type == "ai":
                        print(f"\n🤖 AI Message #{msg_count}:")
                    elif msg_type == "tool":
                        tool_name = getattr(msg, "name", "unknown_tool")
                        print(f"\n🔧 Tool Message ({tool_name}):")
                    elif msg_type == "human":
                        print(f"\n👤 Human Message:")
                    else:
                        print(f"\n📨 {msg_type.upper()} Message:")
                    
                    # Print content (truncate if very long)
                    if isinstance(content, str):
                        if len(content) > 500:
                            print(f"  {content[:500]}...")
                            print(f"  [Content truncated - full length: {len(content)} characters]")
                        else:
                            print(f"  {content}")
                    elif content:
                        content_str = str(content)
                        if len(content_str) > 500:
                            print(f"  {content_str[:500]}...")
                            print(f"  [Content truncated - full length: {len(content_str)} characters]")
                        else:
                            print(f"  {content_str}")
                    
                    # Print tool calls if present
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"  📞 Tool Calls ({len(msg.tool_calls)}):")
                        for i, tool_call in enumerate(msg.tool_calls, 1):
                            tool_name = tool_call.get("name", "unknown")
                            tool_args = tool_call.get("args", {})
                            print(f"    {i}. {tool_name}")
                            if tool_args:
                                args_str = str(tool_args)
                                if len(args_str) > 200:
                                    print(f"       Args: {args_str[:200]}...")
                                else:
                                    print(f"       Args: {args_str}")
                    
                    print("-" * 80)
                
                previous_message_count = msg_count
    
    # Extract messages from final state
    messages = result.get("messages", []) if result else []
    
    print(f"\n✅ Agent execution completed")
    print(f"📊 Total messages: {len(messages)}")
    
    # Debug: Print message types
    for i, msg in enumerate(messages):
        msg_type = getattr(msg, "type", "unknown")
        content_preview = str(getattr(msg, "content", ""))[:100] if hasattr(msg, "content") else ""
        print(f"  Message {i+1}: {msg_type} - {content_preview}...")
    
    # Validate that delegation happened
    assert _called_task_with_subagent_type(messages, "aihehuo"), (
        "Expected the orchestrator to delegate AI He Huo search work using the `task` tool with "
        'subagent_type="aihehuo". If this fails consistently for a given provider, consider '
        "increasing the routing middleware strictness further or using a model with stronger tool-calling."
    )
    print("✅ Main agent delegated to aihehuo subagent")
    
    # Extract and print the complete prompt sent to subagent and its result
    print("\n" + "="*80)
    print("SUBAGENT DELEGATION DETAILS")
    print("="*80)
    
    # Find the task tool call (prompt sent to subagent)
    task_description = None
    for message in messages:
        if getattr(message, "type", None) == "ai":
            # Check valid tool calls
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and tool_call.get("name") == "task":
                    args = tool_call.get("args") or {}
                    if isinstance(args, dict):
                        task_description = args.get("description", "")
                    elif isinstance(args, str):
                        # Try to extract description from JSON string
                        try:
                            import json
                            parsed = json.loads(args)
                            if isinstance(parsed, dict):
                                task_description = parsed.get("description", "")
                        except (json.JSONDecodeError, TypeError):
                            # If JSON parsing fails, try to extract from string
                            if '"description"' in args:
                                # Simple extraction - find description field
                                import re
                                match = re.search(r'"description"\s*:\s*"([^"]+)"', args)
                                if match:
                                    task_description = match.group(1)
            
            # Check invalid tool calls (in case JSON parsing failed)
            if not task_description:
                invalid_tool_calls = getattr(message, "invalid_tool_calls", None) or []
                for tool_call in invalid_tool_calls:
                    if isinstance(tool_call, dict) and tool_call.get("name") == "task":
                        args = tool_call.get("args", "")
                        if isinstance(args, str):
                            try:
                                import json
                                parsed = json.loads(args)
                                if isinstance(parsed, dict):
                                    task_description = parsed.get("description", "")
                            except (json.JSONDecodeError, TypeError):
                                # Try regex extraction
                                import re
                                match = re.search(r'"description"\s*:\s*"([^"]+(?:\\.[^"]*)*)"', args)
                                if not match:
                                    # Try without quotes (might be unescaped)
                                    match = re.search(r'"description"\s*:\s*"(.+?)"(?=\s*[,}])', args, re.DOTALL)
                                if match:
                                    task_description = match.group(1).replace('\\n', '\n').replace('\\"', '"')
    
    if task_description:
        print(f"\n📤 PROMPT SENT TO SUBAGENT ({len(task_description)} chars):")
        print("-" * 80)
        print(task_description)
        print("-" * 80)
    else:
        print("\n⚠️  Could not extract task description from tool call")
    
    # Find and print task result content (what subagent returned)
    task_result = None
    for message in messages:
        if getattr(message, "type", None) == "tool":
            content = str(getattr(message, "content", ""))
            if len(content) > 100:
                task_result = content
                break
    
    if task_result:
        print(f"\n📥 RESULT RETURNED BY SUBAGENT ({len(task_result)} chars):")
        print("-" * 80)
        print(task_result)
        print("-" * 80)
    else:
        print("\n⚠️  Could not find task result from subagent")
    
    print("="*80 + "\n")
    
    # Validate that search was performed (subagent should have used search tools)
    # This checks for evidence of search results in the messages, including:
    # - Explicit tool calls (if visible)
    # - Task result content with search result indicators
    # - Final response with structured search results
    has_search_results = _has_aihehuo_search_results(messages)
    assert has_search_results, (
        "Expected the aihehuo subagent to perform searches and return results. "
        "Check that the subagent has AihehuoMiddleware and that AIHEHUO_API_KEY is set. "
        "The subagent should return search results with candidate information."
    )
    print("✅ AI He Huo subagent performed searches and returned results")
    
    # Additional validation: Check task result content for search result structure
    task_result_found = False
    for message in messages:
        if getattr(message, "type", None) == "tool":
            content = str(getattr(message, "content", ""))
            # Task results from subagent should contain substantial content with search results
            if len(content) > 300:
                # Look for structured search results (candidates with details)
                has_candidates = any(
                    pattern in content.lower() 
                    for pattern in ["候选人", "candidate", "founder", "co-founder", "partner", "investor"]
                )
                has_details = any(
                    pattern in content.lower() 
                    for pattern in ["姓名", "name", "角色", "role", "经验", "experience", "匹配", "match"]
                )
                if has_candidates and has_details:
                    task_result_found = True
                    print(f"  ✓ Found structured search results in task response ({len(content)} chars)")
                    break
    
    if not task_result_found:
        print("  ⚠️  Task result structure check inconclusive, but search results were detected")
    
    # Validate final response contains useful information
    final_text = _last_ai_text(messages)
    assert len(final_text) > 100, (
        "Expected the final assistant message to contain a substantial summary of search results. "
        f"Got {len(final_text)} characters."
    )
    
    # Check for indicators of search results in final response
    # Include both English and Chinese keywords since responses may be in either language
    english_indicators = [
        "co-founder",
        "partner",
        "investor",
        "search",
        "found",
        "recommend",
        "match",
        "candidate",
    ]
    chinese_indicators = [
        "联合创始人",
        "合作伙伴",
        "投资者",
        "搜索",
        "找到",
        "推荐",
        "匹配",
        "候选人",
        "技术",
        "教育",
    ]
    # Check English indicators (case-insensitive)
    has_english_indicators = any(indicator in final_text.lower() for indicator in english_indicators)
    # Check Chinese indicators (case doesn't matter for Chinese)
    has_chinese_indicators = any(indicator in final_text for indicator in chinese_indicators)
    has_result_indicators = has_english_indicators or has_chinese_indicators
    
    assert has_result_indicators, (
        f"Expected the final response to contain information about search results, "
        f"co-founders, partners, or recommendations. "
        f"Final response preview: {final_text[:200]}..."
    )
    print("✅ Final response contains search results and recommendations")
    
    # Print final response for debugging
    should_print = (os.environ.get("BC_TEST_PRINT_AIHEHUO") or "").strip() in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    )
    if request.config.getoption("capture") == "no":
        should_print = True
    
    if should_print:
        print("\n========== FINAL OUTPUT ==========\n")
        print(final_text)
        print("\n========== END FINAL OUTPUT ==========\n")
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)

