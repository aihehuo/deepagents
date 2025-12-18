"""Integration test for AI He Huo search middleware.

This test verifies that the AihehuoMiddleware provides working search tools
for finding members and ideas on the AI He Huo platform.
"""

import os
import re
import time
from pathlib import Path
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_core.messages import HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.aihehuo import AihehuoMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware

from tests.model_provider import create_test_model, load_test_model_config
from tests.test_backends import DirectoryOnlyBackend
from tests.timing_middleware import TimingMiddleware


@pytest.mark.timeout(300)  # 5 minutes for real LLM and API calls
def test_aihehuo_search_with_concrete_idea(tmp_path: Path) -> None:
    """Test AI He Huo search functionality with a concrete business idea.
    
    This test:
    1. Sets up an agent with AihehuoMiddleware
    2. Provides a concrete business idea
    3. Validates that the agent can:
       - Use aihehuo_search_members to find co-founders and investors
       - Use aihehuo_search_ideas to find related ideas
       - Create multiple targeted searches for different roles
       - Generate a summary of findings
       - Write the final report to a file using write_file tool
    """
    repo_root = Path(__file__).parent.parent
    
    # Check for API key
    aihehuo_api_key = os.environ.get("AIHEHUO_API_KEY")
    if not aihehuo_api_key:
        pytest.skip("AIHEHUO_API_KEY not found in environment variables")
    
    # Load model configuration
    cfg = load_test_model_config(repo_root=repo_root)
    
    # Create model
    model = create_test_model(cfg=cfg)
    
    # Create agent with AihehuoMiddleware
    # 
    # Virtual Filesystem Setup:
    # =========================
    # The agent sees a virtual filesystem where:
    # - Virtual root: "/" (agent can use paths like "/file.md", "/report.md")
    # - Base backend root: tmp_path (pytest temporary directory, e.g., /tmp/pytest-xxx/test_xxx)
    #   - This is where the base FilesystemBackend would normally write files
    #   - virtual_mode=False (default), so paths starting with "/" are treated as absolute paths
    # - Actual write location: test_file_dir (same directory as this test file)
    #   - DirectoryOnlyBackend intercepts all writes/edits and maps them to test_file_dir
    #   - Example: Agent writes to "/report.md" → Actually written to test_file_dir/report.md
    #
    # Path Resolution Flow:
    # 1. Agent calls write_file("/report.md", content)
    # 2. DirectoryOnlyBackend.write() intercepts and calls _map_write_path("/report.md")
    # 3. Extracts filename "report.md" and maps to test_file_dir/report.md
    # 4. Calls base_backend.write(test_file_dir/report.md, content)
    # 5. Base backend resolves: tmp_path is root_dir, but path is absolute, so uses as-is
    # 6. File is written to: /Users/yc/workspace/deepagents/tests/report.md
    #
    # Note: Reads/list operations are NOT restricted - they use the base backend with tmp_path as root
    test_file_dir = Path(__file__).parent
    print(f"TEST VIRTUAL FILESYSTEM ROOT: {tmp_path}")
    base_backend = FilesystemBackend(root_dir=str(tmp_path))
    restricted_backend = DirectoryOnlyBackend(
        backend=base_backend,
        test_dir=test_file_dir,
    )
    timing_middleware = TimingMiddleware(verbose=True)
    agent = create_agent(
        model=model,
        middleware=[
            timing_middleware,
            TodoListMiddleware(),
            FilesystemMiddleware(backend=restricted_backend),
            AihehuoMiddleware(),
        ],
        tools=[],
        system_prompt="""You are a business networking assistant. Your job is to help entrepreneurs find co-founders, investors, and partners on the AI He Huo platform.

When given a business idea:
1. Analyze the idea and identify what roles/people are needed
2. Create multiple targeted searches using aihehuo_search_members for different roles:
   - Technical co-founders (if technical expertise is needed)
   - Business co-founders (if business/market expertise is needed)
   - Investors (if funding is mentioned or relevant)
   - Domain experts (if specific industry knowledge is needed)
3. Use aihehuo_search_ideas to find related business ideas
4. Generate a comprehensive summary of your findings
5. **IMPORTANT: Save the final report to a file using the write_file tool** - this is required

Important:
- Use natural language queries (full sentences, not just keywords)
- Create separate searches for different roles/needs
- Each search should be specific and targeted
- Query must be longer than 5 characters for member searches
- Use the investor parameter when searching for investors
- Summarize the results and provide recommendations
- **You MUST save the final report to a file** using write_file with an absolute path (e.g., /search_report.md or /recommendations.md)""",
    )
    
    # Concrete business idea
    user_request = """I have a business idea for an AI-powered language learning platform for children aged 5-12.

The platform will use AI to personalize learning paths, gamify the experience, and help children learn languages through interactive stories and games. We're targeting parents who want their children to learn English, Spanish, or Mandarin.

I need to find:
1. A technical co-founder with AI/ML and mobile app development experience
2. An educational content expert with experience in children's language learning
3. Investors interested in EdTech and AI-driven educational products
4. Any similar projects or ideas I can learn from

Please search the AI He Huo platform and help me find potential partners and investors.

**Important: After completing your searches, please write your final report, search results, and recommendations to a file using the write_file tool. The report should include a summary of your findings and recommendations.**"""

    print("\n" + "="*80)
    print("TEST: AI HE HUO SEARCH WITH CONCRETE IDEA")
    print("="*80)
    print(f"\n📝 Business Idea:\n{user_request}\n")
    
    # Execute the agent
    print("⏳ Starting agent execution...\n")
    input_state = {"messages": [HumanMessage(content=user_request)]}
    
    invoke_start = time.time()
    result = None
    try:
        result = agent.invoke(input_state)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        
        timing_middleware.total_time = invoke_duration
        
        print(f"\n  ✅ Agent execution completed")
        print(f"  📊 Final state has {len(result.get('messages', []))} messages")
        print(f"  ⏱️  Total invoke() time: {invoke_duration:.2f}s ({invoke_duration*1000:.2f}ms)")
    except Exception as e:
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
        
        print(f"\n  ❌ Agent execution failed or timed out")
        print(f"  ⏱️  Total invoke() time before failure: {invoke_duration:.2f}s ({invoke_duration*1000:.2f}ms)")
        print(f"  Error: {type(e).__name__}: {str(e)[:200]}")
        raise
    
    # Validate results
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    if not result:
        raise AssertionError("Agent execution failed - result is None")
    
    messages = result.get("messages", [])
    print(f"\n📊 Messages: {len(messages)} total")
    
    # Check for tool usage
    search_members_used = False
    search_ideas_used = False
    tool_calls_count = 0
    
    for message in messages:
        if message.type == "ai" and hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls_count += 1
                tool_name = tool_call.get('name', '')
                if tool_name == "aihehuo_search_members":
                    search_members_used = True
                    print(f"  ✓ Found aihehuo_search_members tool call")
                    # Print query if available
                    args = tool_call.get('args', {})
                    if 'query' in args:
                        print(f"    Query: {args['query'][:100]}...")
                elif tool_name == "aihehuo_search_ideas":
                    search_ideas_used = True
                    print(f"  ✓ Found aihehuo_search_ideas tool call")
                    # Print query if available
                    args = tool_call.get('args', {})
                    if 'query' in args:
                        print(f"    Query: {args['query'][:100]}...")
        
        if message.type == "tool":
            content = str(message.content)
            if "aihehuo_search_members" in content.lower() or "aihehuo" in content.lower():
                search_members_used = True
            if "aihehuo_search_ideas" in content.lower():
                search_ideas_used = True
    
    # Check for multiple searches (should have multiple searches for different roles)
    member_search_count = 0
    idea_search_count = 0
    
    for message in messages:
        if message.type == "ai" and hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name', '')
                if tool_name == "aihehuo_search_members":
                    member_search_count += 1
                elif tool_name == "aihehuo_search_ideas":
                    idea_search_count += 1
    
    print(f"\n📈 Search Statistics:")
    print(f"  - Total tool calls: {tool_calls_count}")
    print(f"  - Member searches: {member_search_count}")
    print(f"  - Idea searches: {idea_search_count}")
    
    # Validations
    print("\n" + "="*80)
    print("ASSERTIONS")
    print("="*80)
    
    # 1. Should have used search_members tool
    assert search_members_used, "Agent should have used aihehuo_search_members tool"
    print("✅ aihehuo_search_members tool was used")
    
    # 2. Should have multiple member searches (for different roles)
    assert member_search_count >= 2, f"Should have at least 2 member searches for different roles, got {member_search_count}"
    print(f"✅ Multiple member searches created ({member_search_count} searches)")
    
    # 3. Should have used search_ideas tool
    assert search_ideas_used, "Agent should have used aihehuo_search_ideas tool"
    print("✅ aihehuo_search_ideas tool was used")
    
    # 4. Should have generated a summary or response
    final_messages = [m for m in messages if m.type == "ai"]
    assert len(final_messages) > 0, "Should have at least one AI response"
    
    last_message = final_messages[-1]
    last_content = str(last_message.content)
    assert len(last_content) > 100, "Final response should contain a summary"
    print(f"✅ Generated summary ({len(last_content)} characters)")
    
    # 5. Check for investor search (should have investor=True in at least one search)
    investor_search_found = False
    for message in messages:
        if message.type == "ai" and hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get('name') == "aihehuo_search_members":
                    args = tool_call.get('args', {})
                    if args.get('investor') is True:
                        investor_search_found = True
                        break
    
    if investor_search_found:
        print("✅ Investor search found (with investor=True)")
    else:
        print("⚠️  No explicit investor search found (but this is optional)")
    
    # 6. Check for file writing
    print("\n" + "="*80)
    print("FILE WRITING VALIDATION")
    print("="*80)
    
    report_file_path = None
    report_file_content = None
    
    # Check AI messages for write_file tool calls
    for message in messages:
        if message.type == "ai" and hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get('name') == 'write_file':
                    args = tool_call.get('args', {})
                    if 'file_path' in args:
                        report_file_path = args['file_path']
                    elif 'path' in args:
                        report_file_path = args['path']
                    break
            if report_file_path:
                break
    
    # Check tool messages for write_file operations
    if not report_file_path:
        for message in messages:
            if message.type == "tool":
                content = str(message.content)
                if hasattr(message, 'name') and message.name == "write_file":
                    # Try to extract actual path from "Updated file /path/to/file" format
                    updated_match = re.search(r'Updated file\s+([/\w\.\-]+)', content, re.IGNORECASE)
                    if updated_match:
                        report_file_path = updated_match.group(1)
                        break
                    # Try to extract path from content
                    path_match = re.search(r'path[:\s]+([/\w\.\-]+)', content, re.IGNORECASE)
                    if path_match:
                        report_file_path = path_match.group(1)
                        break
                
                # Look for file paths in tool messages
                if "write_file" in content.lower() or "saved" in content.lower() or "written" in content.lower() or "updated file" in content.lower():
                    path_patterns = [
                        r'Updated file\s+([/\w\.\-]+)',  # "Updated file /path/to/file"
                        r'path[:\s]+([/\w\.\-]+\.(?:txt|md|json|csv|html))',
                        r'saved to[:\s]+([/\w\.\-]+)',
                        r'written to[:\s]+([/\w\.\-]+)',
                        r'file[:\s]+([/\w\.\-]+\.(?:txt|md|json|csv|html))',
                        r'at[:\s]+([/\w\.\-]+\.(?:txt|md|json|csv|html))',
                    ]
                    for pattern in path_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            report_file_path = matches[0]
                            break
                    if report_file_path:
                        break
    
    # Check test file directory for report files (where writes are restricted to)
    test_file_dir = Path(__file__).parent
    if not report_file_path:
        report_patterns = ["report", "summary", "findings", "recommendations", "search_results", "partners", "ai_language_learning"]
        for pattern in report_patterns:
            for ext in [".txt", ".md", ".json", ".html"]:
                potential_file = test_file_dir / f"{pattern}{ext}"
                if potential_file.exists():
                    report_file_path = str(potential_file)
                    break
            if report_file_path:
                break
    elif report_file_path and Path(report_file_path).name:
        # If we have a virtual path, also check if the file exists in test directory with that name
        virtual_filename = Path(report_file_path).name
        potential_file = test_file_dir / virtual_filename
        if potential_file.exists() and not Path(report_file_path).exists():
            # Use the mapped path instead
            report_file_path = str(potential_file)
    
    # Read and display file if found
    if report_file_path:
        try:
            file_path_obj = Path(report_file_path)
            test_file_dir = Path(__file__).parent
            
            # If the path is a virtual path (starts with / but not a real absolute path),
            # map it to the test directory using the same logic as DirectoryOnlyBackend
            if file_path_obj.is_absolute() and not file_path_obj.exists():
                # This might be a virtual path - try mapping it to test directory
                filename = file_path_obj.name
                mapped_path = test_file_dir / filename
                if mapped_path.exists():
                    file_path_obj = mapped_path
            elif not file_path_obj.is_absolute():
                # Relative path - resolve from test file directory
                file_path_obj = test_file_dir / report_file_path
            
            if file_path_obj.exists():
                report_file_content = file_path_obj.read_text(encoding='utf-8')
                print(f"\n📄 Report file found: {report_file_path}")
                print(f"   Full path: {file_path_obj.resolve()}")
                print(f"   File size: {len(report_file_content)} characters")
                print(f"\n📝 Report Content:")
                print("-" * 80)
                print(report_file_content)
                print("-" * 80)
                
                # Validate file content
                assert len(report_file_content) > 100, "Report file should contain substantial content"
                print("✅ Report file contains substantial content")
            else:
                print(f"⚠️  Report file path found but file does not exist: {file_path_obj}")
        except Exception as e:
            print(f"⚠️  Could not read report file at {report_file_path}: {e}")
    else:
        print("⚠️  No report file found")
        print("   Checked for write_file tool calls and common file names in tmp_path")
    
    # Assert that file was written
    assert report_file_path is not None, "Agent should have written a report file using write_file tool"
    assert report_file_content is not None, "Report file should exist and be readable"
    print("✅ Report file was successfully written")
    
    # Print final response excerpt
    print("\n" + "="*80)
    print("FINAL RESPONSE EXCERPT")
    print("="*80)
    print(f"\n{last_content[:1000]}...")
    if len(last_content) > 1000:
        print(f"\n[Response truncated - full length: {len(last_content)} characters]")
    
    # Print timing summary
    timing_middleware.print_summary()
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)

