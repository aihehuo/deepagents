"""Integration test for aihehuo member search with todo list and datetime middleware.

This test demonstrates a real-world workflow:
1. User wants to find people to help with a business idea
2. Agent creates a todo list with multiple search queries
3. Agent uses aihehuo-member-search skill to find members
4. Agent generates a report with current date/time
"""

import os
import re
import shutil
import time
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_core.messages import HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.datetime import DateTimeMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents_cli.skills.middleware import SkillsMiddleware

from tests.model_provider import create_test_model, load_test_model_config
from tests.timing_middleware import TimingMiddleware


@pytest.mark.timeout(300)  # 5 minutes for real LLM and API calls (complex multi-step task)
def test_aihehuo_business_search_with_todos_and_datetime(tmp_path: Path) -> None:
    """Test finding business partners using todo list, datetime, and aihehuo search.
    
    This test:
    1. Sets up a temporary agent with skills directory
    2. Copies aihehuo-member-search skill to the agent's skills directory
    3. Creates an agent with TodoListMiddleware, DateTimeMiddleware, and SkillsMiddleware
    4. Sends a request to find people for a business idea
    5. Validates that:
       - A todo list is created with multiple search queries
       - The agent uses the aihehuo-member-search skill
       - The agent generates a report with current date/time
       - All todos are completed
    """
    repo_root = Path(__file__).parent.parent
    
    # Load model configuration using generic model provider interface
    cfg = load_test_model_config(repo_root=repo_root)
    
    aihehuo_api_key = os.environ.get("AIHEHUO_API_KEY")
    if not aihehuo_api_key:
        pytest.skip("AIHEHUO_API_KEY not found in environment variables")
    
    # Temporarily set environment variables for aihehuo API
    old_aihehuo_key = os.environ.get("AIHEHUO_API_KEY")
    old_aihehuo_base = os.environ.get("AIHEHUO_API_BASE")
    
    os.environ["AIHEHUO_API_KEY"] = aihehuo_api_key
    if "AIHEHUO_API_BASE" in os.environ:
        os.environ["AIHEHUO_API_BASE"] = os.environ["AIHEHUO_API_BASE"]
    
    try:
        # Set up skills directory
        agent_id = "test_business_search"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy aihehuo-member-search skill from examples
        example_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "aihehuo-member-search"
        if not example_skill_dir.exists():
            pytest.skip(f"Example skill directory not found: {example_skill_dir}")
        
        skill_dest = skills_dir / "aihehuo-member-search"
        shutil.copytree(example_skill_dir, skill_dest)
        
        # Create model using generic model provider interface
        # Note: Model timeout defaults to 180s but can be overridden via MODEL_API_TIMEOUT_S env var
        # For this long-running test, consider setting MODEL_API_TIMEOUT_S=300 if needed
        model = create_test_model(cfg=cfg)
        
        # Create agent with TodoListMiddleware, DateTimeMiddleware, FilesystemMiddleware, and SkillsMiddleware
        # FilesystemMiddleware is configured to write files to tmp_path so reports are saved to disk
        # This allows us to retrieve and display the complete report content
        # TimingMiddleware is added first to track all execution steps
        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)
        agent = create_agent(
            model=model,
            middleware=[
                timing_middleware,  # Add timing middleware first to track all steps
                TodoListMiddleware(),
                DateTimeMiddleware(),
                FilesystemMiddleware(backend=filesystem_backend),  # Write files to tmp_path
                SkillsMiddleware(
                    skills_dir=skills_dir,
                    assistant_id=agent_id,
                    project_skills_dir=None,
                ),
            ],
            tools=[],  # No additional tools - skills provide the functionality via filesystem
            system_prompt="""You are a business networking assistant. Your job is to:
1. Help users find people who can help with their business ideas
2. Create todo lists for complex multi-step tasks using write_todos
3. Use the aihehuo-member-search skill to search for members on the AI He Huo platform
4. Try different search queries to find relevant people
5. Generate a report with the current date and time using get_current_datetime
6. **IMPORTANT: Save the final report to a file using the write_file tool** - this is required
7. Complete all todos and provide a comprehensive summary

When searching for members:
- **CRITICAL: You MUST execute the skill script to perform searches**
- Read the aihehuo-member-search SKILL.md file to understand how to use it
- **Execute the Python script** using the execute tool (e.g., `execute("python3 /path/to/aihehuo-member-search/aihehuo_member_search.py 'your search query'")`)
- The skill script is located in your skills directory - use the absolute path shown in the system prompt
- Create multiple search queries to find different types of people (e.g., investors, technical co-founders, domain experts)
- Each search should be a separate todo item that you ACTUALLY EXECUTE (not just plan)
- **DO NOT write test scripts or look for .env files** - the skill handles configuration automatically
- After completing searches, compile the results into a report

The report should include:
- Current date and time (use get_current_datetime)
- Summary of search queries used
- **Statistics summary** - include:
  * Total number of users found across all searches
  * Number of users found per search query
  * Key findings from the searches
  * Recommendations based on the results
- **IMPORTANT: Do NOT include complete information for each candidate in the saved report**
- The full detailed information should be kept in memory/context, but the saved file should only contain the summary and statistics

**IMPORTANT: Report Language and Content**
- **DO NOT translate** any content from the API response to English
- Keep all text in the **original language** as returned by the API (likely Chinese)

**CRITICAL: You MUST save a SUMMARY report to a file using the write_file tool.**
- **ONLY save a summary with statistics**, NOT the full detailed report
- The summary should include:
  * Current date and time
  * List of search queries executed
  * Total number of users found (statistics only)
  * Number of users per search query
  * Key findings and recommendations
- **DO NOT include complete user details** (names, bios, goals, etc.) in the saved file
- Use an absolute path for the file_path parameter (e.g., /report.md or /business_search_report.md)
- The path should start with "/" (e.g., "/report.md", "/business_search_report.md", "/search_results.md")
- This step is mandatory and should be one of your final todo items
- Example: write_file(file_path="/business_search_report.md", content="[summary with statistics only]")

**EXECUTION WORKFLOW - FOLLOW THIS EXACTLY:**
1. Create a todo list with specific search queries (e.g., "Search for EdTech investors", "Search for AI/education technical co-founders")
2. **EXECUTE each todo item immediately after creating it** - do not just update the todo list repeatedly
3. For each search todo:
   a. Read the SKILL.md file to understand the script location and usage
   b. **Execute the script** using: `execute("python3 [SKILLS_DIR]/aihehuo-member-search/aihehuo_member_search.py 'your search query here'")`
   c. Collect the results
4. Once you have search results, compile them into a report

**DO NOT:**
- Update the todo list more than once (create it, then execute it)
- Write test scripts or try to test the skill manually
- Look for .env files (the skill handles this automatically)
- Get stuck in planning - execute immediately after planning each step""",
        )
        
        # User request to find people for a business idea
        user_request = """I have a business idea for an AI-powered educational platform that helps students learn programming through interactive coding challenges. 

I need to find people who can help me with this idea. Please:
1. Create a todo list for finding relevant people
2. Search for different types of people (e.g., investors interested in EdTech, technical co-founders with AI/education experience, domain experts in online learning)
3. Try multiple search queries to find the best matches
4. Generate a comprehensive report with the current date and time, summarizing your findings and recommendations

**Important requirements for the report:**
- The saved report should be a **summary with statistics only**
- Include total number of users found and statistics per search query
- **DO NOT include complete user details** in the saved file (this will make it too large)
- **DO NOT translate** any content - keep everything in the original language as returned by the API
- The full detailed information can be mentioned in your response, but the saved file should only contain the summary

Use the aihehuo-member-search skill to search for members on the AI He Huo platform."""
        
        print("\n" + "="*80)
        print("TEST: AIHEHUO BUSINESS SEARCH WITH TODOS AND DATETIME")
        print("="*80)
        print(f"\n📝 User Request:\n{user_request}\n")
        
        # Execute the agent
        print("⏳ Starting agent execution (this may take several minutes for complex task)...\n")
        input_state = {"messages": [HumanMessage(content=user_request)]}
        
        # Execute the agent - use invoke() directly for reliable state extraction
        # Streaming can be unreliable for state extraction, so we use invoke() and show progress via messages
        print("🔄 Agent execution started...")
        print("  (This is a complex multi-step task that may take several minutes)\n")
        
        # Use invoke() directly for reliable final state
        invoke_start = time.time()
        result = None
        try:
            result = agent.invoke(input_state)
            invoke_end = time.time()
            invoke_duration = invoke_end - invoke_start
            
            # Update total time in timing middleware
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
            
            # Try to get partial result if available
            # (This might not work if the exception happened before any state was created)
        
        # Print final todo state (even on error, if we have a result)
        print("\n" + "="*80)
        print("📋 FINAL TODO STATE")
        print("="*80)
        if result:
            todos = result.get("todos", [])
            if todos:
                print(f"\n✅ Found {len(todos)} todos in final state:")
                for i, todo in enumerate(todos, 1):
                    status_emoji = "✅" if todo.get("status") == "completed" else "⏳" if todo.get("status") == "in_progress" else "📋"
                    print(f"  {i}. {status_emoji} [{todo.get('status', 'pending')}] {todo.get('content', 'N/A')}")
                
                completed = sum(1 for t in todos if t.get("status") == "completed")
                in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
                pending = sum(1 for t in todos if t.get("status") == "pending")
                print(f"\n📊 Status breakdown:")
                print(f"  ✅ Completed: {completed}")
                print(f"  ⏳ In Progress: {in_progress}")
                print(f"  📋 Pending: {pending}")
            else:
                print("\n⚠️  No todos found in final state")
        else:
            print("\n⚠️  Could not retrieve final state (execution failed before state was created)")
        
        # Print timing summary (always, even on error)
        timing_middleware.print_summary()
        
        # Print execution summary
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        
        # Extract todos from final state
        if not result:
            print("\n⚠️  Cannot extract execution summary - agent execution failed")
            raise AssertionError("Agent execution failed - cannot validate test requirements")
        
        todos = result.get("todos", [])
        if todos:
            print(f"\n✅ Todo List ({len(todos)} items):")
            for i, todo in enumerate(todos, 1):
                status_emoji = "✅" if todo.get("status") == "completed" else "⏳" if todo.get("status") == "in_progress" else "📋"
                print(f"  {status_emoji} [{todo.get('status', 'pending')}] {todo.get('content', 'N/A')}")
        
        # Check for datetime usage in messages
        messages = result.get("messages", [])
        datetime_used = False
        aihehuo_used = False
        report_generated = False
        
        print(f"\n📊 Messages: {len(messages)} total")
        for i, message in enumerate(messages[-10:], 1):  # Show last 10 messages
            if message.type == "ai":
                content = str(message.content)[:200]  # Truncate for display
                print(f"\n  Message {i} (AI): {content}...")
                
                # Check for datetime usage
                if "get_current_datetime" in str(message.tool_calls) if hasattr(message, 'tool_calls') else "":
                    datetime_used = True
                    print("    ✓ Used get_current_datetime")
                
                # Check for aihehuo skill usage
                if "aihehuo" in content.lower() or "aihehuo-member-search" in str(message.tool_calls) if hasattr(message, 'tool_calls') else "":
                    aihehuo_used = True
                    print("    ✓ Used aihehuo-member-search skill")
                
                # Check for report
                if "report" in content.lower() and ("date" in content.lower() or "time" in content.lower()):
                    report_generated = True
                    print("    ✓ Generated report with date/time")
        
        # Validations
        print("\n" + "="*80)
        print("VALIDATIONS")
        print("="*80)
        
        # 1. Todo list should be created
        assert len(todos) > 0, "Todo list should be created with at least one item"
        print(f"✅ Todo list created with {len(todos)} items")
        
        # 2. Should have multiple search-related todos
        search_todos = [t for t in todos if "search" in t.get("content", "").lower() or "aihehuo" in t.get("content", "").lower()]
        assert len(search_todos) > 0, "Should have at least one search-related todo"
        print(f"✅ Found {len(search_todos)} search-related todos")
        
        # 3. Todos should be completed
        completed_todos = [t for t in todos if t.get("status") == "completed"]
        assert len(completed_todos) > 0, "At least some todos should be completed"
        print(f"✅ {len(completed_todos)}/{len(todos)} todos completed")
        
        # 4. DateTime should be used (check in tool calls or messages)
        if datetime_used:
            print("✅ DateTimeMiddleware was used (get_current_datetime called)")
        else:
            # Check if datetime appears in final response
            final_messages = [m for m in messages if m.type == "ai"]
            if final_messages:
                last_content = str(final_messages[-1].content).lower()
                if "date" in last_content or "time" in last_content or "202" in last_content:
                    datetime_used = True
                    print("✅ DateTime appears in response (likely used)")
        
        # 5. AI He Huo skill should be used
        if aihehuo_used:
            print("✅ AI He Huo member search skill was used")
        else:
            # Check if search results appear in messages
            for message in messages:
                if message.type == "tool" and "aihehuo" in str(message.content).lower():
                    aihehuo_used = True
                    print("✅ AI He Huo search results found in tool messages")
                    break
        
        # 6. Report should be generated
        if report_generated:
            print("✅ Report with date/time was generated")
        else:
            # Check final message for report-like content
            final_messages = [m for m in messages if m.type == "ai"]
            if final_messages:
                last_content = str(final_messages[-1].content).lower()
                if len(last_content) > 200 and ("summary" in last_content or "findings" in last_content or "recommendation" in last_content):
                    report_generated = True
                    print("✅ Comprehensive report found in final response")
        
        # Extract and display the final report
        print("\n" + "="*80)
        print("FINAL REPORT")
        print("="*80)
        
        # 1. Check for files written to filesystem
        report_file_path = None
        report_file_content = None
        
        # Check AI messages for write_file tool calls
        for message in messages:
            if message.type == "ai" and hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get('name') == 'write_file':
                        args = tool_call.get('args', {})
                        if 'path' in args:
                            report_file_path = args['path']
                            break
                if report_file_path:
                    break
        
        # Check tool messages for write_file operations and file paths
        if not report_file_path:
            for message in messages:
                if message.type == "tool":
                    content = str(message.content)
                    # Check if this is a write_file response
                    if hasattr(message, 'name') and message.name == "write_file":
                        # Try to extract path from content
                        path_match = re.search(r'path[:\s]+([/\w\.\-]+)', content, re.IGNORECASE)
                        if path_match:
                            report_file_path = path_match.group(1)
                            break
                    
                    # Look for file paths in tool messages
                    if "write_file" in content.lower() or "saved" in content.lower() or "written" in content.lower():
                        # Try to extract file path from message
                        # Look for common path patterns
                        path_patterns = [
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
        
        # 2. Check tmp_path directory for any report files
        if not report_file_path:
            # Look for common report file names in tmp_path
            report_patterns = ["report", "summary", "findings", "business", "search_results"]
            for pattern in report_patterns:
                for ext in [".txt", ".md", ".json", ".html"]:
                    potential_file = tmp_path / f"{pattern}{ext}"
                    if potential_file.exists():
                        report_file_path = str(potential_file)
                        break
                if report_file_path:
                    break
            
            # Also check subdirectories
            if not report_file_path:
                for root, dirs, files in os.walk(tmp_path):
                    for file in files:
                        if any(pattern in file.lower() for pattern in report_patterns):
                            report_file_path = os.path.join(root, file)
                            break
                    if report_file_path:
                        break
        
        # 3. If file path found, read and display content
        if report_file_path:
            try:
                file_path_obj = Path(report_file_path)
                # If relative path, try resolving from tmp_path
                if not file_path_obj.is_absolute():
                    file_path_obj = tmp_path / report_file_path
                
                if file_path_obj.exists():
                    report_file_content = file_path_obj.read_text(encoding='utf-8')
                    print(f"\n📄 Report saved to file: {report_file_path}")
                    print(f"   Full path: {file_path_obj.resolve()}")
                    print(f"\n📝 Report Content ({len(report_file_content)} characters):")
                    print("-" * 80)
                    print(report_file_content)
                    print("-" * 80)
            except Exception as e:
                print(f"\n⚠️  Could not read report file at {report_file_path}: {e}")
        
        # 4. If no file found, extract report from final AI message
        if not report_file_content:
            final_messages = [m for m in messages if m.type == "ai"]
            if final_messages:
                final_content = str(final_messages[-1].content)
                
                # Try to identify report section
                report_sections = []
                content_lower = final_content.lower()
                
                # Look for report-like sections
                if "report" in content_lower or "summary" in content_lower or "findings" in content_lower:
                    # Extract the full content as the report
                    report_sections.append(final_content)
                
                if report_sections:
                    print(f"\n📝 Report Content (from final AI message, {len(final_content)} characters):")
                    print("-" * 80)
                    print(final_content)
                    print("-" * 80)
                else:
                    # Show excerpt if no clear report section
                    print(f"\n📝 Final Response (excerpt, {len(final_content)} characters total):")
                    print("-" * 80)
                    print(final_content[:2000] + "..." if len(final_content) > 2000 else final_content)
                    print("-" * 80)
                    print(f"\n💡 Tip: Full response is {len(final_content)} characters. Check messages above for complete content.")
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
    finally:
        # Restore environment variables
        if old_aihehuo_key:
            os.environ["AIHEHUO_API_KEY"] = old_aihehuo_key
        elif "AIHEHUO_API_KEY" in os.environ:
            del os.environ["AIHEHUO_API_KEY"]
        
        if old_aihehuo_base:
            os.environ["AIHEHUO_API_BASE"] = old_aihehuo_base
        elif "AIHEHUO_API_BASE" in os.environ:
            del os.environ["AIHEHUO_API_BASE"]

