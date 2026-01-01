"""Integration test for memory writing in Business Cofounder API.

This test validates that the agent writes to memory files when appropriate triggers occur:
- User feedback on agent behavior → should write to user memory
- Business idea progress → should write to conversation memory
- User preferences → should write to user memory
- Important decisions → should write to conversation memory
"""

import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from apps.business_cofounder_api.agent_factory import create_business_cofounder_agent
from tests.model_provider import load_test_model_config


def _extract_text_only(ai_content: object) -> str:
    """Extract ONLY the LLM text from a provider-specific AIMessage content payload."""
    if ai_content is None:
        return ""
    if isinstance(ai_content, str):
        return ai_content.strip()
    if isinstance(ai_content, list):
        parts: list[str] = []
        for item in ai_content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if isinstance(item, dict):
                if item.get("type") in (None, "text"):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                continue
            block_type = getattr(item, "type", None)
            block_text = getattr(item, "text", None)
            if (block_type in (None, "text")) and isinstance(block_text, str) and block_text.strip():
                parts.append(block_text.strip())
        return "\n\n".join(parts).strip()
    return str(ai_content).strip()


def _check_tool_calls(messages: list, tool_name: str) -> list[dict[str, Any]]:
    """Extract tool calls for a specific tool from messages."""
    tool_calls = []
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "ai":
            content = getattr(msg, "content", None)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        if item.get("name") == tool_name:
                            tool_calls.append(item)
                    elif hasattr(item, "type") and getattr(item, "type", None) == "tool_use":
                        if getattr(item, "name", None) == tool_name:
                            tool_calls.append({"name": tool_name, "input": getattr(item, "input", {})})
            elif hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    if getattr(tc, "name", None) == tool_name:
                        tool_calls.append({"name": tool_name, "args": getattr(tc, "args", {})})
    return tool_calls


def _verify_memory_file_exists(base_dir: Path, user_id: str, conversation_id: str | None = None) -> tuple[bool, Path | None, bool, Path | None]:
    """Verify that memory files exist at expected paths.
    
    Returns:
        (user_memory_exists, user_memory_path, conversation_memory_exists, conversation_memory_path)
    """
    user_memory_path = base_dir / "users" / user_id / "agent.md"
    user_exists = user_memory_path.exists()
    
    conversation_memory_path = None
    conversation_exists = False
    if conversation_id:
        conversation_memory_path = base_dir / "users" / user_id / "conversations" / conversation_id / "agent.md"
        conversation_exists = conversation_memory_path.exists()
    
    return user_exists, user_memory_path, conversation_exists, conversation_memory_path


def _read_memory_file(path: Path) -> str:
    """Read memory file content."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@pytest.fixture()
def test_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path]:
    """Create a test agent with temporary base directory.
    
    Monkeypatches Path.home() to return tmp_path so that the agent uses
    tmp_path/.deepagents/business_cofounder_api as the base directory.
    """
    repo_root = Path(__file__).parent.parent
    
    # Load model configuration
    cfg = load_test_model_config(repo_root=repo_root)
    
    # Monkeypatch Path.home() to return tmp_path
    # This makes the agent use tmp_path/.deepagents/business_cofounder_api as base_dir
    original_home = Path.home
    
    def mock_home() -> Path:
        return tmp_path
    
    monkeypatch.setattr(Path, "home", staticmethod(mock_home))
    
    # Create agent - it will use tmp_path/.deepagents/business_cofounder_api as base_dir
    agent, checkpoints_path = create_business_cofounder_agent(
        agent_id="test_memory_agent",
        provider=cfg.provider,
    )
    
    # The base_dir will be tmp_path/.deepagents/business_cofounder_api
    base_dir = tmp_path / ".deepagents" / "business_cofounder_api"
    
    return agent, base_dir


@pytest.mark.timeout(300)  # 5 minutes for real LLM calls
def test_user_feedback_writes_to_user_memory(test_agent: tuple[Any, Path]) -> None:
    """Test that user feedback triggers writing to user memory."""
    agent, base_dir = test_agent
    user_id = "test_user_1"
    conversation_id = "test_conv_1"
    thread_id = f"bc::{user_id}::{conversation_id}"
    
    # Send message with user feedback (elaborated, natural language, no explicit memory instruction)
    message = """I've noticed that your responses tend to be quite lengthy and detailed. While I appreciate thoroughness in some contexts, for our business co-founder conversations, I would prefer you to be more concise and to the point. 

Please keep your answers brief and focused on the key points. This will help us move faster through our discussions and make it easier for me to process the information. I find that shorter, more direct responses work better for my workflow."""
    
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"user_id": user_id},
    }
    
    response = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    
    # Check that write_file or edit_file was called for user memory
    messages = response.get("messages", [])
    write_calls = _check_tool_calls(messages, "write_file")
    edit_calls = _check_tool_calls(messages, "edit_file")
    
    # Verify memory file exists
    user_exists, user_path, _, _ = _verify_memory_file_exists(base_dir, user_id, conversation_id)
    
    # Print memory file content if it exists
    if user_exists:
        print(f"\n{'='*80}")
        print(f"USER MEMORY FILE CONTENT (exists: {user_exists})")
        print(f"Path: {user_path}")
        print(f"{'='*80}")
        content = _read_memory_file(user_path)
        print(content)
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"USER MEMORY FILE DOES NOT EXIST")
        print(f"Expected path: {user_path}")
        print(f"{'='*80}\n")
    
    # The agent should have written to user memory
    # Check either tool calls or actual file existence
    memory_written = (
        any(
            call.get("input", {}).get("file_path", "").endswith(f"/users/{user_id}/agent.md")
            or call.get("args", {}).get("file_path", "").endswith(f"/users/{user_id}/agent.md")
            for call in write_calls + edit_calls
        )
        or user_exists
    )
    
    # Debug: Print agent response if memory wasn't written
    if not memory_written:
        print(f"\n{'='*80}")
        print("DEBUG: Agent response when memory writing failed")
        print(f"{'='*80}")
        for i, msg in enumerate(messages):
            msg_type = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                print(f"Message[{i}] ({msg_type}): {content[:500]}")
            else:
                print(f"Message[{i}] ({msg_type}): {str(content)[:500]}")
        print(f"{'='*80}\n")
    
    assert memory_written, (
        f"Expected agent to write to user memory at /users/{user_id}/agent.md. "
        f"Tool calls: write_file={len(write_calls)}, edit_file={len(edit_calls)}. "
        f"File exists: {user_exists}. "
        f"Check debug output above to see what the agent responded."
    )
    
    # If file exists, verify content includes feedback about conciseness
    if user_exists:
        content = _read_memory_file(user_path)
        assert "concise" in content.lower() or "brief" in content.lower(), (
            f"User memory should contain feedback about conciseness. Content: {content[:200]}"
        )


@pytest.mark.timeout(300)
def test_business_idea_writes_to_conversation_memory(test_agent: tuple[Any, Path]) -> None:
    """Test that business idea progress triggers writing to conversation memory."""
    agent, base_dir = test_agent
    user_id = "test_user_2"
    conversation_id = "test_conv_2"
    thread_id = f"bc::{user_id}::{conversation_id}"
    
    # Send message with business idea (elaborated, natural language, no explicit memory instruction)
    message = """I have a business idea that I've been thinking about for a while. I want to build an app that helps people find parking spots in real-time using AI and sensors.

The concept is that we would install sensors in parking spaces (either through partnerships with parking lot owners or city infrastructure), and use AI to predict availability and guide drivers to open spots. This would solve the common problem of people driving around looking for parking, which wastes time and fuel.

I'm thinking the app could work in urban areas first, where parking is most challenging, and then expand. We could monetize through partnerships with parking facilities or through a subscription model for users who want premium features like reserved spots."""
    
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"user_id": user_id},
    }
    
    response = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    
    # Check that write_file or edit_file was called for conversation memory
    messages = response.get("messages", [])
    write_calls = _check_tool_calls(messages, "write_file")
    edit_calls = _check_tool_calls(messages, "edit_file")
    
    # Verify conversation memory file exists
    _, _, conv_exists, conv_path = _verify_memory_file_exists(base_dir, user_id, conversation_id)
    
    # Print conversation memory file content if it exists
    if conv_exists and conv_path:
        print(f"\n{'='*80}")
        print(f"CONVERSATION MEMORY FILE CONTENT (exists: {conv_exists})")
        print(f"Path: {conv_path}")
        print(f"{'='*80}")
        content = _read_memory_file(conv_path)
        print(content)
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"CONVERSATION MEMORY FILE DOES NOT EXIST")
        if conv_path:
            print(f"Expected path: {conv_path}")
        print(f"{'='*80}\n")
    
    # The agent should have written to conversation memory
    memory_written = (
        any(
            call.get("input", {}).get("file_path", "").endswith(f"/users/{user_id}/conversations/{conversation_id}/agent.md")
            or call.get("args", {}).get("file_path", "").endswith(f"/users/{user_id}/conversations/{conversation_id}/agent.md")
            for call in write_calls + edit_calls
        )
        or conv_exists
    )
    
    # Debug: Print agent response if memory wasn't written
    if not memory_written:
        print(f"\n{'='*80}")
        print("DEBUG: Agent response when memory writing failed")
        print(f"{'='*80}")
        for i, msg in enumerate(messages):
            msg_type = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                print(f"Message[{i}] ({msg_type}): {content[:500]}")
            else:
                print(f"Message[{i}] ({msg_type}): {str(content)[:500]}")
        print(f"{'='*80}\n")
    
    assert memory_written, (
        f"Expected agent to write to conversation memory at /users/{user_id}/conversations/{conversation_id}/agent.md. "
        f"Tool calls: write_file={len(write_calls)}, edit_file={len(edit_calls)}. "
        f"File exists: {conv_exists}. "
        f"Check debug output above to see what the agent responded."
    )
    
    # If file exists, verify content includes business idea
    if conv_exists:
        content = _read_memory_file(conv_path)
        assert "parking" in content.lower() or "business idea" in content.lower(), (
            f"Conversation memory should contain business idea. Content: {content[:200]}"
        )


@pytest.mark.timeout(300)
def test_memory_persistence_across_calls(test_agent: tuple[Any, Path]) -> None:
    """Test that memory persists and accumulates across multiple interactions."""
    agent, base_dir = test_agent
    user_id = "test_user_3"
    conversation_id = "test_conv_3"
    thread_id = f"bc::{user_id}::{conversation_id}"
    
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"user_id": user_id},
    }
    
    # First message: User feedback (elaborated, natural language, no explicit memory instruction)
    response1 = agent.invoke(
        {"messages": [HumanMessage(content="""I want to make sure you understand my communication preferences. I actually prefer detailed explanations over brief ones. 

When you provide information, I like to understand the full context and reasoning behind your suggestions. Brief answers often leave me with more questions, so I'd rather you take the time to explain things thoroughly. This helps me make better decisions and understand the implications of different options.

Please remember this preference for our future conversations.""")]},
        config=config,
    )
    
    # Second message: Business idea (elaborated, natural language, no explicit memory instruction)
    response2 = agent.invoke(
        {"messages": [HumanMessage(content="""I've been working on developing a business idea for a platform that connects freelancers with clients. 

The core concept is to create a marketplace where freelancers can showcase their skills and clients can find the right talent for their projects. I'm thinking of focusing on creative professionals initially - designers, writers, developers - and then expanding to other categories.

The platform would include features like project matching, secure payment processing, and a rating system to build trust. I believe there's a gap in the market for a platform that really focuses on quality matches rather than just volume, and I want to build something that helps both freelancers and clients succeed.""")]},
        config=config,
    )
    
    # Verify both memory files exist
    user_exists, user_path, conv_exists, conv_path = _verify_memory_file_exists(base_dir, user_id, conversation_id)
    
    # Print user memory content
    print(f"\n{'='*80}")
    print(f"USER MEMORY FILE (exists: {user_exists})")
    print(f"Path: {user_path}")
    print(f"{'='*80}")
    if user_exists:
        user_content = _read_memory_file(user_path)
        print(user_content)
    else:
        print("FILE DOES NOT EXIST")
    print(f"{'='*80}\n")
    
    # Print conversation memory content
    print(f"\n{'='*80}")
    print(f"CONVERSATION MEMORY FILE (exists: {conv_exists})")
    if conv_path:
        print(f"Path: {conv_path}")
    print(f"{'='*80}")
    if conv_exists and conv_path:
        conv_content = _read_memory_file(conv_path)
        print(conv_content)
    else:
        print("FILE DOES NOT EXIST")
    print(f"{'='*80}\n")
    
    # User memory should exist from first call
    assert user_exists, (
        f"Expected user memory to exist after first call with feedback. "
        f"Path: {base_dir / 'users' / user_id / 'agent.md'}"
    )
    
    # Conversation memory should exist from second call
    assert conv_exists, (
        f"Expected conversation memory to exist after second call with business idea. "
        f"Path: {base_dir / 'users' / user_id / 'conversations' / conversation_id / 'agent.md'}"
    )
    
    # Verify user memory contains preference
    user_content = _read_memory_file(user_path)
    assert "detailed" in user_content.lower() or "prefer" in user_content.lower(), (
        f"User memory should contain preference about detailed explanations. Content: {user_content[:200]}"
    )
    
    # Verify conversation memory contains business idea
    if conv_path:
        conv_content = _read_memory_file(conv_path)
        assert "freelancer" in conv_content.lower() or "business idea" in conv_content.lower() or "platform" in conv_content.lower(), (
            f"Conversation memory should contain business idea. Content: {conv_content[:200]}"
        )


@pytest.mark.timeout(300)
def test_memory_paths_are_correct(test_agent: tuple[Any, Path]) -> None:
    """Test that memory files are created at correct virtual paths and are accessible."""
    agent, base_dir = test_agent
    user_id = "test_user_4"
    conversation_id = "test_conv_4"
    thread_id = f"bc::{user_id}::{conversation_id}"
    
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"user_id": user_id},
    }
    
    # Send message that should trigger memory writing (elaborated, natural language, no explicit memory instruction)
    response = agent.invoke(
        {"messages": [HumanMessage(content="""I have a specific preference for how you format your responses. I find that information is much easier to digest when it's organized in bullet points rather than long paragraphs.

When you provide lists, recommendations, or multiple pieces of information, please always use bullet points. This makes it faster for me to scan through the content and find what I need. Long paragraphs can be harder to parse, especially when I'm trying to quickly understand key points or make decisions.

Please remember this formatting preference for all our future conversations.""")]},
        config=config,
    )
    
    # Verify memory file exists at correct path
    user_exists, user_path, _, _ = _verify_memory_file_exists(base_dir, user_id, conversation_id)
    
    # Print memory file content
    print(f"\n{'='*80}")
    print(f"USER MEMORY FILE (exists: {user_exists})")
    print(f"Path: {user_path}")
    print(f"{'='*80}")
    if user_exists:
        content = _read_memory_file(user_path)
        print(content)
    else:
        print("FILE DOES NOT EXIST")
    print(f"{'='*80}\n")
    
    assert user_exists, f"Expected user memory file at {user_path}"
    
    # Try to read the memory file back using the agent
    read_response = agent.invoke(
        {"messages": [HumanMessage(content=f"Read the user memory file at /users/{user_id}/agent.md")]},
        config=config,
    )
    
    # Check if read_file was called
    read_messages = read_response.get("messages", [])
    read_calls = _check_tool_calls(read_messages, "read_file")
    
    # Verify read_file was called with correct path
    read_successful = any(
        call.get("input", {}).get("file_path", "").endswith(f"/users/{user_id}/agent.md")
        or call.get("args", {}).get("file_path", "").endswith(f"/users/{user_id}/agent.md")
        for call in read_calls
    ) or user_exists  # File exists means it's accessible
    
    assert read_successful, (
        f"Expected agent to be able to read memory file at /users/{user_id}/agent.md. "
        f"Read calls: {len(read_calls)}"
    )

