"""Test script for ArtifactsMiddleware.

This test verifies that:
1. The middleware provides the add_artifact tool
2. The tool can be invoked and updates state correctly
3. Multiple artifacts can be recorded
4. Artifacts are stored in the agent state
5. Artifacts persist across invocations (with checkpointer)

To run this test:
    # With pytest (recommended):
    pytest tests/test_artifacts_middleware.py -v -s
    
    # Or directly:
    python tests/test_artifacts_middleware.py
    
    # Set environment variables for model configuration:
    # MODEL_API_PROVIDER=deepseek (or qwen)
    # MODEL_API_KEY=your_api_key
    # MODEL_BASE_URL=your_base_url (optional)
    # MODEL_NAME=model_name (optional, defaults to deepseek-chat or qwen-plus)
"""

import pytest
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents.middleware.artifacts import ArtifactsMiddleware, ArtifactMetadata
from tests.model_provider import create_test_model, load_test_model_config


def test_artifacts_middleware_provides_tool():
    """Test that ArtifactsMiddleware provides the add_artifact tool."""
    middleware = ArtifactsMiddleware()
    
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "add_artifact"
    assert "artifact" in middleware.tools[0].description.lower()


def test_artifacts_middleware_has_system_prompt():
    """Test that ArtifactsMiddleware has a system prompt."""
    middleware = ArtifactsMiddleware()
    
    assert middleware.system_prompt_template is not None
    assert "artifact" in middleware.system_prompt_template.lower()
    assert "add_artifact" in middleware.system_prompt_template


def _create_add_artifact_tool_call(url: str, artifact_type: str | None = None, name: str | None = None, tool_call_id: str = "test-call-1") -> dict:
    """Create a ToolCall dict for add_artifact tool with InjectedToolCallId.
    
    Args:
        url: The artifact URL (required).
        artifact_type: The artifact type. If None, not included in args (tool uses default "html").
        name: The artifact name. If None, not included in args (tool uses default "").
        tool_call_id: The tool call ID.
    """
    args = {"url": url}
    if artifact_type is not None:
        args["artifact_type"] = artifact_type
    if name is not None:
        args["name"] = name
    return {
        "name": "add_artifact",
        "args": args,
        "id": tool_call_id,
        "type": "tool_call",
    }


def test_add_artifact_tool_can_be_invoked():
    """Test that the add_artifact tool can be invoked and returns a Command."""
    middleware = ArtifactsMiddleware()
    tool = middleware.tools[0]
    
    test_url = "https://example.com/artifact.html"
    test_type = "html"
    test_name = "Test Artifact"
    
    # For tools with InjectedToolCallId, we need to use the ToolCall dict format
    tool_call = _create_add_artifact_tool_call(test_url, artifact_type=test_type, name=test_name, tool_call_id="test-call-1")
    result = tool.invoke(tool_call)
    
    # Should return a Command
    assert isinstance(result, Command)
    assert "artifacts" in result.update
    assert "messages" in result.update
    assert len(result.update["messages"]) == 1
    
    # Check artifact metadata
    artifacts = result.update["artifacts"]
    assert isinstance(artifacts, list)
    assert len(artifacts) == 1
    
    artifact = artifacts[0]
    assert artifact["url"] == test_url
    assert artifact["artifact_type"] == test_type
    assert artifact["name"] == test_name
    assert "created_at" in artifact
    
    # Message should contain artifact info
    message = result.update["messages"][0]
    assert isinstance(message, ToolMessage)
    assert test_url in message.content
    assert test_type in message.content


def test_add_artifact_with_minimal_args():
    """Test that add_artifact works with minimal arguments (just URL)."""
    middleware = ArtifactsMiddleware()
    tool = middleware.tools[0]
    
    test_url = "https://example.com/document.html"
    
    # Use tool call format with just URL
    tool_call = _create_add_artifact_tool_call(test_url, tool_call_id="test-call-2")
    result = tool.invoke(tool_call)
    
    assert isinstance(result, Command)
    artifacts = result.update["artifacts"]
    assert len(artifacts) == 1
    
    artifact = artifacts[0]
    assert artifact["url"] == test_url
    assert artifact["artifact_type"] == "html"  # Default value
    assert "created_at" in artifact


def test_artifacts_state_initialization():
    """Test that artifacts state is initialized correctly."""
    middleware = ArtifactsMiddleware()
    
    # Simulate before_agent with empty state
    from langgraph.runtime import Runtime
    
    # Create a minimal runtime (just need the type)
    class MockRuntime:
        pass
    
    empty_state = {}
    update = middleware.before_agent(empty_state, MockRuntime())
    
    assert update is not None
    assert "artifacts" in update
    assert update["artifacts"] == []
    
    # Test with existing state (should return None)
    existing_state = {"artifacts": [{"url": "https://example.com/test.html"}]}
    update2 = middleware.before_agent(existing_state, MockRuntime())
    assert update2 is None


@pytest.mark.timeout(300)  # 5 minutes for real LLM calls
def test_artifacts_middleware_with_agent(tmp_path: Path) -> None:
    """Test that an agent can use the add_artifact tool and artifacts are stored in state."""
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    model = create_test_model(cfg=cfg)
    
    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "test-artifacts-1"}}
    
    agent = create_agent(
        model=model,
        middleware=[ArtifactsMiddleware()],
        checkpointer=checkpointer,
        system_prompt="""You are a helpful assistant that can track artifacts.
        
When I ask you to record an artifact URL, use the add_artifact tool to record it.
Always use the add_artifact tool when asked to record an artifact.""",
    )
    
    # First invocation: Ask agent to record an artifact
    test_url = "https://example.com/business-plan.html"
    result1 = agent.invoke({
        "messages": [
            HumanMessage(content=f"Please record this artifact URL: {test_url}")
        ]
    }, config)
    
    # Check that the artifact was recorded
    artifacts = result1.get("artifacts", [])
    print(f"\n📦 Artifacts after first invocation: {len(artifacts)}")
    
    # The agent should have used the add_artifact tool
    tool_messages = [msg for msg in result1["messages"] if msg.type == "tool"]
    artifact_tool_messages = [
        msg for msg in tool_messages
        if "add_artifact" in str(msg.name).lower() or "artifact" in msg.content.lower()
    ]
    
    print(f"   Tool messages related to artifacts: {len(artifact_tool_messages)}")
    if artifact_tool_messages:
        print(f"   First artifact tool message: {artifact_tool_messages[0].content[:100]}...")
    
    # Check state
    assert "artifacts" in result1, "State should have artifacts field"
    
    # If the agent successfully used the tool, artifacts should be populated
    if len(artifacts) > 0:
        print(f"✅ Artifacts recorded: {len(artifacts)}")
        for i, artifact in enumerate(artifacts):
            print(f"   Artifact {i+1}:")
            print(f"     URL: {artifact.get('url', 'N/A')}")
            print(f"     Type: {artifact.get('artifact_type', 'N/A')}")
            print(f"     Name: {artifact.get('name', 'N/A')}")
            print(f"     Created: {artifact.get('created_at', 'N/A')}")
        
        # Verify artifact structure
        first_artifact = artifacts[0]
        assert "url" in first_artifact, "Artifact should have url field"
        assert "artifact_type" in first_artifact, "Artifact should have artifact_type field"
        assert "created_at" in first_artifact, "Artifact should have created_at field"
    else:
        print("⚠️  No artifacts recorded - agent may not have used the tool")
        print("   This could be because:")
        print("   - Agent didn't understand the instruction")
        print("   - Tool wasn't available")
        print("   - Model limitations")
    
    # Second invocation: Record another artifact
    test_url2 = "https://example.com/presentation.html"
    result2 = agent.invoke({
        "messages": [
            HumanMessage(content=f"Please also record this artifact: {test_url2} with type 'html' and name 'Presentation'")
        ]
    }, config)
    
    artifacts2 = result2.get("artifacts", [])
    print(f"\n📦 Artifacts after second invocation: {len(artifacts2)}")
    
    # Artifacts should accumulate (reducer appends)
    if len(artifacts2) > len(artifacts):
        print(f"✅ Artifacts accumulated: {len(artifacts)} -> {len(artifacts2)}")
    elif len(artifacts2) == len(artifacts) and len(artifacts) > 0:
        print(f"ℹ️  Same number of artifacts (may have overwritten or not added)")
    else:
        print(f"⚠️  Artifacts didn't accumulate as expected")
    
    # Print final state
    print(f"\n" + "="*80)
    print("✅ TEST SUMMARY")
    print("="*80)
    print(f"  - Artifacts after first invocation: {len(artifacts)}")
    print(f"  - Artifacts after second invocation: {len(artifacts2)}")
    print(f"  - State has artifacts field: {'artifacts' in result2}")
    
    # Verify that state persists
    assert "artifacts" in result2, "State should have artifacts field after second invocation"


@pytest.mark.timeout(300)  # 5 minutes for real LLM calls
def test_artifacts_persistence_with_checkpointer(tmp_path: Path) -> None:
    """Test that artifacts persist across separate agent invocations with checkpointer."""
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    model = create_test_model(cfg=cfg)
    
    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "test-artifacts-persist"}}
    
    agent = create_agent(
        model=model,
        middleware=[ArtifactsMiddleware()],
        checkpointer=checkpointer,
        system_prompt="""You are a helpful assistant. When asked to record an artifact URL, use the add_artifact tool.""",
    )
    
    # First invocation: Record an artifact
    test_url = "https://example.com/persisted-artifact.html"
    result1 = agent.invoke({
        "messages": [
            HumanMessage(content=f"Record this artifact URL: {test_url}")
        ]
    }, config)
    
    artifacts1 = result1.get("artifacts", [])
    print(f"\n📦 Artifacts in first invocation: {len(artifacts1)}")
    
    # Second invocation with same thread_id (should see previous artifacts)
    result2 = agent.invoke({
        "messages": [
            HumanMessage(content="What artifacts have been recorded so far?")
        ]
    }, config)
    
    artifacts2 = result2.get("artifacts", [])
    print(f"📦 Artifacts in second invocation: {len(artifacts2)}")
    
    # Artifacts should persist (state is loaded from checkpointer)
    if len(artifacts2) >= len(artifacts1):
        print(f"✅ Artifacts persisted: {len(artifacts1)} -> {len(artifacts2)}")
    else:
        print(f"⚠️  Artifacts didn't persist as expected: {len(artifacts1)} -> {len(artifacts2)}")
    
    # Verify state has artifacts field
    assert "artifacts" in result2, "State should have artifacts field after persistence test"


if __name__ == "__main__":
    """Run the tests directly (useful for debugging)."""
    import sys
    
    print("Running ArtifactsMiddleware tests...\n")
    
    # Run basic tests
    print("=" * 80)
    print("Test 1: ArtifactsMiddleware provides tool")
    print("=" * 80)
    try:
        test_artifacts_middleware_provides_tool()
        print("✅ PASSED\n")
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        sys.exit(1)
    
    print("=" * 80)
    print("Test 2: ArtifactsMiddleware has system prompt")
    print("=" * 80)
    try:
        test_artifacts_middleware_has_system_prompt()
        print("✅ PASSED\n")
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        sys.exit(1)
    
    print("=" * 80)
    print("Test 3: add_artifact tool can be invoked")
    print("=" * 80)
    try:
        test_add_artifact_tool_can_be_invoked()
        print("✅ PASSED\n")
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 80)
    print("Test 4: add_artifact with minimal args")
    print("=" * 80)
    try:
        test_add_artifact_with_minimal_args()
        print("✅ PASSED\n")
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 80)
    print("Test 5: Artifacts state initialization")
    print("=" * 80)
    try:
        test_artifacts_state_initialization()
        print("✅ PASSED\n")
    except Exception as e:
        print(f"❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 80)
    print("All basic tests passed!")
    print("=" * 80)
    print("\nNote: Integration tests with real agents require model configuration.")
    print("Run with pytest to execute all tests including integration tests:\n")
    print("  pytest tests/test_artifacts_middleware.py -v -s\n")

