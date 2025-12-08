"""Tests for DateTimeMiddleware."""

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from deepagents.middleware.datetime import DateTimeMiddleware


def test_datetime_middleware_provides_tool():
    """Test that DateTimeMiddleware provides the get_current_datetime tool."""
    middleware = DateTimeMiddleware()
    
    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "get_current_datetime"
    assert "date and time" in middleware.tools[0].description.lower()


def test_datetime_middleware_has_system_prompt():
    """Test that DateTimeMiddleware has a system prompt."""
    middleware = DateTimeMiddleware()
    
    assert middleware.system_prompt is not None
    assert "get_current_datetime" in middleware.system_prompt


def test_datetime_tool_can_be_invoked():
    """Test that the datetime tool can be invoked and returns a Command."""
    middleware = DateTimeMiddleware()
    tool = middleware.tools[0]
    
    # Create a tool call dict
    tool_call = {
        "name": "get_current_datetime",
        "args": {"format": "readable"},
        "id": "test-call-1",
        "type": "tool_call",
    }
    
    result = tool.invoke(tool_call)
    
    # Should return a Command
    from langgraph.types import Command
    assert isinstance(result, Command)
    assert "messages" in result.update
    assert len(result.update["messages"]) == 1
    
    # Message should contain date/time info
    message = result.update["messages"][0]
    assert message.content.startswith("Current date and time:")


@pytest.mark.skipif(
    not pytest.importorskip("langchain_anthropic", reason="langchain_anthropic not installed"),
    reason="Requires langchain_anthropic",
)
class TestDateTimeMiddlewareWithAgent:
    """Integration tests with a real agent."""
    
    def test_agent_can_use_datetime_tool(self):
        """Test that an agent can use the datetime tool."""
        import os
        from pathlib import Path
        from langchain_anthropic import ChatAnthropic
        
        # Try to load DeepSeek config if available
        env_file = Path(__file__).parent.parent / ".env.deepseek"
        if env_file.exists():
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
            
            if base_url and api_key:
                old_base_url = os.environ.get("ANTHROPIC_BASE_URL")
                old_api_key = os.environ.get("ANTHROPIC_API_KEY")
                os.environ["ANTHROPIC_BASE_URL"] = base_url
                os.environ["ANTHROPIC_API_KEY"] = api_key
                
                try:
                    model = ChatAnthropic(
                        model=model_name,
                        base_url=base_url,
                        api_key=api_key,
                    )
                finally:
                    if old_base_url is not None:
                        os.environ["ANTHROPIC_BASE_URL"] = old_base_url
                    if old_api_key is not None:
                        os.environ["ANTHROPIC_API_KEY"] = old_api_key
            else:
                pytest.skip("DeepSeek config incomplete")
        else:
            # Use default Anthropic model (requires API key)
            model = ChatAnthropic(model="claude-sonnet-4-20250514")
        
        agent = create_agent(
            model=model,
            middleware=[DateTimeMiddleware()],
            tools=[],
        )
        
        result = agent.invoke({
            "messages": [HumanMessage(content="What is the current date and time?")]
        })
        
        # Should have used the datetime tool
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        datetime_messages = [
            msg for msg in tool_messages 
            if "date and time" in msg.content.lower() or "get_current_datetime" in str(msg.name).lower()
        ]
        
        assert len(datetime_messages) > 0, "Agent should have used the datetime tool"
        assert "Current date and time:" in datetime_messages[0].content

