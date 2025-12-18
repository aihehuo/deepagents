"""Middleware for providing date and time tools to an agent."""

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Annotated

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command


def _get_datetime_tool() -> StructuredTool:
    """Create a tool that returns the current date and time."""
    
    def get_current_datetime(
        format: Annotated[
            str,
            "Format string for datetime. Use 'iso' for ISO format, 'readable' for human-readable, or any strftime format.",
        ] = "readable",
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Get the current date and time.
        
        Args:
            format: Format for the datetime string. Options:
                - 'iso': ISO 8601 format (e.g., '2024-01-15T14:30:00')
                - 'readable': Human-readable format (e.g., 'January 15, 2024 at 2:30 PM')
                - Any strftime format string (e.g., '%Y-%m-%d %H:%M:%S')
        
        Returns:
            Command with ToolMessage containing the current date and time.
        """
        now = datetime.now()
        
        if format == "iso":
            dt_string = now.isoformat()
        elif format == "readable":
            dt_string = now.strftime("%B %d, %Y at %I:%M %p")
        else:
            try:
                dt_string = now.strftime(format)
            except ValueError:
                dt_string = f"Invalid format string: {format}. Using ISO format: {now.isoformat()}"
        
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Current date and time: {dt_string}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
    
    return StructuredTool.from_function(
        func=get_current_datetime,
        name="get_current_datetime",
        description="""Get the current date and time.
        
Use this tool when you need to know the current date and time. This is useful for:
- Timestamping actions or events
- Scheduling or time-based logic
- Providing context about when something happened
- Creating time-sensitive responses

The format parameter allows you to specify how the datetime should be formatted:
- 'iso': ISO 8601 format (e.g., '2024-01-15T14:30:00')
- 'readable': Human-readable format (e.g., 'January 15, 2024 at 2:30 PM')
- Custom strftime format: Any valid strftime format string (e.g., '%Y-%m-%d %H:%M:%S')

If no format is specified, defaults to 'readable'.""",
    )


DATETIME_SYSTEM_PROMPT = """## `get_current_datetime`

You have access to the `get_current_datetime` tool to get the current date and time.
Use this tool when you need to know what time it is, for timestamping, scheduling, or time-based context.

**Important guidelines:**
- When writing reports or documents that include dates or timestamps, you MUST use this tool to get the accurate current date and time.
- Do NOT guess or make up dates/times - always call `get_current_datetime` to ensure accuracy.
- For human-readable format in reports, use `format='readable'` when calling the tool (e.g., "January 15, 2024 at 2:30 PM").
- This is especially important when timestamping reports, recommendations, or any time-sensitive information.
"""


class DateTimeMiddleware(AgentMiddleware):
    """Middleware for providing date and time tools to an agent.
    
    This middleware adds a `get_current_datetime` tool that returns the current date and time
    in various formats (ISO, readable, or custom strftime format).
    
    Example:
        ```python
        from deepagents.middleware.datetime import DateTimeMiddleware
        from langchain.agents import create_agent
        
        agent = create_agent(
            model="anthropic:claude-sonnet-4-20250514",
            middleware=[DateTimeMiddleware()],
        )
        ```
    """
    
    def __init__(
        self,
        *,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the DateTimeMiddleware.
        
        Args:
            system_prompt: Optional custom system prompt override. If None, uses default.
        """
        super().__init__()
        self.tools = [_get_datetime_tool()]
        self.system_prompt = system_prompt or DATETIME_SYSTEM_PROMPT
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject system prompt about the datetime tool."""
        if self.system_prompt:
            if request.system_prompt:
                new_system_prompt = request.system_prompt + "\n\n" + self.system_prompt
            else:
                new_system_prompt = self.system_prompt
            return handler(request.override(system_prompt=new_system_prompt))
        return handler(request)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        if self.system_prompt:
            if request.system_prompt:
                new_system_prompt = request.system_prompt + "\n\n" + self.system_prompt
            else:
                new_system_prompt = self.system_prompt
            return await handler(request.override(system_prompt=new_system_prompt))
        return await handler(request)

