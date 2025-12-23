"""Middleware for providing AI He Huo platform search tools to an agent."""

import json
import mimetypes
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

try:
    import requests
except ImportError:
    requests = None  # Will be checked when actually needed


AIHEHUO_USER_AGENT = "LLM_AGENT"


class AihehuoState(AgentState):
    """State for the AI He Huo middleware."""

    # No state needed for this middleware, but we keep the schema for consistency
    pass


def _get_api_config() -> tuple[str, str]:
    """Get the API base URL and API key from environment variables.
    
    Returns:
        Tuple of (API base URL, API key)
    """
    api_base = os.getenv('AIHEHUO_API_BASE', 'https://new-api.aihehuo.com')
    api_key = os.getenv('AIHEHUO_API_KEY', '')
    return api_base, api_key


def _search_members_api(
    query: str,
    max_results: int = 10,
    page: int = 1,
    wechat_reachable_only: bool = False,
    investor: bool | None = None,
    excluded_ids: list[str] | None = None,
) -> dict:
    """Search AI He Huo members using the API.
    
    Args:
        query: The search query string (must be longer than 5 characters).
        max_results: Maximum number of results per page (default: 10, minimum: 10).
        page: Page number for pagination (default: 1).
        wechat_reachable_only: Whether to only return users reachable on WeChat.
        investor: Whether to only search for investors (None = search all).
        excluded_ids: List of user IDs to exclude from results.
    
    Returns:
        Dictionary with search results or error information.
    """
    if requests is None:
        return {
            "error": "requests package not installed",
            "message": "Please install requests: pip install requests"
        }
    
    # Validate query length
    if len(query.strip()) <= 5:
        return {
            "error": "Query too short",
            "message": "搜索关键词长度必须大于5个字符",
            "query_length": len(query.strip()),
            "minimum_length": 6
        }
    
    # Validate max_results - API requires minimum of 10
    if max_results < 10:
        max_results = 10
    
    try:
        api_base, api_key = _get_api_config()
        url = f"{api_base}/users/search"
        
        # Check if API key is available
        if not api_key:
            return {
                "error": "API key not configured",
                "message": "AIHEHUO_API_KEY not found in environment variables",
                "hint": "Please set AIHEHUO_API_KEY environment variable"
            }
        
        # Build query parameters
        paginate_obj = {
            "page": page,
            "per": max_results
        }
        
        payload = {
            "query": query,
            "paginate": paginate_obj,
            "vector_search": True,
            "wechat_reachable_only": wechat_reachable_only
        }
        
        # Add optional parameters
        if investor is not None:
            payload["investor"] = investor
        if excluded_ids is not None:
            payload["excluded_ids"] = excluded_ids
        
        # Make API request with authentication
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": AIHEHUO_USER_AGENT
        }
        
        # Use GET request (API accepts JSON body with GET)
        resp = requests.get(url, json=payload, headers=headers, timeout=15)
        
        # Check status code before parsing
        if resp.status_code >= 400:
            error_result = {
                "error": f"API request failed with status {resp.status_code}",
                "status_code": resp.status_code,
                "message": f"HTTP {resp.status_code} error",
            }
            # Try to get error details from response
            try:
                error_data = resp.json()
                if isinstance(error_data, dict):
                    error_result.update(error_data)
            except (json.JSONDecodeError, ValueError):
                # If response is not JSON, include text
                error_result["response_text"] = resp.text[:500]  # Limit length
            
            return error_result
        
        resp.encoding = 'utf-8'
        return resp.json()
        
    except requests.exceptions.RequestException as e:
        # Network errors, timeouts, etc.
        return {
            "error": f"API request failed: {str(e)}",
            "error_type": type(e).__name__,
        }
    except Exception as e:
        return {
            "error": f"Error querying AI He Huo: {str(e)}"
        }


def _search_ideas_api(
    query: str,
    max_results: int = 10,
    page: int = 1,
) -> dict:
    """Search AI He Huo ideas/projects using the API.
    
    Args:
        query: The search query string.
        max_results: Maximum number of results per page (default: 10).
        page: Page number for pagination (default: 1).
    
    Returns:
        Dictionary with search results or error information.
    """
    if requests is None:
        return {
            "error": "requests package not installed",
            "message": "Please install requests: pip install requests"
        }
    
    try:
        api_base, api_key = _get_api_config()
        url = f"{api_base}/ideas/search"
        
        # Check if API key is available
        if not api_key:
            return {
                "error": "API key not configured",
                "message": "AIHEHUO_API_KEY not found in environment variables",
                "hint": "Please set AIHEHUO_API_KEY environment variable"
            }
        
        # Build query parameters
        paginate_obj = {
            "page": page,
            "per": max_results
        }
        
        payload = {
            "query": query,
            "paginate": paginate_obj,
            "vector_search": True,
        }
        
        # Make API request with authentication
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": AIHEHUO_USER_AGENT
        }
        
        # Use GET request (API accepts JSON body with GET)
        resp = requests.get(url, json=payload, headers=headers, timeout=15)
        
        # Check status code before parsing
        if resp.status_code >= 400:
            error_result = {
                "error": f"API request failed with status {resp.status_code}",
                "status_code": resp.status_code,
                "message": f"HTTP {resp.status_code} error",
            }
            # Try to get error details from response
            try:
                error_data = resp.json()
                if isinstance(error_data, dict):
                    error_result.update(error_data)
            except (json.JSONDecodeError, ValueError):
                # If response is not JSON, include text
                error_result["response_text"] = resp.text[:500]  # Limit length
            
            return error_result
        
        resp.encoding = 'utf-8'
        return resp.json()
        
    except requests.exceptions.RequestException as e:
        # Network errors, timeouts, etc.
        return {
            "error": f"API request failed: {str(e)}",
            "error_type": type(e).__name__,
        }
    except Exception as e:
        return {
            "error": f"Error querying AI He Huo: {str(e)}"
        }


SEARCH_MEMBERS_TOOL_DESCRIPTION = """Search for members, entrepreneurs, and investors on the AI He Huo (爱合伙) platform.

This tool uses semantic vector search to find relevant people based on your query. Use coherent, descriptive sentences rather than simple keywords for best results.

Usage:
- The query parameter must be longer than 5 characters
- Use natural language descriptions (e.g., "寻找有AI技术背景的创业者" instead of "AI 技术")
- The max_results parameter defaults to 10 (minimum is 10)
- Use pagination with the page parameter to get more results
- Filter by investor status using the investor parameter
- Filter for WeChat-reachable members using wechat_reachable_only
- Exclude specific users using excluded_ids

Examples:
- Search for AI entrepreneurs: query="寻找有AI技术背景的创业者，希望合作开发智能产品"
- Search for investors: query="寻找对教育科技领域感兴趣的投资人", investor=True
- Search WeChat-reachable members: query="寻找有技术背景的创业者", wechat_reachable_only=True

The results are returned in JSON format with user information, backgrounds, and project details."""

SEARCH_IDEAS_TOOL_DESCRIPTION = """Search for business ideas and projects on the AI He Huo (爱合伙) platform.

This tool uses semantic vector search to find relevant business ideas and projects based on your query.

Usage:
- Use natural language descriptions for best results
- The max_results parameter defaults to 10
- Use pagination with the page parameter to get more results

Examples:
- Search for AI projects: query="AI驱动的教育平台项目"
- Search for mobile apps: query="移动应用开发项目"

The results are returned in JSON format with project information, descriptions, and related details."""

def _search_members_tool_generator(
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the search_members tool.
    
    Args:
        custom_description: Optional custom description for the tool.
    
    Returns:
        Configured search_members tool.
    """
    tool_description = custom_description or SEARCH_MEMBERS_TOOL_DESCRIPTION

    def sync_search_members(
        query: str,
        runtime: ToolRuntime[None, AihehuoState],
        max_results: int = 10,
        page: int = 1,
        wechat_reachable_only: bool = False,
        investor: bool | None = None,
        excluded_ids: list[str] | None = None,
    ) -> str:
        """Synchronous wrapper for search_members tool."""
        result = _search_members_api(
            query=query,
            max_results=max_results,
            page=page,
            wechat_reachable_only=wechat_reachable_only,
            investor=investor,
            excluded_ids=excluded_ids,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def async_search_members(
        query: str,
        runtime: ToolRuntime[None, AihehuoState],
        max_results: int = 10,
        page: int = 1,
        wechat_reachable_only: bool = False,
        investor: bool | None = None,
        excluded_ids: list[str] | None = None,
    ) -> str:
        """Asynchronous wrapper for search_members tool."""
        result = _search_members_api(
            query=query,
            max_results=max_results,
            page=page,
            wechat_reachable_only=wechat_reachable_only,
            investor=investor,
            excluded_ids=excluded_ids,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    return StructuredTool.from_function(
        name="aihehuo_search_members",
        description=tool_description,
        func=sync_search_members,
        coroutine=async_search_members,
    )


def _search_ideas_tool_generator(
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the search_ideas tool.
    
    Args:
        custom_description: Optional custom description for the tool.
    
    Returns:
        Configured search_ideas tool.
    """
    tool_description = custom_description or SEARCH_IDEAS_TOOL_DESCRIPTION

    def sync_search_ideas(
        query: str,
        runtime: ToolRuntime[None, AihehuoState],
        max_results: int = 10,
        page: int = 1,
    ) -> str:
        """Synchronous wrapper for search_ideas tool."""
        result = _search_ideas_api(
            query=query,
            max_results=max_results,
            page=page,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    async def async_search_ideas(
        query: str,
        runtime: ToolRuntime[None, AihehuoState],
        max_results: int = 10,
        page: int = 1,
    ) -> str:
        """Asynchronous wrapper for search_ideas tool."""
        result = _search_ideas_api(
            query=query,
            max_results=max_results,
            page=page,
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    return StructuredTool.from_function(
        name="aihehuo_search_ideas",
        description=tool_description,
        func=sync_search_ideas,
        coroutine=async_search_ideas,
    )


AIHEHUO_SYSTEM_PROMPT = """## AI He Huo Platform Tools `aihehuo_search_members`, `aihehuo_search_ideas`

You have access to the AI He Huo (爱合伙) platform, a Chinese entrepreneurship and networking platform.

- aihehuo_search_members: Search for members, entrepreneurs, and investors using semantic vector search
- aihehuo_search_ideas: Search for business ideas and projects

**Important Notes:**
- Use coherent, descriptive sentences for queries (not just keywords)
- Query must be longer than 5 characters for member search
- Results are returned in JSON format
- Use pagination to get more results if needed"""


class AihehuoMiddleware(AgentMiddleware):
    """Middleware for providing AI He Huo platform search tools to an agent.
    
    This middleware adds search tools to the agent:
    - aihehuo_search_members: Search for members, entrepreneurs, and investors
    - aihehuo_search_ideas: Search for business ideas and projects
    
    The middleware requires the `requests` package and environment variables:
    - AIHEHUO_API_KEY: API key for authentication (required)
    - AIHEHUO_API_BASE: API base URL (defaults to https://new-api.aihehuo.com)
    
    Args:
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
    
    Example:
        ```python
        from deepagents.middleware.aihehuo import AihehuoMiddleware
        from langchain.agents import create_agent
        
        # Set environment variables before creating agent
        import os
        os.environ["AIHEHUO_API_KEY"] = "your_api_key"
        
        agent = create_agent(
            middleware=[AihehuoMiddleware()],
        )
        ```
    """
    
    state_schema = AihehuoState
    
    def __init__(
        self,
        *,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
    ) -> None:
        """Initialize the AI He Huo middleware.
        
        Args:
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
        """
        # Check configuration during initialization
        print("[AihehuoMiddleware] Initializing AI He Huo middleware...")
        
        # Check for requests package
        if requests is None:
            print("[AihehuoMiddleware] WARNING: 'requests' package not installed!")
            print("[AihehuoMiddleware]   Install with: pip install requests")
            print("[AihehuoMiddleware]   Tools will not function without this package.")
        else:
            print("[AihehuoMiddleware] ✓ 'requests' package is available")
        
        # Check environment variables
        api_base, api_key = _get_api_config()
        
        print("[AihehuoMiddleware] Configuration check:")
        print(f"  AIHEHUO_API_BASE: {api_base}")
        
        if api_key:
            # Mask API key for security (show first 8 and last 4 chars)
            masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            print(f"  AIHEHUO_API_KEY: {masked_key} (present, length: {len(api_key)})")
            print("[AihehuoMiddleware] ✓ API key is configured")
        else:
            print("  AIHEHUO_API_KEY: NOT SET")
            print("[AihehuoMiddleware] ⚠ WARNING: AIHEHUO_API_KEY is not configured!")
            print("[AihehuoMiddleware]   The middleware will be initialized, but API calls will fail.")
            print("[AihehuoMiddleware]   Set AIHEHUO_API_KEY environment variable to enable functionality.")
        
        # Set system prompt (allow full override or None to use default)
        self._custom_system_prompt = system_prompt
        
        # Build tools
        if custom_tool_descriptions is None:
            custom_tool_descriptions = {}
        
        self.tools = [
            _search_members_tool_generator(custom_tool_descriptions.get("aihehuo_search_members")),
            _search_ideas_tool_generator(custom_tool_descriptions.get("aihehuo_search_ideas")),
        ]
        
        print(f"[AihehuoMiddleware] ✓ Initialized with {len(self.tools)} tools:")
        print(f"  - aihehuo_search_members")
        print(f"  - aihehuo_search_ideas")
        print("[AihehuoMiddleware] Initialization complete.")
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt with AI He Huo tool instructions.
        
        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.
        
        Returns:
            The model response from the handler.
        """
        # Use custom system prompt if provided, otherwise use default
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            system_prompt = AIHEHUO_SYSTEM_PROMPT
        
        if system_prompt:
            request = request.override(
                system_prompt=request.system_prompt + "\n\n" + system_prompt 
                if request.system_prompt 
                else system_prompt
            )
        
        return handler(request)
    
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt with AI He Huo tool instructions.
        
        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.
        
        Returns:
            The model response from the handler.
        """
        # Use custom system prompt if provided, otherwise use default
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            system_prompt = AIHEHUO_SYSTEM_PROMPT
        
        if system_prompt:
            request = request.override(
                system_prompt=request.system_prompt + "\n\n" + system_prompt 
                if request.system_prompt 
                else system_prompt
            )
        
        return await handler(request)

