#!/usr/bin/env python3
"""AI He Huo Member Search.

Searches the AI He Huo (爱合伙) platform for entrepreneurs and members using semantic vector search.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import requests
except ImportError:
    requests = None  # Will be checked when actually needed


def get_api_config() -> tuple[str, str]:
    """Get the API base URL and API key from environment variables.
    
    Returns:
        Tuple of (API base URL, API key)
    """
    
    # Default fallback - try environment variables
    api_base = os.getenv('AIHEHUO_API_BASE', 'https://new-api.aihehuo.com')
    api_key = os.getenv('AIHEHUO_API_KEY', '')
    return api_base, api_key


def search_members(
    query: str,
    max_results: int = 10,
    page: int = 1,
    wechat_reachable_only: bool = False,
    investor: Optional[bool] = None,
    excluded_ids: Optional[list] = None,
) -> str:
    """Search AI He Huo members based on the provided search query.

    Parameters
    ----------
    query : str
        The search query string (must be longer than 5 characters).
        Use coherent, descriptive sentences rather than simple keywords.
    max_results : int
        Maximum number of results to retrieve per page (default: 10, minimum: 10)
    page : int
        Page number for pagination (default: 1)
    wechat_reachable_only : bool
        Whether to only return users reachable on WeChat (default: False)
    investor : bool, optional
        Whether to only search for investors (default: None, searches all)
    excluded_ids : list, optional
        List of user IDs to exclude from results

    Returns:
        Formatted search results as JSON string or error message.
    """
    if requests is None:
        return json.dumps({
            "error": "requests package not installed",
            "message": "Please install requests: pip install requests"
        }, ensure_ascii=False, indent=2)
    
    # Validate query length
    if len(query.strip()) <= 5:
        return json.dumps({
            "error": "Query too short",
            "message": "搜索关键词长度必须大于5个字符",
            "query_length": len(query.strip()),
            "minimum_length": 6
        }, ensure_ascii=False, indent=2)
    
    # Validate max_results - API requires minimum of 10
    if max_results < 10:
        max_results = 10
    
    try:
        # Get API base URL and API key from .env.aihehuo
        api_base, api_key = get_api_config()
        url = f"{api_base}/users/search"
        
        # Check if API key is available
        if not api_key:
            return json.dumps({
                "error": "API key not configured",
                "message": "AIHEHUO_API_KEY not found in .env.aihehuo file or environment variables",
                "hint": "Please set AIHEHUO_API_KEY in .env.aihehuo file in the repository root"
            }, ensure_ascii=False, indent=2)
        
        # Build query parameters for GET request
        # Note: The API expects pagination as nested structure in JSON body for GET requests
        # Based on MCP server implementation, we use json=payload with GET
        # API requires per to be at least 10
        paginate_obj = {
            "page": page,
            "per": max_results
        }
        
        # Debug: Print pagination values
        print(f"DEBUG: Pagination values - page: {page} (type: {type(page).__name__}), per: {max_results} (type: {type(max_results).__name__})", file=sys.stderr)
        print(f"DEBUG: Paginate object: {paginate_obj}", file=sys.stderr)
        
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
        
        # Debug: Print full payload
        print(f"DEBUG: Full payload being sent: {json.dumps(payload, ensure_ascii=False, indent=2)}", file=sys.stderr)
        
        # Make API request with authentication
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "DeepAgents-Skill"
        }
        
        # Use GET request (API accepts JSON body with GET, as per MCP server implementation)
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
            
            return json.dumps(error_result, ensure_ascii=False, indent=2)
        
        resp.encoding = 'utf-8'
        data = resp.json()
        
        # Format and return results
        return json.dumps(data, ensure_ascii=False, indent=2)
        
    except requests.exceptions.RequestException as e:
        # Network errors, timeouts, etc. (not HTTP status errors)
        error_result = {
            "error": f"API request failed: {str(e)}",
            "error_type": type(e).__name__,
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Error querying AI He Huo: {str(e)}"
        }, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search AI He Huo (爱合伙) platform for entrepreneurs and members"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query string (must be longer than 5 characters, use coherent sentences)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results per page (default: 10)"
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="Page number for pagination (default: 1)"
    )
    parser.add_argument(
        "--wechat-reachable-only",
        action="store_true",
        help="Only return users reachable on WeChat"
    )
    parser.add_argument(
        "--investor",
        action="store_true",
        help="Only search for investors"
    )
    parser.add_argument(
        "--excluded-ids",
        type=str,
        help="Comma-separated list of user IDs to exclude"
    )
    
    args = parser.parse_args()
    
    excluded_ids = None
    if args.excluded_ids:
        excluded_ids = [id.strip() for id in args.excluded_ids.split(',')]
    
    result = search_members(
        query=args.query,
        max_results=args.max_results,
        page=args.page,
        wechat_reachable_only=args.wechat_reachable_only,
        investor=args.investor if args.investor else None,
        excluded_ids=excluded_ids,
    )
    
    print(result)


if __name__ == "__main__":
    main()

