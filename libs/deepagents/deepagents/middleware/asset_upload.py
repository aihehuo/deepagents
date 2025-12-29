"""Middleware for uploading assets/files to external services."""

import json
import mimetypes
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

try:
    import requests
except ImportError:
    requests = None  # Will be checked when actually needed


class AssetUploadState(AgentState):
    """State for the asset upload middleware."""

    # No state needed for this middleware, but we keep the schema for consistency
    pass


def _get_aihehuo_api_config() -> tuple[str, str]:
    """Get the AI He Huo API base URL and API key from environment variables.
    
    Returns:
        Tuple of (API base URL, API key)
    """
    api_base = os.getenv('AIHEHUO_API_BASE', 'https://new-api.aihehuo.com')
    api_key = os.getenv('AIHEHUO_API_KEY', '')
    return api_base, api_key


AIHEHUO_USER_AGENT = "LLM_AGENT"


def _upload_file_to_aihehuo_api(
    file_path: str,
    timeout: int = 60,
) -> dict:
    """Upload a file to AI He Huo cloud storage using the API.
    
    Args:
        file_path: The absolute path to the file to upload.
        timeout: Request timeout in seconds (default: 60).
    
    Returns:
        Dictionary with upload results (including file URL) or error information.
    """
    if requests is None:
        return {
            "error": "requests package not installed",
            "message": "Please install requests: pip install requests"
        }
    
    try:
        api_base, api_key = _get_aihehuo_api_config()
        url = f"{api_base}/micro/upload"
        
        # DEBUG: Print configuration before upload
        print(f"[AssetUploadMiddleware] Upload file - Configuration:")
        print(f"  API Base: {api_base}")
        print(f"  Upload URL: {url}")
        print(f"  API Key present: {bool(api_key)}")
        # Mask API key for debugging (show first 8 and last 4 chars)
        masked_key = None
        if api_key:
            masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            print(f"  API Key (masked): {masked_key}")
        print(f"  Timeout: {timeout}s")
        
        # Check if API key is available
        if not api_key:
            print("[AssetUploadMiddleware] ERROR: API key not configured")
            return {
                "error": "API key not configured",
                "message": "AIHEHUO_API_KEY not found in environment variables",
                "hint": "Please set AIHEHUO_API_KEY environment variable"
            }
        
        # Validate file exists
        # Resolve absolute path to handle any path issues
        abs_file_path = os.path.abspath(os.path.expanduser(file_path))
        
        print(f"[AssetUploadMiddleware] Upload file - File path validation:")
        print(f"  Original path: {file_path}")
        print(f"  Absolute path: {abs_file_path}")
        print(f"  Path exists: {os.path.exists(abs_file_path)}")
        print(f"  Is file: {os.path.isfile(abs_file_path) if os.path.exists(abs_file_path) else 'N/A'}")
        
        # Check both original and absolute paths
        original_exists = os.path.exists(file_path)
        abs_exists = os.path.exists(abs_file_path)
        
        if not original_exists and not abs_exists:
            # Try to get more info about the directory
            dir_path = os.path.dirname(abs_file_path)
            print(f"[AssetUploadMiddleware] ERROR: File not found!")
            print(f"  Checked paths:")
            print(f"    - Original: {file_path} (exists: {original_exists})")
            print(f"    - Absolute: {abs_file_path} (exists: {abs_exists})")
            print(f"  Directory info:")
            print(f"    - Directory: {dir_path}")
            print(f"    - Directory exists: {os.path.exists(dir_path)}")
            if os.path.exists(dir_path):
                print(f"    - Directory is writable: {os.access(dir_path, os.W_OK)}")
                # List files in directory (first 10)
                try:
                    files_in_dir = os.listdir(dir_path)[:10]
                    print(f"    - Files in directory (first 10): {files_in_dir}")
                    # Check if a similar filename exists
                    target_filename = os.path.basename(abs_file_path)
                    similar_files = [f for f in files_in_dir if target_filename.lower() in f.lower() or f.lower() in target_filename.lower()]
                    if similar_files:
                        print(f"    - Similar filenames found: {similar_files}")
                except Exception as e:
                    print(f"    - Could not list directory: {e}")
            
            return {
                "error": "File not found",
                "message": f"File does not exist at path: {file_path}",
                "file_path": file_path,
                "absolute_path": abs_file_path,
                "directory": dir_path,
                "directory_exists": os.path.exists(dir_path) if dir_path else False,
            }
        
        # Use the path that exists (prefer absolute if both exist, or whichever exists)
        if abs_exists:
            actual_file_path = abs_file_path
        elif original_exists:
            actual_file_path = file_path
        else:
            # This shouldn't happen due to the check above, but just in case
            actual_file_path = abs_file_path
        
        # Get file info (use the actual file path that exists)
        file_size = os.path.getsize(actual_file_path)
        original_filename = os.path.basename(actual_file_path)
        file_ext = os.path.splitext(actual_file_path)[1].lower()
        
        # Generate unique filename with timestamp to prevent overwrites in OSS
        # Format: originalname_YYYYMMDD_HHMMSS.ext
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Split filename into name and extension
        filename_without_ext = os.path.splitext(original_filename)[0]
        # Append timestamp before extension
        unique_filename = f"{filename_without_ext}_{timestamp_str}{file_ext}"
        
        # DEBUG: Print file information
        print(f"[AssetUploadMiddleware] Upload file - File information:")
        print(f"  Original path: {file_path}")
        print(f"  Actual file path: {actual_file_path}")
        print(f"  Original filename: {original_filename}")
        print(f"  Unique filename (with timestamp): {unique_filename}")
        print(f"  File extension: {file_ext}")
        print(f"  File size: {file_size} bytes")
        
        # Explicit MIME type mapping for common file types
        mime_type_map = {
            '.md': 'text/markdown',
            '.markdown': 'text/markdown',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.json': 'application/json',
            '.jsonl': 'application/jsonl',
            '.xml': 'application/xml',
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
        }
        
        # Determine MIME type
        if file_ext in mime_type_map:
            mime_type = mime_type_map[file_ext]
        else:
            mime_type, _ = mimetypes.guess_type(actual_file_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
        
        print(f"  MIME type: {mime_type}")
        
        # Prepare upload headers
        upload_headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": AIHEHUO_USER_AGENT
        }
        
        # DEBUG: Print request details (without exposing full API key)
        print(f"[AssetUploadMiddleware] Upload file - Request details:")
        print(f"  Method: POST")
        print(f"  URL: {url}")
        # Create a safe headers dict for printing (mask Authorization header)
        safe_headers = {}
        for k, v in upload_headers.items():
            if k == 'Authorization':
                safe_headers[k] = f'Bearer {masked_key if masked_key else "N/A"}'
            else:
                safe_headers[k] = v
        print(f"  Headers: {safe_headers}")
        print(f"  Files: {{'file': ('{unique_filename}', <file>, '{mime_type}')}}")
        print(f"[AssetUploadMiddleware] Sending upload request...")
        print(f"[AssetUploadMiddleware] Note: Using unique filename '{unique_filename}' to prevent overwrites in OSS")
        
        # Upload file using multipart/form-data
        # Use unique_filename to ensure each upload gets a unique name in OSS
        with open(actual_file_path, 'rb') as f:
            files = {
                'file': (unique_filename, f, mime_type)
            }
            
            resp = requests.post(url, headers=upload_headers, files=files, timeout=timeout)
        
        # DEBUG: Print response details
        print(f"[AssetUploadMiddleware] Upload file - Response received:")
        print(f"  Status code: {resp.status_code}")
        print(f"  Response headers: {dict(resp.headers)}")
        print(f"  Response encoding: {resp.encoding}")
        
        # Check status code before parsing
        if resp.status_code >= 400:
            print(f"[AssetUploadMiddleware] ERROR: Upload failed with status {resp.status_code}")
            error_result = {
                "error": f"API request failed with status {resp.status_code}",
                "status_code": resp.status_code,
                "message": f"HTTP {resp.status_code} error",
            }
            # Try to get error details from response
            try:
                error_data = resp.json()
                print(f"  Response JSON: {json.dumps(error_data, ensure_ascii=False, indent=2)}")
                if isinstance(error_data, dict):
                    error_result.update(error_data)
            except (json.JSONDecodeError, ValueError):
                # If response is not JSON, include text
                response_text = resp.text[:500]  # Limit length
                print(f"  Response text (first 500 chars): {response_text}")
                error_result["response_text"] = response_text
            
            return error_result
        
        # Success case
        resp.encoding = 'utf-8'
        response_data = resp.json()
        print(f"[AssetUploadMiddleware] Upload file - Success!")
        print(f"  Response JSON: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
        return response_data
        
    except requests.exceptions.RequestException as e:
        # Network errors, timeouts, etc.
        print(f"[AssetUploadMiddleware] ERROR: Request exception: {type(e).__name__}: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Response text: {e.response.text[:500]}")
        return {
            "error": f"API request failed: {str(e)}",
            "error_type": type(e).__name__,
        }
    except Exception as e:
        print(f"[AssetUploadMiddleware] ERROR: Unexpected exception: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return {
            "error": f"Error uploading file: {str(e)}"
        }


def _resolve_file_path_for_upload(file_path: str, runtime: ToolRuntime[None, AssetUploadState], backend_root: str | None = None, docs_dir: str | None = None) -> str:
    """Resolve a virtual file path to the actual filesystem path for upload.
    
    This function handles path resolution for files that may have been written through
    a virtual filesystem backend. It tries multiple strategies:
    1. If the path exists as-is, use it
    2. If it's an absolute path that doesn't exist, try resolving relative to backend root
    3. Try resolving relative to current working directory
    4. Check if it's just a filename that should be in docs_dir
    
    Args:
        file_path: The file path to resolve (may be virtual or absolute)
        runtime: The tool runtime (may contain backend information)
        backend_root: Optional backend root directory
        docs_dir: Optional docs directory where files are stored
    
    Returns:
        Resolved absolute file path
    """
    # First, check if the path exists as-is
    if os.path.exists(file_path):
        return os.path.abspath(file_path)
    
    # Try to get backend information if not provided
    if backend_root is None:
        try:
            backend_root = str(Path.cwd())
        except Exception:
            pass
    
    if docs_dir is None:
        try:
            # Check common docs_dir location for Business Co-founder API
            docs_dir = str(Path.home() / ".deepagents" / "business_cofounder_api" / "docs")
        except Exception:
            pass
    
    # Strategy 1: Handle virtual paths (paths starting with "/" that are virtual, not absolute)
    # With virtual_mode=True, paths like /tmp/file.html are virtual paths relative to backend_root
    # But DocsOnlyWriteBackend redirects all writes to docs_dir, so check docs_dir first
    path_obj = Path(file_path)
    
    # Extract filename for DocsOnlyWriteBackend resolution (it only uses the filename)
    filename = path_obj.name
    
    # Priority 1: Check docs_dir + filename (DocsOnlyWriteBackend maps all writes to docs_dir)
    # This is the most likely location since all writes go to docs_dir regardless of virtual path
    if docs_dir and os.path.exists(docs_dir) and filename:
        candidate = Path(docs_dir) / filename
        if os.path.exists(candidate):
            print(f"[AssetUploadMiddleware] Resolved path via docs_dir: {file_path} -> {candidate}")
            return str(candidate.resolve())
    
    # Priority 2: If it's an absolute path that doesn't exist, try resolving relative to backend root
    if path_obj.is_absolute() and not os.path.exists(file_path):
        # Try backend root + filename
        if backend_root and filename:
            candidate = Path(backend_root) / filename
            if os.path.exists(candidate):
                print(f"[AssetUploadMiddleware] Resolved path: {file_path} -> {candidate}")
                return str(candidate.resolve())
        
        # Try resolving the virtual path relative to backend root
        # e.g., /tmp/file.html -> base_dir/tmp/file.html (if virtual_mode=True)
        if backend_root:
            # Remove leading slash and resolve relative to backend root
            relative_path = file_path.lstrip("/")
            candidate = Path(backend_root) / relative_path
            if os.path.exists(candidate):
                print(f"[AssetUploadMiddleware] Resolved virtual path to backend root: {file_path} -> {candidate}")
                return str(candidate.resolve())
    
    # Strategy 2: If it starts with "/" but isn't a real absolute path, treat as virtual path
    if file_path.startswith("/") and not path_obj.is_absolute():
        # Remove leading slash and resolve relative to backend root
        if backend_root:
            relative_path = file_path.lstrip("/")
            candidate = Path(backend_root) / relative_path
            if os.path.exists(candidate):
                print(f"[AssetUploadMiddleware] Resolved virtual path: {file_path} -> {candidate}")
                return str(candidate.resolve())
    
    # Strategy 3: Try relative to current working directory
    candidate = Path.cwd() / file_path
    if os.path.exists(candidate):
        print(f"[AssetUploadMiddleware] Resolved relative path: {file_path} -> {candidate}")
        return str(candidate.resolve())
    
    # If all strategies fail, return the original path (will be handled by upload function's error handling)
    print(f"[AssetUploadMiddleware] Could not resolve path: {file_path} (will try as-is)")
    return file_path


UPLOAD_ASSET_TOOL_DESCRIPTION = """Upload a file (markdown, HTML, or other supported formats) to cloud storage and get a shareable URL.

This tool uploads files to the AI He Huo (爱合伙) platform cloud storage and returns a URL/link that can be shared or used to access the uploaded file.

Usage:
- file_path: The path to the file you want to upload (can be virtual or absolute path)
- Supported file types include: markdown (.md, .markdown), HTML (.html, .htm), text (.txt), JSON (.json), PDF (.pdf), images (.png, .jpg, .jpeg, .gif, .svg), and more
- The file must exist at the specified path (the tool will try to resolve virtual paths automatically)
- The tool returns a dictionary containing the file URL and other metadata

Examples:
- Upload a markdown report: file_path="/path/to/report.md"
- Upload an HTML document: file_path="/path/to/document.html"

The response includes a URL that can be used to access the uploaded file on the platform."""


def _upload_asset_tool_generator(
    custom_description: str | None = None,
    backend_root: str | None = None,
    docs_dir: str | None = None,
) -> BaseTool:
    """Generate the upload_asset tool.
    
    Args:
        custom_description: Optional custom description for the tool.
        backend_root: Optional backend root directory for path resolution.
        docs_dir: Optional docs directory for path resolution.
    
    Returns:
        Configured upload_asset tool.
    """
    tool_description = custom_description or UPLOAD_ASSET_TOOL_DESCRIPTION

    def sync_upload_asset(
        file_path: str,
        runtime: ToolRuntime[None, AssetUploadState],
        timeout: int = 60,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str | Command:
        """Synchronous wrapper for upload_asset tool."""
        # Resolve the file path through the backend if available
        resolved_path = _resolve_file_path_for_upload(file_path, runtime, backend_root, docs_dir)
        result = _upload_file_to_aihehuo_api(
            file_path=resolved_path,
            timeout=timeout,
        )
        
        # Parse the result to check if upload was successful
        result_dict = result if isinstance(result, dict) else json.loads(result) if isinstance(result, str) else {}
        
        # Check if upload was successful (has URL, no error)
        if isinstance(result_dict, dict) and "url" in result_dict and "error" not in result_dict:
            file_url = result_dict.get("url")
            file_name = result_dict.get("filename", "")
            
            # Determine artifact type from file extension
            artifact_type = "html"
            if file_name:
                file_ext = file_name.lower().split(".")[-1] if "." in file_name else ""
                if file_ext in ("md", "markdown"):
                    artifact_type = "md"
                elif file_ext == "pdf":
                    artifact_type = "pdf"
                elif file_ext in ("txt", "text"):
                    artifact_type = "txt"
            
            # Create artifact metadata
            artifact = {
                "url": file_url,
                "artifact_type": artifact_type,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            if file_name:
                artifact["name"] = file_name
            
            # Return Command with both the tool message and artifacts state update
            return Command(
                update={
                    "artifacts": [artifact],  # Reducer will append to existing list
                    "messages": [
                        ToolMessage(
                            content=json.dumps(result_dict, ensure_ascii=False, indent=2),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )
        
        # If upload failed or no URL, just return the result as a string
        return json.dumps(result_dict, ensure_ascii=False, indent=2)

    async def async_upload_asset(
        file_path: str,
        runtime: ToolRuntime[None, AssetUploadState],
        timeout: int = 60,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> str | Command:
        """Asynchronous wrapper for upload_asset tool."""
        # Resolve the file path through the backend if available
        resolved_path = _resolve_file_path_for_upload(file_path, runtime, backend_root, docs_dir)
        result = _upload_file_to_aihehuo_api(
            file_path=resolved_path,
            timeout=timeout,
        )
        
        # Parse the result to check if upload was successful
        result_dict = result if isinstance(result, dict) else json.loads(result) if isinstance(result, str) else {}
        
        # Check if upload was successful (has URL, no error)
        if isinstance(result_dict, dict) and "url" in result_dict and "error" not in result_dict:
            file_url = result_dict.get("url")
            file_name = result_dict.get("filename", "")
            
            # Determine artifact type from file extension
            artifact_type = "html"
            if file_name:
                file_ext = file_name.lower().split(".")[-1] if "." in file_name else ""
                if file_ext in ("md", "markdown"):
                    artifact_type = "md"
                elif file_ext == "pdf":
                    artifact_type = "pdf"
                elif file_ext in ("txt", "text"):
                    artifact_type = "txt"
            
            # Create artifact metadata
            artifact = {
                "url": file_url,
                "artifact_type": artifact_type,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            if file_name:
                artifact["name"] = file_name
            
            # Return Command with both the tool message and artifacts state update
            return Command(
                update={
                    "artifacts": [artifact],  # Reducer will append to existing list
                    "messages": [
                        ToolMessage(
                            content=json.dumps(result_dict, ensure_ascii=False, indent=2),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )
        
        # If upload failed or no URL, just return the result as a string
        return json.dumps(result_dict, ensure_ascii=False, indent=2)

    return StructuredTool.from_function(
        name="upload_asset",
        description=tool_description,
        func=sync_upload_asset,
        coroutine=async_upload_asset,
    )


ASSET_UPLOAD_SYSTEM_PROMPT = """## Asset Upload Tool `upload_asset`

You have access to an asset upload tool that can upload files to cloud storage and return shareable URLs.

- upload_asset: Upload files (markdown, HTML, etc.) to cloud storage and get a shareable URL

**Important Notes:**
- Files are uploaded to the AI He Huo (爱合伙) platform cloud storage
- Uploaded files return a URL that can be shared or accessed on the platform
- The tool automatically resolves virtual file paths to actual filesystem paths
- Files written via write_file are typically stored in the docs directory"""


class AssetUploadMiddleware(AgentMiddleware):
    """Middleware for uploading assets/files to external services.
    
    This middleware adds an upload tool to the agent:
    - upload_asset: Upload files (markdown, HTML, etc.) to cloud storage and get a shareable URL
    
    The middleware requires the `requests` package and environment variables:
    - AIHEHUO_API_KEY: API key for authentication (required for AI He Huo uploads)
    - AIHEHUO_API_BASE: API base URL (defaults to https://new-api.aihehuo.com)
    
    The middleware is aware of virtual filesystem backends (like DocsOnlyWriteBackend)
    and can resolve virtual paths to actual file locations.
    
    Args:
        backend_root: Optional backend root directory for path resolution.
        docs_dir: Optional docs directory where files are stored (for DocsOnlyWriteBackend).
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
    
    Example:
        ```python
        from deepagents.middleware.asset_upload import AssetUploadMiddleware
        from langchain.agents import create_agent
        
        # Set environment variables before creating agent
        import os
        os.environ["AIHEHUO_API_KEY"] = "your_api_key"
        
        # For Business Co-founder API with DocsOnlyWriteBackend
        docs_dir = str(Path.home() / ".deepagents" / "business_cofounder_api" / "docs")
        backend_root = str(Path.cwd())
        
        agent = create_agent(
            middleware=[
                AssetUploadMiddleware(
                    backend_root=backend_root,
                    docs_dir=docs_dir,
                )
            ],
        )
        ```
    """
    
    state_schema = AssetUploadState
    
    def __init__(
        self,
        *,
        backend_root: str | None = None,
        docs_dir: str | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
    ) -> None:
        """Initialize the asset upload middleware.
        
        Args:
            backend_root: Optional backend root directory for path resolution.
            docs_dir: Optional docs directory where files are stored.
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
        """
        # Check configuration during initialization
        print("[AssetUploadMiddleware] Initializing asset upload middleware...")
        
        # Check for requests package
        if requests is None:
            print("[AssetUploadMiddleware] WARNING: 'requests' package not installed!")
            print("[AssetUploadMiddleware]   Install with: pip install requests")
            print("[AssetUploadMiddleware]   Tools will not function without this package.")
        else:
            print("[AssetUploadMiddleware] ✓ 'requests' package is available")
        
        # Check environment variables
        api_base, api_key = _get_aihehuo_api_config()
        
        print("[AssetUploadMiddleware] Configuration check:")
        print(f"  AIHEHUO_API_BASE: {api_base}")
        
        if api_key:
            # Mask API key for security (show first 8 and last 4 chars)
            masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
            print(f"  AIHEHUO_API_KEY: {masked_key} (present, length: {len(api_key)})")
            print("[AssetUploadMiddleware] ✓ API key is configured")
        else:
            print("  AIHEHUO_API_KEY: NOT SET")
            print("[AssetUploadMiddleware] ⚠ WARNING: AIHEHUO_API_KEY is not configured!")
            print("[AssetUploadMiddleware]   The middleware will be initialized, but API calls will fail.")
            print("[AssetUploadMiddleware]   Set AIHEHUO_API_KEY environment variable to enable functionality.")
        
        # Set backend root and docs_dir
        if backend_root is None:
            backend_root = str(Path.cwd())
        if docs_dir is None:
            docs_dir = str(Path.home() / ".deepagents" / "business_cofounder_api" / "docs")
        
        self.backend_root = backend_root
        self.docs_dir = docs_dir
        
        # Show file path mapping information
        print("[AssetUploadMiddleware] File path mapping (virtual -> actual):")
        print(f"  Backend root directory: {backend_root}")
        
        if docs_dir and Path(docs_dir).exists():
            print(f"  Docs directory: {docs_dir}")
            print(f"    → Files written via write_file are stored here")
            print(f"    → Virtual paths like '/Users/.../file.md' map to '{docs_dir}/file.md'")
        else:
            print(f"  Docs directory: {docs_dir} (does not exist)")
        
        # Show example mappings
        print(f"  Path resolution examples:")
        print(f"    - Virtual: '/Users/yc/Documents/file.md'")
        print(f"      → Tries: '{backend_root}/file.md'")
        if docs_dir and Path(docs_dir).exists():
            print(f"      → Tries: '{docs_dir}/file.md' (most likely location)")
        print(f"    - Virtual: '/path/to/file.md'")
        print(f"      → Resolves to: '{backend_root}/path/to/file.md'")
        print(f"    - Relative: 'file.md'")
        print(f"      → Resolves to: '{Path.cwd() / 'file.md'}'")
        
        # Set system prompt (allow full override or None to use default)
        self._custom_system_prompt = system_prompt
        
        # Build tools
        if custom_tool_descriptions is None:
            custom_tool_descriptions = {}
        
        self.tools = [
            _upload_asset_tool_generator(
                custom_tool_descriptions.get("upload_asset"),
                backend_root=self.backend_root,
                docs_dir=self.docs_dir,
            ),
        ]
        
        print(f"[AssetUploadMiddleware] ✓ Initialized with {len(self.tools)} tool:")
        print(f"  - upload_asset")
        print("[AssetUploadMiddleware] Initialization complete.")
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt with asset upload tool instructions.
        
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
            system_prompt = ASSET_UPLOAD_SYSTEM_PROMPT
        
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
        """(async) Update the system prompt with asset upload tool instructions.
        
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
            system_prompt = ASSET_UPLOAD_SYSTEM_PROMPT
        
        if system_prompt:
            request = request.override(
                system_prompt=request.system_prompt + "\n\n" + system_prompt 
                if request.system_prompt 
                else system_prompt
            )
        
        return await handler(request)

