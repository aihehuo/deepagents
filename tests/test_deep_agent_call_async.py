from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest


def _base_url() -> str:
    """Base URL for the Business Co-Founder API."""
    return os.environ.get("BC_API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")


def _http_json(method: str, url: str, payload: dict | None = None, *, timeout_s: float = 60.0) -> tuple[int, dict]:
    """Make an HTTP request and return (status_code, response_dict)."""
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.URLError:
        # Re-raise for the caller to decide whether to skip/fail (e.g., server not running).
        raise
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if hasattr(e, "read") else ""
        try:
            parsed = json.loads(body) if body else {"error": body}
        except Exception:  # noqa: BLE001
            parsed = {"error": body}
        return e.code, parsed


def _create_callback_handler(callback_queue: queue.Queue) -> type[BaseHTTPRequestHandler]:
    """Create a callback handler class with the given queue."""
    
    class CallbackHandler(BaseHTTPRequestHandler):
        """Simple HTTP handler that collects callback requests in a queue."""

        def do_POST(self) -> None:
            """Handle POST requests (callbacks from the async endpoint)."""
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            
            try:
                payload = json.loads(body.decode("utf-8"))
                callback_queue.put(payload)
                print(f"Received callback: {payload}")
            except Exception as e:  # noqa: BLE001
                # Still put error in queue for test to detect
                callback_queue.put({"error": str(e), "raw_body": body.decode("utf-8", errors="replace")})
            
            # Send success response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:
            """Override to reduce noise in test output."""
            # Only log errors or if debug is enabled
            if "error" in format.lower() or os.environ.get("CALLBACK_SERVER_DEBUG"):
                super().log_message(format, *args)
    
    return CallbackHandler


class CallbackServer:
    """Simple HTTP server to receive callbacks from the async endpoint."""

    def __init__(self, port: int = 0) -> None:  # port=0 means auto-assign
        self.port = port
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.callback_queue: queue.Queue = queue.Queue()
        self.server_address: tuple[str, int] | None = None

    def start(self) -> str:
        """Start the callback server in a background thread and return its URL."""
        # Create handler class with the queue
        handler_class = _create_callback_handler(self.callback_queue)
        
        self.server = HTTPServer(("127.0.0.1", self.port), handler_class)
        self.server_address = self.server.server_address
        actual_port = self.server_address[1]
        
        def run_server() -> None:
            assert self.server is not None
            self.server.serve_forever()

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

        # Wait a moment for server to start
        time.sleep(0.1)
        
        return f"http://127.0.0.1:{actual_port}"

    def stop(self) -> None:
        """Stop the callback server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2.0)

    def get_callbacks(self, timeout_s: float = 60.0) -> list[dict[str, Any]]:
        """Collect all callbacks received within the timeout period."""
        callbacks: list[dict[str, Any]] = []
        deadline = time.time() + timeout_s
        
        while time.time() < deadline:
            try:
                # Use a short timeout so we can check the deadline
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                callback = self.callback_queue.get(timeout=min(1.0, remaining))
                callbacks.append(callback)
            except queue.Empty:
                continue
        
        return callbacks


@pytest.mark.timeout(300)
def test_deep_agent_call_async() -> None:
    """
    Test the /deep_agent/call_async endpoint with a callback server.

    This test:
    1. Starts a simple HTTP server to receive callbacks
    2. Calls the /deep_agent/call_async endpoint with the callback URL
    3. Verifies the endpoint returns immediately with session_id
    4. Waits for and verifies that callbacks are received

    This test is skipped unless BC_API_LIVE=1 is set, because it requires:
      - the server already running
      - model credentials configured for the running process
    """
    if os.environ.get("BC_API_LIVE") not in {"1", "true", "TRUE", "yes", "YES"}:
        pytest.skip("Set BC_API_LIVE=1 to run live-server integration tests.")

    base = _base_url()

    # Verify server is reachable
    try:
        status, payload = _http_json("GET", f"{base}/health", timeout_s=10.0)
    except urllib.error.URLError as e:
        strict = os.environ.get("BC_API_LIVE_STRICT") in {"1", "true", "TRUE", "yes", "YES"}
        msg = (
            f"Live server not reachable at {base} ({e}).\n\n"
            "Start it in another terminal, e.g.:\n"
            '  PYTHONPATH="libs/deepagents:libs/deepagents-cli" '
            "uvicorn apps.business_cofounder_api.app:app --host 0.0.0.0 --port 8001\n\n"
            "Or set BC_API_BASE_URL to point at the running server."
        )
        if strict:
            pytest.fail(msg)
        pytest.skip(msg)
    
    assert status == 200, payload
    assert payload.get("status") == "ok"

    # Start callback server
    callback_server = CallbackServer()
    callback_url = callback_server.start()
    
    try:
        # Prepare test data - use a message that will trigger tool calls (todo creation and file operations)
        user_id = f"pytest-async-{int(time.time())}"
        conversation_id = "default"
        # Use a business idea that will trigger the agent to create todos and use tools
        test_message = """I have a business idea: An AI-powered personal finance app that helps people track expenses and save money.
        
Target customers: Young professionals aged 25-35 who struggle with budgeting.
Value proposition: Automatically categorize expenses and provide personalized saving tips.

Please help me develop this idea. Create a todo list and start working through the first few steps."""

        # Call the async endpoint
        status, payload = _http_json(
            "POST",
            f"{base}/deep_agent/call_async",
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message": test_message,
                "callback": callback_url,
                "metadata": {"source": "pytest-async-test"},
            },
            timeout_s=30.0,
        )

        # Verify immediate response
        assert status == 200, payload
        assert payload.get("success") is True, payload
        session_id = payload.get("session_id")
        assert session_id is not None, payload
        assert session_id == f"bc::{user_id}::{conversation_id}", payload
        assert isinstance(payload.get("message"), str), payload

        # Wait for callbacks (give it time for the agent to start streaming)
        # The actual timeout depends on how long the agent takes to process
        callback_timeout = float(os.environ.get("BC_API_ASYNC_CALLBACK_TIMEOUT_S", "120.0"))
        callbacks = callback_server.get_callbacks(timeout_s=callback_timeout)

        # Verify we received at least some callbacks
        assert len(callbacks) > 0, (
            f"No callbacks received within {callback_timeout}s. "
            f"This might indicate the background thread didn't start or the callback URL is unreachable."
        )

        # Track different types of callbacks we receive
        assistant_messages = []
        tool_call_messages = []
        tool_completed_messages = []
        processing_messages = []
        error_messages = []

        # Verify callback structure and categorize messages
        for callback in callbacks:
            # Each callback should have either "message" (assistant content) or "status" (status update)
            assert "message" in callback or "status" in callback, (
                f"Callback missing both 'message' and 'status' keys: {callback}"
            )
            
            # Handle assistant messages (extracted content, no "Assistant:" prefix)
            if "message" in callback:
                message = callback["message"]
                assert isinstance(message, str), f"Callback message should be a string, got: {type(message)} - {message}"
                assert len(message) > 0, "Callback message should not be empty"
                assistant_messages.append(message)
                print(f"Received assistant message: {message}")
            
            # Handle status updates (tool calls, completions, errors, processing, etc.)
            if "status" in callback:
                status = callback["status"]
                assert isinstance(status, str), f"Callback status should be a string, got: {type(status)} - {status}"
                assert len(status) > 0, "Callback status should not be empty"
                
                # Categorize status messages
                if status.startswith("Error:"):
                    error_messages.append(status)
                    print(f"Received error callback: {status}")
                elif "calling" in status.lower() and ("tool" in status.lower() or any(
                    tool in status.lower() for tool in ["write_file", "read_file", "write_todos", "execute", "glob", "grep"]
                )):
                    tool_call_messages.append(status)
                    print(f"Received tool call status: {status}")
                elif status.startswith("Tool ") and "completed" in status:
                    tool_completed_messages.append(status)
                    print(f"Received tool completed status: {status}")
                elif "processing" in status.lower():
                    processing_messages.append(status)
                    print(f"Received processing status: {status}")
                else:
                    print(f"Received status update: {status}")

        print(f"\nCallback Summary:")
        print(f"  Total callbacks: {len(callbacks)}")
        print(f"  Assistant messages: {len(assistant_messages)}")
        print(f"  Tool call messages: {len(tool_call_messages)}")
        print(f"  Tool completed messages: {len(tool_completed_messages)}")
        print(f"  Processing messages: {len(processing_messages)}")
        print(f"  Error messages: {len(error_messages)}")

        # Verify we received tool-related callbacks (this is the key test)
        # We should see at least some tool calls or tool completions
        total_tool_related = len(tool_call_messages) + len(tool_completed_messages)
        assert total_tool_related > 0, (
            f"Expected to receive tool call or tool completed messages, but got none. "
            f"Received {len(callbacks)} total callbacks. "
            f"This might indicate the agent didn't trigger any tool calls, or the callback message format is incorrect."
        )
        
        # Verify we received some assistant responses
        assert len(assistant_messages) > 0 or len(processing_messages) > 0, (
            f"Expected to receive assistant or processing messages, but got none. "
            f"Received {len(callbacks)} total callbacks."
        )

        print(f"Successfully received {len(callbacks)} callbacks, including {total_tool_related} tool-related messages")

    finally:
        callback_server.stop()


@pytest.mark.timeout(60)
def test_deep_agent_call_async_immediate_response() -> None:
    """
    Test that /deep_agent/call_async returns immediately even if callback server is unreachable.

    This verifies the endpoint doesn't block waiting for callback connectivity.
    """
    if os.environ.get("BC_API_LIVE") not in {"1", "true", "TRUE", "yes", "YES"}:
        pytest.skip("Set BC_API_LIVE=1 to run live-server integration tests.")

    base = _base_url()

    # Verify server is reachable
    try:
        status, _ = _http_json("GET", f"{base}/health", timeout_s=10.0)
        assert status == 200
    except urllib.error.URLError:
        pytest.skip("Server not reachable")

    # Use an unreachable callback URL
    user_id = f"pytest-async-unreachable-{int(time.time())}"
    conversation_id = "default"
    
    # Call with unreachable callback URL (should still return immediately)
    status, payload = _http_json(
        "POST",
        f"{base}/deep_agent/call_async",
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "Test message",
            "callback": "http://127.0.0.1:99999/unreachable",  # Unreachable port
        },
        timeout_s=30.0,
    )

    # Should still return success immediately (the callback failures happen in background)
    assert status == 200, payload
    assert payload.get("success") is True, payload
    assert payload.get("session_id") == f"bc::{user_id}::{conversation_id}", payload

