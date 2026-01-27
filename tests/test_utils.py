"""Shared test utilities for integration tests."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


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
        self._callback_url: str | None = None

    def start(self) -> str:
        """Start the callback server in a background thread and return its URL."""
        if self._callback_url:
            return self._callback_url
        
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
        
        self._callback_url = f"http://127.0.0.1:{actual_port}"
        return self._callback_url

    def stop(self) -> None:
        """Stop the callback server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2.0)
        self._callback_url = None

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

    @property
    def url(self) -> str:
        """Get the callback URL (starts server if not already started)."""
        if not self._callback_url:
            return self.start()
        return self._callback_url


def extract_reply_from_callbacks(callbacks: list[dict[str, Any]]) -> str:
    """Extract the agent's reply from callback messages.
    
    Looks for callbacks with type="message" and concatenates them.
    Filters out status updates and other non-message callbacks.
    
    Args:
        callbacks: List of callback payload dictionaries
        
    Returns:
        Concatenated reply text from message callbacks
    """
    messages = []
    for callback in callbacks:
        if callback.get("type") == "message" and "message" in callback:
            message = callback["message"]
            if isinstance(message, str) and message.strip():
                messages.append(message.strip())
    
    # Join messages with spaces (handles multiple message callbacks)
    return " ".join(messages) if messages else ""
