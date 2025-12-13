from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import pytest


def _base_url() -> str:
    # Default matches your local uvicorn command.
    return os.environ.get("BC_API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")


def _http_json(method: str, url: str, payload: dict | None = None, *, timeout_s: float = 60.0) -> tuple[int, dict]:
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


@pytest.mark.timeout(180)
def test_live_server_health_chat_reset() -> None:
    """
    Live-server integration test.

    This test calls the REAL running server (default: http://127.0.0.1:8001).
    It is skipped unless you opt in, because it requires:
      - the server already running
      - model credentials configured for the running process (if /chat triggers an LLM call)
    """
    if os.environ.get("BC_API_LIVE") not in {"1", "true", "TRUE", "yes", "YES"}:
        pytest.skip("Set BC_API_LIVE=1 to run live-server integration tests.")

    base = _base_url()

    # 1) health
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
    assert isinstance(payload.get("checkpoints_path"), str)

    # 2) reset (ensure a clean thread before chatting)
    user_id = f"pytest-user-{int(time.time())}"
    conversation_id = "default"
    status, payload = _http_json(
        "POST",
        f"{base}/reset",
        {"user_id": user_id, "conversation_id": conversation_id},
        timeout_s=10.0,
    )
    assert status == 200, payload
    assert payload.get("ok") is True
    assert payload.get("thread_id") == f"bc::{user_id}::{conversation_id}"

    # 3) chat
    status, payload = _http_json(
        "POST",
        f"{base}/chat",
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "Hi! Please reply with a short acknowledgement.",
            "metadata": {"source": "pytest-live"},
        },
        timeout_s=120.0,
    )
    assert status == 200, payload
    assert payload.get("thread_id") == f"bc::{user_id}::{conversation_id}"
    reply_1 = str(payload.get("reply") or "")
    assert reply_1.strip(), f"Empty reply: {payload!r}"

    # 4) chat again (same thread) - should still return a reply
    status, payload = _http_json(
        "POST",
        f"{base}/chat",
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "Thanks. Now ask me one clarifying question about my business idea.",
            "metadata": {"source": "pytest-live"},
        },
        timeout_s=120.0,
    )
    assert status == 200, payload
    reply_2 = str(payload.get("reply") or "")
    assert reply_2.strip(), f"Empty reply: {payload!r}"


