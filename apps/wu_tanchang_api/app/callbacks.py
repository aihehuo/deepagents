"""Callback streaming helpers for Wu Tanchang API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from langchain_core.messages import AIMessageChunk, HumanMessage

from apps.wu_tanchang_api.app.utils import get_progress_status

_logger = logging.getLogger("uvicorn.error")

DEFAULT_ALLOWED_CALLBACK_BASE_URLS = (
    "http://host.docker.internal:3001/wu_tanchang_callbacks/",
    "http://localhost:3001/wu_tanchang_callbacks/",
    "http://127.0.0.1:3001/wu_tanchang_callbacks/",
)


class CallbackUrlError(ValueError):
    """Raised when a callback URL is outside the configured backend boundary."""


def _allowed_callback_base_urls() -> list[str]:
    raw = os.environ.get("WU_CALLBACK_ALLOWED_BASE_URLS")
    if not raw:
        return list(DEFAULT_ALLOWED_CALLBACK_BASE_URLS)
    return [base.strip() for base in raw.split(",") if base.strip()]


def sanitize_url_for_log(callback_url: str) -> str:
    """Return a URL without query/fragment so callback tokens do not land in logs."""
    parts = urlsplit(callback_url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def validate_callback_url(callback_url: str) -> str:
    """Validate callback URL against configured Rails callback base URL prefixes."""
    parts = urlsplit(callback_url)
    if parts.scheme not in {"http", "https"}:
        raise CallbackUrlError("callback URL must use http or https")
    if not parts.hostname:
        raise CallbackUrlError("callback URL must include a hostname")

    for base_url in _allowed_callback_base_urls():
        base_parts = urlsplit(base_url)
        if base_parts.scheme not in {"http", "https"} or not base_parts.hostname:
            _logger.warning(
                "[WuCallback] Ignoring invalid allowed callback base URL: %s",
                sanitize_url_for_log(base_url),
            )
            continue
        same_origin = (
            parts.scheme == base_parts.scheme
            and parts.hostname == base_parts.hostname
            and (parts.port or _default_port(parts.scheme))
            == (base_parts.port or _default_port(base_parts.scheme))
        )
        base_path = (
            base_parts.path if base_parts.path.endswith("/") else f"{base_parts.path}/"
        )
        if same_origin and parts.path.startswith(base_path):
            return callback_url
    raise CallbackUrlError("callback URL is not under an allowed callback base URL")


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        _logger.warning("Invalid %s=%r; using default %.1f", name, raw, default)
        return default
    if value <= 0:
        _logger.warning("Invalid %s=%r; using default %.1f", name, raw, default)
        return default
    return value


def callback_headers() -> dict[str, str]:
    """Build HTTP headers for callback POSTs."""
    headers = {"Content-Type": "application/json"}
    agent_key = os.environ.get("WU_CALLBACK_AGENT_KEY") or os.environ.get(
        "WU_TANCHANG_CALLBACK_TOKEN"
    )
    if agent_key:
        headers["X-Agent-Key"] = agent_key
    return headers


def _jsonable(value: Any) -> Any:
    """Convert LangChain/Pydantic objects into JSON-serializable data."""
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def invoke_callback(callback_url: str, payload: dict[str, Any]) -> bool:
    """POST a callback payload and return True when Rails requests interrupt."""
    try:
        import requests

        response = requests.post(
            callback_url,
            json=_jsonable(payload),
            headers=callback_headers(),
            timeout=30,
            allow_redirects=False,
        )
        response_text = (response.text or "").strip()
        interrupted = False
        if response_text:
            try:
                data = response.json()
                interrupted = (
                    isinstance(data, dict) and data.get("action") == "interrupt"
                )
            except (ValueError, json.JSONDecodeError):
                _logger.warning(
                    "[WuCallback] Non-JSON callback response: %s", response_text[:500]
                )
        response.raise_for_status()
        return interrupted
    except Exception as exc:  # noqa: BLE001
        _logger.error(
            "[WuCallback] Failed callback POST url=%s type=%s error=%s: %s",
            sanitize_url_for_log(callback_url),
            payload.get("type"),
            type(exc).__name__,
            str(exc),
        )
        return False


def send_heartbeat(
    callback_url: str, session_id: str, metadata: dict[str, Any]
) -> None:
    """Send a liveness heartbeat to the backend callback receiver."""
    heartbeat_url = f"{callback_url.rstrip('/')}/heartbeat"
    payload = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata,
    }
    try:
        import requests

        response = requests.post(
            heartbeat_url,
            json=_jsonable(payload),
            headers=callback_headers(),
            timeout=10,
            allow_redirects=False,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "[WuCallback] Heartbeat failed url=%s error=%s: %s",
            sanitize_url_for_log(heartbeat_url),
            type(exc).__name__,
            str(exc),
        )


def _message_id_from_chunk(message: Any) -> str | None:
    if hasattr(message, "id"):
        return getattr(message, "id", None)
    if isinstance(message, dict):
        return message.get("id")
    return None


def _extract_text_delta(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def run_async_stream_with_callback(
    *,
    agent: Any,
    user_message: str,
    thread_id: str,
    user_id: str,
    conversation_id: str,
    agent_name: str,
    metadata: dict[str, Any],
    callback_url: str,
    on_complete: Any | None = None,
) -> None:
    """Run a Wu Tanchang stream in a background thread and POST callbacks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run() -> None:
        heartbeat_stop = asyncio.Event()

        async def _heartbeat_loop() -> None:
            while not heartbeat_stop.is_set():
                await loop.run_in_executor(
                    None, send_heartbeat, callback_url, thread_id, metadata
                )
                try:
                    await asyncio.wait_for(heartbeat_stop.wait(), timeout=10)
                except asyncio.TimeoutError:
                    continue

        heartbeat_task = asyncio.create_task(_heartbeat_loop())
        base_payload = {
            "session_id": thread_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "agent_name": agent_name,
        }
        fallback_message_id = f"wu_{uuid.uuid4().hex}"
        active_message_id: str | None = None

        try:
            config = {
                "configurable": {"thread_id": thread_id},
                "metadata": {"user_id": user_id, "agent_name": agent_name, **metadata},
            }
            last_status: str | None = None
            timeout_seconds = _env_float("WU_CALLBACK_STREAM_TIMEOUT_S", 300.0)
            async with asyncio.timeout(timeout_seconds):
                async for chunk in agent.astream(
                    {"messages": [HumanMessage(content=user_message)]},
                    config=config,
                    stream_mode=["messages", "updates"],
                    subgraphs=True,
                ):
                    if not isinstance(chunk, tuple) or len(chunk) != 3:
                        continue
                    subgraph_path, stream_mode, data = chunk

                    # Skip subagent raw text chunks to avoid polluting the main chat bubble
                    if stream_mode == "messages" and subgraph_path:
                        continue

                    if (
                        stream_mode == "messages"
                        and not subgraph_path
                        and isinstance(data, tuple)
                        and data
                    ):
                        msg_chunk = data[0]
                        if isinstance(msg_chunk, AIMessageChunk):
                            text = _extract_text_delta(msg_chunk)
                            if not text:
                                continue
                            payload = {
                                **base_payload,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "type": "message",
                                "message_id": _message_id_from_chunk(msg_chunk)
                                or fallback_message_id,
                                "message": text,
                            }
                            active_message_id = payload["message_id"]
                            if invoke_callback(callback_url, payload):
                                break

                    if stream_mode == "updates":
                        status = get_progress_status(subgraph_path, stream_mode, data)
                        if status and status != last_status:
                            last_status = status
                            payload = {
                                **base_payload,
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                                "type": "status",
                                "status": status,
                            }
                            if invoke_callback(callback_url, payload):
                                return

            final_payload = {
                **base_payload,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "type": "status",
                "status": "stream_completed",
            }
            if active_message_id:
                final_payload["message_id"] = active_message_id
            invoke_callback(callback_url, final_payload)
        except Exception as exc:  # noqa: BLE001
            _logger.exception("[WuCallback] Stream failed thread_id=%s", thread_id)
            invoke_callback(
                callback_url,
                {
                    **base_payload,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "type": "status",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        finally:
            heartbeat_stop.set()
            try:
                await asyncio.wait_for(heartbeat_task, timeout=2)
            except Exception:  # noqa: BLE001
                heartbeat_task.cancel()

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()
        if on_complete is not None:
            on_complete(thread_id)


def build_callback_thread(**kwargs: Any) -> threading.Thread:
    """Build the callback stream thread without starting it."""
    on_complete = kwargs.pop("on_complete", None)

    def _target() -> None:
        run_async_stream_with_callback(on_complete=on_complete, **kwargs)

    return threading.Thread(
        target=_target,
        daemon=False,
        name=f"wu-async-{kwargs.get('thread_id', 'unknown')}",
    )


def start_callback_thread(**kwargs: Any) -> threading.Thread:
    """Start the callback stream in a non-daemon thread."""
    thread = build_callback_thread(**kwargs)
    thread.start()
    return thread
