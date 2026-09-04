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

from langchain_core.messages import HumanMessage

from pathlib import Path
from apps.wu_tanchang_api.app.stream_history import (
    extract_main_agent_ai_delta,
    normalize_message_text,
    should_emit_streamed_ai_text,
    snapshot_pre_turn_ai,
)
from apps.wu_tanchang_api.app.user_progress import (
    reset_progress_sink,
    set_progress_sink,
)
from apps.wu_tanchang_api.app.utils import (
    aget_agent_state,
    get_progress_status,
    get_agent_checkpointer,
)
from apps.wu_tanchang_api.agent_factory.utils import get_workspace_agent_id

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
    callback_url: str,
    session_id: str,
    metadata: dict[str, Any],
    *,
    elapsed_s: int = 0,
) -> None:
    """Send a liveness heartbeat to the backend callback receiver."""
    heartbeat_url = f"{callback_url.rstrip('/')}/heartbeat"
    payload = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "elapsed_s": elapsed_s,
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
        from apps.wu_tanchang_api.agent_factory.agent import (
            register_active_agent,
            unregister_active_agent,
        )

        register_active_agent(thread_id, agent)
        heartbeat_stop = asyncio.Event()
        heartbeat_started = datetime.utcnow()

        async def _heartbeat_loop() -> None:
            while not heartbeat_stop.is_set():
                elapsed_s = int(
                    (datetime.utcnow() - heartbeat_started).total_seconds()
                )
                await loop.run_in_executor(
                    None,
                    lambda elapsed=elapsed_s: send_heartbeat(
                        callback_url, thread_id, metadata, elapsed_s=elapsed
                    ),
                )
                try:
                    await asyncio.wait_for(heartbeat_stop.wait(), timeout=10)
                except asyncio.TimeoutError:
                    continue

        heartbeat_task = asyncio.create_task(_heartbeat_loop())
        workspace_name = getattr(agent, "workspace_name", "workspace")
        backend_root = Path(__file__).resolve().parent.parent
        workspace_path = backend_root / workspace_name
        parsed_id = get_workspace_agent_id(workspace_path)
        if parsed_id == "andy01":
            payload_agent_name = "default"
        elif parsed_id == "andy01_owner":
            payload_agent_name = "owner"
        else:
            payload_agent_name = parsed_id

        base_payload = {
            "session_id": thread_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "agent_name": payload_agent_name,
        }
        fallback_message_id = f"wu_{uuid.uuid4().hex}"
        active_message_id: str | None = None
        last_status: str | None = None

        def _emit_user_status(status: str) -> None:
            nonlocal last_status
            if not status or status == last_status:
                return
            last_status = status
            invoke_callback(
                callback_url,
                {
                    **base_payload,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "type": "status",
                    "status": status,
                },
            )

        sink_token = set_progress_sink(_emit_user_status)
        try:
            turn_id = f"{thread_id}::{uuid.uuid4().hex}"
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "deepagents_turn_id": turn_id,
                },
                "metadata": {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "agent_name": agent_name,
                    "callback_url": callback_url,
                    **metadata,
                },
            }
            timeout_seconds = _env_float("WU_CALLBACK_STREAM_TIMEOUT_S", 300.0)
            _emit_user_status("已收到，正在思考…")
            agent_state = await aget_agent_state(agent, config)
            pre_messages = (
                agent_state.values.get("messages", [])
                if agent_state and agent_state.values
                else []
            )
            pre_turn_ai = snapshot_pre_turn_ai(pre_messages)
            msg_count_before = len(pre_messages)
            skipped_history = 0
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

                    if stream_mode == "messages":
                        if (
                            not subgraph_path
                            and isinstance(data, tuple)
                            and data
                        ):
                            msg_chunk = data[0]
                            replay_text = normalize_message_text(msg_chunk)
                            if replay_text and not should_emit_streamed_ai_text(
                                msg_chunk, pre_turn_ai
                            ):
                                skipped_history += 1
                        delta = extract_main_agent_ai_delta(
                            subgraph_path, stream_mode, data, pre_turn_ai
                        )
                        if delta is None:
                            continue
                        text, chunk_message_id = delta
                        payload = {
                            **base_payload,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "type": "message",
                            "message_id": chunk_message_id or fallback_message_id,
                            "message": text,
                        }
                        active_message_id = payload["message_id"]
                        if invoke_callback(callback_url, payload):
                            break

                    if stream_mode == "updates":
                        status = get_progress_status(
                            subgraph_path,
                            stream_mode,
                            data,
                            workspace_name=workspace_name,
                        )
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

            # Calendar UI uses /call_async (not /chat). Persist prep here when
            # the model printed 【会议准备】 but skipped save_meeting_prep.
            post_state = await aget_agent_state(agent, config)
            post_messages = (
                list(post_state.values.get("messages", []))
                if post_state and post_state.values
                else []
            )
            from apps.wu_tanchang_api.agent_factory.meeting_prep_persist import (
                ensure_meeting_prep_persisted,
            )

            ensure_meeting_prep_persisted(
                new_messages=post_messages[msg_count_before:],
                config=config,
            )

            final_payload = {
                **base_payload,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "type": "status",
                "status": "stream_completed",
            }
            if active_message_id:
                final_payload["message_id"] = active_message_id
            if skipped_history:
                _logger.info(
                    "[WuCallback] skipped checkpoint-history stream chunks "
                    "thread_id=%s count=%d pre_ai_ids=%d",
                    thread_id,
                    skipped_history,
                    len(pre_turn_ai.ids),
                )
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
            reset_progress_sink(sink_token)
            unregister_active_agent(thread_id)
            checkpointer = get_agent_checkpointer(agent)
            if checkpointer is not None and hasattr(checkpointer, "flush"):
                checkpointer.flush()
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
