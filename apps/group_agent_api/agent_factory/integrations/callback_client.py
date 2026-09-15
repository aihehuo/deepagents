"""Callback client with SSRF allowlist, HMAC signature, and exponential backoff retries (REQ-009 / RESP-009-FIX)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import posixpath
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException

from apps.group_agent_api.agent_factory.integrations.config import (
    callback_allowed_base_urls,
    callback_hmac_secret,
    callback_max_retries,
    callback_timeout_s,
)

_logger = logging.getLogger("uvicorn.error")

HEADER_CALLBACK_VERSION = "X-GA-Callback-Version"
HEADER_CALLBACK_TS = "X-GA-Callback-Timestamp"
HEADER_CALLBACK_NONCE = "X-GA-Callback-Nonce"
HEADER_CALLBACK_SIGNATURE = "X-GA-Callback-Signature"

_CALLBACK_VERSION = "GA-CALLBACK-V1"
_ACK_KEYS = frozenset(
    {
        "ack_version",
        "run_id",
        "sent_seq",
        "ack_last_seq",
        "disposition",
        "ignored_reason",
        "terminal",
        "run_status",
        "error_code",
    }
)
_RUN_STATUSES = frozenset(
    {
        "pending",
        "calling",
        "admission_retry_wait",
        "queued",
        "running",
        "failed_start",
        "completed",
        "errored",
        "timed_out",
    }
)
_TERMINAL_RUN_STATUSES = frozenset(
    {"failed_start", "completed", "errored", "timed_out"}
)


class CallbackProtocolError(RuntimeError):
    """Fail-closed callback/ACK protocol failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CallbackRejected(CallbackProtocolError):
    """Micro authoritatively rejected a callback without advancing seq."""


class CallbackRunTerminal(CallbackProtocolError):
    """Micro reports that the run is already terminal."""

    def __init__(self, run_status: str) -> None:
        self.run_status = run_status
        super().__init__("run_terminal")


@dataclass(frozen=True)
class CallbackAckV1:
    run_id: str
    sent_seq: int
    ack_last_seq: int
    disposition: str
    ignored_reason: str | None
    terminal: bool
    run_status: str
    error_code: str | None

# Re-export: mouth reject types live in bb.brain.repair (BSD-01 P2).
from apps.group_agent_api.agent_factory.brain_repair.reject import (  # noqa: E402
    MouthIngressRejected,
    parse_mouth_reject_body,
)

__all__ = [
    "MouthIngressRejected",
    "CallbackAckV1",
    "CallbackProtocolError",
    "CallbackRejected",
    "CallbackRunTerminal",
    "parse_mouth_reject_body",
    "send_callback_event",
    "serialize_callback_body",
    "validate_and_normalize_callback_url",
]


def validate_and_normalize_callback_url(url: str) -> str:
    """SSRF protection: validate candidate callback_url against allowed base URLs with strict URL parsing & normalization."""
    raw = (url or "").strip()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail={"error": "callback_url_empty", "message": "callback_url must not be empty"},
        )

    if "\\" in raw:
        raise HTTPException(
            status_code=400,
            detail={"error": "callback_url_invalid_path", "message": "Backslashes in URL are forbidden"},
        )

    parsed = urllib.parse.urlparse(raw)

    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=400,
            detail={"error": "callback_url_invalid_scheme", "message": f"Unsupported scheme: {parsed.scheme}"},
        )

    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail={"error": "callback_url_userinfo_forbidden", "message": "Userinfo in URL is forbidden"},
        )

    if parsed.fragment:
        raise HTTPException(
            status_code=400,
            detail={"error": "callback_url_fragment_forbidden", "message": "Fragment in URL is forbidden"},
        )

    if parsed.query:
        raise HTTPException(
            status_code=400,
            detail={"error": "callback_url_query_forbidden", "message": "Query string in callback_url is forbidden"},
        )

    raw_lower = raw.lower()
    if ".." in parsed.path or "%2e%2e" in raw_lower or "%2e." in raw_lower or ".%2e" in raw_lower:
        raise HTTPException(
            status_code=400,
            detail={"error": "callback_url_path_traversal", "message": "Path traversal dot-segments forbidden"},
        )

    cand_scheme = parsed.scheme.lower()
    cand_host = (parsed.hostname or "").lower()
    if not cand_host:
        raise HTTPException(
            status_code=400,
            detail={"error": "callback_url_invalid_host", "message": "Missing host in callback_url"},
        )

    try:
        cand_port = parsed.port or (80 if cand_scheme == "http" else 443)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "callback_url_invalid_port", "message": f"Invalid port: {exc}"},
        ) from exc

    cand_path = posixpath.normpath(parsed.path or "/")

    allowed_bases = callback_allowed_base_urls()
    is_allowed = False

    for base in allowed_bases:
        base_parsed = urllib.parse.urlparse(base.strip())
        base_scheme = base_parsed.scheme.lower()
        base_host = (base_parsed.hostname or "").lower()
        if not base_host:
            continue
        try:
            base_port = base_parsed.port or (80 if base_scheme == "http" else 443)
        except ValueError:
            continue

        if (cand_scheme, cand_host, cand_port) != (base_scheme, base_host, base_port):
            continue

        base_path = posixpath.normpath(base_parsed.path or "/").rstrip("/")

        if cand_path == base_path or cand_path.startswith(base_path + "/"):
            is_allowed = True
            break

    if not is_allowed:
        _logger.warning("SSRF blocked host=%s", cand_host)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "callback_url_not_allowed",
                "message": "callback_url is not allowed by allowlist",
            },
        )

    # Reconstruct canonical URL (lowercased scheme/host, default port omitted, normalized path)
    is_default_port = (cand_scheme == "http" and cand_port == 80) or (cand_scheme == "https" and cand_port == 443)
    host_port = cand_host if is_default_port else f"{cand_host}:{cand_port}"
    return f"{cand_scheme}://{host_port}{cand_path}"


def validate_callback_url(raw: str) -> None:
    """Validate callback_url against SSRF rules."""
    validate_and_normalize_callback_url(raw)


def sign_callback_payload(
    *,
    secret: str,
    method: str,
    path: str,
    body_bytes: bytes,
    ts: str | int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Generate callback HMAC headers."""
    sec = (secret or "").strip()
    if not sec:
        raise ValueError("missing callback HMAC secret")

    ts_s = str(ts if ts is not None else int(time.time()))
    nonce_s = nonce or secrets.token_hex(16)
    body_sha = hashlib.sha256(body_bytes).hexdigest()

    canon = "\n".join([
        _CALLBACK_VERSION,
        f"method={method.upper()}",
        f"path={path}",
        f"body_sha256={body_sha}",
        f"ts={ts_s}",
        f"nonce={nonce_s}",
    ])

    sig = hmac.new(sec.encode("utf-8"), canon.encode("utf-8"), hashlib.sha256).hexdigest()

    return {
        HEADER_CALLBACK_VERSION: _CALLBACK_VERSION,
        HEADER_CALLBACK_TS: ts_s,
        HEADER_CALLBACK_NONCE: nonce_s,
        HEADER_CALLBACK_SIGNATURE: sig,
        "Content-Type": "application/json",
    }


def serialize_callback_body(envelope_dict: dict[str, Any]) -> bytes:
    """Serialize once; retry callers must retain these exact bytes."""
    return json.dumps(
        envelope_dict,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except Exception as exc:  # noqa: BLE001
        raise CallbackProtocolError("callback_ack_invalid") from exc
    if type(value) is not dict:
        raise CallbackProtocolError("callback_ack_invalid")
    return value


def _parse_ack(
    response: httpx.Response,
    *,
    expected_run_id: str,
    expected_seq: int,
) -> CallbackAckV1:
    body = _strict_json_object(response)
    if set(body) != _ACK_KEYS:
        raise CallbackProtocolError("callback_ack_invalid")
    if type(body["ack_version"]) is not int or body["ack_version"] != 1:
        raise CallbackProtocolError("callback_ack_invalid")
    if type(body["sent_seq"]) is not int or type(body["ack_last_seq"]) is not int:
        raise CallbackProtocolError("callback_ack_invalid")
    if body["sent_seq"] < 0 or body["ack_last_seq"] < 0:
        raise CallbackProtocolError("callback_ack_invalid")
    if body["run_id"] != expected_run_id or body["sent_seq"] != expected_seq:
        raise CallbackProtocolError("callback_ack_invalid")
    if type(body["terminal"]) is not bool or body["run_status"] not in _RUN_STATUSES:
        raise CallbackProtocolError("callback_ack_invalid")
    if body["terminal"] != (body["run_status"] in _TERMINAL_RUN_STATUSES):
        raise CallbackProtocolError("callback_ack_invalid")
    disposition = body["disposition"]
    ignored_reason = body["ignored_reason"]
    error_code = body["error_code"]
    if disposition not in {"accepted", "ignored", "rejected"}:
        raise CallbackProtocolError("callback_ack_invalid")
    if ignored_reason not in {None, "duplicate_seq", "run_terminal"}:
        raise CallbackProtocolError("callback_ack_invalid")
    if error_code is not None and (type(error_code) is not str or not error_code or len(error_code) > 64):
        raise CallbackProtocolError("callback_ack_invalid")
    if disposition == "ignored":
        if ignored_reason is None or error_code is not None:
            raise CallbackProtocolError("callback_ack_invalid")
    elif ignored_reason is not None:
        raise CallbackProtocolError("callback_ack_invalid")
    if disposition == "rejected":
        if error_code is None:
            raise CallbackProtocolError("callback_ack_invalid")
    elif error_code is not None:
        raise CallbackProtocolError("callback_ack_invalid")

    status = response.status_code
    if disposition == "accepted":
        if status != 200 or body["ack_last_seq"] != expected_seq:
            raise CallbackProtocolError("callback_ack_invalid")
    elif ignored_reason == "duplicate_seq":
        if status != 200 or body["terminal"] or body["ack_last_seq"] < expected_seq:
            raise CallbackProtocolError("callback_ack_invalid")
        if body["ack_last_seq"] > expected_seq:
            raise CallbackProtocolError("callback_seq_diverged")
    elif ignored_reason == "run_terminal":
        if status != 200 or not body["terminal"]:
            raise CallbackProtocolError("callback_ack_invalid")
    elif disposition == "rejected":
        if status not in {409, 422, 503}:
            raise CallbackProtocolError("callback_ack_invalid")
        if body["terminal"]:
            raise CallbackProtocolError("callback_ack_invalid")
        if status in {409, 422} and body["ack_last_seq"] >= expected_seq:
            raise CallbackProtocolError("callback_ack_invalid")
        if status == 503 and (
            error_code != "seq_gap" or body["ack_last_seq"] >= expected_seq
        ):
            raise CallbackProtocolError("callback_ack_invalid")
    return CallbackAckV1(
        run_id=body["run_id"],
        sent_seq=body["sent_seq"],
        ack_last_seq=body["ack_last_seq"],
        disposition=disposition,
        ignored_reason=ignored_reason,
        terminal=body["terminal"],
        run_status=body["run_status"],
        error_code=error_code,
    )


async def send_callback_event(
    *,
    callback_url: str,
    envelope_dict: dict[str, Any],
    secret: str | None = None,
    max_retries: int | None = None,
    timeout_s: float | None = None,
    body_bytes: bytes | None = None,
    require_ack: bool = False,
) -> bool:
    """Post callback event JSON to callback_url with exponential backoff retries."""
    validate_callback_url(callback_url)

    sec = secret if secret is not None else callback_hmac_secret()
    retries = max_retries if max_retries is not None else callback_max_retries()
    t_out = timeout_s if timeout_s is not None else callback_timeout_s()

    frozen_body = body_bytes if body_bytes is not None else serialize_callback_body(envelope_dict)
    if not isinstance(frozen_body, bytes):
        raise TypeError("callback body_bytes must be bytes")
    parsed = urllib.parse.urlparse(callback_url)
    expected_run_id = str(envelope_dict.get("run_id") or "")
    expected_seq = envelope_dict.get("seq")
    if require_ack and (not expected_run_id or type(expected_seq) is not int or expected_seq < 1):
        raise CallbackProtocolError("callback_ack_invalid")

    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            # Authentication belongs to each HTTP transport attempt. The JSON
            # bytes remain frozen while timestamp/nonce/signature are renewed.
            headers = sign_callback_payload(
                secret=sec,
                method="POST",
                path=parsed.path or "/",
                body_bytes=frozen_body,
            )
            async with httpx.AsyncClient(
                timeout=t_out,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                resp = await client.post(callback_url, content=frozen_body, headers=headers)

            if require_ack and resp.status_code in {200, 409, 422, 503}:
                ack = _parse_ack(
                    resp,
                    expected_run_id=expected_run_id,
                    expected_seq=expected_seq,
                )
                terminal_event = envelope_dict.get("event") in {"final", "error"}
                if ack.disposition == "accepted" and ack.terminal != terminal_event:
                    raise CallbackProtocolError("callback_ack_invalid")
                if ack.disposition == "accepted" or ack.ignored_reason == "duplicate_seq":
                    return True
                if ack.ignored_reason == "run_terminal":
                    raise CallbackRunTerminal(ack.run_status)
                if resp.status_code == 422 and envelope_dict.get("event") == "final":
                    raise MouthIngressRejected(
                        ack.error_code,
                        status_code=422,
                        disposition="rejected",
                        raw_body={
                            "disposition": "rejected",
                            "error_code": ack.error_code,
                        },
                    )
                raise CallbackRejected(ack.error_code or "callback_ack_invalid")

            if require_ack and resp.status_code in {401, 403}:
                raise CallbackProtocolError("callback_auth_rejected")
            if require_ack and 200 <= resp.status_code < 300:
                raise CallbackProtocolError("callback_ack_invalid")

            if 200 <= resp.status_code < 300:
                _logger.info(
                    "Callback succeeded origin=%s path=%s run_id=%s seq=%s status=%d",
                    parsed.netloc,
                    parsed.path,
                    envelope_dict.get("run_id"),
                    envelope_dict.get("seq"),
                    resp.status_code,
                )
                return True

            # BSD-01 P0/P1: mouth ingress reject — not transport; do not retry same payload.
            if resp.status_code == 422:
                body: dict[str, Any] | None
                try:
                    parsed_body = resp.json()
                    body = parsed_body if isinstance(parsed_body, dict) else None
                except Exception:  # noqa: BLE001
                    body = None
                rejected = parse_mouth_reject_body(body)
                if rejected is not None:
                    _logger.warning(
                        "Callback mouth_rejected origin=%s path=%s run_id=%s "
                        "reason_code=%s repairable_by=%s",
                        parsed.netloc,
                        parsed.path,
                        envelope_dict.get("run_id"),
                        rejected.reason_code,
                        rejected.repairable_by,
                    )
                    raise rejected

            # If 3xx redirect or non-retryable 4xx (except 429), exit immediately without retry
            if (300 <= resp.status_code < 400) or (400 <= resp.status_code < 500 and resp.status_code != 429):
                _logger.error(
                    "Callback failed non-retryable status origin=%s path=%s status=%d",
                    parsed.netloc,
                    parsed.path,
                    resp.status_code,
                )
                return False

            _logger.warning(
                "Callback attempt %d/%d failed origin=%s path=%s status=%d",
                attempt,
                retries + 1,
                parsed.netloc,
                parsed.path,
                resp.status_code,
            )

        except MouthIngressRejected:
            raise
        except CallbackProtocolError:
            raise
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "Callback attempt %d/%d exception origin=%s path=%s error_type=%s",
                attempt,
                retries + 1,
                parsed.netloc,
                parsed.path,
                type(exc).__name__,
            )

        if attempt <= retries:
            backoff = 0.5 * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)

    _logger.error(
        "Callback exhausted all retries origin=%s path=%s run_id=%s seq=%s",
        parsed.netloc,
        parsed.path,
        envelope_dict.get("run_id"),
        envelope_dict.get("seq"),
    )
    return False
