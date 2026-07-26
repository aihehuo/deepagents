"""Callback client with SSRF allowlist, HMAC signature, and exponential backoff retries (REQ-009 / RESP-009-FIX)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import posixpath
import time
import urllib.parse
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
    nonce_s = nonce or hashlib.sha256(f"{ts_s}:{time.time_ns()}".encode()).hexdigest()[:32]
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


async def send_callback_event(
    *,
    callback_url: str,
    envelope_dict: dict[str, Any],
    secret: str | None = None,
    max_retries: int | None = None,
    timeout_s: float | None = None,
) -> bool:
    """Post callback event JSON to callback_url with exponential backoff retries."""
    validate_callback_url(callback_url)

    sec = secret if secret is not None else callback_hmac_secret()
    retries = max_retries if max_retries is not None else callback_max_retries()
    t_out = timeout_s if timeout_s is not None else callback_timeout_s()

    body_bytes = json.dumps(envelope_dict, ensure_ascii=False).encode("utf-8")
    parsed = urllib.parse.urlparse(callback_url)
    headers = sign_callback_payload(
        secret=sec,
        method="POST",
        path=parsed.path or "/",
        body_bytes=body_bytes,
    )

    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            async with httpx.AsyncClient(timeout=t_out, follow_redirects=False) as client:
                resp = await client.post(callback_url, content=body_bytes, headers=headers)

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
