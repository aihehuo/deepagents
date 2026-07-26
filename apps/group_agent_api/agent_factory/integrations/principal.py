"""Trusted session principal for HTTP mode (REQ-007 FIX2).

HTTP mode does NOT trust bare X-GA-* headers. Calendar/BFF must send an
HMAC-SHA256 signature over a canonical principal (user_id, unionid,
token digest, ts, nonce, method, path). Verification proves the pair came
from an OAuth session holder of the shared secret — that is the
authoritative unionid ↔ user_id bind.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from fastapi import HTTPException, Request

from apps.group_agent_api.agent_factory.integrations.config import (
    integration_mode,
    principal_hmac_secret,
    principal_max_skew_s,
)

HEADER_USER_ID = "X-GA-User-Id"
HEADER_UNIONID = "X-GA-Unionid"
HEADER_USER_TOKEN = "X-GA-User-Token"
HEADER_GROUP_TOKEN = "X-GA-Group-Token"
HEADER_TS = "X-GA-Ts"
HEADER_NONCE = "X-GA-Nonce"
HEADER_SIGNATURE = "X-GA-Signature"

_CANON_VERSION = "GA-PRINCIPAL-V1"

# nonce → expiry_epoch (monotonic wall for simplicity use time.time)
_nonce_lock = threading.Lock()
_seen_nonces: OrderedDict[str, float] = OrderedDict()
_MAX_NONCES = 10_000


@dataclass(frozen=True)
class SessionPrincipal:
    user_id: str
    unionid: str | None
    user_token: str | None
    source: str  # "signed_oauth_principal" | "body_stub"
    group_token: str | None = None  # X-GA-Group-Token when signed


def _token_digest(user_token: str | None) -> str:
    tok = (user_token or "").strip()
    if not tok:
        return "-"
    return hashlib.sha256(tok.encode("utf-8")).hexdigest()


def canonical_principal_payload(
    *,
    user_id: str,
    unionid: str,
    user_token: str | None,
    group_token: str | None,
    ts: str,
    nonce: str,
    method: str,
    path: str,
) -> str:
    return "\n".join(
        [
            _CANON_VERSION,
            f"user_id={user_id}",
            f"unionid={unionid}",
            f"token_sha256={_token_digest(user_token)}",
            f"group_token_sha256={_token_digest(group_token)}",
            f"ts={ts}",
            f"nonce={nonce}",
            f"method={method.upper()}",
            f"path={path}",
        ]
    )


def sign_principal(
    *,
    user_id: str,
    unionid: str,
    user_token: str | None = None,
    group_token: str | None = None,
    ts: str | int | None = None,
    nonce: str | None = None,
    method: str = "POST",
    path: str = "/chat",
    secret: str | None = None,
) -> dict[str, str]:
    """Build signed principal headers (Calendar/BFF + tests)."""
    key = (secret if secret is not None else principal_hmac_secret()).encode("utf-8")
    if not key:
        raise ValueError("missing principal HMAC secret")
    ts_s = str(ts if ts is not None else int(time.time()))
    nonce_s = nonce or hashlib.sha256(f"{user_id}:{ts_s}:{time.time_ns()}".encode()).hexdigest()[:32]
    payload = canonical_principal_payload(
        user_id=user_id,
        unionid=unionid,
        user_token=user_token,
        group_token=group_token,
        ts=ts_s,
        nonce=nonce_s,
        method=method,
        path=path,
    )
    sig = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {
        HEADER_USER_ID: user_id,
        HEADER_UNIONID: unionid,
        HEADER_TS: ts_s,
        HEADER_NONCE: nonce_s,
        HEADER_SIGNATURE: sig,
    }
    if user_token:
        headers[HEADER_USER_TOKEN] = user_token
    if group_token:
        headers[HEADER_GROUP_TOKEN] = group_token
    return headers


def _remember_nonce(nonce: str, *, ttl_s: float) -> None:
    now = time.time()
    with _nonce_lock:
        # prune expired
        while _seen_nonces:
            oldest_n, exp = next(iter(_seen_nonces.items()))
            if exp >= now and len(_seen_nonces) < _MAX_NONCES:
                break
            if exp < now:
                _seen_nonces.popitem(last=False)
            elif len(_seen_nonces) >= _MAX_NONCES:
                _seen_nonces.popitem(last=False)
            else:
                break
        if nonce in _seen_nonces and _seen_nonces[nonce] >= now:
            raise HTTPException(
                status_code=401,
                detail={"error": "principal_nonce_replay"},
            )
        _seen_nonces[nonce] = now + ttl_s


def clear_nonce_cache() -> None:
    """Test helper."""
    with _nonce_lock:
        _seen_nonces.clear()


def _bind_map() -> dict[str, str] | None:
    """Optional extra bind table. None = not configured (HMAC is authority)."""
    raw = (os.environ.get("GROUP_AGENT_UNIONID_BIND_JSON") or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": "invalid_unionid_bind_json", "message": str(exc)},
        ) from exc
    if not isinstance(data, dict) or not data:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "invalid_unionid_bind_json",
                "message": "empty_or_not_object",
            },
        )
    return {str(k): str(v) for k, v in data.items()}


def assert_unionid_user_bind(*, unionid: str, user_id: str) -> None:
    """If optional bind table is set, enforce it. Empty/missing table is OK
    only because HMAC-signed OAuth principal already binds the pair.
    A configured but empty object is rejected at parse time.
    """
    mapping = _bind_map()
    if mapping is None:
        return
    expected = mapping.get(unionid)
    if expected is None or expected != user_id:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "unionid_user_id_mismatch",
                "unionid": unionid,
                "user_id": user_id,
            },
        )


def _verify_hmac_principal(
    request: Request,
) -> tuple[str, str, str | None, str | None]:
    secret = principal_hmac_secret()
    if not secret:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "principal_hmac_secret_missing",
                "message": "GROUP_AGENT_PRINCIPAL_HMAC_SECRET required in HTTP mode",
            },
        )

    hdr_uid = (request.headers.get(HEADER_USER_ID) or "").strip()
    hdr_union = (request.headers.get(HEADER_UNIONID) or "").strip()
    hdr_tok = (request.headers.get(HEADER_USER_TOKEN) or "").strip() or None
    hdr_group = (request.headers.get(HEADER_GROUP_TOKEN) or "").strip() or None
    ts = (request.headers.get(HEADER_TS) or "").strip()
    nonce = (request.headers.get(HEADER_NONCE) or "").strip()
    signature = (request.headers.get(HEADER_SIGNATURE) or "").strip()

    if not hdr_uid or not hdr_union or not ts or not nonce or not signature:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_signed_principal",
                "message": (
                    f"HTTP mode requires signed headers: {HEADER_USER_ID}, "
                    f"{HEADER_UNIONID}, {HEADER_TS}, {HEADER_NONCE}, {HEADER_SIGNATURE}"
                ),
            },
        )

    try:
        ts_i = int(ts)
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail={"error": "principal_ts_invalid"},
        ) from exc

    skew = principal_max_skew_s()
    now = int(time.time())
    if abs(now - ts_i) > skew:
        raise HTTPException(
            status_code=401,
            detail={"error": "principal_ts_expired", "skew_s": skew},
        )

    method = (request.method or "POST").upper()
    path = request.url.path if request.url else "/"
    payload = canonical_principal_payload(
        user_id=hdr_uid,
        unionid=hdr_union,
        user_token=hdr_tok,
        group_token=hdr_group,
        ts=ts,
        nonce=nonce,
        method=method,
        path=path,
    )
    expected = hmac.new(
        secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(
            status_code=401,
            detail={"error": "principal_signature_invalid"},
        )

    _remember_nonce(nonce, ttl_s=float(skew * 2))
    return hdr_uid, hdr_union, hdr_tok, hdr_group


def resolve_session_principal(
    request: Request,
    *,
    body_user_id: str | None,
    body_unionid: str | None = None,
    body_user_token: str | None = None,
    force_mode: str | None = None,
) -> SessionPrincipal:
    """Resolve caller identity.

    - stub: body fields (local/test only).
    - http: HMAC-signed principal headers only.
    """
    mode = (force_mode or integration_mode()).strip().lower()
    body_uid = (body_user_id or "").strip()
    body_union = (body_unionid or "").strip() or None
    body_tok = (body_user_token or "").strip() or None

    if mode != "http":
        if not body_uid:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_user_id",
                    "message": "stub requires body user_id",
                },
            )
        return SessionPrincipal(
            user_id=body_uid,
            unionid=body_union,
            user_token=body_tok,
            source="body_stub",
        )

    hdr_uid, hdr_union, hdr_tok, hdr_group = _verify_hmac_principal(request)

    if body_uid and body_uid != hdr_uid:
        raise HTTPException(
            status_code=400,
            detail={"error": "identity_injection_forbidden", "field": "user_id"},
        )
    if body_union and body_union != hdr_union:
        raise HTTPException(
            status_code=400,
            detail={"error": "identity_injection_forbidden", "field": "unionid"},
        )
    if body_tok and body_tok != hdr_tok:
        raise HTTPException(
            status_code=400,
            detail={"error": "identity_injection_forbidden", "field": "user_token"},
        )

    # Signed pair is authoritative OAuth bind; optional table is extra fail-closed.
    assert_unionid_user_bind(unionid=hdr_union, user_id=hdr_uid)
    return SessionPrincipal(
        user_id=hdr_uid,
        unionid=hdr_union,
        user_token=hdr_tok,
        group_token=hdr_group,
        source="signed_oauth_principal",
    )
