"""AES-256-GCM payload encryption for execution ledger envelopes (REQ-032)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Any, Mapping

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from apps.group_agent_api.execution.models import EncryptedPayload


class PayloadCryptoError(Exception):
    """Stable, non-secret crypto failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(raw: str) -> bytes:
    return base64.b64decode(raw.encode("ascii"), validate=True)


def build_aad(
    *,
    run_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    schema_version: int,
    mode_binding: Mapping[str, Any] | None = None,
) -> bytes:
    """Authenticated additional data binding the ciphertext to Run identity.

    Args:
        run_id: Immutable accepted run identifier.
        idempotency_key: Original request's opaque deduplication key.
        request_fingerprint: Original accepted body fingerprint.
        schema_version: Frozen request version; old encodings stay unchanged.
        mode_binding: Experimental v3-only identity/mode/conversation/episode.
    """
    canon = {
        "run_id": run_id,
        "idempotency_key": idempotency_key,
        "request_fingerprint": request_fingerprint,
        "schema_version": int(schema_version),
    }
    if schema_version == 3:
        fields = {"user_id", "execution_mode", "conversation_id", "episode_id"}
        if mode_binding is None or set(mode_binding) != fields or any(type(mode_binding[name]) is not str or not mode_binding[name] for name in fields):
            raise PayloadCryptoError("payload_binding_invalid")
        if mode_binding["execution_mode"] not in {"cafe_host", "group_agent"}:
            raise PayloadCryptoError("payload_binding_invalid")
        canon.update(mode_binding)
    elif mode_binding is not None:
        raise PayloadCryptoError("payload_binding_invalid")
    return json.dumps(canon, ensure_ascii=schema_version != 3, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encrypt_envelope(
    plaintext: Mapping[str, Any] | dict[str, Any],
    *,
    key: bytes,
    key_version: str,
    run_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    schema_version: int,
    mode_binding: Mapping[str, Any] | None = None,
) -> EncryptedPayload:
    """Encrypt a trusted request envelope with AES-256-GCM.

    Logs must never print ciphertext, nonce, tag, or plaintext.
    """
    if len(key) != 32:
        raise PayloadCryptoError("payload_key_length_invalid")
    aad = build_aad(
        run_id=run_id,
        idempotency_key=idempotency_key,
        request_fingerprint=request_fingerprint,
        schema_version=schema_version,
        mode_binding=mode_binding,
    )
    nonce = os.urandom(12)
    aesgcm = AESGCM(key)
    raw = json.dumps(plaintext, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    sealed = aesgcm.encrypt(nonce, raw, aad)
    # cryptography returns ciphertext || tag (16 bytes)
    ciphertext, tag = sealed[:-16], sealed[-16:]
    return EncryptedPayload(
        key_version=key_version,
        nonce_b64=_b64e(nonce),
        ciphertext_b64=_b64e(ciphertext),
        tag_b64=_b64e(tag),
    )


def decrypt_envelope(
    payload: EncryptedPayload,
    *,
    keys: Mapping[str, bytes],
    run_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    schema_version: int,
    mode_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decrypt envelope; unknown version or auth failure → payload_decrypt_failed."""
    key = keys.get(payload.key_version)
    if key is None:
        raise PayloadCryptoError("payload_decrypt_failed")
    if len(key) != 32:
        raise PayloadCryptoError("payload_decrypt_failed")
    try:
        nonce = _b64d(payload.nonce_b64)
        ciphertext = _b64d(payload.ciphertext_b64)
        tag = _b64d(payload.tag_b64)
    except Exception as exc:  # noqa: BLE001
        raise PayloadCryptoError("payload_decrypt_failed") from exc
    aad = build_aad(
        run_id=run_id,
        idempotency_key=idempotency_key,
        request_fingerprint=request_fingerprint,
        schema_version=schema_version,
        mode_binding=mode_binding,
    )
    aesgcm = AESGCM(key)
    try:
        raw = aesgcm.decrypt(nonce, ciphertext + tag, aad)
    except Exception as exc:  # noqa: BLE001
        raise PayloadCryptoError("payload_decrypt_failed") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PayloadCryptoError("payload_decrypt_failed") from exc
    if not isinstance(data, dict):
        raise PayloadCryptoError("payload_decrypt_failed")
    if schema_version == 3:
        request = data.get("request")
        if not isinstance(request, dict) or any(request.get(name) != value for name, value in {"run_id": run_id, "idempotency_key": idempotency_key, "request_fingerprint": request_fingerprint, "request_schema_version": schema_version, **dict(mode_binding or {})}.items()):
            raise PayloadCryptoError("payload_decrypt_failed")
    return data


def pending_callback_aad(
    *,
    key_version: str,
    run_id: str,
    request_fingerprint: str,
    seq: int,
    body_sha256: str,
) -> bytes:
    """Frozen PendingCallbackV1 AES-GCM AAD."""
    value = {
        "purpose": "group-agent-pending-callback",
        "key_version": key_version,
        "run_id": run_id,
        "request_fingerprint": request_fingerprint,
        "seq": seq,
        "body_sha256": body_sha256,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encrypt_callback_body(
    body_bytes: bytes,
    *,
    key: bytes,
    key_version: str,
    run_id: str,
    request_fingerprint: str,
    seq: int,
    body_sha256: str,
) -> EncryptedPayload:
    """Encrypt exact callback bytes before their first transport attempt."""
    if len(key) != 32 or not isinstance(body_bytes, bytes):
        raise PayloadCryptoError("pending_callback_invalid")
    nonce = os.urandom(12)
    aad = pending_callback_aad(
        key_version=key_version,
        run_id=run_id,
        request_fingerprint=request_fingerprint,
        seq=seq,
        body_sha256=body_sha256,
    )
    sealed = AESGCM(key).encrypt(nonce, body_bytes, aad)
    return EncryptedPayload(
        key_version=key_version,
        nonce_b64=_b64e(nonce),
        ciphertext_b64=_b64e(sealed[:-16]),
        tag_b64=_b64e(sealed[-16:]),
    )


def decrypt_callback_body(
    payload: EncryptedPayload,
    *,
    keys: Mapping[str, bytes],
    run_id: str,
    request_fingerprint: str,
    seq: int,
    body_sha256: str,
) -> bytes:
    """Decrypt exact pending callback bytes and verify their stored digest."""
    key = keys.get(payload.key_version)
    if key is None or len(key) != 32:
        raise PayloadCryptoError("pending_callback_invalid")
    try:
        nonce = _b64d(payload.nonce_b64)
        ciphertext = _b64d(payload.ciphertext_b64)
        tag = _b64d(payload.tag_b64)
        raw = AESGCM(key).decrypt(
            nonce,
            ciphertext + tag,
            pending_callback_aad(
                key_version=payload.key_version,
                run_id=run_id,
                request_fingerprint=request_fingerprint,
                seq=seq,
                body_sha256=body_sha256,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        raise PayloadCryptoError("pending_callback_invalid") from exc
    if "sha256:" + hashlib.sha256(raw).hexdigest() != body_sha256:
        raise PayloadCryptoError("pending_callback_invalid")
    return raw


def digest_id(value: str) -> str:
    """Stable non-reversible digest for quota / metrics keys (never log raw ids)."""
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


def digest_token(token: str) -> str:
    """SHA-256 hex digest of a lease token (store digest only)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
