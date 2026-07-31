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
) -> bytes:
    """Authenticated additional data binding the ciphertext to Run identity."""
    canon = {
        "run_id": run_id,
        "idempotency_key": idempotency_key,
        "request_fingerprint": request_fingerprint,
        "schema_version": int(schema_version),
    }
    return json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encrypt_envelope(
    plaintext: Mapping[str, Any] | dict[str, Any],
    *,
    key: bytes,
    key_version: str,
    run_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    schema_version: int,
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
    return data


def digest_id(value: str) -> str:
    """Stable non-reversible digest for quota / metrics keys (never log raw ids)."""
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


def digest_token(token: str) -> str:
    """SHA-256 hex digest of a lease token (store digest only)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
