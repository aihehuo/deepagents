"""RFC 8785-compatible canonical JSON and AEAD helpers for routing state."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from apps.group_agent_api.execution.models import EncryptedPayload

_MAX_SAFE_INTEGER = 9_007_199_254_740_991


class RoutingCryptoError(RuntimeError):
    """Fail-closed routing crypto or canonicalization failure."""

    def __init__(self, code: str = "routing_snapshot_invalid") -> None:
        self.code = code
        super().__init__(code)


def _quote(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RoutingCryptoError("schema_invalid") from exc
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _canonical(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if type(value) is int:
        if abs(value) > _MAX_SAFE_INTEGER:
            raise RoutingCryptoError("schema_invalid")
        return str(value)
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RoutingCryptoError("schema_invalid")
        # V1 contract keys are ASCII. UTF-16 ordering keeps this helper correct
        # if a later closed object admits non-ASCII property names.
        ordered = sorted(value, key=lambda item: item.encode("utf-16-be"))
        return "{" + ",".join(f"{_quote(key)}:{_canonical(value[key])}" for key in ordered) + "}"
    raise RoutingCryptoError("schema_invalid")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize the frozen safe-integer JSON subset as canonical UTF-8."""
    return _canonical(value).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Return the contract lowercase prefixed SHA-256 digest."""
    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def encrypt_canonical(
    value: Any,
    *,
    key: bytes,
    key_version: str,
    aad: Mapping[str, Any],
    nonce: bytes | None = None,
) -> EncryptedPayload:
    """Encrypt canonical JSON using AES-256-GCM and explicit canonical AAD."""
    if len(key) != 32:
        raise RoutingCryptoError()
    nonce = nonce or os.urandom(12)
    if len(nonce) != 12:
        raise RoutingCryptoError()
    sealed = AESGCM(key).encrypt(nonce, canonical_json_bytes(value), canonical_json_bytes(aad))
    return EncryptedPayload(
        key_version=key_version,
        nonce_b64=base64.b64encode(nonce).decode("ascii"),
        ciphertext_b64=base64.b64encode(sealed[:-16]).decode("ascii"),
        tag_b64=base64.b64encode(sealed[-16:]).decode("ascii"),
    )


def decrypt_canonical(
    envelope: EncryptedPayload,
    *,
    keys: Mapping[str, bytes],
    aad: Mapping[str, Any],
    error_code: str,
) -> dict[str, Any]:
    """Decrypt one closed JSON object or raise a stable fail-closed code."""
    key = keys.get(envelope.key_version)
    if key is None or len(key) != 32:
        raise RoutingCryptoError(error_code)
    try:
        nonce = base64.b64decode(envelope.nonce_b64, validate=True)
        ciphertext = base64.b64decode(envelope.ciphertext_b64, validate=True)
        tag = base64.b64decode(envelope.tag_b64, validate=True)
        raw = AESGCM(key).decrypt(
            nonce,
            ciphertext + tag,
            canonical_json_bytes(aad),
        )
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - cryptographic failures share one code.
        raise RoutingCryptoError(error_code) from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise RoutingCryptoError(error_code)
    return value
