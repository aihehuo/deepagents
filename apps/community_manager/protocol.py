"""Signed, bounded service protocol; staff scope always comes from Micro's run."""
import hashlib
import hmac
import json
import os
import time
import uuid
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError


class ProtocolError(RuntimeError):
    """Safe fixed error code, never provider data or credentials."""


def secret() -> str:
    """Secret.
    """
    value = os.environ.get("COMMUNITY_MANAGER_SECRET", "")
    if len(value.encode()) < 32:
        raise ProtocolError("manager_unconfigured")
    return value


def signature(body: bytes, timestamp: str, nonce: str) -> str:
    """Signature.

    Args:
        body: Exact bytes to sign or verify.
        timestamp: UTC epoch seconds as a string.
        nonce: One-time request UUID.
    """
    data = f"community-manager-v1\n{timestamp}\n{nonce}\n".encode() + body
    return hmac.new(secret().encode(), data, hashlib.sha256).hexdigest()


def headers(body: bytes) -> dict[str, str]:
    """Headers.

    Args:
        body: Exact bytes to sign or verify.
    """
    timestamp, nonce = str(int(time.time())), str(uuid.uuid4())
    return {"Content-Type": "application/json", "X-CM-Time": timestamp,
            "X-CM-Nonce": nonce, "X-CM-Signature": signature(body, timestamp, nonce)}


def verify(body: bytes, supplied: dict[str, str], redis_client: object) -> None:
    """Authenticate bounded input and consume its nonce in the manager Redis namespace.

    Args:
        body: Exact bytes to sign or verify.
        supplied: Lowercase signed HTTP headers.
        redis_client: Manager-only replay cache.
    """
    timestamp, nonce = supplied.get("x-cm-time", ""), supplied.get("x-cm-nonce", "")
    try:
        parsed = uuid.UUID(nonce)
        valid = len(timestamp) == 10 and timestamp.isascii() and timestamp.isdecimal() and abs(int(time.time()) - int(timestamp)) <= 60 and str(parsed) == nonce
    except (ValueError, TypeError):
        valid = False
    if not valid or not hmac.compare_digest(signature(body, timestamp, nonce), supplied.get("x-cm-signature", "")):
        raise ProtocolError("manager_unauthorized")
    if not redis_client.set("community_manager:v1:nonce:" + nonce, "1", nx=True, ex=120):
        raise ProtocolError("manager_replayed_request")


class NoRedirect(HTTPRedirectHandler):
    """Never forward a signed request or key to a redirect target."""
    def redirect_request(self, req: object, fp: object, code: int, msg: str, hdrs: object, newurl: str) -> None:
        """Redirect request.

        Args:
            req: Original urllib request.
            fp: Original response stream.
            code: HTTP redirect status.
            msg: HTTP response message.
            hdrs: HTTP response headers.
            newurl: Redirect target, which is refused.
        """
        return None


def rpc(action: str, **values: object) -> dict:
    """Call the fixed OnePenny facade using a service key AND a fresh body signature.

    Args:
        action: Fixed business operation name.
        values: Operation payload validated by Micro.
    """
    base = os.environ.get("COMMUNITY_MANAGER_OPERATIONS_URL", "")
    target = urlparse(base)
    key = os.environ.get("COMMUNITY_MANAGER_AGENT_KEY", "")
    if target.scheme not in {"http", "https"} or not target.hostname or target.username or target.password or target.query or target.fragment or target.path not in {"", "/"} or not key:
        raise ProtocolError("manager_upstream_unconfigured")
    body = json.dumps({"action": action, **values}, ensure_ascii=False, separators=(",", ":")).encode()
    if len(body) > 40_000:
        raise ProtocolError("manager_payload_too_large")
    signed = headers(body)
    signed["X-Agent-Key"] = key
    request = Request(base.rstrip("/") + "/backend/agent/community_manager/dispatch", data=body, headers=signed, method="POST")
    try:
        with build_opener(NoRedirect()).open(request, timeout=25) as response:
            raw = response.read(300_001)
            if len(raw) > 300_000:
                raise ProtocolError("manager_response_too_large")
            data = json.loads(raw)
            if not isinstance(data, dict) or data.get("success") is not True or not isinstance(data.get("result"), dict):
                raise ProtocolError("manager_invalid_response")
            return data["result"]
    except HTTPError as error:
        if action == 'tool' and values.get('name') in {'search_users', 'filter_public_candidates'} and error.code in {500, 502, 503, 504}:
            # Searches have no business writes; keep completed receipts after a failed read.
            return {'tool_error': 'candidate_search_unavailable', 'status': error.code,
                    'note': '本次只读搜索失败，不自动重试；使用已取得候选进行核验或明确报告不足。'}
        if action == 'tool' and error.code in {400, 403, 404, 409, 422, 429}:
            # Definitive business refusal is safe to explain; uncertain network/5xx writes abort.
            return {'tool_error': 'business_request_refused', 'status': error.code}
        raise ProtocolError("manager_upstream_http_" + str(error.code)) from None
