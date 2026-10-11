"""Signed, bounded service protocol; staff scope always comes from Micro's run."""
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError


BUSINESS_ERRORS = frozenset(['activity_data_unavailable', 'assessment_user_mismatch', 'community_service_disabled', 'contact_binding_conflict', 'context_revision_conflict', 'context_stale', 'customer_assessment_disabled', 'customer_assessment_input_too_large', 'customer_assessment_tools_forbidden', 'evidence_role_mismatch', 'group_action_not_supported', 'invalid_actor', 'invalid_context', 'invalid_customer_evaluation', 'invalid_evidence', 'invalid_evidence_quote', 'invalid_manager_action', 'invalid_manager_card', 'invalid_manager_event', 'invalid_manager_message', 'invalid_parameters', 'invalid_plan', 'invalid_plan_window', 'invalid_strategy_state', 'invalid_strategy_version', 'invalid_task_window', 'manager_binding_unavailable', 'manager_busy', 'manager_card_forbidden', 'manager_card_stale', 'manager_card_unconfigured', 'manager_cohort_too_large', 'manager_contact_lead_forbidden', 'manager_customer_blocked', 'manager_disabled', 'manager_enqueue_failed', 'manager_events_disabled', 'manager_feedback_not_due', 'manager_follow_up_conflict', 'manager_follow_up_missing', 'manager_forbidden', 'manager_global_write_forbidden', 'manager_legacy_card_disabled', 'manager_memory_conflict', 'manager_need_unknown', 'manager_notifications_disabled', 'manager_owner_capacity', 'manager_owner_forbidden', 'manager_owner_required', 'manager_plan_unavailable', 'manager_reason_required', 'manager_recipient_forbidden', 'manager_replayed_request', 'manager_request_conflict', 'manager_run_closed', 'manager_run_not_found', 'manager_run_not_running', 'manager_run_unauthorized', 'manager_scope_unconfigured', 'manager_source_changed', 'manager_task_closed', 'manager_task_not_found', 'manager_task_owned', 'manager_task_revision_conflict', 'manager_tool_budget_exceeded', 'manager_tool_forbidden', 'manager_tool_response_too_large', 'manager_unauthorized', 'manager_unconfigured', 'manager_upstream_unconfigured', 'manager_user_out_of_scope', 'manager_user_paused', 'manager_user_temporarily_blocked', 'not_found', 'payload_too_large', 'plan_version_conflict', 'request_id_conflict', 'source_changed', 'source_unavailable', 'strategy_changed', 'strategy_expired', 'strategy_review_forbidden', 'strategy_revision_conflict', 'too_many_evidence_refs', 'user_unavailable'])
LOGGER = logging.getLogger(__name__)


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
        if action == 'tool' and values.get('name') in {'search_users', 'search_recent_users', 'filter_public_candidates'} and error.code in {500, 502, 503, 504}:
            # Searches have no business writes; keep completed receipts after a failed read.
            return {'tool_error': 'candidate_search_unavailable', 'status': error.code,
                    'note': '本次只读搜索失败，不自动重试；使用已取得候选进行核验或明确报告不足。'}
        if action == 'tool' and error.code in {400, 403, 404, 409, 422, 429}:
            # Definitive business refusal is safe to explain; uncertain network/5xx writes abort.
            code = 'business_request_refused'
            try:
                raw = error.read(4097)
                data = json.loads(raw) if len(raw) <= 4096 else None
                candidate = data.get('error') if isinstance(data, dict) and data.get('success') is False else None
                if isinstance(candidate, str) and candidate in BUSINESS_ERRORS:
                    code = candidate
            except (ValueError, TypeError, OSError):
                pass
            # Never log raw bodies, tool arguments, provider messages or credentials.
            name = values.get('name')
            safe_name = name if isinstance(name, str) and name.isascii() and name.replace('_', '').isalnum() and len(name) <= 64 else 'unknown'
            LOGGER.warning('manager_tool_refused tool=%s status=%s code=%s', safe_name, error.code, code)
            return {'tool_error': code, 'status': error.code}
        raise ProtocolError("manager_upstream_http_" + str(error.code)) from None
