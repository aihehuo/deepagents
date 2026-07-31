"""Admission backpressure and fairness quotas (REQ-032-FIX1)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from redis import Redis
from redis.exceptions import RedisError

from apps.group_agent_api.execution.config import DurableQueueConfig
from apps.group_agent_api.execution.crypto import digest_id

_logger = logging.getLogger("uvicorn.error")

_LUA_CHECK_RESERVE = """
local gq = KEYS[1]
local uq = KEYS[2]
local grp = KEYS[3]
local gr = KEYS[4]
local pr = KEYS[5]
local cr = KEYS[6]
local max_q = tonumber(ARGV[1])
local max_r = tonumber(ARGV[2])
local max_uq = tonumber(ARGV[3])
local max_gq = tonumber(ARGV[4])
local max_pr = tonumber(ARGV[5])
local ttl = tonumber(ARGV[6])
local queued = tonumber(redis.call('GET', gq) or '0')
local running = tonumber(redis.call('GET', gr) or '0')
local user_q = tonumber(redis.call('GET', uq) or '0')
local group_q = tonumber(redis.call('GET', grp) or '0')
local prov_r = tonumber(redis.call('GET', pr) or '0')
local conv_r = tonumber(redis.call('GET', cr) or '0')
if user_q >= max_uq then
  return {'deny', 'queue_limit_exceeded', 'user_max_queued'}
end
if conv_r >= 1 then
  return {'deny', 'queue_limit_exceeded', 'conversation_running'}
end
if queued >= max_q then
  return {'deny', 'queue_saturated', 'queue_max_depth'}
end
if running >= max_r then
  return {'deny', 'queue_saturated', 'max_running'}
end
if prov_r >= max_pr then
  return {'deny', 'queue_saturated', 'provider_max_running'}
end
if group_q >= max_gq then
  return {'deny', 'queue_saturated', 'group_max_queued'}
end
redis.call('INCR', gq)
redis.call('EXPIRE', gq, ttl)
redis.call('INCR', uq)
redis.call('EXPIRE', uq, ttl)
redis.call('INCR', grp)
redis.call('EXPIRE', grp, ttl)
redis.call('INCR', KEYS[7])
return {'ok'}
"""

_LUA_START_RUNNING = """
local gq = KEYS[1]
local gr = KEYS[2]
local pr = KEYS[3]
local cr = KEYS[4]
local ver = KEYS[5]
local ttl = tonumber(ARGV[1])
local q = tonumber(redis.call('GET', gq) or '0')
if q > 0 then
  redis.call('DECR', gq)
end
redis.call('INCR', gr)
redis.call('EXPIRE', gr, ttl)
redis.call('INCR', pr)
redis.call('EXPIRE', pr, ttl)
redis.call('INCR', cr)
redis.call('EXPIRE', cr, ttl)
redis.call('INCR', ver)
return {'ok'}
"""

_LUA_FINISH = """
local uq = KEYS[1]
local grp = KEYS[2]
local gr = KEYS[3]
local pr = KEYS[4]
local cr = KEYS[5]
local ver = KEYS[6]
local was_queued = ARGV[1]
local function dec(key)
  local v = tonumber(redis.call('GET', key) or '0')
  if v > 0 then
    redis.call('DECR', key)
  end
end
if was_queued == '1' then
  dec(uq)
  dec(grp)
end
dec(gr)
dec(pr)
dec(cr)
redis.call('INCR', ver)
return {'ok'}
"""

_LUA_RETRY_WAIT = """
local gr = KEYS[1]
local pr = KEYS[2]
local cr = KEYS[3]
local gq = KEYS[4]
local uq = KEYS[5]
local grp = KEYS[6]
local ver = KEYS[7]
local ttl = tonumber(ARGV[1])
local function dec(key)
  local v = tonumber(redis.call('GET', key) or '0')
  if v > 0 then
    redis.call('DECR', key)
  end
end
dec(gr)
dec(pr)
dec(cr)
for _, key in ipairs({gq, uq, grp}) do
  redis.call('INCR', key)
  redis.call('EXPIRE', key, ttl)
end
redis.call('INCR', ver)
return {'ok'}
"""

_LUA_LEASE_EXPIRED = """
local gr = KEYS[1]
local pr = KEYS[2]
local cr = KEYS[3]
local gq = KEYS[4]
local uq = KEYS[5]
local grp = KEYS[6]
local ver = KEYS[7]
local ttl = tonumber(ARGV[1])
local has_conv = ARGV[2] == '1'
local has_user = ARGV[3] == '1'
local has_group = ARGV[4] == '1'
local function dec(key)
  local v = tonumber(redis.call('GET', key) or '0')
  if v > 0 then
    redis.call('DECR', key)
  end
end
dec(gr)
dec(pr)
if has_conv then
  dec(cr)
end
redis.call('INCR', gq)
redis.call('EXPIRE', gq, ttl)
if has_user then
  redis.call('INCR', uq)
  redis.call('EXPIRE', uq, ttl)
end
if has_group then
  redis.call('INCR', grp)
  redis.call('EXPIRE', grp, ttl)
end
redis.call('INCR', ver)
return {'ok'}
"""

_LUA_RELEASE_QUEUED = """
local ver = KEYS[4]
local function dec(key)
  local v = tonumber(redis.call('GET', key) or '0')
  if v > 0 then
    redis.call('DECR', key)
  end
end
dec(KEYS[1])
dec(KEYS[2])
dec(KEYS[3])
redis.call('INCR', ver)
return {'ok'}
"""

_LUA_RECONCILE_APPLY = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
local expected = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
if current ~= expected then
  return {'version_conflict', tostring(current)}
end
for i = 2, #KEYS do
  local desired = tonumber(ARGV[i + 1])
  if desired <= 0 then
    redis.call('DEL', KEYS[i])
  else
    redis.call('SET', KEYS[i], desired, 'EX', ttl)
  end
end
local committed = redis.call('INCR', KEYS[1])
return {'ok', tostring(committed)}
"""


@dataclass(frozen=True)
class BackpressureDecision:
    allowed: bool
    error_code: str | None = None
    http_status: int | None = None
    retry_after_s: int | None = None
    reason: str | None = None


class BackpressureController:
    """Atomic Redis quota check+reserve with TTL safety."""

    def __init__(self, client: Redis, config: DurableQueueConfig) -> None:
        self._r = client
        self._cfg = config

    def _k(self, *parts: str) -> str:
        return self._cfg.key("metrics", *parts)

    def check_and_reserve(
        self,
        *,
        user_id: str,
        group_id: str,
        conversation_id: str,
        provider: str,
    ) -> BackpressureDecision:
        """Atomically check quotas and increment queued counters when allowed."""
        user_d = digest_id(user_id)
        group_d = digest_id(group_id)
        conv_d = digest_id(conversation_id)
        provider_key = (provider or "default").strip() or "default"
        ttl = max(60, int(self._cfg.admission_timeout_s * 4))
        try:
            result = self._r.eval(
                _LUA_CHECK_RESERVE,
                7,
                self._k("queued_global"),
                self._k("queued_user", user_d),
                self._k("queued_group", group_d),
                self._k("running_global"),
                self._k("running_provider", provider_key),
                self._k("running_conv", conv_d),
                self._k("metrics_version"),
                str(self._cfg.queue_max_depth),
                str(self._cfg.max_running),
                str(self._cfg.user_max_queued),
                str(self._cfg.group_max_queued),
                str(self._cfg.provider_max_running),
                str(ttl),
            )
        except RedisError:
            return BackpressureDecision(
                allowed=False,
                error_code="queue_unavailable",
                http_status=503,
                retry_after_s=5,
                reason="redis_counters_unavailable",
            )
        if str(result[0]) == "ok":
            return BackpressureDecision(allowed=True)
        code = str(result[1])
        reason = str(result[2]) if len(result) > 2 else code
        status = 429 if code == "queue_limit_exceeded" else 503
        return BackpressureDecision(
            allowed=False,
            error_code=code,
            http_status=status,
            retry_after_s=int(self._cfg.admission_timeout_s),
            reason=reason,
        )

    def on_start_running(self, *, conversation_id: str, provider: str) -> None:
        ttl = max(60, int(self._cfg.hard_time_limit_s * 2))
        conv_d = digest_id(conversation_id)
        provider_key = (provider or "default").strip() or "default"
        try:
            self._r.eval(
                _LUA_START_RUNNING,
                5,
                self._k("queued_global"),
                self._k("running_global"),
                self._k("running_provider", provider_key),
                self._k("running_conv", conv_d),
                self._k("metrics_version"),
                str(ttl),
            )
        except RedisError as exc:
            _logger.warning("backpressure_start_failed error_type=%s", type(exc).__name__)

    def on_finish(
        self,
        *,
        user_id: str,
        group_id: str,
        conversation_id: str,
        provider: str,
        was_queued: bool = False,
    ) -> None:
        """Release running (+ optional queued) after terminal attempt completion."""
        user_d = digest_id(user_id)
        group_d = digest_id(group_id)
        conv_d = digest_id(conversation_id)
        provider_key = (provider or "default").strip() or "default"
        try:
            self._r.eval(
                _LUA_FINISH,
                6,
                self._k("queued_user", user_d),
                self._k("queued_group", group_d),
                self._k("running_global"),
                self._k("running_provider", provider_key),
                self._k("running_conv", conv_d),
                self._k("metrics_version"),
                "1" if was_queued else "0",
            )
        except RedisError as exc:
            _logger.warning("backpressure_finish_failed error_type=%s", type(exc).__name__)

    def on_retry_wait(
        self,
        *,
        user_id: str,
        group_id: str,
        conversation_id: str,
        provider: str,
    ) -> None:
        """running → retry_wait: drop running counters, restore queued occupancy."""
        user_d = digest_id(user_id)
        group_d = digest_id(group_id)
        conv_d = digest_id(conversation_id)
        provider_key = (provider or "default").strip() or "default"
        ttl = max(60, int(self._cfg.record_ttl_s))
        try:
            self._r.eval(
                _LUA_RETRY_WAIT,
                7,
                self._k("running_global"),
                self._k("running_provider", provider_key),
                self._k("running_conv", conv_d),
                self._k("queued_global"),
                self._k("queued_user", user_d),
                self._k("queued_group", group_d),
                self._k("metrics_version"),
                str(ttl),
            )
        except RedisError as exc:
            _logger.warning("backpressure_retry_wait_failed error_type=%s", type(exc).__name__)

    def on_lost_lease(self) -> None:
        """Old attempt lost ownership — do not touch counters (takeover owns them)."""
        return

    def on_lease_expired(
        self,
        *,
        user_id: str = "",
        group_id: str = "",
        conversation_id: str = "",
        provider: str = "default",
        user_id_digest: str = "",
        group_id_digest: str = "",
    ) -> None:
        """running → enqueue_failed via recovery: drop running, restore queued."""
        user_d = user_id_digest or (digest_id(user_id) if user_id else "")
        group_d = group_id_digest or (digest_id(group_id) if group_id else "")
        conv_d = digest_id(conversation_id) if conversation_id else ""
        provider_key = (provider or "default").strip() or "default"
        ttl = max(60, int(self._cfg.record_ttl_s))
        noop_conv = self._k("noop", "running_conv")
        noop_user = self._k("noop", "queued_user")
        noop_group = self._k("noop", "queued_group")
        try:
            self._r.eval(
                _LUA_LEASE_EXPIRED,
                7,
                self._k("running_global"),
                self._k("running_provider", provider_key),
                self._k("running_conv", conv_d) if conv_d else noop_conv,
                self._k("queued_global"),
                self._k("queued_user", user_d) if user_d else noop_user,
                self._k("queued_group", group_d) if group_d else noop_group,
                self._k("metrics_version"),
                str(ttl),
                "1" if conv_d else "0",
                "1" if user_d else "0",
                "1" if group_d else "0",
            )
        except RedisError as exc:
            _logger.warning("backpressure_lease_expired_failed error_type=%s", type(exc).__name__)

    def on_retry_promoted(self, *, user_id: str = "", group_id: str = "", user_id_digest: str = "", group_id_digest: str = "") -> None:
        """retry_wait → enqueue_failed: queued counters already restored at on_retry_wait."""
        return

    def release_queued_reservation(self, *, user_id: str, group_id: str) -> None:
        """Release queued counters when admission fails before ledger occupancy."""
        user_d = digest_id(user_id)
        group_d = digest_id(group_id)
        try:
            self._r.eval(
                _LUA_RELEASE_QUEUED,
                4,
                self._k("queued_global"),
                self._k("queued_user", user_d),
                self._k("queued_group", group_d),
                self._k("metrics_version"),
            )
        except RedisError as exc:
            _logger.warning("backpressure_release_failed error_type=%s", type(exc).__name__)

    def reconcile_from_ledger(self, store: object) -> bool:
        """Version-gated rebuild with atomic Lua compare-and-apply (FIX5).

        Returns True when applied; False when aborted due to concurrent mutation.
        Apply never writes counters unless metrics_version still equals start_ver.
        """
        from collections import defaultdict

        from apps.group_agent_api.execution.models import ExecutionStatus
        from apps.group_agent_api.execution.redis_store import ExecutionStore

        if not isinstance(store, ExecutionStore):
            return False

        ver_key = self._k("metrics_version")
        try:
            start_ver = int(self._r.get(ver_key) or 0)
        except (RedisError, TypeError, ValueError):
            return False

        queued_statuses = {
            ExecutionStatus.QUEUED,
            ExecutionStatus.ACCEPTED,
            ExecutionStatus.ENQUEUE_FAILED,
            ExecutionStatus.RETRY_WAIT,
        }
        desired: dict[str, int] = defaultdict(int)

        for st in (
            ExecutionStatus.QUEUED,
            ExecutionStatus.ACCEPTED,
            ExecutionStatus.ENQUEUE_FAILED,
            ExecutionStatus.RETRY_WAIT,
            ExecutionStatus.RUNNING,
        ):
            for rec in store.scan_status(st, count=200):
                if rec.status in queued_statuses:
                    desired[self._k("queued_global")] += 1
                    if rec.user_id_digest:
                        desired[self._k("queued_user", rec.user_id_digest)] += 1
                    if rec.group_id_digest:
                        desired[self._k("queued_group", rec.group_id_digest)] += 1
                elif rec.status == ExecutionStatus.RUNNING:
                    desired[self._k("running_global")] += 1
                    provider = (rec.provider or "default").strip() or "default"
                    desired[self._k("running_provider", provider)] += 1
                    if rec.conversation_id:
                        desired[self._k("running_conv", digest_id(rec.conversation_id))] += 1

        try:
            end_ver = int(self._r.get(ver_key) or 0)
        except (RedisError, TypeError, ValueError):
            return False
        if end_ver != start_ver:
            _logger.info(
                "reconcile_aborted reason=metrics_version_changed start=%s end=%s",
                start_ver,
                end_ver,
            )
            return False

        pattern = self._cfg.key("metrics", "*")
        existing_keys: list[str] = []
        try:
            for key in self._r.scan_iter(match=pattern, count=500):
                k = str(key)
                if k == ver_key:
                    continue
                existing_keys.append(k)
        except RedisError:
            return False

        for k in existing_keys:
            if k not in desired:
                desired[k] = 0

        ttl = 120
        ordered = sorted(desired.items())
        try:
            result = self._r.eval(
                _LUA_RECONCILE_APPLY,
                1 + len(ordered),
                ver_key,
                *(key for key, _ in ordered),
                str(start_ver),
                str(ttl),
                *(str(want) for _, want in ordered),
            )
            if str(result[0]) != "ok":
                _logger.info(
                    "reconcile_aborted reason=%s start=%s current=%s",
                    str(result[0]),
                    start_ver,
                    str(result[1]) if len(result) > 1 else "",
                )
                return False
            return True
        except RedisError:
            return False

