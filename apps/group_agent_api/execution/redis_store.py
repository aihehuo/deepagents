"""Redis execution ledger with Lua CAS transitions (REQ-032)."""

from __future__ import annotations

import json
import logging
import secrets
import uuid
from dataclasses import dataclass
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from apps.group_agent_api.execution.config import DurableQueueConfig
from apps.group_agent_api.execution.crypto import digest_token
from apps.group_agent_api.execution.models import (
    EncryptedPayload,
    ExecutionRecord,
    ExecutionStatus,
    LeaseClaim,
    TERMINAL_STATUSES,
)

_logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class ClaimOutcome:
    """Result of an atomic claim attempt."""

    kind: str
    claim: LeaseClaim | None = None
    detail: str | None = None

# create-or-read with binding conflict detection
_LUA_CREATE_OR_GET = """
local run_key = KEYS[1]
local idem_key = KEYS[2]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local run_id = ARGV[3]
local idem = ARGV[4]
local schema = ARGV[5]
local fp = ARGV[6]
local qschema = ARGV[7]
local fields_json = ARGV[8]

if redis.call('EXISTS', idem_key) == 1 then
  local existing_run = redis.call('GET', idem_key)
  if existing_run ~= run_id then
    return {'conflict', 'idempotency_conflict'}
  end
  if redis.call('EXISTS', run_key) == 1 then
    local e_schema = redis.call('HGET', run_key, 'request_schema_version')
    local e_fp = redis.call('HGET', run_key, 'request_fingerprint')
    local e_idem = redis.call('HGET', run_key, 'idempotency_key')
    if e_schema ~= schema or e_fp ~= fp or e_idem ~= idem then
      return {'conflict', 'idempotency_conflict'}
    end
    return {'hit', run_id}
  end
end

if redis.call('EXISTS', run_key) == 1 then
  local e_idem = redis.call('HGET', run_key, 'idempotency_key')
  if e_idem ~= idem then
    return {'conflict', 'run_binding_conflict'}
  end
  local e_schema = redis.call('HGET', run_key, 'request_schema_version')
  local e_fp = redis.call('HGET', run_key, 'request_fingerprint')
  if e_schema ~= schema or e_fp ~= fp then
    return {'conflict', 'idempotency_conflict'}
  end
  redis.call('SET', idem_key, run_id, 'EX', ttl)
  return {'hit', run_id}
end

local fields = cjson.decode(fields_json)
for k, v in pairs(fields) do
  redis.call('HSET', run_key, k, v)
end
redis.call('EXPIRE', run_key, ttl)
redis.call('SET', idem_key, run_id, 'EX', ttl)
return {'created', run_id}
"""

# FIX2: only exact expected → queued. Already-queued is NOT ok (blocks multi-publish).
_LUA_MARK_QUEUED = """
local run_key = KEYS[1]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local expected = ARGV[3]
local delivery_id = ARGV[4]
local status = redis.call('HGET', run_key, 'status')
if not status then
  return {'missing'}
end
if status == 'queued' then
  return {'already_queued', redis.call('HGET', run_key, 'last_delivery_id') or ''}
end
if status ~= expected then
  return {'conflict', status}
end
redis.call('HSET', run_key, 'status', 'queued')
redis.call('HSET', run_key, 'queued_at', tostring(now))
if delivery_id ~= '' then
  redis.call('HSET', run_key, 'last_delivery_id', delivery_id)
end
redis.call('HDEL', run_key, 'last_error_code')
redis.call('EXPIRE', run_key, ttl)
return {'ok', 'queued'}
"""

# FIX2: admission-only — accepted → enqueue_failed (never touch running/terminal).
_LUA_MARK_ACCEPTED_ENQUEUE_FAILED = """
local run_key = KEYS[1]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local err = ARGV[3]
local status = redis.call('HGET', run_key, 'status')
if not status then
  return {'missing'}
end
if status ~= 'accepted' and status ~= 'enqueue_failed' then
  return {'conflict', status}
end
redis.call('HSET', run_key, 'status', 'enqueue_failed')
redis.call('HSET', run_key, 'last_error_code', err)
redis.call('EXPIRE', run_key, ttl)
return {'ok', 'enqueue_failed'}
"""

# FIX2: DLQ replay — dead_lettered → enqueue_failed with operator + replay_id audit.
_LUA_MARK_DLQ_REPLAY = """
local run_key = KEYS[1]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local err = ARGV[3]
local operator_id = ARGV[4]
local replay_id = ARGV[5]
local reason = ARGV[6]
local status = redis.call('HGET', run_key, 'status')
if not status then
  return {'missing'}
end
if status ~= 'dead_lettered' then
  return {'conflict', status}
end
if operator_id == '' or replay_id == '' then
  return {'missing_operator'}
end
local audit_raw = redis.call('HGET', run_key, 'replay_audit_json') or '[]'
local ok, audit = pcall(cjson.decode, audit_raw)
if not ok then
  audit = {}
end
audit[#audit+1] = {
  operator_id = operator_id,
  reason = reason,
  replay_id = replay_id,
  at = tostring(now)
}
redis.call('HSET', run_key, 'replay_audit_json', cjson.encode(audit))
redis.call('HSET', run_key, 'status', 'enqueue_failed')
redis.call('HSET', run_key, 'last_error_code', err)
redis.call('HDEL', run_key, 'terminal_fence')
redis.call('HDEL', run_key, 'finished_at')
redis.call('HDEL', run_key, 'lease_owner')
redis.call('HDEL', run_key, 'lease_token_digest')
redis.call('HDEL', run_key, 'lease_expires_at')
redis.call('HDEL', run_key, 'current_attempt_id')
redis.call('EXPIRE', run_key, ttl)
return {'ok', 'enqueue_failed'}
"""

# FIX2: atomic recovery delivery reservation — at most one publisher per run cycle.
_LUA_CLAIM_RECOVERY_DELIVERY = """
local run_key = KEYS[1]
local claim_key = KEYS[2]
local now = tonumber(ARGV[1])
local claim_ttl = tonumber(ARGV[2])
local expected = ARGV[3]
local delivery_id = ARGV[4]
local status = redis.call('HGET', run_key, 'status')
if not status then
  return {'missing'}
end
if status ~= expected then
  return {'conflict', status}
end
local ok = redis.call('SET', claim_key, delivery_id, 'NX', 'EX', claim_ttl)
if not ok then
  local existing = redis.call('GET', claim_key)
  if existing == delivery_id then
    return {'ok', delivery_id}
  end
  return {'busy', existing or ''}
end
return {'ok', delivery_id}
"""

# FIX3: poison only from non-running recoverable states (never revoke active lease).
_LUA_POISON_TO_DLQ = """
local run_key = KEYS[1]
local conv_key = KEYS[2]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local err = ARGV[3]
local fence = ARGV[4]
local expected = ARGV[5]
local status = redis.call('HGET', run_key, 'status')
if not status then
  return {'missing'}
end
if status == 'succeeded' or status == 'failed' or status == 'dead_lettered' then
  return {'terminal', status}
end
if status == 'running' then
  return {'running_protected', status}
end
if status ~= 'queued' and status ~= 'accepted' and status ~= 'enqueue_failed'
   and status ~= 'retry_wait' then
  return {'conflict', status}
end
if expected ~= '' and status ~= expected then
  return {'conflict', status}
end
if redis.call('HGET', run_key, 'terminal_fence') then
  return {'already_terminal'}
end
redis.call('HSET', run_key, 'status', 'dead_lettered')
redis.call('HSET', run_key, 'finished_at', tostring(now))
redis.call('HSET', run_key, 'terminal_fence', fence)
redis.call('HSET', run_key, 'last_error_code', err)
redis.call('HDEL', run_key, 'lease_owner')
redis.call('HDEL', run_key, 'lease_token_digest')
redis.call('HDEL', run_key, 'lease_expires_at')
redis.call('HDEL', run_key, 'current_attempt_id')
redis.call('HDEL', run_key, 'next_attempt_at')
redis.call('EXPIRE', run_key, ttl)
local run_id = redis.call('HGET', run_key, 'run_id')
local conv_val = redis.call('GET', conv_key)
if conv_val == run_id then
  redis.call('DEL', conv_key)
end
return {'ok', 'dead_lettered'}
"""

# FIX3: compare-and-delete recovery claim (atomic).
_LUA_RELEASE_RECOVERY_CLAIM = """
local claim_key = KEYS[1]
local delivery_id = ARGV[1]
local cur = redis.call('GET', claim_key)
if not cur then
  return {'missing'}
end
if cur ~= delivery_id then
  return {'mismatch', cur}
end
redis.call('DEL', claim_key)
return {'ok'}
"""

# FIX4: user×group global monotonic fencing CAS (fail-closed when fenced).
_LUA_CAS_PROFILE_FENCE = """
local fence_key = KEYS[1]
local meta_key = KEYS[2]
local run_key = KEYS[3]
local epoch_key = KEYS[4]
local token = tonumber(ARGV[1])
local attempt_id = ARGV[2]
local owner_digest = ARGV[3]
local now = tonumber(ARGV[4])
local run_id = ARGV[5]
local user_digest = ARGV[6]
local group_digest = ARGV[7]
local audit_ttl = tonumber(ARGV[8])
if not token or token <= 0 then
  return {'fence_token_required'}
end
local status = redis.call('HGET', run_key, 'status')
if status ~= 'running' then
  return {'fence_not_running', status or 'missing'}
end
if redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'fence_attempt_mismatch'}
end
if redis.call('HGET', run_key, 'lease_token_digest') ~= owner_digest then
  return {'fence_token_mismatch'}
end
if redis.call('HGET', run_key, 'user_id_digest') ~= user_digest
   or redis.call('HGET', run_key, 'group_id_digest') ~= group_digest then
  return {'fence_identity_mismatch'}
end
local expires = redis.call('HGET', run_key, 'lease_expires_at')
if not expires or tonumber(expires) < now then
  return {'fence_lease_expired'}
end
local stored = tonumber(redis.call('HGET', run_key, 'fencing_token') or '0')
if token ~= stored then
  return {'fence_epoch_mismatch', tostring(stored)}
end
local issued = tonumber(redis.call('GET', epoch_key) or '0')
if issued <= 0 then
  return {'fence_epoch_missing'}
end
if token ~= issued then
  return {'stale', tostring(issued)}
end
local current = tonumber(redis.call('GET', fence_key) or '0')
if current > 0 then
  if token < current then
    return {'stale', tostring(current)}
  end
  if token == current then
    local meta = redis.call('GET', meta_key) or ''
    -- same epoch only allows same attempt (+ run) idempotent replay
    local expected = run_id .. ':' .. attempt_id
    if meta ~= '' and meta ~= expected then
      return {'fence_attempt_conflict', meta}
    end
  end
end
redis.call('SET', fence_key, tostring(token), 'EX', audit_ttl)
redis.call('SET', meta_key, run_id .. ':' .. attempt_id, 'EX', audit_ttl)
return {'ok', tostring(token)}
"""

_LUA_CLAIM = """
local run_key = KEYS[1]
local conv_key = KEYS[2]
local epoch_key = KEYS[3]
local now = tonumber(ARGV[1])
local lease_ttl = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local owner = ARGV[4]
local token_digest = ARGV[5]
local attempt_id = ARGV[6]
local status = redis.call('HGET', run_key, 'status')
if not status then
  return {'missing'}
end
if status == 'succeeded' or status == 'failed' or status == 'dead_lettered' then
  return {'terminal', status}
end
-- FIX1: only queued → running. Never claim accepted/enqueue_failed/retry_wait.
if status ~= 'queued' then
  return {'not_queued', status}
end
if redis.call('EXISTS', conv_key) == 1 then
  local conv_owner = redis.call('GET', conv_key)
  local conv_run = redis.call('HGET', run_key, 'run_id')
  if conv_owner and conv_owner ~= conv_run then
    return {'conversation_busy', conv_owner}
  end
end
local new_expires = now + lease_ttl
redis.call('HSET', run_key, 'status', 'running')
redis.call('HSET', run_key, 'lease_owner', owner)
redis.call('HSET', run_key, 'lease_token_digest', token_digest)
redis.call('HSET', run_key, 'lease_expires_at', tostring(new_expires))
redis.call('HSET', run_key, 'heartbeat_at', tostring(now))
redis.call('HSET', run_key, 'current_attempt_id', attempt_id)
redis.call('HINCRBY', run_key, 'attempt_count', 1)
-- FIX4: user×group global monotonic epoch (not per-run restart at 1).
local fencing = redis.call('INCR', epoch_key)
redis.call('HSET', run_key, 'fencing_token', tostring(fencing))
if not redis.call('HGET', run_key, 'started_at') then
  redis.call('HSET', run_key, 'started_at', tostring(now))
end
redis.call('EXPIRE', run_key, ttl)
local run_id = redis.call('HGET', run_key, 'run_id')
redis.call('SET', conv_key, run_id, 'EX', math.ceil(lease_ttl) + 5)
return {'claimed', attempt_id, tostring(new_expires), tostring(fencing)}
"""

_LUA_EXPIRE_LEASE = """
local run_key = KEYS[1]
local conv_key = KEYS[2]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local status = redis.call('HGET', run_key, 'status')
if status ~= 'running' then
  return {'skip', status or 'missing'}
end
local expires = redis.call('HGET', run_key, 'lease_expires_at')
if expires and tonumber(expires) > now then
  return {'active', expires}
end
-- FIX1: go to enqueue_failed (recoverable), never bare queued without publish.
redis.call('HSET', run_key, 'status', 'enqueue_failed')
redis.call('HSET', run_key, 'last_error_code', 'lease_expired')
redis.call('HDEL', run_key, 'lease_owner')
redis.call('HDEL', run_key, 'lease_token_digest')
redis.call('HDEL', run_key, 'lease_expires_at')
redis.call('HDEL', run_key, 'current_attempt_id')
redis.call('EXPIRE', run_key, ttl)
local run_id = redis.call('HGET', run_key, 'run_id')
local conv_val = redis.call('GET', conv_key)
if conv_val == run_id then
  redis.call('DEL', conv_key)
end
return {'ok', 'enqueue_failed'}
"""

_LUA_PROMOTE_RETRY = """
local run_key = KEYS[1]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local status = redis.call('HGET', run_key, 'status')
if status ~= 'retry_wait' then
  return {'skip', status or 'missing'}
end
local next_at = redis.call('HGET', run_key, 'next_attempt_at')
if next_at and tonumber(next_at) > now then
  return {'not_due', next_at}
end
redis.call('HSET', run_key, 'status', 'enqueue_failed')
redis.call('HSET', run_key, 'last_error_code', 'retry_due')
redis.call('HDEL', run_key, 'next_attempt_at')
redis.call('EXPIRE', run_key, ttl)
return {'ok', 'enqueue_failed'}
"""

_LUA_SCHEDULE_CONV_WAIT = """
local run_key = KEYS[1]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local next_at = ARGV[3]
local status = redis.call('HGET', run_key, 'status')
if status ~= 'queued' then
  return {'conflict', status or 'missing'}
end
redis.call('HSET', run_key, 'status', 'retry_wait')
redis.call('HSET', run_key, 'next_attempt_at', next_at)
redis.call('HSET', run_key, 'last_error_code', 'conversation_busy')
redis.call('EXPIRE', run_key, ttl)
return {'ok', 'retry_wait'}
"""

_LUA_RENEW = """
local run_key = KEYS[1]
local conv_key = KEYS[2]
local now = tonumber(ARGV[1])
local lease_ttl = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local owner = ARGV[4]
local token_digest = ARGV[5]
local attempt_id = ARGV[6]
local status = redis.call('HGET', run_key, 'status')
if status ~= 'running' then
  return {'conflict', status or 'missing'}
end
if redis.call('HGET', run_key, 'lease_owner') ~= owner then
  return {'not_owner'}
end
if redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest then
  return {'bad_token'}
end
if redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'bad_attempt'}
end
local expires = redis.call('HGET', run_key, 'lease_expires_at')
if not expires or tonumber(expires) < now then
  return {'expired'}
end
local new_expires = now + lease_ttl
redis.call('HSET', run_key, 'lease_expires_at', tostring(new_expires))
redis.call('HSET', run_key, 'heartbeat_at', tostring(now))
redis.call('EXPIRE', run_key, ttl)
local run_id = redis.call('HGET', run_key, 'run_id')
redis.call('SET', conv_key, run_id, 'EX', math.ceil(lease_ttl) + 5)
return {'ok', tostring(new_expires)}
"""

_LUA_FINISH = """
local run_key = KEYS[1]
local conv_key = KEYS[2]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local owner = ARGV[3]
local token_digest = ARGV[4]
local attempt_id = ARGV[5]
local new_status = ARGV[6]
local err = ARGV[7]
local fence = ARGV[8]
local status = redis.call('HGET', run_key, 'status')
if status ~= 'running' then
  return {'conflict', status or 'missing'}
end
if redis.call('HGET', run_key, 'lease_owner') ~= owner then
  return {'not_owner'}
end
if redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest then
  return {'bad_token'}
end
if redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'bad_attempt'}
end
local expires = redis.call('HGET', run_key, 'lease_expires_at')
if not expires or tonumber(expires) < now then
  return {'expired'}
end
if redis.call('HGET', run_key, 'terminal_fence') then
  return {'already_terminal'}
end
redis.call('HSET', run_key, 'status', new_status)
redis.call('HSET', run_key, 'finished_at', tostring(now))
redis.call('HSET', run_key, 'terminal_fence', fence)
if err ~= '' then
  redis.call('HSET', run_key, 'last_error_code', err)
end
redis.call('HDEL', run_key, 'lease_owner')
redis.call('HDEL', run_key, 'lease_token_digest')
redis.call('HDEL', run_key, 'lease_expires_at')
redis.call('EXPIRE', run_key, ttl)
local run_id = redis.call('HGET', run_key, 'run_id')
local conv_val = redis.call('GET', conv_key)
if conv_val == run_id then
  redis.call('DEL', conv_key)
end
return {'ok', new_status}
"""

_LUA_RETRY_WAIT = """
local run_key = KEYS[1]
local conv_key = KEYS[2]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local owner = ARGV[3]
local token_digest = ARGV[4]
local attempt_id = ARGV[5]
local next_at = ARGV[6]
local err = ARGV[7]
local status = redis.call('HGET', run_key, 'status')
if status ~= 'running' then
  return {'conflict', status or 'missing'}
end
if redis.call('HGET', run_key, 'lease_owner') ~= owner then
  return {'not_owner'}
end
if redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest then
  return {'bad_token'}
end
if redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'bad_attempt'}
end
redis.call('HSET', run_key, 'status', 'retry_wait')
redis.call('HSET', run_key, 'next_attempt_at', next_at)
redis.call('HSET', run_key, 'last_error_code', err)
redis.call('HDEL', run_key, 'lease_owner')
redis.call('HDEL', run_key, 'lease_token_digest')
redis.call('HDEL', run_key, 'lease_expires_at')
redis.call('HDEL', run_key, 'current_attempt_id')
redis.call('EXPIRE', run_key, ttl)
local run_id = redis.call('HGET', run_key, 'run_id')
local conv_val = redis.call('GET', conv_key)
if conv_val == run_id then
  redis.call('DEL', conv_key)
end
return {'ok', 'retry_wait'}
"""


class ExecutionStoreError(Exception):
    """Ledger / Redis failure with stable code."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(message or code)


class ExecutionStore:
    """Redis-backed execution ledger."""

    def __init__(self, client: Redis, config: DurableQueueConfig) -> None:
        self._r = client
        self._cfg = config

    @classmethod
    def from_config(cls, config: DurableQueueConfig) -> ExecutionStore:
        client = Redis.from_url(config.redis_url, decode_responses=True)
        return cls(client, config)

    def ping(self) -> bool:
        return bool(self._r.ping())

    def redis_time(self) -> float:
        """Authoritative Redis server time (seconds)."""
        secs, micros = self._r.time()
        return float(secs) + float(micros) / 1_000_000.0

    def run_key(self, run_id: str) -> str:
        return self._cfg.key("run", run_id)

    def idem_key(self, idempotency_key: str) -> str:
        return self._cfg.key("idem", idempotency_key)

    def conv_key(self, conversation_id: str) -> str:
        return self._cfg.key("conv", conversation_id)

    def get(self, run_id: str) -> ExecutionRecord | None:
        data = self._r.hgetall(self.run_key(run_id))
        if not data:
            return None
        return ExecutionRecord.from_redis_mapping(data)

    def get_by_idempotency(self, idempotency_key: str) -> ExecutionRecord | None:
        run_id = self._r.get(self.idem_key(idempotency_key))
        if not run_id:
            return None
        return self.get(str(run_id))

    def create_or_get(
        self,
        *,
        record: ExecutionRecord,
    ) -> tuple[str, ExecutionRecord]:
        """Atomically create or return existing bound record.

        Returns:
            (created|hit, record)

        Raises:
            ExecutionStoreError: On binding conflicts or Redis failure.
        """
        fields = record.to_redis_mapping()
        try:
            result = self._r.eval(
                _LUA_CREATE_OR_GET,
                2,
                self.run_key(record.run_id),
                self.idem_key(record.idempotency_key),
                str(self.redis_time()),
                str(self._cfg.record_ttl_s),
                record.run_id,
                record.idempotency_key,
                str(record.request_schema_version),
                record.request_fingerprint,
                str(record.queue_schema_version),
                json.dumps(fields, separators=(",", ":")),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc

        kind = str(result[0])
        if kind == "conflict":
            raise ExecutionStoreError(str(result[1]))
        loaded = self.get(record.run_id)
        if loaded is None:
            raise ExecutionStoreError("queue_unavailable", "record_missing_after_create")
        return kind, loaded

    def mark_queued(
        self,
        run_id: str,
        *,
        expected_status: str,
        delivery_id: str = "",
    ) -> ExecutionRecord:
        """CAS expected → queued. Already-queued is a conflict (FIX2 multi-publish guard)."""
        try:
            result = self._r.eval(
                _LUA_MARK_QUEUED,
                1,
                self.run_key(run_id),
                str(self.redis_time()),
                str(self._cfg.record_ttl_s),
                expected_status,
                delivery_id or "",
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind == "already_queued":
            raise ExecutionStoreError("already_queued", str(result[1] if len(result) > 1 else ""))
        if kind != "ok":
            raise ExecutionStoreError("status_conflict", str(result))
        loaded = self.get(run_id)
        assert loaded is not None
        return loaded

    def mark_accepted_enqueue_failed(
        self, run_id: str, error_code: str = "enqueue_failed"
    ) -> ExecutionRecord:
        """Admission-only: accepted|enqueue_failed → enqueue_failed. Never clears lease/terminal."""
        try:
            result = self._r.eval(
                _LUA_MARK_ACCEPTED_ENQUEUE_FAILED,
                1,
                self.run_key(run_id),
                str(self.redis_time()),
                str(self._cfg.record_ttl_s),
                error_code,
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        if str(result[0]) != "ok":
            raise ExecutionStoreError("status_conflict", str(result))
        loaded = self.get(run_id)
        assert loaded is not None
        return loaded

    def mark_dlq_replay_to_enqueue_failed(
        self,
        run_id: str,
        *,
        operator_id: str,
        replay_id: str,
        reason: str,
        error_code: str = "dlq_replay",
    ) -> ExecutionRecord:
        """Operator DLQ replay: dead_lettered → enqueue_failed with audit (clears terminal fence)."""
        try:
            result = self._r.eval(
                _LUA_MARK_DLQ_REPLAY,
                1,
                self.run_key(run_id),
                str(self.redis_time()),
                str(self._cfg.record_ttl_s),
                error_code,
                operator_id,
                replay_id,
                reason,
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        if str(result[0]) != "ok":
            raise ExecutionStoreError("status_conflict", str(result))
        loaded = self.get(run_id)
        assert loaded is not None
        return loaded

    def claim_recovery_delivery(
        self,
        run_id: str,
        *,
        expected_status: str,
        delivery_id: str,
        claim_ttl_s: float | None = None,
    ) -> str:
        """Atomically reserve the right to publish one recovery delivery.

        Returns:
            'ok' | 'busy' | 'conflict' | 'missing'
        """
        ttl = int(claim_ttl_s or max(30, self._cfg.admission_timeout_s * 2))
        claim_key = self._cfg.key("recovery_claim", run_id)
        try:
            result = self._r.eval(
                _LUA_CLAIM_RECOVERY_DELIVERY,
                2,
                self.run_key(run_id),
                claim_key,
                str(self.redis_time()),
                str(ttl),
                expected_status,
                delivery_id,
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        return str(result[0])

    def release_recovery_claim(self, run_id: str, *, delivery_id: str) -> str:
        """Atomically compare-and-delete recovery claim (FIX3)."""
        claim_key = self._cfg.key("recovery_claim", run_id)
        try:
            result = self._r.eval(
                _LUA_RELEASE_RECOVERY_CLAIM,
                1,
                claim_key,
                delivery_id,
            )
        except RedisError:
            return "error"
        return str(result[0])

    def claim_publish_delivery(
        self,
        run_id: str,
        *,
        expected_status: str,
        delivery_id: str,
        claim_ttl_s: float | None = None,
    ) -> str:
        """Shared publish ownership for admission repair and recovery (FIX3)."""
        # TTL must exceed broker publish max block (visibility / socket).
        default_ttl = max(
            float(self._cfg.visibility_timeout_s),
            float(self._cfg.admission_timeout_s) * 4,
            60.0,
        )
        return self.claim_recovery_delivery(
            run_id,
            expected_status=expected_status,
            delivery_id=delivery_id,
            claim_ttl_s=claim_ttl_s or default_ttl,
        )

    def cas_profile_write_fence(
        self,
        *,
        user_id: str,
        group_id: str,
        claim: LeaseClaim,
    ) -> str:
        """Atomically authorize a profile write for this fencing epoch (FIX4).

        Uses user×group global fence key; missing/non-positive token fail-closed;
        same epoch only allows same run:attempt idempotent replay.
        """
        from apps.group_agent_api.execution.crypto import digest_id, digest_token

        ud = digest_id(user_id)
        gd = digest_id(group_id)
        fence_key = self._cfg.key("profile_fence", ud, gd)
        meta_key = self._cfg.key("profile_fence_meta", ud, gd)
        try:
            result = self._r.eval(
                _LUA_CAS_PROFILE_FENCE,
                4,
                fence_key,
                meta_key,
                self.run_key(claim.run_id),
                self._cfg.key("profile_epoch", ud, gd),
                str(claim.fencing_token),
                claim.attempt_id,
                digest_token(claim.lease_token),
                str(self.redis_time()),
                claim.run_id,
                ud,
                gd,
                str(self._cfg.record_ttl_s),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        return str(result[0])

    def epoch_key_for_record(self, record: ExecutionRecord | None, *, conversation_id: str) -> str:
        """Resolve the durable user×group epoch key or fail closed.

        A conversation-scoped fallback would allocate the token in a different
        comparison domain from `cas_profile_write_fence`, so malformed or legacy
        records must not acquire a write-capable lease.
        """
        del conversation_id
        if record is None:
            raise ExecutionStoreError("missing_record")
        if not record.user_id_digest or not record.group_id_digest:
            raise ExecutionStoreError("fence_identity_missing")
        return self._cfg.key(
            "profile_epoch",
            record.user_id_digest,
            record.group_id_digest,
        )

    def poison_to_dlq(
        self,
        run_id: str,
        *,
        conversation_id: str,
        error_code: str,
        expected_status: str = "",
        fence: str | None = None,
    ) -> ExecutionRecord:
        """CAS into dead_lettered — never from running (FIX3)."""
        fence_val = fence or f"{run_id}:poison:{error_code}"
        try:
            result = self._r.eval(
                _LUA_POISON_TO_DLQ,
                2,
                self.run_key(run_id),
                self.conv_key(conversation_id or run_id),
                str(self.redis_time()),
                str(self._cfg.terminal_ttl_s),
                error_code,
                fence_val,
                expected_status or "",
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind == "terminal":
            loaded = self.get(run_id)
            if loaded is not None:
                return loaded
            raise ExecutionStoreError("already_terminal")
        if kind == "running_protected":
            raise ExecutionStoreError("running_protected", kind)
        if kind != "ok":
            raise ExecutionStoreError("poison_dlq_rejected", kind)
        loaded = self.get(run_id)
        assert loaded is not None
        return loaded

    def claim_lease(
        self,
        *,
        run_id: str,
        conversation_id: str,
        owner: str,
    ) -> ClaimOutcome:
        """Claim lease only from ``queued`` → ``running``.

        fencing_token is allocated from the user×group global epoch (FIX4).
        """
        attempt_id = str(uuid.uuid4())
        lease_token = secrets.token_urlsafe(32)
        token_digest = digest_token(lease_token)
        record = self.get(run_id)
        try:
            epoch_key = self.epoch_key_for_record(record, conversation_id=conversation_id)
        except ExecutionStoreError as exc:
            return ClaimOutcome(kind=exc.code)
        try:
            result = self._r.eval(
                _LUA_CLAIM,
                3,
                self.run_key(run_id),
                self.conv_key(conversation_id or run_id),
                epoch_key,
                str(self.redis_time()),
                str(self._cfg.lease_ttl_s),
                str(self._cfg.record_ttl_s),
                owner,
                token_digest,
                attempt_id,
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc

        kind = str(result[0])
        if kind == "claimed":
            fencing = int(result[3]) if len(result) > 3 else 0
            return ClaimOutcome(
                kind="claimed",
                claim=LeaseClaim(
                    run_id=run_id,
                    attempt_id=str(result[1]),
                    lease_token=lease_token,
                    lease_owner=owner,
                    lease_expires_at=float(result[2]),
                    fencing_token=fencing,
                ),
            )
        detail = str(result[1]) if len(result) > 1 else None
        _logger.info("lease_claim_skip run_id=%s reason=%s detail=%s", run_id, kind, detail)
        return ClaimOutcome(kind=kind, detail=detail)

    def schedule_conversation_wait(self, run_id: str, *, delay_s: float = 2.0) -> ExecutionRecord:
        """CAS queued → retry_wait when conversation is busy (avoids bare queued ACK-loss)."""
        next_at = self.redis_time() + delay_s
        try:
            result = self._r.eval(
                _LUA_SCHEDULE_CONV_WAIT,
                1,
                self.run_key(run_id),
                str(self.redis_time()),
                str(self._cfg.record_ttl_s),
                str(next_at),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        if str(result[0]) != "ok":
            raise ExecutionStoreError("status_conflict", str(result))
        loaded = self.get(run_id)
        assert loaded is not None
        return loaded

    def promote_retry_wait(self, run_id: str) -> str:
        """Lua CAS: retry_wait (due) → enqueue_failed for safe republish."""
        try:
            result = self._r.eval(
                _LUA_PROMOTE_RETRY,
                1,
                self.run_key(run_id),
                str(self.redis_time()),
                str(self._cfg.record_ttl_s),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        return str(result[0])

    def renew_lease(self, claim: LeaseClaim, *, conversation_id: str) -> float:
        try:
            result = self._r.eval(
                _LUA_RENEW,
                2,
                self.run_key(claim.run_id),
                self.conv_key(conversation_id or claim.run_id),
                str(self.redis_time()),
                str(self._cfg.lease_ttl_s),
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        if str(result[0]) != "ok":
            raise ExecutionStoreError("lease_renew_failed", str(result[0]))
        return float(result[1])

    def finish(
        self,
        claim: LeaseClaim,
        *,
        conversation_id: str,
        status: ExecutionStatus,
        error_code: str = "",
        fence: str | None = None,
    ) -> ExecutionRecord:
        if status not in TERMINAL_STATUSES:
            raise ExecutionStoreError("invalid_terminal_status")
        fence_val = fence or f"{claim.run_id}:{claim.attempt_id}:{status.value}"
        ttl = self._cfg.terminal_ttl_s
        try:
            result = self._r.eval(
                _LUA_FINISH,
                2,
                self.run_key(claim.run_id),
                self.conv_key(conversation_id or claim.run_id),
                str(self.redis_time()),
                str(ttl),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                status.value,
                error_code or "",
                fence_val,
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind != "ok":
            raise ExecutionStoreError("finish_rejected", kind)
        loaded = self.get(claim.run_id)
        assert loaded is not None
        return loaded

    def schedule_retry(
        self,
        claim: LeaseClaim,
        *,
        conversation_id: str,
        next_attempt_at: float,
        error_code: str,
    ) -> ExecutionRecord:
        try:
            result = self._r.eval(
                _LUA_RETRY_WAIT,
                2,
                self.run_key(claim.run_id),
                self.conv_key(conversation_id or claim.run_id),
                str(self.redis_time()),
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                str(next_attempt_at),
                error_code,
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        if str(result[0]) != "ok":
            raise ExecutionStoreError("retry_rejected", str(result[0]))
        loaded = self.get(claim.run_id)
        assert loaded is not None
        return loaded

    def expire_lease_if_needed(self, run_id: str, conversation_id: str) -> str:
        try:
            result = self._r.eval(
                _LUA_EXPIRE_LEASE,
                2,
                self.run_key(run_id),
                self.conv_key(conversation_id or run_id),
                str(self.redis_time()),
                str(self._cfg.record_ttl_s),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        return str(result[0])

    def append_replay_audit(
        self,
        run_id: str,
        *,
        operator_id: str,
        reason: str,
        replay_id: str,
    ) -> None:
        record = self.get(run_id)
        if record is None:
            raise ExecutionStoreError("missing_record")
        entry = {
            "operator_id": operator_id,
            "reason": reason,
            "replay_id": replay_id,
            "at": str(self.redis_time()),
        }
        record.replay_audit.append(entry)
        self._r.hset(
            self.run_key(run_id),
            "replay_audit_json",
            json.dumps(record.replay_audit, separators=(",", ":")),
        )

    def set_status_queued_from_retry(self, run_id: str) -> ExecutionRecord:
        """Promote due retry_wait → enqueue_failed (publish happens outside)."""
        kind = self.promote_retry_wait(run_id)
        loaded = self.get(run_id)
        if loaded is None:
            raise ExecutionStoreError("missing_record")
        if kind == "ok":
            return loaded
        return loaded

    def scan_status(
        self,
        status: ExecutionStatus,
        *,
        count: int = 100,
        limit: int | None = None,
    ) -> list[ExecutionRecord]:
        """Full cursor scan of run keys for a status (FIX4: no silent 5000 cap).

        Args:
            count: SCAN COUNT hint per iteration.
            limit: Optional max records to return; None = all matching.
        """
        pattern = self._cfg.key("run", "*")
        out: list[ExecutionRecord] = []
        for key in self._r.scan_iter(match=pattern, count=count):
            data = self._r.hgetall(key)
            if not data:
                continue
            if str(data.get("status")) != status.value:
                continue
            out.append(ExecutionRecord.from_redis_mapping(data))
            if limit is not None and len(out) >= limit:
                break
        return out

    def update_payload(self, run_id: str, payload: EncryptedPayload) -> None:
        self._r.hset(
            self.run_key(run_id),
            mapping={
                "payload_key_version": payload.key_version,
                "payload_nonce_b64": payload.nonce_b64,
                "payload_ciphertext_b64": payload.ciphertext_b64,
                "payload_tag_b64": payload.tag_b64,
            },
        )
