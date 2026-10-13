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
    TERMINAL_STATUSES,
    EncryptedPayload,
    ExecutionRecord,
    ExecutionStatus,
    LeaseClaim,
    PendingCallback,
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

_LUA_CREATE_ROUTING_SNAPSHOT = """
local run_key = KEYS[1]
local ttl = tonumber(ARGV[1])
local owner = ARGV[2]
local token_digest = ARGV[3]
local attempt_id = ARGV[4]
local version = ARGV[5]
local snapshot_hash = ARGV[6]
local created_at = ARGV[7]
local key_version = ARGV[8]
local nonce_b64 = ARGV[9]
local ciphertext_b64 = ARGV[10]
local tag_b64 = ARGV[11]
local ledger_json = ARGV[12]
if redis.call('HGET', run_key, 'status') ~= 'running' then
  return {'rejected', 'fence_not_running'}
end
if redis.call('HGET', run_key, 'lease_owner') ~= owner
   or redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest
   or redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'rejected', 'fence_attempt_mismatch'}
end
local existing = redis.call('HGET', run_key, 'routing_snapshot_hash')
if existing then
  if existing == snapshot_hash
     and redis.call('HGET', run_key, 'routing_snapshot_version') == version
     and redis.call('HGET', run_key, 'routing_snapshot_key_version') == key_version then
    return {'hit', existing}
  end
  return {'conflict', existing}
end
redis.call('HSET', run_key,
  'routing_snapshot_version', version,
  'routing_snapshot_hash', snapshot_hash,
  'routing_snapshot_created_at', created_at,
  'routing_snapshot_key_version', key_version,
  'routing_snapshot_nonce_b64', nonce_b64,
  'routing_snapshot_ciphertext_b64', ciphertext_b64,
  'routing_snapshot_tag_b64', tag_b64,
  'receipt_ledger_json', ledger_json)
redis.call('EXPIRE', run_key, ttl)
return {'created', snapshot_hash}
"""

_LUA_RESERVE_QUERY = """
local function encode_ledger(value)
  local marker = '__intent_empty_array_6d84c925__'
  if next(value.receipt_refs) == nil then value.receipt_refs = {marker} end
  if next(value.audit_refs) == nil then value.audit_refs = {marker} end
  if next(value.adoption_refs) == nil then value.adoption_refs = {marker} end
  local encoded = cjson.encode(value)
  local normalized = string.gsub(encoded, '%["' .. marker .. '"%]', '[]')
  return normalized
end
local run_key = KEYS[1]
local reservation_prefix = KEYS[2]
local signature_key = KEYS[3]
local ttl = tonumber(ARGV[1])
local owner = ARGV[2]
local token_digest = ARGV[3]
local attempt_id = ARGV[4]
local snapshot_hash = ARGV[5]
local step_id = ARGV[6]
local max_requests = tonumber(ARGV[7])
local max_pages = tonumber(ARGV[8])
local max_items = tonumber(ARGV[9])
local max_elapsed = tonumber(ARGV[10])
local hard_timeout = tonumber(ARGV[11])
if redis.call('HGET', run_key, 'status') ~= 'running'
   or redis.call('HGET', run_key, 'lease_owner') ~= owner
   or redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest
   or redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'rejected', 'fence_attempt_mismatch'}
end
if redis.call('HGET', run_key, 'routing_snapshot_hash') ~= snapshot_hash then
  return {'rejected', 'routing_snapshot_invalid'}
end
if redis.call('EXISTS', signature_key) == 1 then
  return {'duplicate'}
end
local raw = redis.call('HGET', run_key, 'receipt_ledger_json')
local ok, ledger = pcall(cjson.decode, raw or '')
if not ok or ledger.routing_snapshot_hash ~= snapshot_hash then
  return {'rejected', 'receipt_invalid'}
end
if type(ledger.receipt_refs) ~= 'table'
   or type(ledger.audit_refs) ~= 'table'
   or type(ledger.adoption_refs) ~= 'table'
   or type(ledger.budget) ~= 'table' then
  return {'rejected', 'receipt_invalid'}
end
local budget = ledger.budget
if budget.requests_reserved + 1 > max_requests
   or budget.pages_reserved + 1 > max_pages
   or budget.items_committed >= max_items
   or budget.tool_elapsed_committed_ms + budget.tool_elapsed_reserved_ms + hard_timeout > max_elapsed then
  return {'exhausted'}
end
local idx = tonumber(budget.next_request_index_by_step[step_id] or 1)
budget.next_request_index_by_step[step_id] = idx + 1
budget.requests_reserved = budget.requests_reserved + 1
budget.pages_reserved = budget.pages_reserved + 1
budget.tool_elapsed_reserved_ms = budget.tool_elapsed_reserved_ms + hard_timeout
redis.call('HSET', run_key, 'receipt_ledger_json', encode_ledger(ledger))
local reservation_key = reservation_prefix .. ':' .. tostring(idx)
redis.call('HSET', reservation_key,
  'run_id', ledger.run_id, 'routing_snapshot_hash', snapshot_hash,
  'plan_step_id', step_id, 'request_index', tostring(idx),
  'hard_timeout_ms', tostring(hard_timeout), 'attempt_id', attempt_id)
redis.call('EXPIRE', reservation_key, ttl)
redis.call('SET', signature_key, tostring(idx), 'EX', ttl)
redis.call('EXPIRE', run_key, ttl)
return {'reserved', tostring(idx)}
"""

_LUA_COMMIT_RECEIPT = """
local function encode_ledger(value)
  local marker = '__intent_empty_array_6d84c925__'
  if next(value.receipt_refs) == nil then value.receipt_refs = {marker} end
  if next(value.audit_refs) == nil then value.audit_refs = {marker} end
  if next(value.adoption_refs) == nil then value.adoption_refs = {marker} end
  local encoded = cjson.encode(value)
  local normalized = string.gsub(encoded, '%["' .. marker .. '"%]', '[]')
  return normalized
end
local run_key = KEYS[1]
local reservation_key = KEYS[2]
local receipt_key = KEYS[3]
local ttl = tonumber(ARGV[1])
local owner = ARGV[2]
local token_digest = ARGV[3]
local attempt_id = ARGV[4]
local snapshot_hash = ARGV[5]
local step_id = ARGV[6]
local request_index = ARGV[7]
local budget_item_count = tonumber(ARGV[8])
local elapsed_ms = tonumber(ARGV[9])
local receipt_id = ARGV[10]
local receipt_fields = cjson.decode(ARGV[11])
local max_items = tonumber(ARGV[12])
if redis.call('HGET', run_key, 'status') ~= 'running'
   or redis.call('HGET', run_key, 'lease_owner') ~= owner
   or redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest
   or redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'rejected', 'fence_attempt_mismatch'}
end
if redis.call('HGET', run_key, 'routing_snapshot_hash') ~= snapshot_hash then
  return {'rejected', 'routing_snapshot_invalid'}
end
local existing = redis.call('HGET', receipt_key, 'receipt_id')
if existing then
  if existing == receipt_id then return {'hit', receipt_id} end
  return {'rejected', 'receipt_invalid'}
end
if redis.call('HGET', reservation_key, 'attempt_id') ~= attempt_id
   or redis.call('HGET', reservation_key, 'routing_snapshot_hash') ~= snapshot_hash then
  return {'rejected', 'receipt_invalid'}
end
local hard_timeout = tonumber(redis.call('HGET', reservation_key, 'hard_timeout_ms') or '0')
local raw = redis.call('HGET', run_key, 'receipt_ledger_json')
local ok, ledger = pcall(cjson.decode, raw or '')
if not ok then return {'rejected', 'receipt_invalid'} end
if type(ledger.receipt_refs) ~= 'table'
   or type(ledger.audit_refs) ~= 'table'
   or type(ledger.adoption_refs) ~= 'table'
   or type(ledger.budget) ~= 'table' then
  return {'rejected', 'receipt_invalid'}
end
if ledger.budget.items_committed + budget_item_count > max_items then
  return {'exhausted'}
end
ledger.budget.tool_elapsed_reserved_ms = math.max(0, ledger.budget.tool_elapsed_reserved_ms - hard_timeout)
ledger.budget.tool_elapsed_committed_ms = ledger.budget.tool_elapsed_committed_ms + elapsed_ms
ledger.budget.items_committed = ledger.budget.items_committed + budget_item_count
ledger.receipt_refs[#ledger.receipt_refs+1] = {
  plan_step_id = step_id, request_index = tonumber(request_index), receipt_id = receipt_id
}
for k, v in pairs(receipt_fields) do redis.call('HSET', receipt_key, k, v) end
redis.call('HSET', receipt_key, 'receipt_id', receipt_id)
redis.call('EXPIRE', receipt_key, ttl)
redis.call('DEL', reservation_key)
redis.call('HSET', run_key, 'receipt_ledger_json', encode_ledger(ledger))
redis.call('EXPIRE', run_key, ttl)
return {'created', receipt_id}
"""

_LUA_COMMIT_AUDIT = """
local function encode_ledger(value)
  local marker = '__intent_empty_array_6d84c925__'
  if next(value.receipt_refs) == nil then value.receipt_refs = {marker} end
  if next(value.audit_refs) == nil then value.audit_refs = {marker} end
  if next(value.adoption_refs) == nil then value.adoption_refs = {marker} end
  local encoded = cjson.encode(value)
  local normalized = string.gsub(encoded, '%["' .. marker .. '"%]', '[]')
  return normalized
end
local run_key = KEYS[1]
local reservation_key = KEYS[2]
local audit_key = KEYS[3]
local ttl = tonumber(ARGV[1])
local owner = ARGV[2]
local token_digest = ARGV[3]
local attempt_id = ARGV[4]
local snapshot_hash = ARGV[5]
local step_id = ARGV[6]
local request_index = ARGV[7]
local status = ARGV[8]
local elapsed_ms = tonumber(ARGV[9])
local audit_json = ARGV[10]
if redis.call('HGET', run_key, 'status') ~= 'running'
   or redis.call('HGET', run_key, 'lease_owner') ~= owner
   or redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest
   or redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'rejected', 'fence_attempt_mismatch'}
end
if redis.call('EXISTS', audit_key) == 1 then return {'hit'} end
if redis.call('HGET', run_key, 'routing_snapshot_hash') ~= snapshot_hash then
  return {'rejected', 'routing_snapshot_invalid'}
end
if redis.call('HGET', reservation_key, 'attempt_id') ~= attempt_id
   or redis.call('HGET', reservation_key, 'routing_snapshot_hash') ~= snapshot_hash then
  return {'rejected', 'receipt_invalid'}
end
local hard_timeout = tonumber(redis.call('HGET', reservation_key, 'hard_timeout_ms') or '0')
local raw = redis.call('HGET', run_key, 'receipt_ledger_json')
local ok, ledger = pcall(cjson.decode, raw or '')
if not ok then return {'rejected', 'receipt_invalid'} end
if type(ledger.receipt_refs) ~= 'table'
   or type(ledger.audit_refs) ~= 'table'
   or type(ledger.adoption_refs) ~= 'table'
   or type(ledger.budget) ~= 'table' then
  return {'rejected', 'receipt_invalid'}
end
ledger.budget.tool_elapsed_reserved_ms = math.max(0, ledger.budget.tool_elapsed_reserved_ms - hard_timeout)
ledger.budget.tool_elapsed_committed_ms = ledger.budget.tool_elapsed_committed_ms + elapsed_ms
ledger.audit_refs[#ledger.audit_refs+1] = {
  plan_step_id = step_id, request_index = tonumber(request_index), status = status
}
redis.call('SET', audit_key, audit_json, 'EX', ttl)
redis.call('DEL', reservation_key)
redis.call('HSET', run_key, 'receipt_ledger_json', encode_ledger(ledger))
redis.call('EXPIRE', run_key, ttl)
return {'created'}
"""

_LUA_ADOPT_RECEIPT = """
local function encode_ledger(value)
  local marker = '__intent_empty_array_6d84c925__'
  if next(value.receipt_refs) == nil then value.receipt_refs = {marker} end
  if next(value.audit_refs) == nil then value.audit_refs = {marker} end
  if next(value.adoption_refs) == nil then value.adoption_refs = {marker} end
  local encoded = cjson.encode(value)
  local normalized = string.gsub(encoded, '%["' .. marker .. '"%]', '[]')
  return normalized
end
local run_key = KEYS[1]
local adoption_key = KEYS[2]
local ttl = tonumber(ARGV[1])
local owner = ARGV[2]
local token_digest = ARGV[3]
local attempt_id = ARGV[4]
local snapshot_hash = ARGV[5]
local receipt_id = ARGV[6]
local attempt_number = tonumber(ARGV[7])
local adoption_json = ARGV[8]
if redis.call('HGET', run_key, 'status') ~= 'running'
   or redis.call('HGET', run_key, 'lease_owner') ~= owner
   or redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest
   or redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'rejected', 'fence_attempt_mismatch'}
end
if redis.call('EXISTS', adoption_key) == 1 then return {'hit'} end
local raw = redis.call('HGET', run_key, 'receipt_ledger_json')
local ok, ledger = pcall(cjson.decode, raw or '')
if not ok or ledger.routing_snapshot_hash ~= snapshot_hash then
  return {'rejected', 'receipt_invalid'}
end
if type(ledger.receipt_refs) ~= 'table'
   or type(ledger.audit_refs) ~= 'table'
   or type(ledger.adoption_refs) ~= 'table' then
  return {'rejected', 'receipt_invalid'}
end
local receipt_found = false
for _, ref in ipairs(ledger.receipt_refs) do
  if ref.receipt_id == receipt_id then receipt_found = true end
end
if not receipt_found then return {'rejected', 'receipt_invalid'} end
ledger.adoption_refs[#ledger.adoption_refs+1] = {
  receipt_id = receipt_id, adopted_by_attempt_number = attempt_number
}
redis.call('SET', adoption_key, adoption_json, 'EX', ttl)
redis.call('HSET', run_key, 'receipt_ledger_json', encode_ledger(ledger))
redis.call('EXPIRE', run_key, ttl)
return {'created'}
"""

_PENDING_CALLBACK_FIELDS = (
    "pending_callback_version",
    "pending_callback_seq",
    "pending_callback_body_sha256",
    "pending_callback_producer_attempt_id",
    "pending_callback_producer_attempt_number",
    "pending_callback_adopted_by_attempt_id",
    "pending_callback_adopted_by_attempt_number",
    "pending_callback_routing_snapshot_hash",
    "pending_callback_created_at",
    "pending_callback_key_version",
    "pending_callback_nonce_b64",
    "pending_callback_ciphertext_b64",
    "pending_callback_tag_b64",
)

_LUA_CREATE_PENDING_CALLBACK = """
local run_key = KEYS[1]
local ttl = tonumber(ARGV[1])
local owner = ARGV[2]
local token_digest = ARGV[3]
local attempt_id = ARGV[4]
local attempt_number = tonumber(ARGV[5])
local seq = tonumber(ARGV[6])
local digest = ARGV[7]
if redis.call('HGET', run_key, 'status') ~= 'running'
   or redis.call('HGET', run_key, 'lease_owner') ~= owner
   or redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest
   or redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id
   or tonumber(redis.call('HGET', run_key, 'attempt_count') or '0') ~= attempt_number then
  return {'rejected', 'fence_attempt_mismatch'}
end
local pending_seq = redis.call('HGET', run_key, 'pending_callback_seq')
if pending_seq then
  if tonumber(pending_seq) == seq
     and redis.call('HGET', run_key, 'pending_callback_body_sha256') == digest then
    return {'hit'}
  end
  return {'rejected', 'callback_seq_diverged'}
end
local last_seq = tonumber(redis.call('HGET', run_key, 'last_callback_seq') or '0')
if seq ~= last_seq + 1 then return {'rejected', 'callback_seq_diverged'} end
redis.call('HSET', run_key,
  'pending_callback_version', '1',
  'pending_callback_seq', tostring(seq),
  'pending_callback_body_sha256', digest,
  'pending_callback_producer_attempt_id', attempt_id,
  'pending_callback_producer_attempt_number', tostring(attempt_number),
  'pending_callback_routing_snapshot_hash', ARGV[8],
  'pending_callback_created_at', ARGV[9],
  'pending_callback_key_version', ARGV[10],
  'pending_callback_nonce_b64', ARGV[11],
  'pending_callback_ciphertext_b64', ARGV[12],
  'pending_callback_tag_b64', ARGV[13])
redis.call('HDEL', run_key,
  'pending_callback_adopted_by_attempt_id',
  'pending_callback_adopted_by_attempt_number')
redis.call('EXPIRE', run_key, ttl)
return {'created'}
"""

_LUA_ADOPT_PENDING_CALLBACK = """
local run_key = KEYS[1]
local ttl = tonumber(ARGV[1])
local owner = ARGV[2]
local token_digest = ARGV[3]
local attempt_id = ARGV[4]
local attempt_number = tonumber(ARGV[5])
if redis.call('HGET', run_key, 'status') ~= 'running'
   or redis.call('HGET', run_key, 'lease_owner') ~= owner
   or redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest
   or redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id
   or tonumber(redis.call('HGET', run_key, 'attempt_count') or '0') ~= attempt_number then
  return {'rejected', 'fence_attempt_mismatch'}
end
local pending_seq = tonumber(redis.call('HGET', run_key, 'pending_callback_seq') or '0')
local last_seq = tonumber(redis.call('HGET', run_key, 'last_callback_seq') or '0')
if pending_seq == 0 then return {'none'} end
if pending_seq ~= last_seq + 1 then return {'rejected', 'callback_seq_diverged'} end
local producer_number = tonumber(redis.call('HGET', run_key, 'pending_callback_producer_attempt_number') or '0')
if producer_number <= 0 or producer_number > attempt_number then
  return {'rejected', 'pending_callback_invalid'}
end
redis.call('HSET', run_key,
  'pending_callback_adopted_by_attempt_id', attempt_id,
  'pending_callback_adopted_by_attempt_number', tostring(attempt_number))
redis.call('EXPIRE', run_key, ttl)
return {'adopted'}
"""

_LUA_COMMIT_CALLBACK_ACK = """
local run_key = KEYS[1]
local ttl = tonumber(ARGV[1])
local owner = ARGV[2]
local token_digest = ARGV[3]
local attempt_id = ARGV[4]
local seq = tonumber(ARGV[5])
local digest = ARGV[6]
if redis.call('HGET', run_key, 'status') ~= 'running'
   or redis.call('HGET', run_key, 'lease_owner') ~= owner
   or redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest
   or redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'rejected', 'fence_attempt_mismatch'}
end
if tonumber(redis.call('HGET', run_key, 'pending_callback_seq') or '0') ~= seq
   or redis.call('HGET', run_key, 'pending_callback_body_sha256') ~= digest then
  return {'rejected', 'pending_callback_invalid'}
end
local last_seq = tonumber(redis.call('HGET', run_key, 'last_callback_seq') or '0')
if seq ~= last_seq + 1 then return {'rejected', 'callback_seq_diverged'} end
redis.call('HSET', run_key, 'last_callback_seq', tostring(seq))
redis.call('HDEL', run_key, unpack(cjson.decode(ARGV[7])))
redis.call('EXPIRE', run_key, ttl)
return {'committed', tostring(seq)}
"""

_LUA_DISCARD_PENDING_CALLBACK = """
local run_key = KEYS[1]
local ttl = tonumber(ARGV[1])
local owner = ARGV[2]
local token_digest = ARGV[3]
local attempt_id = ARGV[4]
local seq = tonumber(ARGV[5])
local digest = ARGV[6]
if redis.call('HGET', run_key, 'status') ~= 'running'
   or redis.call('HGET', run_key, 'lease_owner') ~= owner
   or redis.call('HGET', run_key, 'lease_token_digest') ~= token_digest
   or redis.call('HGET', run_key, 'current_attempt_id') ~= attempt_id then
  return {'rejected', 'fence_attempt_mismatch'}
end
if tonumber(redis.call('HGET', run_key, 'pending_callback_seq') or '0') ~= seq
   or redis.call('HGET', run_key, 'pending_callback_body_sha256') ~= digest then
  return {'rejected', 'pending_callback_invalid'}
end
redis.call('HDEL', run_key, unpack(cjson.decode(ARGV[7])))
redis.call('EXPIRE', run_key, ttl)
return {'discarded'}
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

    def routing_crypto_material(self) -> tuple[dict[str, bytes], str]:
        """Return configured keyring and current key version for routing state."""
        return dict(self._cfg.payload_keys), self._cfg.payload_current_version

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
        if record.request_schema_version == 5:
            return self._cfg.key("assistant_epoch", record.run_id)
        if record.request_schema_version in {3, 4}:
            if not record.mode_binding or record.mode_binding.get("execution_mode") not in {"cafe_host", "group_agent"}:
                raise ExecutionStoreError("binding_conflict")
            if record.mode_binding["execution_mode"] == "cafe_host":
                return self._cfg.key("greeting_epoch", record.run_id)
        if record.request_schema_version == 2:
            if record.input_kind == "venue_event":
                # A read-only greeting must not supersede the ordinary run's
                # user×group profile write epoch. It still has run fencing.
                return self._cfg.key("greeting_epoch", record.run_id)
            if record.input_kind != "user_message":
                raise ExecutionStoreError("binding_conflict")
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

    def create_pending_callback(
        self,
        claim: LeaseClaim,
        *,
        attempt_number: int,
        pending: PendingCallback,
    ) -> str:
        """Persist the one callback slot before its first network send."""
        try:
            result = self._r.eval(
                _LUA_CREATE_PENDING_CALLBACK,
                1,
                self.run_key(claim.run_id),
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                str(attempt_number),
                str(pending.seq),
                pending.body_sha256,
                pending.routing_snapshot_hash or "",
                pending.created_at,
                pending.ciphertext.key_version,
                pending.ciphertext.nonce_b64,
                pending.ciphertext.ciphertext_b64,
                pending.ciphertext.tag_b64,
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind not in {"created", "hit"}:
            raise ExecutionStoreError(
                str(result[1]) if len(result) > 1 else "pending_callback_invalid"
            )
        return kind

    def adopt_pending_callback(
        self, claim: LeaseClaim, *, attempt_number: int
    ) -> PendingCallback | None:
        """Claim an older attempt's pending bytes before any new callback."""
        try:
            result = self._r.eval(
                _LUA_ADOPT_PENDING_CALLBACK,
                1,
                self.run_key(claim.run_id),
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                str(attempt_number),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind == "none":
            return None
        if kind != "adopted":
            raise ExecutionStoreError(
                str(result[1]) if len(result) > 1 else "pending_callback_invalid"
            )
        record = self.get(claim.run_id)
        if record is None or record.pending_callback is None:
            raise ExecutionStoreError("pending_callback_invalid")
        return record.pending_callback

    def commit_callback_ack(
        self,
        claim: LeaseClaim,
        *,
        seq: int,
        body_sha256: str,
    ) -> int:
        """Atomically advance run-global seq and clear the matching pending slot."""
        try:
            result = self._r.eval(
                _LUA_COMMIT_CALLBACK_ACK,
                1,
                self.run_key(claim.run_id),
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                str(seq),
                body_sha256,
                json.dumps(_PENDING_CALLBACK_FIELDS),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        if str(result[0]) != "committed":
            raise ExecutionStoreError(
                str(result[1]) if len(result) > 1 else "pending_callback_invalid"
            )
        return int(result[1])

    def discard_pending_callback(
        self,
        claim: LeaseClaim,
        *,
        seq: int,
        body_sha256: str,
    ) -> None:
        """Clear a definitively uncommitted mouth-rejected callback."""
        try:
            result = self._r.eval(
                _LUA_DISCARD_PENDING_CALLBACK,
                1,
                self.run_key(claim.run_id),
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                str(seq),
                body_sha256,
                json.dumps(_PENDING_CALLBACK_FIELDS),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        if str(result[0]) != "discarded":
            raise ExecutionStoreError(
                str(result[1]) if len(result) > 1 else "pending_callback_invalid"
            )

    def create_routing_snapshot(
        self,
        claim: LeaseClaim,
        *,
        snapshot_version: int,
        snapshot_hash: str,
        created_at: str,
        envelope: EncryptedPayload,
        ledger: dict[str, Any],
    ) -> str:
        """Create the encrypted routing snapshot exactly once under lease CAS."""
        try:
            result = self._r.eval(
                _LUA_CREATE_ROUTING_SNAPSHOT,
                1,
                self.run_key(claim.run_id),
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                str(snapshot_version),
                snapshot_hash,
                created_at,
                envelope.key_version,
                envelope.nonce_b64,
                envelope.ciphertext_b64,
                envelope.tag_b64,
                json.dumps(ledger, sort_keys=True, separators=(",", ":")),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind not in {"created", "hit"}:
            raise ExecutionStoreError("routing_snapshot_invalid", str(result))
        return kind

    def _reservation_key(
        self, run_id: str, snapshot_hash: str, step_id: str, request_index: int | None = None
    ) -> str:
        parts = ["query_reservation", run_id, snapshot_hash, step_id]
        if request_index is not None:
            parts.append(str(request_index))
        return self._cfg.key(*parts)

    def _receipt_key(
        self, run_id: str, snapshot_hash: str, step_id: str, request_index: int
    ) -> str:
        return self._cfg.key("query_receipt", run_id, snapshot_hash, step_id, str(request_index))

    def _query_signature_key(
        self, run_id: str, snapshot_hash: str, step_id: str, parameters_digest: str
    ) -> str:
        return self._cfg.key(
            "query_signature", run_id, snapshot_hash, step_id, parameters_digest
        )

    def reserve_query_request(
        self,
        claim: LeaseClaim,
        *,
        snapshot_hash: str,
        step_id: str,
        max_requests: int,
        max_pages: int,
        max_items: int,
        max_elapsed_ms: int,
        hard_timeout_ms: int,
        parameters_digest: str,
    ) -> int:
        """Atomically reserve the run-global budget and next step request index."""
        try:
            result = self._r.eval(
                _LUA_RESERVE_QUERY,
                3,
                self.run_key(claim.run_id),
                self._reservation_key(claim.run_id, snapshot_hash, step_id),
                self._query_signature_key(
                    claim.run_id, snapshot_hash, step_id, parameters_digest
                ),
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                snapshot_hash,
                step_id,
                str(max_requests),
                str(max_pages),
                str(max_items),
                str(max_elapsed_ms),
                str(hard_timeout_ms),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind in {"duplicate", "exhausted"}:
            raise ExecutionStoreError("query_budget_exhausted")
        if kind != "reserved":
            raise ExecutionStoreError(str(result[1]) if len(result) > 1 else "receipt_invalid")
        return int(result[1])

    def commit_query_receipt(
        self,
        claim: LeaseClaim,
        *,
        snapshot_hash: str,
        step_id: str,
        request_index: int,
        budget_item_count: int,
        elapsed_ms: int,
        receipt_id: str,
        fields: dict[str, str],
        max_items: int,
    ) -> str:
        """Atomically create an immutable encrypted receipt and advance its budget."""
        try:
            result = self._r.eval(
                _LUA_COMMIT_RECEIPT,
                3,
                self.run_key(claim.run_id),
                self._reservation_key(
                    claim.run_id, snapshot_hash, step_id, request_index
                ),
                self._receipt_key(claim.run_id, snapshot_hash, step_id, request_index),
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                snapshot_hash,
                step_id,
                str(request_index),
                str(budget_item_count),
                str(elapsed_ms),
                receipt_id,
                json.dumps(fields, sort_keys=True, separators=(",", ":")),
                str(max_items),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind == "exhausted":
            raise ExecutionStoreError("query_budget_exhausted")
        if kind not in {"created", "hit"}:
            raise ExecutionStoreError(str(result[1]) if len(result) > 1 else "receipt_invalid")
        return kind

    def commit_query_audit(
        self,
        claim: LeaseClaim,
        *,
        snapshot_hash: str,
        step_id: str,
        request_index: int,
        status: str,
        elapsed_ms: int,
        audit: dict[str, Any],
    ) -> str:
        """Commit a failed/timed-out audit; it can never satisfy the plan."""
        audit_key = self._cfg.key(
            "query_audit", claim.run_id, snapshot_hash, step_id, str(request_index)
        )
        try:
            result = self._r.eval(
                _LUA_COMMIT_AUDIT,
                3,
                self.run_key(claim.run_id),
                self._reservation_key(
                    claim.run_id, snapshot_hash, step_id, request_index
                ),
                audit_key,
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                snapshot_hash,
                step_id,
                str(request_index),
                status,
                str(elapsed_ms),
                json.dumps(audit, sort_keys=True, separators=(",", ":")),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind not in {"created", "hit"}:
            raise ExecutionStoreError(str(result[1]) if len(result) > 1 else "receipt_invalid")
        return kind

    def get_receipt_ledger(self, run_id: str) -> dict[str, Any]:
        """Load the closed receipt ledger or fail closed on corruption."""
        raw = self._r.hget(self.run_key(run_id), "receipt_ledger_json")
        try:
            value = json.loads(str(raw))
        except Exception as exc:  # noqa: BLE001
            raise ExecutionStoreError("receipt_invalid") from exc
        if not isinstance(value, dict):
            raise ExecutionStoreError("receipt_invalid")
        return value

    def get_query_receipt_fields(
        self, run_id: str, snapshot_hash: str, step_id: str, request_index: int
    ) -> dict[str, str]:
        """Load one encrypted receipt record referenced by the ledger."""
        value = self._r.hgetall(
            self._receipt_key(run_id, snapshot_hash, step_id, request_index)
        )
        if not value:
            raise ExecutionStoreError("receipt_invalid")
        return {str(key): str(item) for key, item in value.items()}

    def adopt_query_receipt(
        self,
        claim: LeaseClaim,
        *,
        snapshot_hash: str,
        receipt_id: str,
        attempt_number: int,
        adoption: dict[str, Any],
    ) -> str:
        """Write one idempotent adoption row under the current lease."""
        key = self._cfg.key(
            "query_adoption", claim.run_id, snapshot_hash, receipt_id, str(attempt_number)
        )
        try:
            result = self._r.eval(
                _LUA_ADOPT_RECEIPT,
                2,
                self.run_key(claim.run_id),
                key,
                str(self._cfg.record_ttl_s),
                claim.lease_owner,
                digest_token(claim.lease_token),
                claim.attempt_id,
                snapshot_hash,
                receipt_id,
                str(attempt_number),
                json.dumps(adoption, sort_keys=True, separators=(",", ":")),
            )
        except RedisError as exc:
            raise ExecutionStoreError("queue_unavailable") from exc
        kind = str(result[0])
        if kind not in {"created", "hit"}:
            raise ExecutionStoreError(str(result[1]) if len(result) > 1 else "receipt_invalid")
        return kind

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
