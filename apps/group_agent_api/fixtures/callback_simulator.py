"""REQ-010 Callback Simulator for group_agent_api E2E testing."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

app = FastAPI(title="Callback Simulator", version="1.0.0")


class EventRecord(BaseModel):
    run_id: str
    idempotency_key: str
    seq: int
    event: str
    occurred_at: str
    user_id: str
    group_id: str
    conversation_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    received_at: float


class CallbackSimulatorState:

    def __init__(self, secret: str | None = None):
        self.secret = secret or os.environ.get("GROUP_AGENT_CALLBACK_HMAC_SECRET", "test_callback_secret_32bytes_long!")
        self.records: list[EventRecord] = []
        self.seq_by_run: dict[str, int] = {}
        self.terminal_by_run: dict[str, str] = {}
        self.nonces: set[str] = set()
        self.hmac_failures: int = 0
        self.seq_failures: int = 0
        self.terminal_failures: int = 0

    def reset(self) -> None:
        self.records.clear()
        self.seq_by_run.clear()
        self.terminal_by_run.clear()
        self.nonces.clear()
        self.hmac_failures = 0
        self.seq_failures = 0
        self.terminal_failures = 0


simulator_state = CallbackSimulatorState()


def verify_hmac_signature(
    method: str,
    path: str,
    body_bytes: bytes,
    signature_header: str,
    secret: str,
    ts_str: str,
    nonce_str: str,
) -> bool:
    """Canonical HMAC-SHA256 signature verification according to GA-CALLBACK-V1 standard."""
    raw_sig = signature_header.split("v1=")[-1].strip() if "v1=" in signature_header else signature_header.strip()
    body_sha = hashlib.sha256(body_bytes).hexdigest()
    canon = "\n".join([
        "GA-CALLBACK-V1",
        f"method={method.upper()}",
        f"path={path}",
        f"body_sha256={body_sha}",
        f"ts={ts_str}",
        f"nonce={nonce_str}",
    ])
    expected_canon = hmac.new(secret.encode("utf-8"), canon.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_canon, raw_sig)


@app.post("/group_agent_callbacks")
async def receive_callback(
    request: Request,
    x_ga_callback_version: str | None = Header(None, alias="X-GA-Callback-Version"),
    x_ga_callback_signature: str | None = Header(None, alias="X-GA-Callback-Signature"),
    x_ga_callback_timestamp: str | None = Header(None, alias="X-GA-Callback-Timestamp"),
    x_ga_callback_nonce: str | None = Header(None, alias="X-GA-Callback-Nonce"),
):
    # Mandatory Header Presence Verification (Zero Bypasses)
    if not x_ga_callback_version or not x_ga_callback_signature or not x_ga_callback_timestamp or not x_ga_callback_nonce:
        simulator_state.hmac_failures += 1
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_callback_header", "message": "All four GA-CALLBACK-V1 security headers are required."}
        )

    # 1. Version Check
    if x_ga_callback_version != "GA-CALLBACK-V1":
        simulator_state.hmac_failures += 1
        raise HTTPException(status_code=400, detail={"error": "invalid_callback_version"})

    # 2. Timestamp Skew Check (<= 300s)
    try:
        ts_val = float(x_ga_callback_timestamp)
        if abs(time.time() - ts_val) > 300:
            simulator_state.hmac_failures += 1
            raise HTTPException(status_code=400, detail={"error": "timestamp_skew_exceeded"})
    except ValueError:
        simulator_state.hmac_failures += 1
        raise HTTPException(status_code=400, detail={"error": "invalid_timestamp_format"})

    raw_body = await request.body()
    path = request.url.path

    # 3. Canonical HMAC Signature Verification
    if not verify_hmac_signature(
        request.method,
        path,
        raw_body,
        x_ga_callback_signature,
        simulator_state.secret,
        x_ga_callback_timestamp,
        x_ga_callback_nonce,
    ):
        simulator_state.hmac_failures += 1
        raise HTTPException(status_code=401, detail={"error": "invalid_hmac_signature"})

    # 4. Nonce Anti-Replay Store
    if x_ga_callback_nonce in simulator_state.nonces:
        raise HTTPException(status_code=409, detail={"error": "nonce_replayed"})
    simulator_state.nonces.add(x_ga_callback_nonce)

    # Parse payload
    data = await request.json()
    run_id = data.get("run_id")
    seq = data.get("seq", 0)
    event = data.get("event")

    # 5. Terminal State Check
    if run_id in simulator_state.terminal_by_run:
        simulator_state.terminal_failures += 1
        raise HTTPException(
            status_code=400,
            detail={
                "error": "terminal_state_violation",
                "message": f"Run '{run_id}' already terminated with '{simulator_state.terminal_by_run[run_id]}'",
            },
        )

    # 6. Strict Sequence Increment Check
    last_seq = simulator_state.seq_by_run.get(run_id, 0)
    if seq <= last_seq:
        simulator_state.seq_failures += 1
        raise HTTPException(
            status_code=400,
            detail={
                "error": "sequence_non_increasing",
                "message": f"seq {seq} <= last_seq {last_seq} for run {run_id}",
            },
        )

    simulator_state.seq_by_run[run_id] = seq

    rec = EventRecord(
        run_id=run_id,
        idempotency_key=data.get("idempotency_key", ""),
        seq=seq,
        event=event,
        occurred_at=data.get("occurred_at", ""),
        user_id=data.get("user_id", ""),
        group_id=data.get("group_id", ""),
        conversation_id=data.get("conversation_id", ""),
        payload=data.get("payload", {}),
        received_at=time.time(),
    )
    simulator_state.records.append(rec)

    if event in {"final", "error"}:
        simulator_state.terminal_by_run[run_id] = event

    return {"status": "ok", "run_id": run_id, "seq": seq}


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "callback_simulator"}


@app.get("/stats")
async def get_stats():
    return {
        "records_count": len(simulator_state.records),
        "active_runs": len(simulator_state.seq_by_run) - len(simulator_state.terminal_by_run),
        "terminal_runs": len(simulator_state.terminal_by_run),
        "hmac_failures": simulator_state.hmac_failures,
        "seq_failures": simulator_state.seq_failures,
        "terminal_failures": simulator_state.terminal_failures,
    }


@app.get("/records")
async def get_all_records():
    return {
        "total": len(simulator_state.records),
        "records": [r.model_dump() for r in simulator_state.records],
    }


@app.get("/records/{run_id}")
async def get_records_by_run(run_id: str):
    records_for_run = [r.model_dump() for r in simulator_state.records if r.run_id == run_id]
    return {
        "run_id": run_id,
        "count": len(records_for_run),
        "terminal_event": simulator_state.terminal_by_run.get(run_id),
        "records": records_for_run,
    }


@app.post("/reset")
async def reset_simulator():
    simulator_state.reset()
    return {"status": "reset"}
