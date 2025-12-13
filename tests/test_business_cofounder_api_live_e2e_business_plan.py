from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pytest

from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver


def _base_url() -> str:
    return os.environ.get("BC_API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")

def _checkpoints_host_path_override() -> str | None:
    """
    Optional: when the API runs in Docker, /health returns an in-container path like
    /root/.deepagents/... which is NOT readable from the host running pytest.

    If you mount the checkpoints directory (e.g. ./data -> /root/.deepagents/business_cofounder_api),
    set BC_API_CHECKPOINTS_HOST_PATH to the host file path of checkpoints.pkl.
    """
    v = os.environ.get("BC_API_CHECKPOINTS_HOST_PATH")
    return v if v else None


def _live_enabled() -> bool:
    return os.environ.get("BC_API_LIVE_E2E") in {"1", "true", "TRUE", "yes", "YES"}


def _strict() -> bool:
    return os.environ.get("BC_API_LIVE_STRICT") in {"1", "true", "TRUE", "yes", "YES"}


def _e2e_chat_timeout_s() -> float:
    """Default per /chat timeout for the E2E test (seconds)."""
    return float(os.environ.get("BC_API_E2E_CHAT_TIMEOUT_S", "420"))


def _e2e_plan_timeout_s() -> float:
    """Timeout for the final 'full business plan' /chat call (seconds)."""
    return float(os.environ.get("BC_API_E2E_PLAN_TIMEOUT_S", "900"))


def _http_json(method: str, url: str, payload: dict | None = None, *, timeout_s: float = 60.0) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if hasattr(e, "read") else ""
        try:
            parsed = json.loads(body) if body else {"error": body}
        except Exception:  # noqa: BLE001
            parsed = {"error": body}
        return e.code, parsed


def _thread_id(*, user_id: str, conversation_id: str) -> str:
    return f"bc::{user_id}::{conversation_id}"


def _get_latest_state(*, checkpoints_path: Path, thread_id: str) -> dict[str, Any]:
    saver = DiskBackedInMemorySaver(file_path=str(checkpoints_path))
    ckpt_tuple = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    if ckpt_tuple is None:
        return {}

    checkpoint = ckpt_tuple.checkpoint if hasattr(ckpt_tuple, "checkpoint") else ckpt_tuple[1]
    if isinstance(checkpoint, dict):
        # LangGraph checkpoints commonly store channel/state values under these keys.
        for k in ("channel_values", "state", "values"):
            v = checkpoint.get(k)
            if isinstance(v, dict):
                return v
        return checkpoint
    return {}


def _wait_for(
    *,
    checkpoints_path: Path,
    thread_id: str,
    predicate: Callable[[dict[str, Any]], bool],
    timeout_s: float = 60.0,
    poll_s: float = 1.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_state: dict[str, Any] = {}
    while time.time() < deadline:
        last_state = _get_latest_state(checkpoints_path=checkpoints_path, thread_id=thread_id)
        if predicate(last_state):
            return last_state
        time.sleep(poll_s)
    return last_state


@pytest.mark.timeout(1800)
def test_live_e2e_full_business_plan_flow() -> None:
    """
    Live E2E test: drive the running API through the full sequential business-idea workflow.

    Opt-in only (costly + requires LLM creds in the running server process):
      BC_API_LIVE_E2E=1

    It asserts progression by reading the real server's DiskBackedInMemorySaver file path
    returned by GET /health, then inspecting the latest checkpoint state for the thread.
    """
    if not _live_enabled():
        pytest.skip("Set BC_API_LIVE_E2E=1 to run the full live E2E business-plan test.")

    base = _base_url()

    # ---- health (also gives us checkpoints_path)
    try:
        status, health = _http_json("GET", f"{base}/health", timeout_s=10.0)
    except urllib.error.URLError as e:
        msg = f"Live server not reachable at {base} ({e}). Start uvicorn on port 8001 or set BC_API_BASE_URL."
        if _strict():
            pytest.fail(msg)
        pytest.skip(msg)

    assert status == 200, health
    # IMPORTANT (Docker): /health returns an in-container path. Allow overriding to a host path.
    override = _checkpoints_host_path_override()
    checkpoints_path = Path(override) if override else Path(health["checkpoints_path"])

    # Unique user id so parallel runs don't stomp on each other.
    user_id = f"pytest-live-user-{int(time.time())}"
    conversation_id = "default"
    tid = _thread_id(user_id=user_id, conversation_id=conversation_id)

    # ---- reset thread first
    status, payload = _http_json(
        "POST",
        f"{base}/reset",
        {"user_id": user_id, "conversation_id": conversation_id},
        timeout_s=15.0,
    )
    assert status == 200, payload
    assert payload.get("ok") is True

    def _chat(message: str, *, timeout_s: float | None = None) -> str:
        if timeout_s is None:
            timeout_s = _e2e_chat_timeout_s()
        try:
            status, payload = _http_json(
                "POST",
                f"{base}/chat",
                {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "message": message,
                    "metadata": {"source": "pytest-live-e2e"},
                },
                timeout_s=timeout_s,
            )
        except TimeoutError as e:
            msg = (
                f"/chat timed out after {timeout_s}s while waiting for the server response.\n"
                "Hint: increase BC_API_E2E_CHAT_TIMEOUT_S / BC_API_E2E_PLAN_TIMEOUT_S.\n"
                f"error={e!s}"
            )
            if _strict():
                pytest.fail(msg)
            pytest.skip(msg)
        if status != 200:
            # With the improved API error handling, you'll usually see:
            # { "detail": { "error_type": "...", "error_message": "...", "thread_id": "..." } }
            detail = payload.get("detail") if isinstance(payload, dict) else None
            msg = (
                f"/chat failed with status={status}.\n"
                f"payload={payload!r}\n\n"
                "Common causes:\n"
                "- missing/invalid model credentials in the server process\n"
                "- provider quota/billing issue (e.g. 'Insufficient Balance')\n"
                "- model name not available\n"
                "\n"
                f"detail={detail!r}"
            )
            if _strict():
                pytest.fail(msg)
            pytest.skip(msg)
        reply = str(payload.get("reply") or "").strip()
        assert reply, f"Empty reply: {payload!r}"
        return reply

    # ---- Step 1: provide a realistic idea (should mark business_idea_complete)
    _chat(
        """I want to build a B2B SaaS product for mid-sized companies to reduce employee burnout in IT teams.

Problem:
- DevOps / SRE / platform engineers are overloaded by alerts, ad-hoc requests, and unclear prioritization.
- Managers lack early warning signals and actionable interventions, so burnout shows up too late (attrition, missed SLAs).

Solution:
- A Slack + Jira + PagerDuty integrated assistant that (1) summarizes workload/alerts, (2) flags risk patterns, (3) recommends weekly focus plans,
  and (4) provides lightweight “manager interventions” templates.

Target users:
- Primary: Engineering managers (10–80 reports across teams) at 200–2000 employee companies.
- Secondary: SRE/DevOps leads who own reliability metrics and on-call health.

Value:
- Reduce high-severity incidents by improving focus time, and reduce attrition/on-call fatigue.
- We estimate 1–2 fewer resignations per year for a 200-person eng org, plus improved reliability.

Business model:
- Subscription priced per employee in engineering (or per active seat), annual contracts.

Competition:
- Notion/Linear/Jira are task tools; OpsGenie/PagerDuty are alert tools; we sit on top as an intelligence + workflow layer.

Team:
- We have 8 years building developer tools, and we previously built internal analytics at a large SaaS company.

Please evaluate whether this is a complete business idea. If complete, call mark_business_idea_complete.""",
        timeout_s=_e2e_chat_timeout_s(),
    )

    state = _wait_for(
        checkpoints_path=checkpoints_path,
        thread_id=tid,
        predicate=lambda s: s.get("business_idea_complete") is True,
        timeout_s=60.0,
    )
    assert state.get("business_idea_complete") is True, state

    # ---- Step 2: persona clarification
    _chat(
        "Now clarify the user persona (primary + secondary). Output a concise persona profile and then call mark_persona_clarified.",
        timeout_s=_e2e_chat_timeout_s(),
    )
    state = _wait_for(
        checkpoints_path=checkpoints_path,
        thread_id=tid,
        predicate=lambda s: s.get("persona_clarified") is True,
        timeout_s=60.0,
    )
    assert state.get("persona_clarified") is True, state

    # ---- Step 3: painpoint enhancement
    _chat(
        "Enhance the pain point with emotional resonance dimensions and then call mark_painpoint_enhanced.",
        timeout_s=_e2e_chat_timeout_s(),
    )
    state = _wait_for(
        checkpoints_path=checkpoints_path,
        thread_id=tid,
        predicate=lambda s: s.get("painpoint_enhanced") is True,
        timeout_s=60.0,
    )
    assert state.get("painpoint_enhanced") is True, state

    # ---- Step 4: 60s pitch
    pitch = _chat(
        "Create a 60-second pitch for this idea (structured). Then call mark_pitch_created.",
        timeout_s=max(_e2e_chat_timeout_s(), 600.0),
    )
    state = _wait_for(
        checkpoints_path=checkpoints_path,
        thread_id=tid,
        predicate=lambda s: s.get("pitch_created") is True,
        timeout_s=60.0,
    )
    assert state.get("pitch_created") is True, state
    assert ("60" in pitch and "pitch" in pitch.lower()) or ("call to action" in pitch.lower())

    # ---- Step 5: pricing
    pricing = _chat(
        "Do baseline pricing and pricing optimization tactics. Then call mark_pricing_optimized.",
        timeout_s=max(_e2e_chat_timeout_s(), 600.0),
    )
    state = _wait_for(
        checkpoints_path=checkpoints_path,
        thread_id=tid,
        predicate=lambda s: s.get("pricing_optimized") is True,
        timeout_s=60.0,
    )
    assert state.get("pricing_optimized") is True, state
    assert "pricing" in pricing.lower()

    # ---- Step 6: pivot exploration
    pivot = _chat(
        "Explore alternative business model archetypes for this product (at least 3) and recommend the most promising.",
        timeout_s=max(_e2e_chat_timeout_s(), 600.0),
    )
    pivot_l = pivot.lower()
    archetype_hits = sum(
        1
        for kw in [
            "subscription",
            "usage-based",
            "membership",
            "transaction",
            "brokerage",
            "service",
            "retail",
        ]
        if kw in pivot_l
    )
    assert archetype_hits >= 2, pivot

    # ---- Step 7: full business plan
    plan = _chat(
        """Now write a complete business plan in English for this startup.

Requirements:
- Include sections: Executive Summary, Problem, Solution, Target Customers & Persona, Market Size (rough), Competitive Landscape,
  Business Model & Pricing, Go-To-Market, Product & Roadmap (next 90 days), Operations, Team, Financial Projections (rough),
  Risks & Mitigations, and Next Steps.
- Use bullets and short paragraphs. Include at least 3 concrete metrics/assumptions.""",
        timeout_s=_e2e_plan_timeout_s(),
    )
    plan_l = plan.lower()
    for required in [
        "executive summary",
        "problem",
        "solution",
        "market",
        "competitive",
        "pricing",
        "go-to-market",
        "roadmap",
        "risks",
        "next steps",
    ]:
        assert required in plan_l, f"Missing section '{required}' in plan reply."


