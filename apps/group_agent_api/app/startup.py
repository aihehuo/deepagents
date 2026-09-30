"""Startup for group_agent_api."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from apps.group_agent_api.agent_factory.agent import APP_NAME, default_runtime_dir
from apps.group_agent_api.agent_factory.integrations.config import (
    assert_startup_security,
    integration_mode,
)
from apps.group_agent_api.agent_factory.member_graphs import build_member_graphs
from apps.group_agent_api.agent_factory.model_builder import (
    create_model,
    create_planner_model,
)
from apps.group_agent_api.app.anonymous_config import anonymous_enabled, anonymous_limits
from apps.group_agent_api.app.state import AppState
from deepagents.observability import UCObserver

_logger = logging.getLogger("uvicorn.error")


def _load_dotenv_if_present() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


async def startup(state_ref: dict[str, AppState | None]) -> None:
    _load_dotenv_if_present()
    assert_startup_security()
    runtime = Path(os.environ.get("GROUP_AGENT_RUNTIME_DIR", str(default_runtime_dir())))
    runtime.mkdir(parents=True, exist_ok=True)
    UCObserver.set_log_dir(runtime / "logs")

    base_model = create_model()
    graphs = build_member_graphs(runtime=runtime, model=base_model)
    anonymous_agent = None
    if anonymous_enabled():
        from apps.group_agent_api.agent_factory.anonymous import create_anonymous_agent

        anonymous_limits()
        anonymous_agent = create_anonymous_agent(skills_dir=runtime / "skills" / "aihehuo", model=base_model)
    polish_model = None
    quality_model = None
    planner_model = None
    greeting_model = None
    cafe_shadow_review_model = None
    try:
        from apps.group_agent_api.agent_factory.integrations.config import (
            llm_polish_enabled,
        )

        review_timeout_s = float(
            os.environ.get("GROUP_AGENT_CAFE_REVIEW_TIMEOUT_S", "20")
        )
        core_timeout_s = float(
            os.environ.get("GROUP_AGENT_CAFE_MODEL_TIMEOUT_S", "25")
        )
        if llm_polish_enabled():
            polish_model = create_model(
                log_prefix="[GroupAgentPolish]",
                timeout_s=review_timeout_s,
                max_retries=0,
            )
        # Match-ready judge: reuse polish model when present, else dedicated instance.
        quality_model = polish_model or create_model(
            log_prefix="[GroupAgentQuality]",
            timeout_s=review_timeout_s,
            max_retries=0,
        )
        planner_model = create_planner_model()
        greeting_model = create_model(
            log_prefix="[GroupAgentVenueGreeting]",
            timeout_s=core_timeout_s,
            max_retries=0,
        )
        from apps.group_agent_api.agent_factory.cafe_shadow_review import (
            shadow_review_enabled,
        )

        if shadow_review_enabled():
            cafe_shadow_review_model = create_model(
                log_prefix="[GroupAgentCafeShadowReview]",
                timeout_s=review_timeout_s,
                max_retries=0,
            )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("polish/quality model unavailable: %s", exc)

    durable_config = None
    durable_store = None
    backpressure = None
    from apps.group_agent_api.execution.config import (
        durable_queue_enabled,
        load_durable_queue_config,
    )

    if durable_queue_enabled():
        durable_config = load_durable_queue_config(require_enabled=True)
        assert durable_config is not None
        from apps.group_agent_api.execution.backpressure import BackpressureController
        from apps.group_agent_api.execution.redis_store import ExecutionStore

        durable_store = ExecutionStore.from_config(durable_config)
        if not durable_store.ping():
            raise RuntimeError("durable Redis ping failed at startup")
        backpressure = BackpressureController(durable_store._r, durable_config)
        _logger.info(
            "durable_queue enabled prefix=%s queue=%s",
            durable_config.redis_prefix,
            durable_config.celery_queue,
        )

    state_ref["state"] = AppState(
        **graphs,
        anonymous_agent=anonymous_agent,
        base_dir=runtime,
        polish_model=polish_model,
        quality_model=quality_model,
        planner_model=planner_model,
        durable_config=durable_config,
        durable_store=durable_store,
        backpressure=backpressure,
        greeting_model=greeting_model,
        cafe_shadow_review_model=cafe_shadow_review_model,
    )
    _logger.info(
        "%s ready runtime=%s integration=%s durable=%s",
        APP_NAME,
        runtime,
        integration_mode(),
        "1" if durable_queue_enabled() else "0",
    )
