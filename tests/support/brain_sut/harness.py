"""Install production-seam patches so the real brain talks to fakes only.

Seam map (what production imports):

* ``match_backend.run_match`` ← hand search (called from ``search_candidates`` tool)
* ``profile_client.persist_group_profile`` ← hand write (also used from agent tool)
* ``membership_client.fetch_membership`` ← ear probe
* ``callback_client.send_callback_event`` ← mouth
* ``async_manager.send_callback_event`` ← mouth (re-exported import site)

The agent under test is the **same** ``create_agent`` / ``/chat`` / model_builder
path used in production — only neighbors are replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.support.brain_sut.fakes import FakeEar, FakeHand, FakeMouth


@dataclass
class BrainSutHarness:
    hand: FakeHand
    ear: FakeEar
    mouth: FakeMouth


def install_brain_sut(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hand: FakeHand | None = None,
    ear: FakeEar | None = None,
    mouth: FakeMouth | None = None,
    force_profile_http: bool = True,
) -> BrainSutHarness:
    """Patch neighbor clients to fakes. Does not change brain internals.

    Args:
        force_profile_http: When True, ``save_group_profile`` uses the http branch
            so FakeHand.persist is exercised (production tool path), without
            starting Micro. Startup may still use INTEGRATION=stub.
    """
    h = hand or FakeHand()
    e = ear or FakeEar()
    m = mouth or FakeMouth()

    # Hand · search — the model tool is the only caller; patch its import site.
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_backend.run_match",
        h.run_match,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        h.run_match,
    )

    # Hand · profile
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.profile_client.persist_group_profile",
        h.persist_group_profile,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.persist_group_profile",
        h.persist_group_profile,
    )
    if force_profile_http:
        monkeypatch.setattr(
            "apps.group_agent_api.agent_factory.agent.integration_mode",
            lambda: "http",
        )

    # Ear · membership
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.fetch_membership",
        e.fetch_membership,
    )

    # Mouth · callback (both definition and async_manager binding)
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.callback_client.send_callback_event",
        m.send_callback_event,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.app.async_manager.send_callback_event",
        m.send_callback_event,
    )

    # Isolation: any leftover HTTP post must fail loudly
    def _forbid_requests(*_a: Any, **_k: Any) -> None:
        raise AssertionError("brain_sut isolation: unexpected requests.* call")

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_client.requests.post",
        _forbid_requests,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.profile_client.requests.post",
        _forbid_requests,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        _forbid_requests,
    )

    return BrainSutHarness(hand=h, ear=e, mouth=m)


def install_instrumented_real_model(monkeypatch: pytest.MonkeyPatch, recorder: Any) -> Any:
    """Build the production ChatOpenAI client and inject it into create_agent.

    Fails the test if the stub model is constructed — that is the proof that
    GROUP_AGENT_MODEL_MODE=real actually reached the live LLM path.
    """
    import importlib

    from langchain_openai import ChatOpenAI

    from apps.group_agent_api.agent_factory.model_builder import create_model

    live_model = create_model()
    if not isinstance(live_model, ChatOpenAI):
        raise AssertionError(
            "brain_sut real-LLM: expected langchain_openai.ChatOpenAI, "
            f"got {type(live_model).__name__} (stub path is not allowed)"
        )
    if type(live_model).__name__ == "StubGroupAgentChatModel":
        raise AssertionError("brain_sut real-LLM: stub model leaked into real mode")
    recorder.attach(live_model)

    startup_mod = importlib.import_module("apps.group_agent_api.app.startup")
    original_create_agent = startup_mod.create_agent  # type: ignore[attr-defined]

    def patched_create_agent(
        *,
        base_dir: Path | None = None,
        model: Any | None = None,
        checkpointer: Any | None = None,
    ) -> tuple[Any, Any, Path]:
        del model  # never allow a stub to replace the live client
        return original_create_agent(
            base_dir=base_dir,
            model=live_model,
            checkpointer=checkpointer,
        )

    monkeypatch.setattr(startup_mod, "create_agent", patched_create_agent)
    return live_model
