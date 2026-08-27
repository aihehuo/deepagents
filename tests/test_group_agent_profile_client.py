"""Authoritative group-profile persistence client tests."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import pytest
import requests
from langchain_core.messages import AIMessage, ToolMessage

from apps.group_agent_api.agent_factory.agent import save_group_profile
from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.agent_factory.integrations.profile_client import (
    PROFILE_PATH,
    ProfileHttpError,
    canonical_profile_digest,
    persist_group_profile,
)
from apps.group_agent_api.agent_factory.profile_schema import (
    GroupProfile,
    profile_from_flat,
)
from apps.group_agent_api.agent_factory.profile_store import load_profile, save_profile
from apps.group_agent_api.app.async_manager import execute_async_run
from apps.group_agent_api.app.endpoints.chat import _invoke_config
from apps.group_agent_api.app.models import AsyncCallRequest, RolloutContext
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState


class _Response:
    def __init__(self, status_code: int, body: dict[str, Any] | None) -> None:
        self.status_code = status_code
        self._body = body
        self.content = json.dumps(body).encode("utf-8") if body is not None else b""

    def json(self) -> dict[str, Any] | None:
        return self._body


class _InvalidJsonResponse(_Response):
    def __init__(self) -> None:
        super().__init__(200, None)
        self.content = b"{bad-json"

    def json(self) -> dict[str, Any]:
        raise ValueError("bad JSON")


def _ack(**overrides: Any) -> dict[str, Any]:
    body = {
        "status": "created",
        "user_id": "u1",
        "group_id": "g1",
        "profile_version": 1,
        "schema_version": 1,
        "updated_at": "2026-07-27T10:11:12.123456Z",
        "profile_digest": canonical_profile_digest(_profile()),
    }
    body.update(overrides)
    return body


def _profile() -> GroupProfile:
    return profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="做 AI Agent",
        need="找渠道",
        offer="产品能力",
        need_disclosure="match_only",
    )


def test_canonical_profile_digest_matches_micro_fixed_order_json() -> None:
    assert canonical_profile_digest(_profile()) == (
        "3d2317e5c9507fa4fefc5b8661300e9afaf58110afed412f69c0a0e7f98c0c11"
    )


def test_persist_group_profile_signs_exact_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured["url"] = url
        captured.update(kwargs)
        return _Response(200, _ack())

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.profile_client.requests.post",
        fake_post,
    )
    result = persist_group_profile(
        profile=_profile(),
        run_id="run_1",
        base_url="http://micro.test",
        secret="test-secret",
        timeout_s=2,
    )

    assert result["status"] == "created"
    assert captured["url"] == f"http://micro.test{PROFILE_PATH}"
    sent = json.loads(captured["data"])
    assert sent == {
        "run_id": "run_1",
        "user_id": "u1",
        "group_id": "g1",
        "profile": {
            "doing": {
                "value": "做 AI Agent",
                "disclosure": "inferred_unconfirmed",
            },
            "need": {"value": "找渠道", "disclosure": "match_only"},
            "offer": {
                "value": "产品能力",
                "disclosure": "inferred_unconfirmed",
            },
            "schema_version": 1,
        },
    }
    headers = captured["headers"]
    body_sha = hashlib.sha256(captured["data"]).hexdigest()
    canonical = "\n".join(
        [
            "GA-CALLBACK-V1",
            "method=POST",
            f"path={PROFILE_PATH}",
            f"body_sha256={body_sha}",
            f"ts={headers['X-GA-Callback-Timestamp']}",
            f"nonce={headers['X-GA-Callback-Nonce']}",
        ]
    )
    expected_signature = hmac.new(
        b"test-secret",
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert headers["X-GA-Callback-Signature"] == expected_signature
    assert captured["allow_redirects"] is False


@pytest.mark.parametrize(
    ("status", "version"),
    [("idempotent", 1), ("updated", 2)],
)
def test_persist_group_profile_accepts_digest_bound_same_run_ack(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    version: int,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.profile_client.requests.post",
        lambda *_args, **_kwargs: _Response(
            200,
            _ack(status=status, profile_version=version),
        ),
    )

    result = persist_group_profile(
        profile=_profile(),
        run_id="run_1",
        base_url="http://micro.test",
        secret="test-secret",
    )

    assert result["status"] == status
    assert result["profile_digest"] == canonical_profile_digest(_profile())


@pytest.mark.parametrize(
    ("status_code", "body", "expected"),
    [
        (422, {"error": "invalid_profile"}, "http_422"),
        (500, {"error": "internal_error"}, "http_500"),
        (200, _ack(user_id="other"), "user_id_mismatch"),
        (200, _ack(group_id="other"), "group_id_mismatch"),
        (200, _ack(status="unknown"), "invalid_status"),
        (200, _ack(profile_version=0), "invalid_profile_version"),
        (200, _ack(profile_version=True), "invalid_profile_version"),
        (200, _ack(status="created", profile_version=2), "invalid_profile_version"),
        (200, _ack(schema_version=2), "schema_version_mismatch"),
        (200, _ack(updated_at="2026-07-27T10:11:12Z"), "invalid_updated_at"),
        (200, _ack(updated_at="not-a-date.123456Z"), "invalid_updated_at"),
        (200, _ack(profile_digest="A" * 64), "invalid_profile_digest"),
        (200, _ack(profile_digest="0" * 64), "profile_digest_mismatch"),
    ],
)
def test_persist_group_profile_rejects_bad_response(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    body: dict[str, Any],
    expected: str,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.profile_client.requests.post",
        lambda *_args, **_kwargs: _Response(status_code, body),
    )

    with pytest.raises(ProfileHttpError, match=expected):
        persist_group_profile(
            profile=_profile(),
            run_id="run_1",
            base_url="http://micro.test",
            secret="test-secret",
        )


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout("timed out"),
        requests.ConnectionError("network down"),
    ],
)
def test_persist_group_profile_reports_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: requests.RequestException,
) -> None:
    def fail_post(*_args: Any, **_kwargs: Any) -> _Response:
        raise error

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.profile_client.requests.post",
        fail_post,
    )

    with pytest.raises(ProfileHttpError, match=f"transport_error:{type(error).__name__}"):
        persist_group_profile(
            profile=_profile(),
            run_id="run_1",
            base_url="http://micro.test",
            secret="test-secret",
            timeout_s=0.01,
        )


def test_persist_group_profile_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.profile_client.requests.post",
        lambda *_args, **_kwargs: _InvalidJsonResponse(),
    )

    with pytest.raises(ProfileHttpError, match="invalid_json"):
        persist_group_profile(
            profile=_profile(),
            run_id="run_1",
            base_url="http://micro.test",
            secret="test-secret",
        )


def test_persist_group_profile_rejects_empty_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.profile_client.requests.post",
        lambda *_args, **_kwargs: _Response(200, None),
    )

    with pytest.raises(ProfileHttpError, match="invalid_json"):
        persist_group_profile(
            profile=_profile(),
            run_id="run_1",
            base_url="http://micro.test",
            secret="test-secret",
        )


def test_persist_group_profile_requires_run_id() -> None:
    with pytest.raises(ProfileHttpError, match="missing_run_id"):
        persist_group_profile(
            profile=_profile(),
            run_id="",
            base_url="http://micro.test",
            secret="test-secret",
        )


def _invoke_save_tool(base_dir: Path, *, run_id: str = "run_1") -> str:
    return str(
        save_group_profile.invoke(
            {
                "doing": "做 AI Agent",
                "need": "找渠道",
                "offer": "产品能力",
                "need_disclosure": "match_only",
            },
            config={
                "metadata": {
                    "user_id": "u1",
                    "group_id": "g1",
                    "base_dir": str(base_dir),
                    "run_id": run_id,
                }
            },
        )
    )


class _Checkpointer:
    def flush(self) -> None:
        pass


class _AgentState:
    values = {"messages": []}


class _ToolCallingAgent:
    def __init__(self, profiles: list[GroupProfile]) -> None:
        self.profiles = profiles
        self.calls = 0
        self.messages: list[Any] = []
        self.checkpointer = _Checkpointer()

    async def aget_state(self, _config: dict[str, Any]) -> _AgentState:
        return _AgentState()

    async def ainvoke(
        self,
        payload: dict[str, list[Any]],
        config: dict[str, Any],
    ) -> dict[str, list[Any]]:
        index = min(self.calls, len(self.profiles) - 1)
        profile = self.profiles[index]
        self.calls += 1
        result = save_group_profile.invoke(
            {
                "doing": profile.doing.value,
                "need": profile.need.value,
                "offer": profile.offer.value,
                "doing_disclosure": profile.doing.disclosure.value,
                "need_disclosure": profile.need.disclosure.value,
                "offer_disclosure": profile.offer.disclosure.value,
            },
            config=config,
        )
        self.messages.extend(
            [
                payload["messages"][0],
                ToolMessage(
                    content=str(result),
                    tool_call_id=f"profile_tool_{self.calls}",
                    name="save_group_profile",
                ),
                AIMessage(content=f"tool result: {result}"),
            ]
        )
        return {"messages": list(self.messages)}


def _async_request() -> AsyncCallRequest:
    return AsyncCallRequest(
        run_id="run_profile_1",
        idempotency_key="idem_profile_1",
        user_id="u1",
        unionid="union_1",
        group_id="g1",
        conversation_id="conversation_1",
        message="更新我的画像",
        callback_url="http://micro.test/group_agent_callbacks/run_profile_1",
        membership="in_group",
        run_match=True,
        run_invite=True,
    )


def _trusted_session() -> TrustedSession:
    return TrustedSession(
        principal=SessionPrincipal(
            user_id="u1",
            unionid="union_1",
            user_token=None,
            source="signed_oauth_principal",
        ),
        group_id="g1",
        group_token=None,
        membership=MembershipResult(
            tier=CapabilityTier.in_group,
            source="database_membership",
        ),
    )


async def _run_async_profile_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    agent: _ToolCallingAgent,
    req: AsyncCallRequest | None = None,
) -> list[dict[str, Any]]:
    envelopes: list[dict[str, Any]] = []

    async def capture_callback(
        *,
        callback_url: str,
        envelope_dict: dict[str, Any],
    ) -> bool:
        assert callback_url.endswith("/group_agent_callbacks/run_profile_1")
        envelopes.append(envelope_dict)
        return True

    monkeypatch.setattr(
        "apps.group_agent_api.app.async_manager.send_callback_event",
        capture_callback,
    )
    state = AppState(agent=agent, base_dir=tmp_path)
    await execute_async_run(
        req=req or _async_request(),
        session=_trusted_session(),
        state=state,
        tid="ga::u1::g1::conversation_1",
    )
    return envelopes


def test_http_tool_writes_cache_only_after_remote_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_persist(*, profile: Any, run_id: str, **kwargs: Any) -> dict[str, Any]:
        assert load_profile(tmp_path, "u1", "g1") is None
        calls.append((profile.user_id, profile.group_id, run_id))
        return _ack()

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.integration_mode",
        lambda: "http",
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.persist_group_profile",
        fake_persist,
    )

    result = _invoke_save_tool(tmp_path)

    assert result.startswith("ok: saved profile to database")
    assert calls == [("u1", "g1", "run_1")]
    assert load_profile(tmp_path, "u1", "g1") is not None


def test_http_tool_does_not_write_cache_when_remote_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.integration_mode",
        lambda: "http",
    )

    def fail_persist(**_kwargs: Any) -> dict[str, Any]:
        raise ProfileHttpError("transport_error:Timeout")

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.persist_group_profile",
        fail_persist,
    )

    result = _invoke_save_tool(tmp_path)

    assert result == "error: profile_database:transport_error:Timeout"
    assert load_profile(tmp_path, "u1", "g1") is None


def test_http_tool_persists_authoritative_profile_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.integration_mode",
        lambda: "http",
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.persist_group_profile",
        lambda **_kwargs: _ack(status="updated", profile_version=7),
    )

    result = _invoke_save_tool(tmp_path)

    assert "version=7" in result
    cached = load_profile(tmp_path, "u1", "g1")
    assert cached is not None
    assert cached.profile_version == 7


def test_http_tool_keeps_cache_unchanged_for_stale_ack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="权威缓存中的新项目",
        need="新需求",
        offer="新资源",
    )
    save_profile(tmp_path, existing)
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.integration_mode",
        lambda: "http",
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.persist_group_profile",
        lambda **_kwargs: _ack(status="stale_ignored", profile_version=4),
    )

    result = _invoke_save_tool(tmp_path)

    assert result == (
        "ok: profile_superseded; database kept a newer profile "
        "(version=4); local cache unchanged"
    )
    cached = load_profile(tmp_path, "u1", "g1")
    assert cached is not None
    assert cached.updated_at == existing.updated_at
    assert cached.doing.value == "权威缓存中的新项目"


def test_stub_tool_never_calls_remote_and_keeps_local_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.integration_mode",
        lambda: "stub",
    )

    def unexpected_remote(**_kwargs: Any) -> dict[str, Any]:
        pytest.fail("stub mode must not call Micro")

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.persist_group_profile",
        unexpected_remote,
    )

    result = _invoke_save_tool(tmp_path, run_id="")

    assert result == "ok: saved profile to /users/u1/groups/g1/profile.json"
    assert load_profile(tmp_path, "u1", "g1") is not None


def test_invoke_config_uses_trusted_run_context_over_client_metadata() -> None:
    config = _invoke_config(
        tid="ga::u1::g1::trusted-conversation",
        user_id="u1",
        group_id="g1",
        base_dir="/tmp/runtime",
        membership="in_group",
        metadata={
            "run_id": "client-run",
            "conversation_id": "client-conversation",
            "user_id": "client-user",
            "custom": "preserved",
        },
        run_id="trusted-run",
        conversation_id="trusted-conversation",
    )

    assert config["metadata"] == {
        "custom": "preserved",
        "user_id": "u1",
        "group_id": "g1",
        "base_dir": "/tmp/runtime",
        "membership": "in_group",
        "run_id": "trusted-run",
        "conversation_id": "trusted-conversation",
    }


async def test_async_stale_ack_is_superseded_terminal_without_retry_or_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="数据库中的较新项目",
        need="较新需求",
        offer="较新资源",
    )
    attempted = _profile()
    save_profile(tmp_path, existing)
    agent = _ToolCallingAgent([attempted])
    remote_calls = 0

    def stale_ack(**_kwargs: Any) -> dict[str, Any]:
        nonlocal remote_calls
        remote_calls += 1
        return _ack(
            status="stale_ignored",
            profile_version=4,
            profile_digest=canonical_profile_digest(existing),
        )

    def unexpected_match(**_kwargs: Any) -> None:
        pytest.fail("superseded profiles must not enter matching")

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.integration_mode",
        lambda: "http",
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.persist_group_profile",
        stale_ack,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.search_tool.match_backend.run_match",
        unexpected_match,
    )

    envelopes = await _run_async_profile_flow(
        monkeypatch,
        tmp_path=tmp_path,
        agent=agent,
    )

    assert agent.calls == 1
    assert remote_calls == 1
    assert [envelope["event"] for envelope in envelopes] == ["progress", "final"]
    final = envelopes[-1]["payload"]
    assert final["profile_persisted"] is False
    assert final["profile_status"] == "superseded"
    assert final["match_status"] == "skipped"
    assert final["match_reason"] == "profile_superseded"
    assert final["candidates"] == []
    assert final["invite_text"] is None
    cached = load_profile(tmp_path, "u1", "g1")
    assert cached is not None
    assert cached.updated_at == existing.updated_at
    assert cached.doing.value == existing.doing.value


async def test_async_same_run_idempotent_retry_recovers_local_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = _profile()
    agent = _ToolCallingAgent([profile, profile])
    statuses = iter(("created", "idempotent"))
    remote_digests: list[str] = []

    def persist(**kwargs: Any) -> dict[str, Any]:
        remote_profile = kwargs["profile"]
        digest = canonical_profile_digest(remote_profile)
        remote_digests.append(digest)
        return _ack(status=next(statuses), profile_digest=digest)

    real_save_profile = save_profile
    local_attempts = 0

    def flaky_local_save(base_dir: Path, candidate: GroupProfile) -> Path:
        nonlocal local_attempts
        local_attempts += 1
        if local_attempts == 1:
            raise OSError("simulated local disk failure")
        return real_save_profile(base_dir, candidate)

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.integration_mode",
        lambda: "http",
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.persist_group_profile",
        persist,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.save_profile",
        flaky_local_save,
    )
    monkeypatch.setenv("GROUP_AGENT_GROUNDED_FINAL_ENABLED", "1")
    req = _async_request().model_copy(
        update={
            "run_match": False,
            "run_invite": False,
            "rollout_context": RolloutContext(protocol_mode="grounded_v2"),
        }
    )

    envelopes = await _run_async_profile_flow(
        monkeypatch,
        tmp_path=tmp_path,
        agent=agent,
        req=req,
    )

    assert agent.calls == 2
    assert local_attempts == 2
    assert remote_digests == [
        canonical_profile_digest(profile),
        canonical_profile_digest(profile),
    ]
    cached = load_profile(tmp_path, "u1", "g1")
    assert cached is not None
    assert canonical_profile_digest(cached) == remote_digests[-1]
    final = envelopes[-1]["payload"]
    assert final["profile_persisted"] is True
    assert final["profile_status"] == "persisted"
    assert final["reply_mode"] == "profile_confirmation"
    assert final["profile"]["profile_version"] == 1


async def test_async_same_run_changed_retry_converges_to_updated_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _profile()
    second = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="做企业知识库 Agent",
        need="找首批企业客户",
        offer="Agent 产品与交付能力",
        need_disclosure="match_only",
    )
    agent = _ToolCallingAgent([first, second])
    statuses = iter((("created", 1), ("updated", 2)))
    authoritative_digest = ""

    def persist(**kwargs: Any) -> dict[str, Any]:
        nonlocal authoritative_digest
        status, version = next(statuses)
        authoritative_digest = canonical_profile_digest(kwargs["profile"])
        return _ack(
            status=status,
            profile_version=version,
            profile_digest=authoritative_digest,
        )

    real_save_profile = save_profile
    local_attempts = 0

    def flaky_local_save(base_dir: Path, candidate: GroupProfile) -> Path:
        nonlocal local_attempts
        local_attempts += 1
        if local_attempts == 1:
            raise OSError("simulated local disk failure")
        return real_save_profile(base_dir, candidate)

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.integration_mode",
        lambda: "http",
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.persist_group_profile",
        persist,
    )
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.agent.save_profile",
        flaky_local_save,
    )
    req = _async_request().model_copy(update={"run_match": False, "run_invite": False})

    envelopes = await _run_async_profile_flow(
        monkeypatch,
        tmp_path=tmp_path,
        agent=agent,
        req=req,
    )

    assert agent.calls == 2
    cached = load_profile(tmp_path, "u1", "g1")
    assert cached is not None
    assert cached.doing.value == second.doing.value
    assert canonical_profile_digest(cached) == authoritative_digest
    final = envelopes[-1]["payload"]
    assert final["profile_persisted"] is True
    assert final["profile_status"] == "persisted"
