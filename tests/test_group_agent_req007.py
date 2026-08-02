"""REQ-007 tests: HTTP clients, doing-only topic, LLM polish fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.integrations.match_backend import run_match
from apps.group_agent_api.agent_factory.integrations.match_client import (
    fetch_group_agent_match,
)
from apps.group_agent_api.agent_factory.integrations.membership_backend import (
    resolve_session_capability,
)
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
    fetch_membership,
)
from apps.group_agent_api.agent_factory.invite_copy import derive_common_topic
from apps.group_agent_api.agent_factory.invite_llm import generate_invite_with_optional_llm
from apps.group_agent_api.agent_factory.match_stub import MatchResult, MatchStub, set_match_stub
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat


@pytest.fixture(autouse=True)
def _stub_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    set_match_stub(MatchStub())
    yield
    set_match_stub(MatchStub())


def _profile():
    return profile_from_flat(
        user_id="mock_u1",
        group_id="mock_g1",
        doing="智能宠物喂食器",
        need="联网与 App 固件",
        offer="工厂与供应链",
    )


# ---------------------------------------------------------------------------
# Membership HTTP
# ---------------------------------------------------------------------------


def test_fetch_membership_maps_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict:
            return {"tier": "in_group", "event_id": 1041, "reason": "wie_active_hit"}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _Resp(),
    )
    res = fetch_membership(unionid="u_wx", group_token="tok")
    assert res.tier is CapabilityTier.in_group
    assert res.event_id == "1041"


def test_fetch_membership_exception_soft_fails_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        _boom,
    )
    res = fetch_membership(unionid="u", group_token="t")
    assert res.tier is CapabilityTier.unknown


def test_http_mode_ignores_membership_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")

    class _Resp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict:
            return {"tier": "not_in_group", "reason": "wie_active_miss"}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _Resp(),
    )
    res = resolve_session_capability(
        membership_override="in_group",  # would unlock if stub — must be ignored
        unionid="u",
        group_token="g",
        force_mode="http",
    )
    assert res.tier is CapabilityTier.not_in_group
    assert res.source == "http"


# ---------------------------------------------------------------------------
# Match HTTP · doing-only · no plaintext group_id
# ---------------------------------------------------------------------------


def test_fetch_match_doing_only_and_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        content = b"{}"
        text = "{}"

        def json(self) -> dict:
            return {
                "status": "matched",
                "group_id": "1041",
                "query": "q",
                "reason": "matched_1",
                "candidates": [
                    {
                        "user_id": "202",
                        "group_id": "1041",
                        "source_group_id": "1041",
                        "display_name": "周然",
                        "doing": {
                            "value": "智能小家电量产固件",
                            "disclosure": "confirmed_public",
                        },
                        "need": {
                            "value": "秘密融资",
                            "disclosure": "match_only",
                        },
                        "offer": {
                            "value": "推断技能",
                            "disclosure": "inferred_unconfirmed",
                        },
                        "match_score": 0.8,
                        "match_confidence": "high",
                        "bound": True,
                        "wechat_reachable": True,
                    }
                ],
            }

    def _post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_client.requests.post",
        _post,
    )
    monkeypatch.setenv("AIHEHUO_API_KEY", "bearer-test")
    result = fetch_group_agent_match(
        query="联网固件",
        group_token="group.jwt",
        excluded_ids=["1"],
    )
    assert "group_id" not in captured["json"]
    assert "member_ids" not in captured["json"]
    assert captured["json"]["g"] == "group.jwt"
    assert captured["headers"]["Authorization"] == "Bearer bearer-test"
    assert result.status == "matched"
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert "need" not in c
    assert "offer" not in c
    assert c["doing"]["value"] == "智能小家电量产固件"


def test_fetch_group_agent_match_omits_g_when_token_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        content = b"{}"
        text = "{}"

        def json(self):
            return {
                "status": "empty",
                "candidates": [],
                "query": "q",
                "group_id": "global",
                "reason": "sc05_no_suitable_match",
            }

    def _post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_client.requests.post",
        _post,
    )
    monkeypatch.setenv("AIHEHUO_API_KEY", "bearer-test")
    result = fetch_group_agent_match(query="q", group_token=None)
    assert "g" not in captured["json"]
    assert "group_id" not in captured["json"]
    assert result.group_id == "global"
    assert result.status == "empty"


def test_fetch_group_agent_match_drops_candidates_not_reachable_on_wechat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Resp:
        status_code = 200
        content = b"{}"
        text = "{}"

        def json(self):
            return {
                "status": "matched",
                "candidates": [
                    {
                        "user_id": "101",
                        "display_name": "不可触达用户",
                        "doing": {"value": "芯片设计", "disclosure": "confirmed_public"},
                        "bound": False,
                        "wechat_reachable": False,
                    }
                ],
                "query": "芯片设计",
                "group_id": "global",
                "reason": "matched_1",
            }

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_client.requests.post",
        lambda *args, **kwargs: _Resp(),
    )
    monkeypatch.setenv("AIHEHUO_API_KEY", "bearer-test")

    result = fetch_group_agent_match(query="芯片设计")

    assert result.status == "empty"
    assert result.candidates == []
    assert result.reason == "no_wechat_reachable_candidates"


def test_run_match_http_empty_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")

    class _Resp:
        status_code = 401
        text = "nope"
        content = b"nope"

        def json(self):
            return {}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_client.requests.post",
        lambda *a, **k: _Resp(),
    )
    monkeypatch.setenv("AIHEHUO_API_KEY", "x")
    result = run_match(
        query="q",
        group_id="g1",
        group_token="tok",
        force_mode="http",
    )
    assert result.status == "empty"
    assert result.candidates == []


# ---------------------------------------------------------------------------
# doing-only topic
# ---------------------------------------------------------------------------


def test_derive_topic_uses_candidate_doing_and_initiator_need() -> None:
    cands = [
        {
            "user_id": "x",
            "display_name": "周",
            "doing": {"value": "量产固件", "disclosure": "confirmed_public"},
            "offer": {"value": "不应被使用", "disclosure": "confirmed_public"},
        }
    ]
    topic = derive_common_topic(_profile(), cands)
    # Topic centers initiator need for natural group paste (not candidate ad paste).
    assert "联网" in topic.topic or "固件" in topic.topic
    assert "不应被使用" not in topic.topic
    assert topic.degraded is False


# ---------------------------------------------------------------------------
# LLM polish fallback
# ---------------------------------------------------------------------------


def test_llm_polish_falls_back_when_assert_fails() -> None:
    cands = MatchStub().search(
        query="智能宠物喂食器 联网 固件 工厂",
        group_id="mock_g1",
        excluded_ids=["mock_u1"],
    ).candidates

    class _BadModel:
        def invoke(self, _msgs):
            class _M:
                content = "他很适合你当合伙人 @陌生人"

            return _M()

    result = generate_invite_with_optional_llm(
        profile=_profile(),
        candidates=cands,
        match_status="matched",
        willing_to_at=True,
        model=_BadModel(),
        use_llm=True,
    )
    assert result.ok
    assert "当合伙人" not in result.text
    assert "@" in result.text  # template kept


# ---------------------------------------------------------------------------
# RESP-007-FIX regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
def test_fetch_membership_non_2xx_fail_closed(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    class _Resp:
        status_code = status
        content = b'{"tier":"in_group","event_id":1041}'

        def json(self) -> dict:
            return {"tier": "in_group", "event_id": 1041}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _Resp(),
    )
    res = fetch_membership(unionid="u", group_token="t")
    assert res.tier is CapabilityTier.unknown
    assert res.reason == f"http_{status}"


def test_fetch_membership_bad_json_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 200
        content = b"not-json"

        def json(self):
            raise ValueError("nope")

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _Resp(),
    )
    res = fetch_membership(unionid="u", group_token="t")
    assert res.tier is CapabilityTier.unknown
    assert res.reason == "bad_json"


def test_llm_polish_rejects_at_uncertainty_only_and_falls_back() -> None:
    """Orchestrator repro: polished '@Name + uncertainty' must NOT pass five-element assert."""
    cands = MatchStub().search(
        query="智能宠物喂食器 联网 固件 工厂",
        group_id="mock_g1",
        excluded_ids=["mock_u1"],
    ).candidates
    assert cands
    name = cands[0].get("display_name") or cands[0].get("user_id")

    class _ThinModel:
        def invoke(self, _msgs):
            class _M:
                content = f"@{name} 值得聊一次以确认，不一定合适"

            return _M()

    result = generate_invite_with_optional_llm(
        profile=_profile(),
        candidates=cands,
        match_status="matched",
        willing_to_at=True,
        model=_ThinModel(),
        use_llm=True,
    )
    assert result.ok
    # Must fall back to template (has who/resources/topic/low_pressure)
    assert "我在做" in result.text
    assert "聊聊就好" in result.text
    assert result.text != f"@{name} 值得聊一次以确认，不一定合适"


def test_principal_http_rejects_body_user_id_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
        resolve_session_principal,
        sign_principal,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()
    signed = sign_principal(
        user_id="real_user",
        unionid="wx_u1",
        method="POST",
        path="/chat",
        secret="test-secret",
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/chat",
        "headers": [(k.lower().encode(), v.encode()) for k, v in signed.items()],
    }
    with pytest.raises(HTTPException) as ei:
        resolve_session_principal(
            Request(scope),
            body_user_id="attacker",
            body_unionid=None,
            body_user_token=None,
            force_mode="http",
        )
    assert ei.value.status_code == 400
    assert ei.value.detail["error"] == "identity_injection_forbidden"


def test_principal_http_bind_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
        resolve_session_principal,
        sign_principal,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    monkeypatch.setenv(
        "GROUP_AGENT_UNIONID_BIND_JSON", '{"wx_u1":"real_user"}'
    )
    clear_nonce_cache()
    signed = sign_principal(
        user_id="other_user",
        unionid="wx_u1",
        method="POST",
        path="/chat",
        secret="test-secret",
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/chat",
        "headers": [(k.lower().encode(), v.encode()) for k, v in signed.items()],
    }
    with pytest.raises(HTTPException) as ei:
        resolve_session_principal(
            Request(scope),
            body_user_id="other_user",
            force_mode="http",
        )
    assert ei.value.status_code == 403
    assert ei.value.detail["error"] == "unionid_user_id_mismatch"


def test_group_bind_rejects_plaintext_mismatch() -> None:
    from fastapi import HTTPException
    from apps.group_agent_api.agent_factory.integrations.group_bind import (
        resolve_trusted_group_id,
    )
    from apps.group_agent_api.agent_factory.integrations.membership_client import (
        MembershipResult,
    )

    membership = MembershipResult(
        tier=CapabilityTier.in_group, event_id="1041", source="http"
    )
    with pytest.raises(HTTPException) as ei:
        resolve_trusted_group_id(
            plaintext_group_id="other_group",
            membership=membership,
            force_mode="http",
        )
    assert ei.value.status_code == 400
    assert ei.value.detail["error"] == "group_id_mismatch"


def test_group_bind_accepts_global_session_despite_membership_event() -> None:
    """REQ-032: plaintext global must not collide with leftover group_token event_id."""
    from apps.group_agent_api.agent_factory.integrations.group_bind import (
        resolve_trusted_group_id,
    )
    from apps.group_agent_api.agent_factory.integrations.membership_client import (
        MembershipResult,
    )

    membership = MembershipResult(
        tier=CapabilityTier.in_group, event_id="763", source="http"
    )
    trusted = resolve_trusted_group_id(
        plaintext_group_id="global",
        membership=membership,
        force_mode="http",
    )
    assert trusted == "global"


def test_global_session_unlocks_network_without_group_token() -> None:
    """Authenticated global bucket must not fall to tier=unknown (reply-loop)."""
    res = resolve_session_capability(
        membership_override="in_group",  # ignored in http
        unionid="wx_union_1",
        group_token=None,
        force_mode="http",
        group_id="global",
        user_id="448838",
    )
    assert res.tier == CapabilityTier.in_group
    assert res.reason == "global_session_authenticated"
    assert res.source == "http_global"


def test_align_match_keeps_pool_for_global_bucket() -> None:
    from apps.group_agent_api.agent_factory.integrations.group_bind import (
        align_match_to_trusted_group,
    )

    raw = MatchResult(
        status="matched",
        group_id="763",
        query="q",
        reason="x",
        candidates=[
            {
                "user_id": "1",
                "group_id": "999",
                "source_group_id": "999",
                "display_name": "X",
                "is_reachable": True,
                "doing": {"value": "固件", "disclosure": "confirmed_public"},
            }
        ],
    )
    aligned = align_match_to_trusted_group(raw, trusted_group_id="global")
    assert aligned.status == "matched"
    assert aligned.group_id == "global"
    assert len(aligned.candidates) == 1
    assert aligned.candidates[0]["source_group_id"] == "999"


def test_align_match_drops_foreign_group() -> None:
    from apps.group_agent_api.agent_factory.integrations.group_bind import (
        align_match_to_trusted_group,
    )

    raw = MatchResult(
        status="matched",
        group_id="9999",
        query="q",
        reason="x",
        candidates=[
            {
                "user_id": "1",
                "group_id": "9999",
                "source_group_id": "9999",
                "display_name": "X",
                "doing": {"value": "固件", "disclosure": "confirmed_public"},
            }
        ],
    )
    aligned = align_match_to_trusted_group(raw, trusted_group_id="1041")
    assert aligned.status == "empty"
    assert aligned.candidates == []
    assert aligned.group_id == "1041"


@pytest.mark.asyncio
async def test_http_invite_forbids_caller_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.app.endpoints.invite import invite
    from apps.group_agent_api.app.models import InviteRequest
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.agent_factory.profile_store import save_profile

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    monkeypatch.setenv("GROUP_AGENT_UNIONID_BIND_JSON", '{"wx":"u1"}')

    class _MemResp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict:
            return {"tier": "in_group", "event_id": "1041"}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _MemResp(),
    )

    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="u1",
            group_id="1041",
            doing="固件",
            need="量产",
            offer="工厂",
        ),
    )
    st = AppState(agent=object(), base_dir=tmp_path, polish_model=None)
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
        sign_principal,
    )

    clear_nonce_cache()
    signed = sign_principal(
        user_id="u1", unionid="wx", method="POST", path="/invite", secret="test-secret"
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/invite",
        "headers": [(k.lower().encode(), v.encode()) for k, v in signed.items()],
    }
    req = InviteRequest(
        user_id="u1",
        group_id="1041",
        group_token="tok",
        candidates=[
            {
                "user_id": "fake",
                "group_id": "1041",
                "source_group_id": "1041",
                "display_name": "伪造",
                "doing": {"value": "假 doing", "disclosure": "confirmed_public"},
            }
        ],
    )
    with pytest.raises(HTTPException) as ei:
        await invite(req, st, Request(scope))
    assert ei.value.status_code == 400
    assert ei.value.detail["error"] == "candidates_injection_forbidden"


@pytest.mark.asyncio
async def test_session_resolve_off_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync membership sleep must not block other awaitables (to_thread)."""
    import asyncio
    import time
    from starlette.requests import Request
    from apps.group_agent_api.app.session import resolve_trusted_session

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")

    def _slow_membership(*a, **k):
        time.sleep(0.25)
        return MembershipResult(
            tier=CapabilityTier.in_group,
            event_id="1041",
            reason="ok",
            source="http",
        )

    monkeypatch.setattr(
        "apps.group_agent_api.app.session.resolve_session_capability",
        _slow_membership,
    )
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
        sign_principal,
    )

    clear_nonce_cache()
    signed = sign_principal(
        user_id="u1", unionid="wx", method="POST", path="/match", secret="test-secret"
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/match",
        "headers": [(k.lower().encode(), v.encode()) for k, v in signed.items()],
    }
    req = Request(scope)

    async def _ticker() -> int:
        n = 0
        end = time.monotonic() + 0.2
        while time.monotonic() < end:
            await asyncio.sleep(0.01)
            n += 1
        return n

    t0 = time.monotonic()
    ticks, _session = await asyncio.gather(
        _ticker(),
        resolve_trusted_session(
            req,
            body_user_id="u1",
            body_group_id="1041",
            body_membership="unknown",
            body_unionid=None,
            body_group_token="tok",
            body_user_token=None,
        ),
    )
    elapsed = time.monotonic() - t0
    assert ticks >= 5, "event loop should keep ticking during membership sleep"
    assert elapsed >= 0.2


# ---------------------------------------------------------------------------
# RESP-007-FIX2 regressions · HMAC principal + owner guards + startup
# ---------------------------------------------------------------------------


def _signed_scope(
    *,
    user_id: str,
    unionid: str,
    path: str,
    method: str = "POST",
    secret: str = "test-secret",
    user_token: str | None = None,
    group_token: str | None = None,
    ts: int | None = None,
    nonce: str | None = None,
    bad_sig: bool = False,
    mutate_group_token: str | None = None,
) -> dict:
    from apps.group_agent_api.agent_factory.integrations.principal import sign_principal

    signed = sign_principal(
        user_id=user_id,
        unionid=unionid,
        user_token=user_token,
        group_token=group_token,
        method=method,
        path=path,
        secret=secret,
        ts=ts,
        nonce=nonce,
    )
    if bad_sig:
        signed["X-GA-Signature"] = "0" * 64
    if mutate_group_token is not None:
        signed["X-GA-Group-Token"] = mutate_group_token
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in signed.items()],
    }


def test_forged_x_ga_headers_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
        resolve_session_principal,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()
    # Bare headers without signature — previously accepted
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/chat",
        "headers": [
            (b"x-ga-user-id", b"victim"),
            (b"x-ga-unionid", b"attacker_unionid"),
        ],
    }
    with pytest.raises(HTTPException) as ei:
        resolve_session_principal(
            Request(scope), body_user_id="victim", force_mode="http"
        )
    assert ei.value.status_code == 401
    assert ei.value.detail["error"] == "missing_signed_principal"


def test_bad_hmac_signature_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
        resolve_session_principal,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()
    scope = _signed_scope(
        user_id="victim", unionid="u", path="/chat", bad_sig=True
    )
    with pytest.raises(HTTPException) as ei:
        resolve_session_principal(
            Request(scope), body_user_id="victim", force_mode="http"
        )
    assert ei.value.detail["error"] == "principal_signature_invalid"


def test_hmac_valid_principal_accepted_without_static_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signed OAuth principal is the authoritative bind; empty JSON must not be required."""
    from starlette.requests import Request
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
        resolve_session_principal,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    monkeypatch.delenv("GROUP_AGENT_UNIONID_BIND_JSON", raising=False)
    clear_nonce_cache()
    scope = _signed_scope(user_id="u1", unionid="wx1", path="/chat")
    principal = resolve_session_principal(
        Request(scope), body_user_id="u1", force_mode="http"
    )
    assert principal.user_id == "u1"
    assert principal.unionid == "wx1"
    assert principal.source == "signed_oauth_principal"


def test_nonce_replay_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
        resolve_session_principal,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()
    scope = _signed_scope(
        user_id="u1", unionid="wx", path="/chat", nonce="fixed-nonce-1"
    )
    resolve_session_principal(Request(scope), body_user_id="u1", force_mode="http")
    with pytest.raises(HTTPException) as ei:
        resolve_session_principal(
            Request(scope), body_user_id="u1", force_mode="http"
        )
    assert ei.value.detail["error"] == "principal_nonce_replay"


def test_startup_fail_closed_http_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apps.group_agent_api.agent_factory.integrations.config import (
        assert_startup_security,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.delenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", raising=False)
    monkeypatch.delenv("GROUP_AGENT_ENV", raising=False)
    monkeypatch.delenv("GROUP_AGENT_REQUIRE_TRUSTED_PRINCIPAL", raising=False)
    with pytest.raises(RuntimeError, match="HMAC_SECRET"):
        assert_startup_security()


def test_startup_fail_closed_prod_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.group_agent_api.agent_factory.integrations.config import (
        assert_startup_security,
    )

    monkeypatch.setenv("GROUP_AGENT_ENV", "production")
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "x")
    with pytest.raises(RuntimeError, match="forbids GROUP_AGENT_INTEGRATION=stub"):
        assert_startup_security()


@pytest.mark.asyncio
async def test_http_profile_cross_user_forbidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.app.endpoints import profile as profile_ep
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.agent_factory.profile_store import save_profile
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()

    class _MemResp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict:
            return {"tier": "in_group", "event_id": "1041"}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _MemResp(),
    )
    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="victim", group_id="1041", doing="秘", need="n", offer="o"
        ),
    )
    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="attacker", group_id="1041", doing="a", need="n", offer="o"
        ),
    )
    st = AppState(agent=object(), base_dir=tmp_path, polish_model=None)
    # Attacker signs as self but asks for victim profile
    scope = _signed_scope(
        user_id="attacker",
        unionid="wx_a",
        path="/profile",
        method="GET",
        group_token="tok",
    )
    with pytest.raises(HTTPException) as ei:
        await profile_ep.get_profile(
            user_id="victim",
            group_id="1041",
            state=st,
            request=Request(scope),
        )
    assert ei.value.status_code in {400, 403}


@pytest.mark.asyncio
async def test_http_profile_self_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from starlette.requests import Request
    from apps.group_agent_api.app.endpoints import profile as profile_ep
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.agent_factory.profile_store import save_profile
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()

    class _MemResp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict:
            return {"tier": "in_group", "event_id": "1041"}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _MemResp(),
    )
    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="u1", group_id="1041", doing="固件", need="n", offer="o"
        ),
    )
    st = AppState(agent=object(), base_dir=tmp_path, polish_model=None)
    scope = _signed_scope(
        user_id="u1",
        unionid="wx",
        path="/profile",
        method="GET",
        group_token="tok",
    )
    resp = await profile_ep.get_profile(
        user_id="u1",
        group_id="1041",
        state=st,
        request=Request(scope),
    )
    assert resp.exists is True
    assert resp.profile["doing"]["value"] == "固件"


@pytest.mark.asyncio
async def test_http_reset_cross_user_keeps_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.app.endpoints import reset as reset_ep
    from apps.group_agent_api.app.models import ResetRequest
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.agent_factory.profile_store import (
        disk_profile_path,
        save_profile,
    )
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()

    class _MemResp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict:
            return {"tier": "in_group", "event_id": "1041"}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _MemResp(),
    )
    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="victim", group_id="1041", doing="keep", need="n", offer="o"
        ),
    )
    victim_path = disk_profile_path(tmp_path, "victim", "1041")
    assert victim_path.exists()

    st = AppState(agent=object(), base_dir=tmp_path, polish_model=None)
    scope = _signed_scope(
        user_id="attacker", unionid="wx_a", path="/reset", method="POST"
    )
    with pytest.raises(HTTPException) as ei:
        await reset_ep.reset(
            ResetRequest(
                user_id="victim",
                group_id="1041",
                clear_profile=True,
                group_token="tok",
            ),
            st,
            Request(scope),
        )
    assert ei.value.status_code in {400, 403}
    assert victim_path.exists(), "victim profile must remain"


@pytest.mark.asyncio
async def test_http_reset_cross_group_keeps_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.app.endpoints import reset as reset_ep
    from apps.group_agent_api.app.models import ResetRequest
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.agent_factory.profile_store import (
        disk_profile_path,
        save_profile,
    )
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()

    class _MemResp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict:
            return {"tier": "in_group", "event_id": "1041"}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _MemResp(),
    )
    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="u1", group_id="9999", doing="other", need="n", offer="o"
        ),
    )
    other_path = disk_profile_path(tmp_path, "u1", "9999")
    assert other_path.exists()

    st = AppState(agent=object(), base_dir=tmp_path, polish_model=None)
    scope = _signed_scope(user_id="u1", unionid="wx", path="/reset", method="POST")
    with pytest.raises(HTTPException) as ei:
        await reset_ep.reset(
            ResetRequest(
                user_id="u1",
                group_id="9999",  # mismatches membership event_id 1041
                clear_profile=True,
                group_token="tok",
            ),
            st,
            Request(scope),
        )
    assert ei.value.status_code == 400
    assert ei.value.detail["error"] == "group_id_mismatch"
    assert other_path.exists()


@pytest.mark.asyncio
async def test_http_reset_self_clears_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from starlette.requests import Request
    from apps.group_agent_api.app.endpoints import reset as reset_ep
    from apps.group_agent_api.app.models import ResetRequest
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.agent_factory.profile_store import (
        disk_profile_path,
        save_profile,
    )
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()

    class _MemResp:
        status_code = 200
        content = b"{}"

        def json(self) -> dict:
            return {"tier": "in_group", "event_id": "1041"}

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.requests.post",
        lambda *a, **k: _MemResp(),
    )
    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="u1", group_id="1041", doing="x", need="n", offer="o"
        ),
    )
    path = disk_profile_path(tmp_path, "u1", "1041")
    assert path.exists()
    st = AppState(agent=object(), base_dir=tmp_path, polish_model=None)
    scope = _signed_scope(user_id="u1", unionid="wx", path="/reset", method="POST")
    resp = await reset_ep.reset(
        ResetRequest(
            user_id="u1",
            group_id="1041",
            clear_profile=True,
            group_token="tok",
        ),
        st,
        Request(scope),
    )
    assert resp.profile_cleared is True
    assert not path.exists()


# ---------------------------------------------------------------------------
# RESP-007-FIX3 · no token-in-URL on /profile
# ---------------------------------------------------------------------------


def test_profile_openapi_has_no_token_query_params() -> None:
    from apps.group_agent_api.app import create_app

    app = create_app()
    schema = app.openapi()
    profile = schema["paths"]["/profile"]["get"]
    params = profile.get("parameters") or []
    names = {p.get("name") for p in params if p.get("in") == "query"}
    assert "group_token" not in names
    assert "user_token" not in names
    assert "unionid" not in names
    assert "user_id" in names
    assert "group_id" in names


def test_tampered_group_token_header_fails_hmac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from starlette.requests import Request
    from fastapi import HTTPException
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
        resolve_session_principal,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()
    scope = _signed_scope(
        user_id="u1",
        unionid="wx",
        path="/profile",
        method="GET",
        group_token="real-tok",
        mutate_group_token="tampered-tok",
    )
    with pytest.raises(HTTPException) as ei:
        resolve_session_principal(
            Request(scope), body_user_id="u1", force_mode="http"
        )
    assert ei.value.detail["error"] == "principal_signature_invalid"


@pytest.mark.asyncio
async def test_profile_reads_group_token_from_header_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Access path uses header group token bound into HMAC — not URL query."""
    from starlette.requests import Request
    from apps.group_agent_api.app.endpoints import profile as profile_ep
    from apps.group_agent_api.app.state import AppState
    from apps.group_agent_api.agent_factory.profile_store import save_profile
    from apps.group_agent_api.agent_factory.integrations.principal import (
        clear_nonce_cache,
    )

    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "http")
    monkeypatch.setenv("GROUP_AGENT_PRINCIPAL_HMAC_SECRET", "test-secret")
    clear_nonce_cache()

    captured: dict[str, str] = {}

    def _mem(*, unionid: str, group_token: str, **k):
        captured["group_token"] = group_token
        return MembershipResult(
            tier=CapabilityTier.in_group, event_id="1041", source="http"
        )

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_backend.fetch_membership",
        _mem,
    )
    # membership_backend imports fetch_membership - patch at client used by backend
    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.membership_client.fetch_membership",
        _mem,
    )

    save_profile(
        tmp_path,
        profile_from_flat(
            user_id="u1", group_id="1041", doing="固件", need="n", offer="o"
        ),
    )
    st = AppState(agent=object(), base_dir=tmp_path, polish_model=None)
    scope = _signed_scope(
        user_id="u1",
        unionid="wx",
        path="/profile",
        method="GET",
        group_token="header-group-jwt",
    )
    # Simulate a Request whose URL still contains leaked query (must be ignored).
    scope["query_string"] = b"user_id=u1&group_id=1041&group_token=LEAKED&user_token=LEAKED"
    resp = await profile_ep.get_profile(
        user_id="u1",
        group_id="1041",
        state=st,
        request=Request(scope),
    )
    assert resp.exists is True
    assert captured["group_token"] == "header-group-jwt"
    assert "LEAKED" not in captured["group_token"]
