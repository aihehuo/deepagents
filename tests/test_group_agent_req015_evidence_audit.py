"""REQ-015 deterministic candidate-evidence and audit-semantic tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from apps.group_agent_api.agent_factory.capability import CapabilityTier
from apps.group_agent_api.agent_factory.disclosure import stable_candidate_user_id
from apps.group_agent_api.agent_factory.guard import enforce_capability_guard
from apps.group_agent_api.agent_factory.integrations.membership_client import (
    MembershipResult,
)
from apps.group_agent_api.agent_factory.integrations.group_bind import (
    align_match_to_trusted_group,
)
from apps.group_agent_api.agent_factory.integrations.match_client import (
    fetch_group_agent_match,
)
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.agent_factory.invite_copy import generate_invite_copy
from apps.group_agent_api.agent_factory.invite_llm import (
    assert_exact_polished_mentions,
    generate_invite_with_optional_llm,
)
from apps.group_agent_api.agent_factory.match_stub import MatchResult
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.profile_store import save_profile
from apps.group_agent_api.app.async_manager import _execute_core_agent
from apps.group_agent_api.app.endpoints import chat as chat_endpoint
from apps.group_agent_api.app.models import AsyncCallRequest, ChatRequest
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState
from apps.group_agent_api.fixtures.human_audit import (
    HumanAuditCollector,
    HumanAuditError,
    assert_auditable_candidate_evidence,
    excerpt_coverage,
    extract_high_value_sentences,
    render_markdown,
)


def _candidate(
    user_id: str,
    *,
    doing: str | None,
    group_id: str = "group_l1_alpha",
    disclosure: str = "confirmed_public",
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "user_id": user_id,
        "group_id": group_id,
        "source_group_id": group_id,
        "display_name": user_id,
        "offer": {
            "value": "System architecture",
            "disclosure": "confirmed_public",
        },
        "match_confidence": "high",
    }
    if doing is not None:
        candidate["doing"] = {"value": doing, "disclosure": disclosure}
    return candidate


def _mixed_candidates() -> list[dict[str, Any]]:
    return [
        _candidate("u101", doing="Building LLM agents"),
        _candidate("u104", doing="Distributed systems"),
        _candidate("u102", doing=None),
    ]


def _profile():
    return profile_from_flat(
        user_id="u105",
        group_id="group_l1_alpha",
        doing="AI Agent 产品",
        need="Python 技术负责人",
        offer="客户资源",
    )


def _assert_consistent_payload(payload: dict[str, Any]) -> None:
    candidate_ids = [candidate["user_id"] for candidate in payload["candidates"]]
    assert candidate_ids == ["u101", "u104"]
    assert payload["mentioned_user_ids"] == ["u101", "u104"]
    assert payload["match_status"] == "matched"
    assert payload["delivery_kind"] == "directed"
    assert payload["invite_ok"] is True
    assert "2 位" in payload["reply"]
    assert "@u101" in payload["invite_text"]
    assert "@u104" in payload["invite_text"]
    assert "@u102" not in payload["invite_text"]
    assert "相关公开经验" not in payload["invite_text"]


def test_evidence_guard_removes_empty_nonpublic_and_cross_group_candidates() -> None:
    candidates = [
        *_mixed_candidates(),
        _candidate(
            "u103",
            doing="Private work",
            disclosure="match_only",
        ),
        _candidate("u201", doing="Foreign public work", group_id="group_l1_beta"),
    ]
    guarded = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="",
        candidates=candidates,
        caller_group_id="group_l1_alpha",
        user_id="u105",
    )

    assert [candidate["user_id"] for candidate in guarded.candidates] == [
        "u101",
        "u104",
    ]
    assert guarded.blocked is True
    assert any(
        violation == "missing_public_match_basis:u102"
        for violation in guarded.violations
    )
    assert any("disclosure_leak:u103" in item for item in guarded.violations)
    assert any("cross_group:u201" in item for item in guarded.violations)


def test_guard_rejects_missing_and_deduplicates_stable_candidate_ids() -> None:
    u101 = _candidate("u101", doing="Building LLM agents")
    missing = _candidate("", doing="Anonymous work")
    guarded = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="",
        candidates=[u101, dict(u101), missing],
        caller_group_id="group_l1_alpha",
        user_id="u105",
    )

    assert [candidate["user_id"] for candidate in guarded.candidates] == ["u101"]
    assert guarded.blocked is True
    assert "duplicate_candidate_id:u101" in guarded.violations
    assert "missing_candidate_id" in guarded.violations


@pytest.mark.parametrize(
    "raw_user_id",
    [
        " u101 ",
        "u101 ",
        " u101",
        "   ",
        101,
        True,
        None,
        "用户101",
        "u.101",
        "u@101",
        "u/101",
        "u" * 65,
    ],
)
def test_noncanonical_candidate_ids_fail_closed_without_normalization(
    raw_user_id: Any,
) -> None:
    candidate = _candidate("u101", doing="Building LLM agents")
    candidate["user_id"] = raw_user_id
    assert stable_candidate_user_id(candidate) is None

    guarded = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="",
        candidates=[candidate],
        caller_group_id="group_l1_alpha",
        user_id="u105",
    )
    assert guarded.candidates == []
    assert guarded.blocked is True
    assert "missing_candidate_id" in guarded.violations

    aligned = align_match_to_trusted_group(
        MatchResult(
            status="matched",
            candidates=[candidate],
            query="Python Agent",
            group_id="group_l1_alpha",
            reason="matched_1",
        ),
        trusted_group_id="group_l1_alpha",
    )
    assert aligned.status == "empty"
    assert aligned.candidates == []
    assert aligned.reason == "no_stable_candidate_id"

    invite = generate_invite_copy(
        profile=_profile(),
        candidates=[candidate],
        match_status="matched",
        willing_to_at=True,
    )
    assert invite.kind == "undirected"
    assert invite.mentioned_user_ids == []
    assert "@" not in invite.text


def test_canonical_candidate_id_is_byte_identical_across_payload_and_invite() -> None:
    candidate = _candidate("u_101-1", doing="Building LLM agents")
    guarded = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="",
        candidates=[candidate],
        caller_group_id="group_l1_alpha",
        user_id="u105",
    )
    invite = generate_invite_copy(
        profile=_profile(),
        candidates=guarded.candidates,
        match_status="matched",
        willing_to_at=True,
    )

    assert guarded.ok is True
    assert guarded.candidates[0]["user_id"] == "u_101-1"
    assert invite.mentioned_user_ids == ["u_101-1"]
    assert "@u_101-1" in invite.text


@pytest.mark.parametrize("raw_user_id", [" u101 ", 101, True])
def test_new_api_client_preserves_invalid_raw_id_for_fail_closed_alignment(
    monkeypatch: pytest.MonkeyPatch,
    raw_user_id: Any,
) -> None:
    class _Response:
        status_code = 200
        content = b"{}"
        text = "{}"

        def json(self):
            return {
                "status": "matched",
                "group_id": "group_l1_alpha",
                "reason": "matched_1",
                "candidates": [
                    {
                        "user_id": raw_user_id,
                        "group_id": "group_l1_alpha",
                        "doing": {
                            "value": "Building LLM agents",
                            "disclosure": "confirmed_public",
                        },
                    }
                ],
            }

    monkeypatch.setattr(
        "apps.group_agent_api.agent_factory.integrations.match_client.requests.post",
        lambda *args, **kwargs: _Response(),
    )
    result = fetch_group_agent_match(
        query="Python Agent",
        group_token="group.jwt",
        bearer="test-bearer",
    )
    assert result.candidates[0]["user_id"] is raw_user_id
    aligned = align_match_to_trusted_group(
        result,
        trusted_group_id="group_l1_alpha",
    )
    assert aligned.status == "empty"
    assert aligned.candidates == []
    assert aligned.reason == "no_stable_candidate_id"


@pytest.mark.parametrize("unsafe_first", [True, False])
def test_guard_duplicate_order_never_allows_unsafe_record_to_replace_safe(
    unsafe_first: bool,
) -> None:
    safe = _candidate("u101", doing="Building LLM agents")
    unsafe = _candidate(
        "u101",
        doing="Private work",
        disclosure="match_only",
    )
    candidates = [unsafe, safe] if unsafe_first else [safe, unsafe]
    guarded = enforce_capability_guard(
        tier=CapabilityTier.in_group,
        reply="",
        candidates=candidates,
        caller_group_id="group_l1_alpha",
        user_id="u105",
    )

    assert len(guarded.candidates) == 1
    assert guarded.candidates[0]["doing"]["value"] == "Building LLM agents"
    assert "duplicate_candidate_id:u101" in guarded.violations


def test_invite_defense_filters_empty_basis_and_never_fabricates_reason() -> None:
    result = generate_invite_copy(
        profile=_profile(),
        candidates=_mixed_candidates(),
        match_status="matched",
        willing_to_at=True,
    )

    assert result.kind == "directed"
    assert result.mentioned_user_ids == ["u101", "u104"]
    assert "@u102" not in result.text
    assert "Building LLM agents" in result.text
    assert "Distributed systems" in result.text
    assert "相关公开经验" not in result.text


@pytest.mark.parametrize("unsafe_first", [True, False])
def test_direct_invite_deduplicates_and_keeps_first_safe_record(
    unsafe_first: bool,
) -> None:
    safe = _candidate("u101", doing="Building LLM agents")
    unsafe = _candidate(
        "u101",
        doing="Private work",
        disclosure="match_only",
    )
    candidates = [unsafe, safe] if unsafe_first else [safe, unsafe]
    candidates.append(_candidate("", doing="Anonymous work"))

    result = generate_invite_copy(
        profile=_profile(),
        candidates=candidates,
        match_status="matched",
        willing_to_at=True,
    )

    assert result.mentioned_user_ids == ["u101"]
    assert result.text.count("@u101") == 1
    assert "Building LLM agents" in result.text
    assert "Private work" not in result.text


def test_all_candidates_without_basis_degrade_honestly() -> None:
    result = generate_invite_copy(
        profile=_profile(),
        candidates=[_candidate("u102", doing=None)],
        match_status="matched",
        willing_to_at=True,
    )

    assert result.kind == "undirected"
    assert result.match_status == "empty"
    assert result.mentioned_user_ids == []
    assert "@" not in result.text
    assert "暂时没找到" in (result.honest_note or "")


@pytest.mark.parametrize("mutation", ["delete", "add", "duplicate"])
def test_optional_polish_falls_back_on_any_mention_set_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.delenv("GROUP_AGENT_MODEL_MODE", raising=False)
    monkeypatch.delenv("GROUP_AGENT_INTEGRATION", raising=False)
    candidates = _mixed_candidates()[:2]
    base = generate_invite_copy(
        profile=_profile(),
        candidates=candidates,
        match_status="matched",
        willing_to_at=True,
    )
    lines = base.text.splitlines()
    if mutation == "delete":
        polished = "\n".join(line for line in lines if not line.startswith("@u104 "))
    elif mutation == "add":
        polished = f"{base.text}\n@u999 也值得聊一次以确认，不一定合适"
    else:
        u101_line = next(line for line in lines if line.startswith("@u101 "))
        polished = f"{base.text}\n{u101_line}"

    class _FakePolishModel:
        def invoke(self, _messages):
            class _Response:
                content = polished

            return _Response()

    result = generate_invite_with_optional_llm(
        profile=_profile(),
        candidates=candidates,
        match_status="matched",
        willing_to_at=True,
        model=_FakePolishModel(),
        use_llm=True,
    )

    assert result.ok is True
    assert result.text == base.text
    assert result.mentioned_user_ids == ["u101", "u104"]


def test_alignment_all_missing_basis_sets_empty_reason() -> None:
    aligned = align_match_to_trusted_group(
        MatchResult(
            status="matched",
            candidates=[_candidate("u102", doing=None)],
            query="Python Agent",
            group_id="group_l1_alpha",
            reason="matched_1",
        ),
        trusted_group_id="group_l1_alpha",
    )

    assert aligned.status == "empty"
    assert aligned.candidates == []
    assert aligned.reason == "no_auditable_public_match_basis"


@pytest.mark.parametrize("unsafe_first", [True, False])
def test_alignment_deduplicates_after_safety_and_records_identity_issues(
    caplog: pytest.LogCaptureFixture,
    unsafe_first: bool,
) -> None:
    safe = _candidate("u101", doing="Building LLM agents")
    unsafe = _candidate(
        "u101",
        doing="Private work",
        disclosure="match_only",
    )
    candidates = [unsafe, safe] if unsafe_first else [safe, unsafe]
    candidates.append(_candidate("", doing="Anonymous work"))
    with caplog.at_level("WARNING", logger="uvicorn.error"):
        aligned = align_match_to_trusted_group(
            MatchResult(
                status="matched",
                candidates=candidates,
                query="Python Agent",
                group_id="group_l1_alpha",
                reason="matched_3",
            ),
            trusted_group_id="group_l1_alpha",
        )

    assert len(aligned.candidates) == 1
    assert aligned.candidates[0]["doing"]["value"] == "Building LLM agents"
    assert "duplicate_candidate_id:u101" in caplog.text
    assert "missing_candidate_id" in caplog.text


def test_comma_never_becomes_a_standalone_audit_fragment() -> None:
    source = (
        "如果需要，我可以继续帮助你补充筛选条件。"
        "下一步，请确认是否继续。"
        "这里还有一条完整说明。"
    )
    excerpts = extract_high_value_sentences(source, max_chars=600)

    assert excerpts
    assert "如果需要，" not in excerpts
    assert "下一步，" not in excerpts
    assert all(
        excerpt.endswith(("。", "！", "？", "!", "?", "；", ";"))
        for excerpt in excerpts
    )
    assert excerpt_coverage(source, excerpts) <= 0.5


def _audit_collector(tmp_path: Path) -> HumanAuditCollector:
    return HumanAuditCollector(
        enabled=True,
        run_id="req015_audit",
        provider="qwen",
        model="qwen-turbo",
        base_url_configured=True,
        fixture_level="L1",
        group_id="group_l1_alpha",
        caller_id="u105",
        output_dir=tmp_path,
    )


def _capture_audit(collector: HumanAuditCollector) -> dict[str, Any]:
    profile = _profile().to_storage_dict()
    for number in (1, 2):
        collector.capture_round(
            number=number,
            user_input=f"第 {number} 轮固定输入",
            reply=(
                "我已记录 AI Agent 产品和 Python 技术负责人需求。"
                "下一步，可以继续确认合作约束。"
            ),
            llm_delta={
                "llm_starts": 2,
                "llm_ends": 2,
                "tool_calls": {"save_group_profile": 1},
                "tokens": 100,
            },
            latency_s=1,
            profile_before=None if number == 1 else profile,
            profile_after=profile,
        )
    collector.capture_round(
        number=3,
        user_input="请从本群推荐并直接 @。",
        reply=(
            "本群已有 2 位公开信息与需求有交集的人选。"
            "是否匹配仍需要沟通确认。"
        ),
        llm_delta={
            "llm_starts": 2,
            "llm_ends": 2,
            "tool_calls": {},
            "tokens": 100,
        },
        latency_s=1,
        profile_before=profile,
        profile_after=profile,
        candidates=_mixed_candidates()[:2],
        invite_text=(
            "@u101 你公开资料里提到「Building LLM agents」，"
            "值得聊一次以确认是否对得上。"
            "@u104 你公开资料里提到「Distributed systems」，"
            "值得聊一次以确认是否对得上。"
            "先聊具体问题，不预设合作结论。"
        ),
        mentioned_user_ids=["u101", "u104"],
        invite_ok=True,
        guard_blocked=False,
    )
    report = collector.build_report(
        total_llm_invocations=6,
        total_tokens=300,
        total_time_s=3,
        machine_oracles={
            "current_group_only": True,
            "sensitive_leak_count": 0,
        },
    )
    assert report is not None
    return report


def test_human_audit_maps_every_mention_to_real_public_basis(tmp_path: Path) -> None:
    report = _capture_audit(_audit_collector(tmp_path))
    round3 = report["rounds"][2]

    assert round3["mentioned_user_ids"] == ["u101", "u104"]
    assert round3["mentioned_evidence"] == [
        {
            "user_id": "u101",
            "public_match_basis": {"doing": "Building LLM agents"},
        },
        {
            "user_id": "u104",
            "public_match_basis": {"doing": "Distributed systems"},
        },
    ]
    assert round3["automatic_checks"]["all_mentioned_have_public_basis"] is True
    markdown = render_markdown(report)
    assert "@u101" in markdown and "Building LLM agents" in markdown
    assert "@u104" in markdown and "Distributed systems" in markdown
    assert "u102" not in markdown


def test_human_audit_fails_closed_for_empty_or_missing_mentioned_basis(
    tmp_path: Path,
) -> None:
    collector = _audit_collector(tmp_path)
    profile = _profile().to_storage_dict()
    with pytest.raises(HumanAuditError, match="public match basis"):
        collector.capture_round(
            number=3,
            user_input="请推荐。",
            reply="这是一条完整回复。这里是另一条完整回复。",
            llm_delta={},
            latency_s=1,
            profile_before=profile,
            profile_after=profile,
            candidates=[_candidate("u102", doing=None)],
            invite_text="@u102 请交流。先聊具体问题。",
            mentioned_user_ids=["u102"],
            invite_ok=True,
            guard_blocked=False,
        )


def test_human_audit_rejects_noncanonical_candidate_id(tmp_path: Path) -> None:
    collector = _audit_collector(tmp_path)
    profile = _profile().to_storage_dict()
    candidate = _candidate("u101", doing="Building LLM agents")
    candidate["user_id"] = " u101 "
    with pytest.raises(HumanAuditError, match="stable user_id"):
        collector.capture_round(
            number=3,
            user_input="请推荐。",
            reply="这是一条完整回复。这里是另一条完整回复。",
            llm_delta={},
            latency_s=1,
            profile_before=profile,
            profile_after=profile,
            candidates=[candidate],
            invite_text="@u101 请交流。先聊具体问题。",
            mentioned_user_ids=["u101"],
            invite_ok=True,
            guard_blocked=False,
        )


@pytest.mark.parametrize(
    ("mentioned", "candidate_id", "invite_text"),
    [
        ([101], "101", "@101 请交流。先聊具体问题。"),
        ([True], "True", "@True 请交流。先聊具体问题。"),
        ([" u101 "], "u101", "@u101 请交流。先聊具体问题。"),
    ],
)
def test_human_audit_capture_rejects_noncanonical_mentioned_identity(
    tmp_path: Path,
    mentioned: list[Any],
    candidate_id: str,
    invite_text: str,
) -> None:
    collector = _audit_collector(tmp_path)
    profile = _profile().to_storage_dict()
    with pytest.raises(HumanAuditError, match="mentioned identity.*canonical string"):
        collector.capture_round(
            number=3,
            user_input="请推荐。",
            reply="这是一条完整回复。这里是另一条完整回复。",
            llm_delta={},
            latency_s=1,
            profile_before=profile,
            profile_after=profile,
            candidates=[_candidate(candidate_id, doing="Building LLM agents")],
            invite_text=invite_text,
            mentioned_user_ids=mentioned,
            invite_ok=True,
            guard_blocked=False,
        )
    assert collector.captured_rounds == 0


@pytest.mark.parametrize(
    "surface",
    [
        "candidate",
        "mentioned_evidence",
        "mentioned_user_ids",
        "invite_actual_at_user_ids",
    ],
)
@pytest.mark.parametrize("invalid_id", [101, True, " u101 "])
def test_human_audit_report_recheck_rejects_noncanonical_identity(
    tmp_path: Path,
    surface: str,
    invalid_id: Any,
) -> None:
    report = _capture_audit(_audit_collector(tmp_path))
    round3 = report["rounds"][2]
    if surface == "candidate":
        round3["candidates"][0]["user_id"] = invalid_id
    elif surface == "mentioned_evidence":
        round3["mentioned_evidence"][0]["user_id"] = invalid_id
    else:
        round3[surface][0] = invalid_id

    with pytest.raises(HumanAuditError, match="identity.*canonical string"):
        assert_auditable_candidate_evidence(report)


@pytest.mark.parametrize("invalid_id", [101, True, " u101 "])
def test_polish_expected_mentions_reject_noncanonical_identity(
    invalid_id: Any,
) -> None:
    violations = assert_exact_polished_mentions(
        text="@u101 请交流。",
        expected_user_ids=[invalid_id],
    )

    assert violations == ["polished_invalid_expected_mention_id"]


def test_canonical_identity_passes_capture_report_and_polish(
    tmp_path: Path,
) -> None:
    collector = _audit_collector(tmp_path)
    profile = _profile().to_storage_dict()
    collector.capture_round(
        number=3,
        user_input="请推荐。",
        reply="这是一条完整回复。这里是另一条完整回复。",
        llm_delta={},
        latency_s=1,
        profile_before=profile,
        profile_after=profile,
        candidates=[_candidate("u_101-1", doing="Building LLM agents")],
        invite_text="@u_101-1 请交流。先聊具体问题。",
        mentioned_user_ids=["u_101-1"],
        invite_ok=True,
        guard_blocked=False,
    )
    assert collector.captured_rounds == 1
    report = {
        "rounds": [
            {
                "round": 3,
                "candidates": [
                    {
                        "user_id": "u_101-1",
                        "public_match_basis": {"doing": "Building LLM agents"},
                    }
                ],
                "mentioned_evidence": [
                    {
                        "user_id": "u_101-1",
                        "public_match_basis": {"doing": "Building LLM agents"},
                    }
                ],
                "mentioned_user_ids": ["u_101-1"],
                "invite_actual_at_user_ids": ["u_101-1"],
            }
        ]
    }
    assert_auditable_candidate_evidence(report)
    assert (
        assert_exact_polished_mentions(
            text="@u_101-1 请交流。",
            expected_user_ids=["u_101-1"],
        )
        == []
    )


@pytest.mark.parametrize(
    ("invite_text", "mentioned"),
    [
        ("@u101 请交流。这里是完整说明。", ["u101", "u104"]),
        ("@u101 @u104 @u999 请交流。这里是完整说明。", ["u101", "u104"]),
        ("@u101 @u104 @u101 请交流。这里是完整说明。", ["u101", "u104"]),
    ],
)
def test_human_audit_rejects_actual_invite_at_mismatch(
    tmp_path: Path,
    invite_text: str,
    mentioned: list[str],
) -> None:
    collector = _audit_collector(tmp_path)
    profile = _profile().to_storage_dict()
    with pytest.raises(HumanAuditError, match="invite text mentions inconsistent"):
        collector.capture_round(
            number=3,
            user_input="请推荐。",
            reply="这是一条完整回复。这里是另一条完整回复。",
            llm_delta={},
            latency_s=1,
            profile_before=profile,
            profile_after=profile,
            candidates=_mixed_candidates()[:2],
            invite_text=invite_text,
            mentioned_user_ids=mentioned,
            invite_ok=True,
            guard_blocked=False,
        )


def test_human_audit_build_and_write_recheck_actual_mentions(
    tmp_path: Path,
) -> None:
    collector = _audit_collector(tmp_path)
    report = _capture_audit(collector)
    report["rounds"][2]["invite_actual_at_user_ids"] = ["u101"]

    with pytest.raises(HumanAuditError, match="invite text mentions inconsistent"):
        collector.write_report(report)


class _Checkpointer:
    def flush(self) -> None:
        return None


class _PersistingAgent:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.checkpointer = _Checkpointer()

    async def aget_state(self, _config: dict[str, Any]) -> Any:
        class _State:
            values = {"messages": []}

        return _State()

    async def ainvoke(
        self, payload: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        metadata = config["metadata"]
        profile = profile_from_flat(
            user_id=str(metadata["user_id"]),
            group_id=str(metadata["group_id"]),
            doing="AI Agent 产品",
            need="Python 技术负责人",
            offer="客户资源",
        )
        save_profile(self.base_dir, profile)
        return {
            "messages": [
                payload["messages"][0],
                AIMessage(content="画像已经更新。"),
            ]
        }


def _match_result() -> MatchResult:
    u101 = _candidate("u101", doing="Building LLM agents")
    return MatchResult(
        status="matched",
        candidates=[
            u101,
            dict(u101),
            _candidate("u104", doing="Distributed systems"),
        ],
        query="Python Agent",
        group_id="group_l1_alpha",
        reason="matched_3",
    )


def _empty_basis_match_result() -> MatchResult:
    return MatchResult(
        status="matched",
        candidates=[_candidate("u102", doing=None)],
        query="Python Agent",
        group_id="group_l1_alpha",
        reason="matched_1",
    )


@pytest.mark.asyncio
async def test_sync_chat_payload_uses_only_evidence_gated_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setattr(chat_endpoint, "run_match", lambda **_kwargs: _match_result())
    state = AppState(agent=_PersistingAgent(tmp_path), base_dir=tmp_path)

    response = await chat_endpoint.chat(
        ChatRequest(
            user_id="u105",
            group_id="group_l1_alpha",
            conversation_id="req015_sync",
            message="请推荐。",
            membership="in_group",
            run_match=True,
            run_invite=True,
            willing_to_at=True,
        ),
        state,
    )

    _assert_consistent_payload(response.model_dump())


@pytest.mark.asyncio
async def test_async_callback_payload_uses_only_evidence_gated_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.app.async_manager.run_match",
        lambda **_kwargs: _match_result(),
    )
    state = AppState(agent=_PersistingAgent(tmp_path), base_dir=tmp_path)
    session = TrustedSession(
        principal=SessionPrincipal(
            user_id="u105",
            unionid="union_u105",
            user_token=None,
            source="stub",
        ),
        group_id="group_l1_alpha",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )
    request = AsyncCallRequest(
        run_id="req015_async_run",
        idempotency_key="req015_async_idem",
        user_id="u105",
        unionid="union_u105",
        group_id="group_l1_alpha",
        conversation_id="req015_async",
        message="请推荐。",
        callback_url="http://localhost:3009/group_agent_callbacks",
        run_match=True,
        run_invite=True,
        willing_to_at=True,
    )
    final_payload: dict[str, Any] = {}

    async def emit_callback(event_type: str, payload: dict[str, Any]) -> bool:
        if event_type == "final":
            final_payload.update(payload)
        return True

    await _execute_core_agent(
        req=request,
        session=session,
        state=state,
        tid="ga::u105::group_l1_alpha::req015_async",
        emit_callback=emit_callback,
    )

    _assert_consistent_payload(final_payload)


@pytest.mark.asyncio
async def test_sync_chat_all_missing_basis_has_consistent_empty_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_ENV", "test")
    monkeypatch.setattr(
        chat_endpoint,
        "run_match",
        lambda **_kwargs: _empty_basis_match_result(),
    )
    state = AppState(agent=_PersistingAgent(tmp_path), base_dir=tmp_path)

    response = await chat_endpoint.chat(
        ChatRequest(
            user_id="u105",
            group_id="group_l1_alpha",
            conversation_id="req015_sync_empty",
            message="请推荐。",
            membership="in_group",
            run_match=True,
            run_invite=True,
            willing_to_at=True,
        ),
        state,
    )

    assert response.match_status == "empty"
    assert response.match_reason == "no_auditable_public_match_basis"
    assert response.candidates == []
    assert response.mentioned_user_ids == []
    assert response.delivery_kind == "undirected"


@pytest.mark.asyncio
async def test_async_all_missing_basis_has_consistent_empty_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "apps.group_agent_api.app.async_manager.run_match",
        lambda **_kwargs: _empty_basis_match_result(),
    )
    state = AppState(agent=_PersistingAgent(tmp_path), base_dir=tmp_path)
    session = TrustedSession(
        principal=SessionPrincipal(
            user_id="u105",
            unionid="union_u105",
            user_token=None,
            source="stub",
        ),
        group_id="group_l1_alpha",
        group_token=None,
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
    )
    request = AsyncCallRequest(
        run_id="req015_async_empty_run",
        idempotency_key="req015_async_empty_idem",
        user_id="u105",
        unionid="union_u105",
        group_id="group_l1_alpha",
        conversation_id="req015_async_empty",
        message="请推荐。",
        callback_url="http://localhost:3009/group_agent_callbacks",
        run_match=True,
        run_invite=True,
        willing_to_at=True,
    )
    final_payload: dict[str, Any] = {}

    async def emit_callback(event_type: str, payload: dict[str, Any]) -> bool:
        if event_type == "final":
            final_payload.update(payload)
        return True

    await _execute_core_agent(
        req=request,
        session=session,
        state=state,
        tid="ga::u105::group_l1_alpha::req015_async_empty",
        emit_callback=emit_callback,
    )

    assert final_payload["match_status"] == "empty"
    assert final_payload["match_reason"] == "no_auditable_public_match_basis"
    assert final_payload["candidates"] == []
    assert final_payload["mentioned_user_ids"] == []
    assert final_payload["delivery_kind"] == "undirected"
