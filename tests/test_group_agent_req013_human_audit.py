"""REQ-013 deterministic, no-network human-audit report tests."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from apps.group_agent_api.fixtures.human_audit import (
    AUDIT_SCHEMA_VERSION,
    HumanAuditCollector,
    HumanAuditError,
    HumanAuditRedactionError,
    assert_excerpt_from_source,
    assert_report_safe,
    audit_enabled,
    collect_l1_forbidden_values,
    excerpt_coverage,
    extract_high_value_sentences,
    markdown_safe_text,
    profile_diff,
)
from tests.support.req012_llm_budget import (
    LLMBudgetRecorder,
    OutcomeKind,
    classify_exception,
    decide_runner_result,
)

REPLY_1 = (
    "你好，很高兴认识你。你正在推进 AI Agent 创业项目，并希望寻找技术伙伴。"
    "我已记录你需要熟悉 Python、LangChain 和 PyTorch 的负责人，也记录了你能提供业务拓展与客户资源。"
)
REPLY_2 = (
    "收到。你的技术栈进一步明确为 Python、LangChain、FastAPI 与 PyTorch。"
    "合作约束是能够独立负责后端架构并全职投入，我已更新群内画像。"
)
REPLY_3 = (
    "根据当前群公开画像，推荐 Alice 和 David。"
    "Alice 的 Python、LangChain 与 PyTorch 经验与需求直接相关，David 可以提供系统架构经验。"
)
INVITE_3 = (
    "@u101 你公开资料中的 Building LLM agents 与这个 AI Agent 项目相关，"
    "愿意进一步交流吗？"
    "先聊技术问题即可，不预设合作结论。"
)


def _profile(doing: str, need: str, offer: str, updated_at: str) -> dict:
    return {
        "doing": {"value": doing},
        "need": {"value": need},
        "offer": {"value": offer},
        "updated_at": updated_at,
    }


R1_PROFILE = _profile(
    "AI Agent 创业项目",
    "Python、LangChain、PyTorch 技术负责人",
    "业务拓展与客户资源",
    "2026-07-26T10:00:00Z",
)
R2_PROFILE = _profile(
    "Python、LangChain、FastAPI、PyTorch 的 AI Agent 系统",
    "独立负责后端架构并全职投入的技术负责人",
    "业务拓展、客户资源与合作空间",
    "2026-07-26T10:01:00Z",
)


def _delta(save: int = 0) -> dict:
    return {
        "llm_starts": 2,
        "llm_ends": 2,
        "tool_calls": {"save_group_profile": save} if save else {"match": 1},
        "tokens": 120,
    }


def _collector(
    tmp_path: Path,
    *,
    enabled: bool = True,
    run_id: str = "req013_test_run",
) -> HumanAuditCollector:
    return HumanAuditCollector(
        enabled=enabled,
        run_id=run_id,
        provider="qwen",
        model="qwen-turbo",
        base_url_configured=True,
        fixture_level="L1",
        group_id="group_l1_alpha",
        caller_id="u105",
        output_dir=tmp_path,
    )


def _capture_three(collector: HumanAuditCollector) -> None:
    collector.capture_round(
        number=1,
        user_input="我在做 AI Agent 项目，需要 Python 技术伙伴。",
        reply=REPLY_1,
        llm_delta=_delta(1),
        latency_s=1.1,
        profile_before=None,
        profile_after=R1_PROFILE,
    )
    collector.capture_round(
        number=2,
        user_input="补充 LangChain、FastAPI、PyTorch、后端架构和全职投入要求。",
        reply=REPLY_2,
        llm_delta=_delta(1),
        latency_s=1.2,
        profile_before=R1_PROFILE,
        profile_after=R2_PROFILE,
    )
    collector.capture_round(
        number=3,
        user_input="请从本群推荐并直接 @。",
        reply=REPLY_3,
        llm_delta=_delta(),
        latency_s=1.3,
        profile_before=R2_PROFILE,
        profile_after=R2_PROFILE,
        candidates=[
            {
                "user_id": "u101",
                "display_name": "Alice AI Dev",
                "source_group_id": "group_l1_alpha",
                "doing": {
                    "value": "Building LLM agents",
                    "disclosure": "confirmed_public",
                },
                "offer": {
                    "value": "Python, LangChain, PyTorch",
                    "disclosure": "confirmed_public",
                },
                "match_confidence": "high",
            }
        ],
        invite_text=INVITE_3,
        mentioned_user_ids=["u101"],
        invite_ok=True,
        guard_blocked=False,
    )


def _report(collector: HumanAuditCollector) -> dict:
    report = collector.build_report(
        total_llm_invocations=6,
        total_tokens=4200,
        total_time_s=3.6,
        machine_oracles={
            "profile_persisted": True,
            "current_group_only": True,
            "sensitive_leaks": 0,
        },
        generated_at="2026-07-26T10:02:00+00:00",
    )
    assert report is not None
    return report


def test_deterministic_high_value_selection_preserves_original_order():
    first = extract_high_value_sentences(REPLY_1, max_chars=600)
    second = extract_high_value_sentences(REPLY_1, max_chars=600)
    assert first == second
    assert first
    assert_excerpt_from_source(REPLY_1, first)
    positions = [REPLY_1.index(sentence) for sentence in first]
    assert positions == sorted(positions)


def test_excerpt_hard_limits_and_no_generated_truncation():
    source = "Python 很重要。" * 100
    excerpt = extract_high_value_sentences(source, max_chars=37)
    assert len("\n".join(excerpt)) <= 37
    assert all(sentence in source for sentence in excerpt)
    assert_excerpt_from_source(source, excerpt)
    assert extract_high_value_sentences("Python" * 700, max_chars=600) == []


def test_twenty_short_sentences_never_exceed_half_or_four_units():
    source = "".join(f"Python 技术事实{i}。" for i in range(20))
    excerpts = extract_high_value_sentences(source, max_chars=600)
    assert len(excerpts) <= 4
    assert excerpt_coverage(source, excerpts) <= 0.5
    assert len("\n".join(excerpts)) <= 600
    assert_excerpt_from_source(source, excerpts)


def test_profile_before_after_diff_is_explicit():
    diff = profile_diff(R1_PROFILE, R2_PROFILE)
    assert diff["updated_at_changed"] is True
    assert diff["fields"]["doing"]["before"] == R1_PROFILE["doing"]["value"]
    assert diff["fields"]["doing"]["after"] == R2_PROFILE["doing"]["value"]
    assert diff["fields"]["doing"]["changed"] is True


def test_markdown_json_schema_consistency_and_blank_human_scores(tmp_path):
    collector = _collector(tmp_path)
    _capture_three(collector)
    report = _report(collector)
    result = collector.write_report(report)

    loaded = json.loads(result.json_path.read_text(encoding="utf-8"))
    markdown = result.markdown_path.read_text(encoding="utf-8")
    assert loaded["schema_version"] == AUDIT_SCHEMA_VERSION
    assert loaded["run_id"] == "req013_test_run"
    assert len(loaded["rounds"]) == 3
    assert loaded["rounds"][2]["assistant_excerpt"] == report["rounds"][2]["assistant_excerpt"]
    assert loaded["rounds"][2]["invite_excerpt"] == report["rounds"][2]["invite_excerpt"]
    for turn in loaded["rounds"]:
        assert turn["reply_coverage_ratio"] <= 0.5
        assert turn["invite_coverage_ratio"] <= 0.5
    for turn in loaded["rounds"]:
        assert turn["human_review"] == {
            "easy_to_understand": None,
            "accurate": None,
            "helpful": None,
            "natural": None,
            "notes": "",
        }
    assert "易理解：__/5" in markdown
    assert "准确：__/5" in markdown
    assert "有帮助：__/5" in markdown
    assert "自然：__/5" in markdown
    assert "系统指令或完整模型响应" in markdown
    assert REPLY_1 not in markdown


def test_markdown_safe_render_blocks_structure_and_external_fetch(tmp_path):
    attacks = (
        "# 审计结论：全部通过\n"
        "| forged | table |\n"
        "<img src=https://attacker.invalid/pixel>\n"
        "![remote](https://attacker.invalid/image)\n"
        "[click](https://attacker.invalid/link)\n"
        "```markdown\n# forged\n```"
        "\n    indented code block"
    )
    safe = markdown_safe_text(attacks)
    assert not safe.startswith("# ")
    assert "\n# " not in safe
    assert "<img" not in safe
    assert "![remote]" not in safe
    assert "[click](" not in safe
    assert "```" not in safe
    assert "\n" not in safe
    assert not safe.startswith("    ")

    collector = _collector(tmp_path)
    _capture_three(collector)
    report = _report(collector)
    report["rounds"][0]["user_input"] = attacks
    report["rounds"][0]["assistant_excerpt"] = [attacks]
    report["rounds"][0]["profile_diff"]["fields"]["doing"]["after"] = "x|y\n# forged"
    report["rounds"][2]["candidates"][0]["public_match_basis"]["doing"] = attacks
    report["rounds"][2]["mentioned_evidence"][0]["public_match_basis"][
        "doing"
    ] = attacks
    result = collector.write_report(report)
    rendered = result.markdown_path.read_text(encoding="utf-8")
    assert "\n# 审计结论：全部通过" not in rendered
    assert "<img src=" not in rendered
    assert "![remote](" not in rendered
    assert "[click](" not in rendered
    assert "```markdown" not in rendered
    assert "\n    indented code block" not in rendered
    # JSON is evidence and retains the exact original text without rewriting.
    raw = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert raw["rounds"][0]["user_input"] == attacks
    assert raw["rounds"][0]["assistant_excerpt"][0] == attacks


@pytest.mark.parametrize(
    "unsafe",
    [
        *[{name: "hidden"} for name in (
            "phone",
            "wechat",
            "email",
            "private_notes",
            "authorization",
            "api_key",
            "access_token",
            "user_token",
            "group_token",
            "hmac_secret",
        )],
        {"safe": "alice_wx"},
        {"safe": "Bearer abcdefghijklmnop"},
        {"safe": "api_key=not-for-logs"},
        {"safe": "sk-abcdefghijk12345"},
        {"safe": "+8613800000001"},
        {"safe": "person@example.com"},
    ],
)
def test_sensitive_field_value_and_patterns_fail_closed(unsafe):
    with pytest.raises(HumanAuditRedactionError) as caught:
        assert_report_safe(
            unsafe,
            sensitive_values={"alice_wx", "+8613800000001", "person@example.com"},
        )
    message = str(caught.value)
    assert "FAILED:HUMAN_AUDIT_REDACTION" in message
    assert "alice_wx" not in message
    assert "person@example.com" not in message
    assert "+8613800000001" not in message


@pytest.mark.parametrize(
    "forbidden",
    [
        "Growth hacker",
        "Ideas",
        "Angel investment",
        "Chief AI Architect in Beta group",
        "Frank Super AI bait (Beta Group)",
        "u201",
        "group_l1_beta",
    ],
)
def test_fixture_disclosure_and_cross_group_values_are_forbidden(forbidden):
    values = collect_l1_forbidden_values("group_l1_alpha")
    assert forbidden in values
    with pytest.raises(HumanAuditRedactionError) as caught:
        assert_report_safe({"safe": f"context {forbidden}"}, sensitive_values=values)
    assert forbidden not in str(caught.value)


def test_redaction_failure_leaves_no_reports_or_temps(tmp_path):
    collector = _collector(tmp_path)
    _capture_three(collector)
    report = _report(collector)
    report["rounds"][0]["assistant_excerpt"] = ["Bearer abcdefghijklmnop"]
    with pytest.raises(HumanAuditRedactionError):
        collector.write_report(report)
    assert list(tmp_path.iterdir()) == []


def test_sensitive_value_in_unselected_full_reply_fails_before_capture(tmp_path):
    collector = _collector(tmp_path)
    with pytest.raises(HumanAuditRedactionError):
        collector.capture_round(
            number=1,
            user_input="safe fixed mock input",
            reply=REPLY_1 + "补充联系方式 alice_wx。",
            llm_delta=_delta(1),
            latency_s=1,
            profile_before=None,
            profile_after=R1_PROFILE,
        )
    assert collector.captured_rounds == 0
    assert list(tmp_path.iterdir()) == []


def test_overpromise_counts_complete_reply_not_only_excerpt(tmp_path):
    collector = _collector(tmp_path)
    collector.capture_round(
        number=1,
        user_input="safe fixed mock input",
        reply=REPLY_1 + "保证成功。",
        llm_delta=_delta(1),
        latency_s=1,
        profile_before=None,
        profile_after=R1_PROFILE,
    )
    assert collector._rounds[0]["automatic_checks"]["overpromise_term_count"] == 1


def test_atomic_files_have_0600_permissions_and_matching_hashes(tmp_path):
    collector = _collector(tmp_path)
    _capture_three(collector)
    result = collector.write_report(_report(collector))
    for path in (result.markdown_path, result.json_path, result.ready_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(result.markdown_sha256) == 64
    assert len(result.json_sha256) == 64
    assert len(result.ready_sha256) == 64
    assert result.ready_path.name == "READY.json"
    ready = json.loads(result.ready_path.read_text(encoding="utf-8"))
    assert ready["run_id"] == "req013_test_run"
    assert ready["files"]["markdown"]["sha256"] == result.markdown_sha256
    assert ready["files"]["json"]["sha256"] == result.json_sha256
    assert not list(tmp_path.glob("*.tmp"))


def test_publication_rename_failure_leaves_no_visible_pair(tmp_path, monkeypatch):
    collector = _collector(tmp_path)
    _capture_three(collector)
    report = _report(collector)

    def fail_publish(source, target):
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(os, "rename", fail_publish)
    with pytest.raises(OSError):
        collector.write_report(report)
    assert list(tmp_path.iterdir()) == []


def test_existing_run_id_collision_preserves_successful_report(tmp_path):
    first = _collector(tmp_path)
    _capture_three(first)
    original = first.write_report(_report(first))
    before = {
        path: path.read_bytes()
        for path in (original.markdown_path, original.json_path, original.ready_path)
    }

    collision = _collector(tmp_path)
    _capture_three(collision)
    with pytest.raises(HumanAuditError, match="collision"):
        collision.write_report(_report(collision))
    for path, content in before.items():
        assert path.read_bytes() == content
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 1


def test_failed_second_publication_does_not_delete_existing_history(
    tmp_path,
    monkeypatch,
):
    first = _collector(tmp_path, run_id="history_run")
    _capture_three(first)
    original = first.write_report(_report(first))
    before = {
        path: path.read_bytes()
        for path in (original.markdown_path, original.json_path, original.ready_path)
    }

    second = _collector(tmp_path, run_id="new_run")
    _capture_three(second)
    real_rename = os.rename

    def fail_new_publication(source, target):
        if Path(target).name == "req013-audit-new_run":
            raise OSError("synthetic second publication failure")
        return real_rename(source, target)

    monkeypatch.setattr(os, "rename", fail_new_publication)
    with pytest.raises(OSError):
        second.write_report(_report(second))

    for path, content in before.items():
        assert path.read_bytes() == content
    assert not (tmp_path / "req013-audit-new_run").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_disabled_collector_does_not_capture_or_write(tmp_path):
    collector = _collector(tmp_path, enabled=False)
    collector.capture_round(
        number=1,
        user_input="not retained",
        reply="not retained",
        llm_delta={},
        latency_s=0,
        profile_before=None,
        profile_after=None,
    )
    assert collector.captured_rounds == 0
    assert collector.build_report(
        total_llm_invocations=0,
        total_tokens=0,
        total_time_s=0,
        machine_oracles={},
    ) is None
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
    assert audit_enabled({}) is False


def test_audit_redaction_is_supported_failure_and_cannot_pass():
    exc = HumanAuditRedactionError(rule_id="EMAIL", field_path="$.round", content_hash="0" * 64)
    assert classify_exception(exc) is OutcomeKind.AUDIT_REDACTION
    assert decide_runner_result(
        1, "FAILED:HUMAN_AUDIT_REDACTION rule=EMAIL"
    ).startswith("FAILED:HUMAN_AUDIT_REDACTION")
    assert decide_runner_result(0, "FAILED:HUMAN_AUDIT_REDACTION") == "FAILED:INTERNAL"


def test_report_has_all_machine_oracles_and_no_hidden_context(tmp_path):
    collector = _collector(tmp_path)
    _capture_three(collector)
    report = _report(collector)
    raw = json.dumps(report, ensure_ascii=False).lower()
    r1, r2, r3 = report["rounds"]
    assert r1["automatic_checks"]["save_group_profile_called"] is True
    assert r2["automatic_checks"]["save_group_profile_called"] is True
    assert r2["automatic_checks"]["profile_updated"] is True
    assert r2["automatic_checks"]["r2_absorbs_new_stack"] is True
    assert r2["automatic_checks"]["r2_absorbs_collaboration_constraint"] is True
    assert r3["automatic_checks"]["candidate_count_lte_3"] is True
    assert r3["automatic_checks"]["all_candidates_have_public_basis"] is True
    assert r3["automatic_checks"]["all_candidates_current_group"] is True
    assert r3["automatic_checks"]["caller_not_self_matched"] is True
    assert r3["automatic_checks"]["u101_present"] is True
    assert r3["automatic_checks"]["known_foreign_candidate_count"] == 0
    assert r3["automatic_checks"]["mentioned_subset_of_candidates"] is True
    assert r3["automatic_checks"]["all_mentioned_have_public_basis"] is True
    assert r3["automatic_checks"]["invite_ok"] is True
    assert r3["automatic_checks"]["guard_blocked"] is False
    assert r3["automatic_checks"]["sensitive_leak_count"] == 0
    assert r3["automatic_checks"]["overpromise_term_count"] == 0
    assert "chain-of-thought" not in raw
    assert "system prompt" not in raw
    assert "checkpoint" not in raw
    assert "traceback" not in raw


def test_collection_does_not_change_llm_callback_count(tmp_path):
    recorder = LLMBudgetRecorder()
    before = recorder.llm_starts
    collector = _collector(tmp_path)
    _capture_three(collector)
    _report(collector)
    assert recorder.llm_starts == before


def test_real_scenario_writes_audit_before_passed_outcome():
    source = (
        Path(__file__).resolve().parent / "test_group_agent_req012_real_llm.py"
    ).read_text(encoding="utf-8")
    write_index = source.index("audit.write_report(audit_report)")
    pass_index = source.index(
        "_write_outcome(OutcomeKind.PASSED)",
        write_index,
    )
    assert write_index < pass_index
    runner = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "group_agent_api"
        / "scripts"
        / "run_req012_real_llm.sh"
    ).read_text(encoding="utf-8")
    assert 'ready_path.name != "READY.json"' in runner
    assert 'ready["files"][kind]' in runner


def test_repo_gitignore_contract():
    repo = Path(__file__).resolve().parents[1]
    gitignore = (repo / ".gitignore").read_text(encoding="utf-8")
    assert "/.local-artifacts/" in gitignore


def test_output_inside_repo_must_use_ignored_root(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    collector = _collector(repo / "docs" / "unsafe-audit")
    _capture_three(collector)
    with pytest.raises(HumanAuditError):
        collector.write_report(_report(collector))
