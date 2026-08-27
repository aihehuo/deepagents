"""Contract + real-LLM tests for mod.brain.reply_grounding (TSD-14 §4.6.5)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from apps.group_agent_api.agent_factory.checks.reply_grounding import (
    MODULE_ID,
    ReplyGroundingInput,
    Verdict,
    apply_reply_grounding_gate,
    build_ground_from_turn,
    check_reply_grounding,
    default_repair_fn,
    format_check_deny,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.judge import (
    parse_judge_json,
)
from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import (
    CandidateGround,
    FactItem,
    GroundBlock,
    MatchEvidenceItem,
)
from apps.group_agent_api.agent_factory.module_config import (
    load_modules_config,
    reload_modules_config,
    reset_modules_config_cache,
)

REAL_LLM = os.environ.get("GROUP_AGENT_REAL_LLM_TEST", "").strip() in {
    "1",
    "true",
    "yes",
}


def _ground(*, count: int = 0, with_candidate: bool = False) -> GroundBlock:
    cands: list[CandidateGround] = []
    if with_candidate:
        cands.append(
            CandidateGround(
                user_id="cand_1",
                display_name="阿强",
                facts=[FactItem(field="doing", value="做高中数学教研")],
                match_evidence=[MatchEvidenceItem(summary="公开资料提到教研")],
            )
        )
        count = max(count, 1)
    return GroundBlock(
        candidates=cands,
        initiator_profile={
            "doing": "做AI教育产品",
            "need": "找教研合伙人",
            "offer": "有原型",
        },
        receipts=[],
        candidate_count=count,
    )


def _write_modules_yaml(path: Path, *, reply_grounding: bool) -> Path:
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "preset: current",
                "checks:",
                f"  chk.reply_fact_grounding_llm: {'true' if reply_grounding else 'false'}",
                "modules:",
                f"  mod.brain.reply_grounding: {'true' if reply_grounding else 'false'}",
                "reply_grounding:",
                "  max_attempts: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _reset_modules_config() -> Any:
    reset_modules_config_cache()
    yield
    reset_modules_config_cache()


def test_default_yaml_enables_reply_grounding() -> None:
    cfg = load_modules_config()
    assert cfg.reply_grounding_enabled() is True
    assert cfg.is_check_enabled("chk.reply_fact_grounding_llm") is True
    assert cfg.reply_grounding_max_attempts() == 2
    assert cfg.source_path.endswith("modules.yaml")


def test_module_disabled_via_yaml_is_noop(tmp_path: Path) -> None:
    yaml_path = _write_modules_yaml(tmp_path / "off.yaml", reply_grounding=False)
    reload_modules_config(yaml_path)
    original = "匹配到一位候选人，他主导过全国性课标建设。"
    gated = apply_reply_grounding_gate(
        reply=original,
        reply_mode="dialogue",
        candidates=[],
        model=None,
    )
    assert gated.skipped is True
    assert gated.passed is True
    assert gated.reply == original


def test_l0_invented_candidate_fail_closed() -> None:
    payload = ReplyGroundingInput(
        reply="匹配到一位候选人，背景如下：曾主导某教培项目。",
        reply_mode="dialogue",
        ground=_ground(count=0),
    )
    result = check_reply_grounding(payload, model=None)
    assert result.verdict == Verdict.fail
    assert "invented_candidate" in result.codes
    assert result.layer == "l0"
    deny = format_check_deny(result)
    assert MODULE_ID in deny
    assert "invented_candidate" in deny
    assert "<check_deny" in deny


def test_l0_unverified_action_without_receipts() -> None:
    payload = ReplyGroundingInput(
        reply="我已经帮你发到群里并@了对方。",
        reply_mode="dialogue",
        ground=_ground(count=0),
    )
    result = check_reply_grounding(payload, model=None)
    assert result.verdict == Verdict.fail
    assert "unverified_action" in result.codes


def test_l0_count_mismatch() -> None:
    payload = ReplyGroundingInput(
        reply="本群已有 3 位值得进一步聊的人选。",
        reply_mode="recommendation",
        ground=_ground(count=1, with_candidate=True),
    )
    result = check_reply_grounding(payload, model=None)
    assert result.verdict == Verdict.fail
    assert "unsupported_claim" in result.codes


def test_illegal_judge_output_fail_closed() -> None:
    assert parse_judge_json("") is None
    assert parse_judge_json("not json") is None
    assert parse_judge_json('{"verdict":"maybe"}') is None


def test_parse_judge_pass_and_fail() -> None:
    passed = parse_judge_json(
        '{"verdict":"pass","codes":[],"spans":[],"repairable_by":"llm","message":""}'
    )
    assert passed is not None
    assert passed.verdict == Verdict.pass_
    assert passed.codes == []

    failed = parse_judge_json(
        '{"verdict":"fail","codes":["exaggeration"],"spans":["主导过全国性课标建设"],'
        '"repairable_by":"llm","message":"夸大了角色"}'
    )
    assert failed is not None
    assert failed.verdict == Verdict.fail
    assert failed.codes == ["exaggeration"]
    assert "主导过全国性课标建设" in failed.spans


class _FakeJudge:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def invoke(self, _messages: list[Any]) -> Any:
        self.calls += 1

        class _Msg:
            content = self.text

        return _Msg()


def test_l1_faithful_restatement_pass() -> None:
    model = _FakeJudge(
        '{"verdict":"pass","codes":[],"spans":[],"repairable_by":"llm","message":""}'
    )
    payload = ReplyGroundingInput(
        reply="阿强公开资料提到在做高中数学教研，可以聊聊是否合拍。",
        reply_mode="recommendation",
        ground=_ground(count=1, with_candidate=True),
    )
    result = check_reply_grounding(payload, model=model)
    assert result.verdict == Verdict.pass_
    assert result.layer == "l1"
    assert model.calls == 1


def test_l1_exaggeration_fail() -> None:
    model = _FakeJudge(
        '{"verdict":"fail","codes":["exaggeration"],'
        '"spans":["主导过全国性课标建设"],'
        '"repairable_by":"llm","message":"facts 无全国性课标"}'
    )
    payload = ReplyGroundingInput(
        reply="阿强主导过全国性课标建设，高度契合你的需求。",
        reply_mode="recommendation",
        ground=_ground(count=1, with_candidate=True),
    )
    result = check_reply_grounding(payload, model=model)
    assert result.verdict == Verdict.fail
    assert "exaggeration" in result.codes


def test_l1_chitchat_pass() -> None:
    model = _FakeJudge(
        '{"verdict":"pass","codes":[],"spans":[],"repairable_by":"llm","message":""}'
    )
    payload = ReplyGroundingInput(
        reply="好的，你先说说更看重教研还是运营？",
        reply_mode="dialogue",
        ground=_ground(count=0),
    )
    result = check_reply_grounding(payload, model=model)
    assert result.verdict == Verdict.pass_


def test_missing_model_fail_closed_when_l1_needed() -> None:
    payload = ReplyGroundingInput(
        reply="阿强在做高中数学教研，可以进一步聊。",
        reply_mode="recommendation",
        ground=_ground(count=1, with_candidate=True),
    )
    result = check_reply_grounding(payload, model=None)
    assert result.verdict == Verdict.fail
    assert "schema_invalid" in result.codes


def test_gate_repair_then_pass() -> None:
    class _SeqJudge:
        def __init__(self) -> None:
            self.n = 0

        def invoke(self, messages: list[Any]) -> Any:
            self.n += 1
            text = (
                '{"verdict":"fail","codes":["exaggeration"],'
                '"spans":["全国性课标"],"repairable_by":"llm","message":"夸大"}'
                if self.n == 1
                else '{"verdict":"pass","codes":[],"spans":[],'
                '"repairable_by":"llm","message":""}'
            )

            class _Msg:
                content = text

            return _Msg()

    def repair(_deny: str, _prev: str, _ground: GroundBlock) -> str:
        return "阿强公开资料提到在做高中数学教研，是否合拍仍需沟通确认。"

    model = _SeqJudge()
    gated = apply_reply_grounding_gate(
        reply="阿强主导过全国性课标建设。",
        reply_mode="recommendation",
        candidates=[
            {
                "user_id": "cand_1",
                "display_name": "阿强",
                "doing": {"value": "做高中数学教研", "disclosure": "confirmed_public"},
            }
        ],
        model=model,
        repair_fn=repair,
        max_attempts=2,
        enabled=True,
    )
    assert gated.passed is True
    assert gated.abandoned is False
    assert gated.attempts == 2
    assert "全国性课标" not in gated.reply
    assert gated.check_deny is None


def test_gate_abandon_when_repair_exhausted() -> None:
    class _AlwaysFail:
        def invoke(self, _messages: list[Any]) -> Any:
            class _Msg:
                content = (
                    '{"verdict":"fail","codes":["unsupported_claim"],'
                    '"spans":["假履历"],"repairable_by":"llm","message":"无来源"}'
                )

            return _Msg()

    def bad_repair(_deny: str, _prev: str, _ground: GroundBlock) -> str:
        return "他还主导过更多假履历项目。"

    gated = apply_reply_grounding_gate(
        reply="阿强主导过全国性课标建设。",
        reply_mode="recommendation",
        candidates=[
            {
                "user_id": "cand_1",
                "display_name": "阿强",
                "doing": {"value": "做高中数学教研", "disclosure": "confirmed_public"},
            }
        ],
        model=_AlwaysFail(),
        repair_fn=bad_repair,
        max_attempts=2,
        enabled=True,
    )
    assert gated.abandoned is True
    assert "全国性课标" not in gated.reply
    assert "1 位" in gated.reply or "1位" in gated.reply.replace(" ", "")


def test_build_ground_from_turn_maps_public_fields() -> None:
    ground = build_ground_from_turn(
        candidates=[
            {
                "user_id": "u1",
                "display_name": "小王",
                "doing": {"value": "做算法", "disclosure": "confirmed_public"},
                "match_evidence": [{"summary": "技能重合"}],
            }
        ],
        candidate_count=1,
    )
    assert ground.candidate_count == 1
    assert ground.candidates[0].display_name == "小王"
    assert ground.candidates[0].facts[0].value == "做算法"


def test_fail_always_produces_check_deny_payload() -> None:
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="匹配到一位候选人，背景如下：曾主导某教培项目。",
            reply_mode="dialogue",
            ground=_ground(count=0),
        ),
        model=None,
    )
    deny = format_check_deny(result)
    assert 'module="mod.brain.reply_grounding"' in deny
    assert "codes=" in deny
    assert "spans:" in deny
    assert "可做:" in deny
    assert "不可做:" in deny


# ---------------------------------------------------------------------------
# Trickier L0 / gate / parser scenarios (no live LLM)
# ---------------------------------------------------------------------------


def test_l0_empty_reply_passes() -> None:
    result = check_reply_grounding(
        ReplyGroundingInput(reply="   ", reply_mode="dialogue", ground=_ground(count=0)),
        model=None,
    )
    assert result.verdict == Verdict.pass_
    assert result.layer == "l0"


def test_l0_no_match_generic_language_without_named_person_passes_to_l1() -> None:
    """Empty-result wording without inventing a person should not L0-short-circuit."""
    model = _FakeJudge(
        '{"verdict":"pass","codes":[],"spans":[],"repairable_by":"llm","message":""}'
    )
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="这次暂未找到合适人选，你可以放宽城市或行业后再试。",
            reply_mode="no_match",
            ground=_ground(count=0),
        ),
        model=model,
    )
    assert result.verdict == Verdict.pass_
    assert result.layer == "l1"
    assert model.calls == 1


def test_l0_invented_candidate_via_matched_one_person_phrasing() -> None:
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="匹配到一位候选人，他曾主导某教培项目。",
            reply_mode="dialogue",
            ground=_ground(count=0),
        ),
        model=None,
    )
    assert result.verdict == Verdict.fail
    assert "invented_candidate" in result.codes


def test_l0_count_claim_three_vs_one_adopted() -> None:
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="推荐 3 位值得进一步聊的人选。",
            reply_mode="recommendation",
            ground=_ground(count=1, with_candidate=True),
        ),
        model=None,
    )
    assert result.verdict == Verdict.fail
    assert "unsupported_claim" in result.codes
    assert any("3" in s for s in result.spans)


def test_l0_action_claim_with_receipt_skips_l0_action_fail() -> None:
    """Receipts present → L0 must not fail unverified_action; L1 decides."""
    model = _FakeJudge(
        '{"verdict":"pass","codes":[],"spans":[],"repairable_by":"llm","message":""}'
    )
    ground = _ground(count=0)
    ground = ground.model_copy(
        update={"receipts": [{"kind": "manual_copy", "id": "receipt:1"}]}
    )
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="我已经帮你发到群里并@了对方。",
            reply_mode="dialogue",
            ground=ground,
        ),
        model=model,
    )
    # L0 skips action short-circuit when receipts exist; L1 (fake) passes.
    assert result.verdict == Verdict.pass_
    assert result.layer == "l1"


def test_l0_action_claim_variants_without_receipts() -> None:
    for reply in (
        "已经通知管理员了，请等待。",
        "我帮你发到群里了。",
        "已替你在群里@了对方。",
    ):
        result = check_reply_grounding(
            ReplyGroundingInput(
                reply=reply,
                reply_mode="dialogue",
                ground=_ground(count=0),
            ),
            model=None,
        )
        assert result.verdict == Verdict.fail, reply
        assert "unverified_action" in result.codes, reply


def test_parse_judge_json_strips_markdown_fence() -> None:
    wrapped = (
        "```json\n"
        '{"verdict":"fail","codes":["exaggeration"],"spans":["非常适合"],'
        '"repairable_by":"llm","message":"主观拔高"}\n'
        "```"
    )
    parsed = parse_judge_json(wrapped)
    assert parsed is not None
    assert parsed.verdict == Verdict.fail
    assert parsed.codes == ["exaggeration"]


def test_parse_judge_unknown_codes_are_dropped_then_defaulted() -> None:
    parsed = parse_judge_json(
        '{"verdict":"fail","codes":["made_up_code"],"spans":["x"],'
        '"repairable_by":"llm","message":"bad"}'
    )
    assert parsed is not None
    assert parsed.verdict == Verdict.fail
    assert "made_up_code" not in parsed.codes
    # Empty after filter → default unsupported_claim
    assert parsed.codes == ["unsupported_claim"]


def test_l1_mixed_faithful_plus_invented_clause_fails() -> None:
    """One true fact + one invented clause still fails (judge fixture)."""
    model = _FakeJudge(
        '{"verdict":"fail","codes":["unsupported_claim"],'
        '"spans":["主导过全国性课标建设"],'
        '"repairable_by":"llm","message":"后半句无来源"}'
    )
    payload = ReplyGroundingInput(
        reply="阿强在做高中数学教研，还主导过全国性课标建设。",
        reply_mode="recommendation",
        ground=_ground(count=1, with_candidate=True),
    )
    result = check_reply_grounding(payload, model=model)
    assert result.verdict == Verdict.fail
    assert "unsupported_claim" in result.codes


def test_l1_hedged_invention_still_fails() -> None:
    model = _FakeJudge(
        '{"verdict":"fail","codes":["unsupported_claim"],'
        '"spans":["据说主导过全国课标"],'
        '"repairable_by":"llm","message":"传闻式编造仍无 ground"}'
    )
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="阿强据说主导过全国课标，或许可以聊聊。",
            reply_mode="recommendation",
            ground=_ground(count=1, with_candidate=True),
        ),
        model=model,
    )
    assert result.verdict == Verdict.fail
    assert "unsupported_claim" in result.codes


def test_l1_role_title_inflation_fails() -> None:
    model = _FakeJudge(
        '{"verdict":"fail","codes":["exaggeration"],'
        '"spans":["头部教培龙头的全国教研负责人"],'
        '"repairable_by":"llm","message":"职位拔高"}'
    )
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="阿强是头部教培龙头的全国教研负责人，和你需求高度契合。",
            reply_mode="recommendation",
            ground=_ground(count=1, with_candidate=True),
        ),
        model=model,
    )
    assert result.verdict == Verdict.fail
    assert "exaggeration" in result.codes


def test_l1_numeric_inflation_fails() -> None:
    model = _FakeJudge(
        '{"verdict":"fail","codes":["unsupported_claim"],'
        '"spans":["服务过上百所学校"],'
        '"repairable_by":"llm","message":"无数字来源"}'
    )
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="阿强做高中数学教研，已服务过上百所学校。",
            reply_mode="recommendation",
            ground=_ground(count=1, with_candidate=True),
        ),
        model=model,
    )
    assert result.verdict == Verdict.fail
    assert "unsupported_claim" in result.codes


def test_l1_cross_attribute_initiator_need_as_candidate_win_fails() -> None:
    """Must not treat initiator profile as candidate achievement."""
    model = _FakeJudge(
        '{"verdict":"fail","codes":["unsupported_claim"],'
        '"spans":["已经帮你做出AI教育产品原型"],'
        '"repairable_by":"llm","message":"把发起人 offer 安到候选人身上"}'
    )
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="阿强已经帮你做出AI教育产品原型，可直接对接。",
            reply_mode="recommendation",
            ground=_ground(count=1, with_candidate=True),
        ),
        model=model,
    )
    assert result.verdict == Verdict.fail
    assert "unsupported_claim" in result.codes


def test_gate_two_repairs_then_pass() -> None:
    """First rewrite still bad → second rewrite cleans → pass (max_attempts=3)."""
    rewrites = iter(
        [
            "阿强主导过全国性课标，很适合你。",
            "阿强公开资料提到在做高中数学教研，是否合拍仍需沟通确认。",
        ]
    )

    class _Judge:
        def invoke(self, messages: list[Any]) -> Any:
            blob = "".join(str(getattr(m, "content", "") or "") for m in (messages or []))
            reply_tail = blob.split("reply:")[-1] if "reply:" in blob else blob
            if "是否合拍仍需沟通确认" in reply_tail and "全国性课标" not in reply_tail:
                text = (
                    '{"verdict":"pass","codes":[],"spans":[],'
                    '"repairable_by":"llm","message":""}'
                )
            else:
                text = (
                    '{"verdict":"fail","codes":["exaggeration"],'
                    '"spans":["全国性课标"],"repairable_by":"llm","message":"仍夸大"}'
                )

            class _Msg:
                content = text

            return _Msg()

    gated = apply_reply_grounding_gate(
        reply="阿强主导过全国性课标建设。",
        reply_mode="recommendation",
        candidates=[
            {
                "user_id": "cand_1",
                "display_name": "阿强",
                "doing": {"value": "做高中数学教研", "disclosure": "confirmed_public"},
            }
        ],
        model=_Judge(),
        repair_fn=lambda _d, _p, _g: next(rewrites),
        max_attempts=3,
        enabled=True,
    )
    assert gated.passed is True
    assert gated.abandoned is False
    assert gated.attempts == 3
    assert "全国性课标" not in gated.reply
    assert "教研" in gated.reply


def test_gate_repair_must_not_reintroduce_invented_numbers() -> None:
    def repair(_deny: str, _prev: str, _ground: GroundBlock) -> str:
        return "阿强公开资料提到在做高中数学教研，是否合拍仍需沟通确认。"

    class _Seq:
        def __init__(self) -> None:
            self.n = 0

        def invoke(self, _messages: list[Any]) -> Any:
            self.n += 1
            text = (
                '{"verdict":"fail","codes":["unsupported_claim"],'
                '"spans":["200+教师"],"repairable_by":"llm","message":"无数字"}'
                if self.n == 1
                else '{"verdict":"pass","codes":[],"spans":[],'
                '"repairable_by":"llm","message":""}'
            )

            class _Msg:
                content = text

            return _Msg()

    gated = apply_reply_grounding_gate(
        reply="阿强带过 200+ 教师培训，非常适合你。",
        reply_mode="recommendation",
        candidates=[
            {
                "user_id": "cand_1",
                "display_name": "阿强",
                "doing": {"value": "做高中数学教研", "disclosure": "confirmed_public"},
            }
        ],
        model=_Seq(),
        repair_fn=repair,
        max_attempts=2,
        enabled=True,
    )
    assert gated.passed is True
    assert "200+" not in gated.reply
    assert "非常适合" not in gated.reply


def test_judge_exception_fail_closed() -> None:
    class _Boom:
        def invoke(self, _messages: list[Any]) -> Any:
            raise RuntimeError("provider down")

    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="阿强在做高中数学教研。",
            reply_mode="recommendation",
            ground=_ground(count=1, with_candidate=True),
        ),
        model=_Boom(),
    )
    assert result.verdict == Verdict.fail
    assert "schema_invalid" in result.codes


def test_l0_only_abandon_draft_skips_l1() -> None:
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply="本群已有 1 位公开信息与需求有交集的人选；是否匹配仍需沟通确认。",
            reply_mode="recommendation",
            ground=_ground(count=1, with_candidate=True),
        ),
        model=None,
        l0_only=True,
    )
    assert result.verdict == Verdict.pass_
    assert result.layer == "l0"


def _require_real_llm_env() -> None:
    if not REAL_LLM:
        pytest.skip("set GROUP_AGENT_REAL_LLM_TEST=1 to run live LLM tests")
    provider = (os.environ.get("GROUP_AGENT_PROVIDER") or "").strip().lower()
    if provider not in {"qwen", "dashscope", "deepseek", "openai"}:
        # Allow create_model defaults, but still require a key.
        provider = (os.environ.get("GROUP_AGENT_PROVIDER") or "qwen").strip().lower()
    key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not key or key == "EMPTY":
        pytest.skip("real LLM API key missing in process env")
    if not (os.environ.get("GROUP_AGENT_MODEL") or "").strip() and provider in {
        "qwen",
        "dashscope",
    }:
        os.environ.setdefault("GROUP_AGENT_MODEL", "qwen-plus")
    if not (os.environ.get("GROUP_AGENT_BASE_URL") or "").strip() and provider in {
        "qwen",
        "dashscope",
    }:
        os.environ.setdefault(
            "GROUP_AGENT_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _classify_llm_role(messages: list[Any]) -> str:
    """Label judge sub-agent vs rewrite (main-agent repair) from system prompt."""
    blob = ""
    for msg in messages or []:
        typ = getattr(msg, "type", None) or msg.__class__.__name__
        if "System" in str(typ) or typ == "system":
            blob += _message_text(getattr(msg, "content", None))
    # Order matters: rewrite prompt also mentions「事实对照」in passing.
    if "改口步骤" in blob:
        return "REWRITE_MAIN_AGENT"
    if "推荐文案事实对照" in blob or "事实对照」裁判" in blob:
        return "JUDGE_SUBAGENT"
    return "LLM"


class TracingChatModel:
    """Wrap every ``invoke`` so tests can prove text came from a real LLM round-trip."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        import time

        call_no = len(self.calls) + 1
        role = _classify_llm_role(messages)
        rendered_in: list[str] = []
        for i, msg in enumerate(messages or []):
            typ = getattr(msg, "type", None) or msg.__class__.__name__
            rendered_in.append(
                f"--- messages[{i}] {typ} ---\n{_message_text(getattr(msg, 'content', None))}"
            )
        inbound = "\n\n".join(rendered_in)
        print("\n" + "#" * 72)
        print(f"LLM CALL #{call_no} · role={role} · BEGIN")
        print("#" * 72)
        print(inbound)
        print("-" * 72)
        print(f"LLM CALL #{call_no} · invoking provider…")
        started = time.perf_counter()
        result = self._inner.invoke(messages, **kwargs)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        outbound = _message_text(getattr(result, "content", None))
        print("-" * 72)
        print(f"LLM CALL #{call_no} · role={role} · RESPONSE ({elapsed_ms} ms)")
        print("-" * 72)
        print(outbound)
        print("#" * 72)
        print(f"LLM CALL #{call_no} · role={role} · END")
        print("#" * 72 + "\n")
        self.calls.append(
            {
                "n": call_no,
                "role": role,
                "elapsed_ms": elapsed_ms,
                "request": inbound,
                "response": outbound,
            }
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _live_judge_model(*, trace: bool = False) -> Any:
    _require_real_llm_env()
    # Force real — setdefault is wrong when the shell already has stub.
    os.environ["GROUP_AGENT_MODEL_MODE"] = "real"
    os.environ.setdefault("GROUP_AGENT_PROVIDER", "qwen")
    from apps.group_agent_api.agent_factory.model_builder import create_model

    model = create_model(log_prefix="[ReplyGroundingRealLLM]")
    if trace:
        return TracingChatModel(model)
    return model


_FAIL_CODES = frozenset({"exaggeration", "unsupported_claim", "invented_candidate", "unverified_action"})


# (id, reply, forbidden_substrings that must not survive a rewrite if we repair)
_REAL_LLM_TRICKY_FAIL_CASES: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "mixed_faithful_plus_invented",
        "阿强在做高中数学教研，还主导过全国性课标建设。",
        ("全国性课标",),
    ),
    (
        "hedged_invention",
        "阿强据说主导过全国课标，或许可以聊聊。",
        ("全国课标", "课标建设"),
    ),
    (
        "role_title_inflation",
        "阿强是头部教培龙头的全国教研负责人，和你需求高度契合。",
        ("全国教研负责人", "头部教培", "高度契合"),
    ),
    (
        "numeric_inflation",
        "阿强做高中数学教研，已服务过上百所学校。",
        ("上百所", "100所", "百所"),
    ),
    (
        "cross_attribute_initiator_as_candidate",
        "阿强已经帮你做出AI教育产品原型，可直接对接。",
        ("做出AI教育产品原型", "已经帮你做出"),
    ),
    (
        "subjective_perfect_match",
        "阿强在做高中数学教研，完美匹配你的所有需求，一定能成。",
        ("完美匹配", "一定能成"),
    ),
]


@pytest.mark.real_llm
@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "case_id,reply,_forbidden",
    _REAL_LLM_TRICKY_FAIL_CASES,
    ids=[c[0] for c in _REAL_LLM_TRICKY_FAIL_CASES],
)
def test_reply_grounding_real_llm_tricky_fail_cases(
    case_id: str, reply: str, _forbidden: tuple[str, ...]
) -> None:
    """Live judge must fail tricky invented / inflated / cross-attributed copy."""
    model = _live_judge_model()
    result = check_reply_grounding(
        ReplyGroundingInput(
            reply=reply,
            reply_mode="recommendation",
            ground=_ground(count=1, with_candidate=True),
        ),
        model=model,
    )
    print(f"\nCASE {case_id}\nreply={reply}\nresult={result}")
    assert result.verdict == Verdict.fail, (case_id, result)
    assert result.layer in {"l1", "l0"}, result.layer
    assert any(c in _FAIL_CODES for c in result.codes), (case_id, result.codes)
    assert result.spans or result.message, (case_id, result)


@pytest.mark.real_llm
@pytest.mark.timeout(90)
def test_reply_grounding_real_llm_chitchat_and_generic_no_match_pass() -> None:
    """Live judge must pass clarifying Q and empty-result wording with no bios."""
    model = _live_judge_model()
    for reply, mode in (
        ("好的，你更看重教研经验还是运营资源？", "dialogue"),
        ("这次暂未找到合适人选，你可以放宽城市或行业后再试。", "no_match"),
    ):
        result = check_reply_grounding(
            ReplyGroundingInput(
                reply=reply,
                reply_mode=mode,
                ground=_ground(count=0),
            ),
            model=model,
        )
        print(f"\nPASS CASE mode={mode} reply={reply}\nresult={result}")
        assert result.verdict == Verdict.pass_, (mode, reply, result)
        assert result.codes == []


@pytest.mark.real_llm
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "case_id,bad_reply,forbidden",
    [
        _REAL_LLM_TRICKY_FAIL_CASES[0],  # mixed
        _REAL_LLM_TRICKY_FAIL_CASES[2],  # role title
        _REAL_LLM_TRICKY_FAIL_CASES[3],  # numeric
        _REAL_LLM_TRICKY_FAIL_CASES[4],  # cross-attribute
    ],
    ids=["mixed_rewrite", "title_rewrite", "numeric_rewrite", "cross_attr_rewrite"],
)
def test_reply_grounding_real_llm_tricky_check_and_rewrite(
    case_id: str, bad_reply: str, forbidden: tuple[str, ...]
) -> None:
    """Fail → rewrite → pass for tricky cases; forbidden spans must not remain."""
    model = _live_judge_model(trace=True)
    assert isinstance(model, TracingChatModel)
    candidates = [
        {
            "user_id": "cand_1",
            "display_name": "阿强",
            "doing": {"value": "做高中数学教研", "disclosure": "confirmed_public"},
            "match_evidence": [{"summary": "公开资料提到教研"}],
        }
    ]

    print("\n" + "=" * 72)
    print(f"TRICKY REWRITE · {case_id}")
    print("=" * 72)
    print(f"initial:\n{bad_reply}")

    first = check_reply_grounding(
        ReplyGroundingInput(
            reply=bad_reply,
            reply_mode="recommendation",
            ground=_ground(count=1, with_candidate=True),
        ),
        model=model,
    )
    print(
        f"first check: verdict={first.verdict.value} codes={first.codes} "
        f"spans={first.spans} message={first.message}"
    )
    assert first.verdict == Verdict.fail, (case_id, first)

    gated = apply_reply_grounding_gate(
        reply=bad_reply,
        reply_mode="recommendation",
        candidates=candidates,
        candidate_count=1,
        model=model,
        repair_fn=default_repair_fn(model),
        max_attempts=3,
        enabled=True,
    )
    print(
        f"gate: passed={gated.passed} abandoned={gated.abandoned} "
        f"attempts={gated.attempts}\nfinal:\n{gated.reply}"
    )
    assert gated.passed is True, (case_id, gated)
    for token in forbidden:
        assert token not in gated.reply, (case_id, token, gated.reply)
    # Must not invent the classic hallucination tokens either.
    assert "全国性课标" not in gated.reply
    assert "200+" not in gated.reply and "200＋" not in gated.reply
    if not gated.abandoned:
        assert "教研" in gated.reply or "人选" in gated.reply or "阿强" in gated.reply, (
            case_id,
            gated.reply,
        )
        rewrite_roles = [c["role"] for c in model.calls]
        assert "REWRITE_MAIN_AGENT" in rewrite_roles or gated.attempts >= 1, rewrite_roles


@pytest.mark.real_llm
@pytest.mark.timeout(90)
def test_reply_grounding_real_llm_check_catches_exaggeration() -> None:
    """Live judge must fail a reply that invents facts outside ground."""
    model = _live_judge_model()
    payload = ReplyGroundingInput(
        reply=(
            "阿强主导过全国性课标建设，还服务过上百所学校，"
            "和你的需求高度契合，可以直接对接。"
        ),
        reply_mode="recommendation",
        ground=_ground(count=1, with_candidate=True),
    )
    result = check_reply_grounding(payload, model=model)
    assert result.verdict == Verdict.fail, result
    assert result.layer == "l1"
    assert any(
        code in result.codes for code in ("exaggeration", "unsupported_claim")
    ), result.codes
    deny = format_check_deny(result)
    assert MODULE_ID in deny
    assert "spans:" in deny


@pytest.mark.real_llm
@pytest.mark.timeout(90)
def test_reply_grounding_real_llm_faithful_reply_passes() -> None:
    """Live judge must pass a faithful restatement of candidate facts."""
    model = _live_judge_model()
    payload = ReplyGroundingInput(
        reply=(
            "阿强的公开资料提到在做高中数学教研。"
            "是否真正匹配还需要你们聊过后确认。"
        ),
        reply_mode="recommendation",
        ground=_ground(count=1, with_candidate=True),
    )
    result = check_reply_grounding(payload, model=model)
    assert result.verdict == Verdict.pass_, result
    assert result.codes == []


@pytest.mark.real_llm
@pytest.mark.timeout(180)
def test_reply_grounding_real_llm_check_and_rewrite() -> None:
    """Fail → check_deny rewrite → second check must pass without invented spans.

    Every LLM ``invoke`` (judge sub-agent + rewrite) is wrapped so the log shows
    exact request messages and provider responses — proving rewrite text is
    model-generated, not a local template.
    """
    model = _live_judge_model(trace=True)
    assert isinstance(model, TracingChatModel)
    bad_reply = (
        "我给你找到了阿强：他主导过全国性课标建设，"
        "还带过 200+ 教师培训，非常适合你。"
    )
    candidates = [
        {
            "user_id": "cand_1",
            "display_name": "阿强",
            "doing": {"value": "做高中数学教研", "disclosure": "confirmed_public"},
            "match_evidence": [{"summary": "公开资料提到教研"}],
        }
    ]
    ground = _ground(count=1, with_candidate=True)

    def _dump(title: str, body: str = "") -> None:
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        if body:
            print(body)

    _dump(
        "FIXTURE · ground (authoritative facts this turn)",
        ground.model_dump_json(indent=2, ensure_ascii=False),
    )
    _dump("FIXTURE · original recommendation copy (hallucinated)", bad_reply)

    first = check_reply_grounding(
        ReplyGroundingInput(
            reply=bad_reply,
            reply_mode="recommendation",
            ground=ground,
        ),
        model=model,
    )
    _dump(
        "PARSED · first check result (from JUDGE_SUBAGENT LLM call above)",
        (
            f"verdict={first.verdict.value}\n"
            f"layer={first.layer}\n"
            f"codes={first.codes}\n"
            f"spans={first.spans}\n"
            f"repairable_by={first.repairable_by.value}\n"
            f"message={first.message}"
        ),
    )
    assert first.verdict == Verdict.fail, first
    assert first.codes, first
    assert any(c["role"] == "JUDGE_SUBAGENT" for c in model.calls), model.calls

    deny = format_check_deny(first)
    _dump("ASSEMBLED · ctx.check_deny (not an LLM call; local format)", deny)

    calls_before_gate = len(model.calls)
    gated = apply_reply_grounding_gate(
        reply=bad_reply,
        reply_mode="recommendation",
        candidates=candidates,
        candidate_count=1,
        model=model,
        repair_fn=default_repair_fn(model),
        max_attempts=3,
        enabled=True,
    )
    gate_calls = model.calls[calls_before_gate:]
    roles = [c["role"] for c in gate_calls]
    _dump(
        "GATE · LLM roles during apply_reply_grounding_gate",
        (
            f"new_llm_calls={len(gate_calls)} roles={roles}\n"
            f"passed={gated.passed} abandoned={gated.abandoned} "
            f"attempts={gated.attempts}\n"
            f"final user-visible reply:\n{gated.reply}"
        ),
    )

    assert gated.passed is True, gated
    assert "全国性课标" not in gated.reply
    assert "200+" not in gated.reply and "200＋" not in gated.reply
    assert "教研" in gated.reply or "人选" in gated.reply, gated.reply

    if gated.abandoned:
        _dump("DONE · abandoned to safe draft", gated.reply)
        return

    # Prove final reply text appeared as a REWRITE_MAIN_AGENT response body.
    rewrite_responses = [
        c["response"].strip()
        for c in model.calls
        if c["role"] == "REWRITE_MAIN_AGENT"
    ]
    assert rewrite_responses, (
        "expected at least one REWRITE_MAIN_AGENT LLM call; "
        f"roles={[c['role'] for c in model.calls]}"
    )
    assert any(
        gated.reply.strip() == r or gated.reply.strip() in r for r in rewrite_responses
    ), (
        "final reply must match a REWRITE_MAIN_AGENT LLM response\n"
        f"final={gated.reply!r}\n"
        f"rewrite_responses={rewrite_responses!r}"
    )

    second = check_reply_grounding(
        ReplyGroundingInput(
            reply=gated.reply,
            reply_mode="recommendation",
            ground=ground,
        ),
        model=model,
    )
    _dump(
        "PARSED · re-check of gated reply (from another JUDGE_SUBAGENT call)",
        (
            f"verdict={second.verdict.value}\n"
            f"codes={second.codes}\n"
            f"spans={second.spans}\n"
            f"message={second.message}"
        ),
    )
    if second.verdict == Verdict.pass_:
        _dump(
            "DONE · LLM call inventory",
            "\n".join(
                f"  #{c['n']} {c['role']} {c['elapsed_ms']}ms "
                f"response_preview={c['response'][:80]!r}…"
                for c in model.calls
            ),
        )
        return

    repair = default_repair_fn(model)
    assert repair is not None
    rewritten = repair(format_check_deny(second), gated.reply, ground)
    third = check_reply_grounding(
        ReplyGroundingInput(
            reply=rewritten,
            reply_mode="recommendation",
            ground=ground,
        ),
        model=model,
    )
    _dump(
        "PARSED · final re-check after extra rewrite",
        (
            f"verdict={third.verdict.value}\n"
            f"codes={third.codes}\n"
            f"final reply:\n{rewritten}"
        ),
    )
    assert third.verdict == Verdict.pass_, (second, rewritten, third)
    assert "全国性课标" not in rewritten
    assert "200+" not in rewritten
    _dump(
        "DONE · LLM call inventory",
        "\n".join(
            f"  #{c['n']} {c['role']} {c['elapsed_ms']}ms"
            for c in model.calls
        ),
    )
