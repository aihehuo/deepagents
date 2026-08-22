"""REQ-DA-066 · Group Agent Grounded Orchestration & Action Consistency Tests.

Validates:
- WP1: Grounding protocol models & canonical JSON digest
- WP2: save_group_profile v2 with ProfileClaimInput, MatchConstraintInput, trusted source_message_id, numeric extraction
- WP3: MatchContractV2 client rejecting global, missing evidence, invalid disclosure, fail closed
- WP4: Typed final in async_manager with dialogue_text separation, reply_mode determination, ga-grounding-v1 payload
- WP5: Restricted candidate copy in per_candidate_copy without unevidenced hype, facts fallback to '现有资料未说明'
- WP6: Revisit and candidate Q&A with prior_recommendation facts
- WP7: Action claim guard blocking unauthorized group send, @, admin notification completion claims
"""

import json
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from apps.group_agent_api.agent_factory.grounding_protocol import (
    ClaimType,
    ConfidenceLevel,
    DisclosureLevelV2,
    GroundedFinalV1,
    GroupProfileV2,
    MatchConstraintInput,
    MatchConstraintV1,
    MatchEvidenceV1,
    CandidateFactV1,
    CandidateV2,
    MatchResultV2,
    ProfileClaimInput,
    ProfileClaimV2,
    ProfileEvidenceV2,
    ReplyMode,
    canonical_json_bytes,
    canonical_sha256,
    extract_numbers_and_units,
    validate_profile_claim_grounding,
)
from apps.group_agent_api.agent_factory.agent import save_group_profile
from apps.group_agent_api.agent_factory.content_quality import (
    guard_action_claims,
    looks_like_unauthorized_action_claim,
)
from apps.group_agent_api.agent_factory.per_candidate_copy import (
    generate_single_candidate_copy,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.revisit import (
    PriorCandidate,
    PriorCandidateFact,
    PriorRecommendation,
    known_match_system_content,
    parse_prior_recommendation,
)
from apps.group_agent_api.agent_factory.integrations.match_client import (
    fetch_group_agent_match,
)


# ==============================================================================
# WP1: Protocol models & digest
# ==============================================================================

def test_wp1_canonical_json_digest_deterministic() -> None:
    claim = ProfileClaimV2(
        value="正在验证集装箱降温方案",
        disclosure=DisclosureLevelV2.match_only,
        claim_type=ClaimType.hypothesis,
        confidence=ConfidenceLevel.user_stated,
        evidence=[
            ProfileEvidenceV2(
                source_message_id=123,
                evidence_text="我现在有一个集装箱降温的想法",
            )
        ],
    )
    d1 = canonical_sha256(claim)
    d2 = canonical_sha256(claim)
    assert d1 == d2
    assert d1.startswith("sha256:")
    assert len(d1) == 7 + 64


def test_wp1_unknown_constraint_operator_fails() -> None:
    with pytest.raises(ValueError, match="unknown constraint operator"):
        MatchConstraintV1(
            field="city",
            operator="invalid_operator",
            values=["上海"],
        )


def test_wp1_unknown_constraint_field_fails() -> None:
    with pytest.raises(ValueError, match="unknown constraint field"):
        MatchConstraintV1(
            field="unknown_field_xyz",
            operator="in",
            values=["上海"],
        )


# ==============================================================================
# WP2: save_group_profile v2 & grounding validation
# ==============================================================================

def test_wp2_numeric_extraction_check() -> None:
    text = "我们团队有10人，已经服务了50家企业，销售额达到了300万。"
    nums = extract_numbers_and_units(text)
    assert "10人" in nums or "10" in nums
    assert "50家" in nums or "50" in nums
    assert "300万" in nums or "300" in nums


def test_wp2_claim_grounding_validation() -> None:
    source_msg = "我们目前在做AI教育产品，还在初步验证中。"
    claim = ProfileClaimV2(
        value="已服务100家企业",
        claim_type=ClaimType.fact,
        evidence=[
            ProfileEvidenceV2(
                source_message_id=1,
                evidence_text="我们目前在做AI教育产品",
            )
        ],
    )
    violations = validate_profile_claim_grounding(claim, source_msg)
    assert any("unsupported_numbers" in v for v in violations)


def test_wp2_save_group_profile_with_constraints(tmp_path: Path) -> None:
    config = {
        "metadata": {
            "user_id": "u_test",
            "group_id": "g_test",
            "base_dir": str(tmp_path),
            "source_message_id": 999,
            "source_message_text": "我在做外贸SaaS，需要找技术合伙人，只考虑上海的伙伴",
        }
    }
    res = save_group_profile.invoke(
        {
            "doing": {"value": "外贸SaaS", "claim_type": "fact", "disclosure": "confirmed_public", "evidence_text": "我在做外贸SaaS"},
            "need": {"value": "技术合伙人", "claim_type": "goal", "disclosure": "match_only", "evidence_text": "需要找技术合伙人"},
            "offer": {"value": "外贸行业资源", "claim_type": "fact", "disclosure": "match_only"},
            "match_constraints": [
                {"field": "city", "operator": "in", "values": ["上海"], "strength": "hard", "evidence_text": "只考虑上海的伙伴"}
            ],
        },
        config=config,
    )
    assert "ok:" in res


# ==============================================================================
# WP3: MatchContractV2 client
# ==============================================================================

def test_wp3_match_v2_rejects_global_and_missing_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"1"
    mock_resp.json.return_value = {
        "contract_version": "ga-match-v2",
        "status": "matched",
        "reason": "matched_2",
        "candidates": [
            {
                "user_id": "1001",
                "display_name": "张三",
                "source_group_id": "global",  # must be rejected
                "wechat_reachable": True,
                "facts": [{"field": "doing", "value": "AI"}],
                "match_evidence": [{"initiator_field": "need", "candidate_field": "doing", "relation": "match", "summary": "契合"}],
            },
            {
                "user_id": "1002",
                "display_name": "李四",
                "source_group_id": "grp_1",
                "wechat_reachable": True,
                "facts": [],  # missing facts, must be rejected in v2
                "match_evidence": [],
            },
            {
                "user_id": "1003",
                "display_name": "王五",
                "source_group_id": "grp_2",
                "same_group": False,
                "wechat_reachable": True,
                "facts": [
                    {
                        "field": "doing",
                        "value": "后端架构",
                        "disclosure": "match_only",
                        "source_type": "group_agent_profile",
                        "source_ref": "profile:1003:2",
                        "source_version": 2,
                        "source_group_id": "grp_2",
                    }
                ],
                "match_evidence": [{"initiator_field": "need", "candidate_field": "doing", "relation": "match", "summary": "后端经验匹配需求"}],
            },
        ],
    }

    monkeypatch.setattr("requests.post", lambda *a, **kw: mock_resp)
    result = fetch_group_agent_match(
        query="找后端架构师",
        bearer="test_jwt",
        contract_version="ga-match-v2",
    )
    assert result.status == "matched"
    assert len(result.candidates) == 1
    assert result.candidates[0]["user_id"] == "1003"


@pytest.mark.parametrize(
    ("fact_patch", "expected_field"),
    [
        ({"field": "city"}, "field"),
        ({"source_ref": None}, "source_ref"),
        ({"source_version": None}, "source_version"),
        ({"source_group_id": None}, "source_group_id"),
    ],
)
def test_wp3_candidate_fact_requires_micro_authority_fields(
    fact_patch: dict[str, Any],
    expected_field: str,
) -> None:
    fact = {
        "field": "doing",
        "value": "后端架构",
        "disclosure": "match_only",
        "source_type": "group_agent_profile",
        "source_ref": "profile:1003:2",
        "source_version": 2,
        "source_group_id": "grp_2",
    }
    fact.update(fact_patch)

    with pytest.raises(ValueError, match=expected_field):
        CandidateFactV1.model_validate(fact)


def test_wp3_accepts_captured_new_api_match_v2_fixture() -> None:
    fixture_path = Path(__file__).parent / "fixtures/group_agent/new_api_match_v2.json"
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))

    result = MatchResultV2.model_validate(raw)

    assert result.status == "matched"
    assert result.candidates[0].user_id == "202"
    assert result.candidates[0].facts[0].source_ref == "profile:202:2"


@pytest.mark.parametrize(
    "tampered_ref",
    [
        "202",
        "profile:999:2",
        "profile:202:3",
        "profile:202:02",
    ],
)
def test_wp3_rejects_tampered_opaque_profile_ref(tampered_ref: str) -> None:
    fixture_path = Path(__file__).parent / "fixtures/group_agent/new_api_match_v2.json"
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    raw["candidates"][0]["facts"][0]["source_ref"] = tampered_ref

    with pytest.raises(ValueError, match="source_ref"):
        MatchResultV2.model_validate(raw)


# ==============================================================================
# WP4: Typed final in async_manager & Deep -> Micro contract fixtures
# ==============================================================================

def test_wp4_grounded_final_and_micro_contract_fixtures() -> None:
    from apps.group_agent_api.agent_factory.grounding_protocol import (
        calculate_candidate_facts_digest,
        DialogueKind,
        InviteBlock,
        MatchSummaryBlock,
        ProfileSummaryBlock,
        GroundingBlock,
    )
    import hashlib

    # 1. Candidate facts digest algorithm test matching Micro
    candidates = [
        {
            "user_id": "999901",
            "source_group_id": "group_123",
            "display_name": "候选人甲",
            "same_group": False,
            "wechat_reachable": True,
            "facts": [
                {
                    "field": "doing",
                    "value": "跨境电商物流SaaS",
                    "disclosure": "match_only",
                    "source_type": "group_agent_profile",
                    "source_ref": "profile:999901:1",
                    "source_version": 1,
                    "source_group_id": "group_123",
                }
            ],
            "match_evidence": [
                {
                    "initiator_field": "need",
                    "candidate_field": "doing",
                    "relation": "need_matches_doing",
                    "summary": "需求与候选人当前方向对应",
                }
            ],
        }
    ]
    raw_tuple = "999901|doing|跨境电商物流SaaS|match_only"
    expected_digest = f"sha256:{hashlib.sha256(raw_tuple.encode('utf-8')).hexdigest()}"
    calculated_digest = calculate_candidate_facts_digest(candidates)
    assert calculated_digest == expected_digest

    # 2. Recommendation GroundedFinalV1 schema test
    rec_final = GroundedFinalV1(
        protocol_version="ga-grounding-v1",
        run_id="__RUN_ID__",
        reply_mode=ReplyMode.recommendation,
        dialogue_text=None,
        candidate_count=1,
        candidates=candidates,
        profile=ProfileSummaryBlock(
            schema_version=1,
            profile_version=1,
            digest="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            source_group_id="group_123",
            doing={"value": "外贸平台", "claim_type": "fact"},
            need={"value": "物流合伙人", "claim_type": "goal"},
            offer={"value": "海外资源", "claim_type": "fact"},
        ),
        match=MatchSummaryBlock(
            contract_version="ga-match-v2",
            status="matched",
            reason_code="matched_1",
            candidates=candidates,
            candidate_count=1,
        ),
        invite=InviteBlock(
            status="ready",
            delivery_kind="manual_copy",
            text="您好，我们群内有一位匹配的伙伴。",
        ),
        grounding=GroundingBlock(
            candidate_facts_digest=calculated_digest,
            constraint_digest="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
    )
    rec_dict = rec_final.model_dump(mode="json")
    assert rec_dict["protocol_version"] == "ga-grounding-v1"
    assert rec_dict["reply_mode"] == "recommendation"
    assert rec_dict["candidate_count"] == 1
    assert rec_dict["invite"]["status"] == "ready"
    assert rec_dict["invite"]["delivery_kind"] == "manual_copy"
    assert rec_dict["grounding"]["candidate_facts_digest"] == expected_digest
    assert rec_dict["profile"]["doing"]["value"] == "外贸平台"
    assert rec_dict["profile"]["source_group_id"] == "group_123"
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "group_agent"
        / "deep_grounded_final_v1.json"
    )
    assert rec_dict == json.loads(fixture_path.read_text(encoding="utf-8"))

    # 3. Dialogue GroundedFinalV1 schema test with dialogue_kind
    diag_final = GroundedFinalV1(
        protocol_version="ga-grounding-v1",
        run_id="ga_run_456",
        reply_mode=ReplyMode.dialogue,
        dialogue_kind=DialogueKind.clarification_question,
        dialogue_text="请问你想找哪个城市的合伙人？",
        candidate_count=0,
        candidates=[],
        match=MatchSummaryBlock(
            contract_version="ga-match-v2",
            status="empty",
            candidates=[],
            candidate_count=0,
        ),
        invite=InviteBlock(
            status="not_available",
            delivery_kind=None,
            text=None,
        ),
    )
    diag_dict = diag_final.model_dump(mode="json")
    assert diag_dict["protocol_version"] == "ga-grounding-v1"
    assert diag_dict["reply_mode"] == "dialogue"
    assert diag_dict["dialogue_kind"] == "clarification_question"
    assert diag_dict["dialogue_text"] == "请问你想找哪个城市的合伙人？"
    assert diag_dict["invite"]["status"] == "not_available"
    assert diag_dict["invite"]["delivery_kind"] is None


# ==============================================================================
# WP5: Restricted candidate copy
# ==============================================================================


def test_wp5_restricted_copy_no_unsupported_hype() -> None:
    profile = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="跨境电商独立站",
        need="海外支付与合规专家",
        offer="供应链资源",
    )
    candidate = {
        "user_id": "cand_99",
        "display_name": "赵六",
        "doing": {"value": "海外支付与风控合规", "disclosure": "confirmed_public"},
        "match_evidence": [
            {"initiator_field": "need", "candidate_field": "doing", "relation": "need_matches_doing", "summary": "对方有海外支付与合规背景，与您的合规需求方向一致"}
        ],
    }
    copy = generate_single_candidate_copy(profile, candidate)
    # Highlights must come from evidence summary
    assert "对方有海外支付与合规背景" in copy["match_highlights"][0]
    # No unconditional hype
    assert "落地助力" not in copy["match_highlights"][0]
    # Missing facts fallback
    candidate_empty = {
        "user_id": "cand_100",
        "display_name": "孙七",
    }
    copy_empty = generate_single_candidate_copy(profile, candidate_empty)
    assert "现有资料未说明" in copy_empty["match_highlights"][0] or "现有资料未说明" in copy_empty["match_highlights"][1]


# ==============================================================================
# WP6: Revisit and candidate Q&A
# ==============================================================================

def test_wp6_prior_recommendation_parser_and_prompt() -> None:
    raw_meta = {
        "prior_recommendation": {
            "artifact_run_id": "ga_run_123",
            "artifact_digest": "sha256:abc",
            "candidates": [
                {
                    "user_id": "493347",
                    "display_name": "邵**",
                    "same_group": False,
                    "connection_status": "not_requested",
                    "facts": [
                        {"field": "doing", "value": "AI与本地商户相关项目", "disclosure": "match_only"}
                    ],
                }
            ],
        }
    }
    prior_rec = parse_prior_recommendation(raw_meta["prior_recommendation"])
    assert prior_rec is not None
    assert len(prior_rec.candidates) == 1
    assert prior_rec.candidates[0].display_name == "邵**"
    assert prior_rec.candidates[0].same_group is False

    system_content = known_match_system_content(None, prior_rec=prior_rec)
    assert system_content is not None
    assert "邵**" in system_content
    assert "异群" in system_content
    assert "现有资料未说明，可申请对接后向本人确认" in system_content


# ==============================================================================
# WP7: Action claim guard
# ==============================================================================

def test_wp7_action_claim_guard_blocks_unauthorized_claims() -> None:
    claims = [
        "我已经帮您发送到群里并@了对方，请留意消息。",
        "我已经通知管理员了，管理员已收到你的对接申请。",
        "在群里@了对方，对方已回复同意交流。",
        "已有2人触达，对方同意了申请。",
    ]
    for c in claims:
        assert looks_like_unauthorized_action_claim(c) is True
        safe_reply, blocked = guard_action_claims(c)
        assert blocked is True
        assert "我无法直接向群内发送消息或通知管理员" in safe_reply

    safe_text = "我理解你的需求是寻找懂技术的合伙人，你可以点击卡片申请对接。"
    assert looks_like_unauthorized_action_claim(safe_text) is False
    res_text, blocked = guard_action_claims(safe_text)
    assert blocked is False
    assert res_text == safe_text
