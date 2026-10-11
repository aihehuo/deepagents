"""BP evidence remains bounded and distinct from current customer intent."""
import pytest
from apps.community_manager.journey_contract import validate_context, CONVERGENCE_PROMPT


def context() -> tuple[dict, dict]:
    """Return a host-owned BP source with original page metadata."""
    value = {"schema_version": 1, "user_number": "123456", "sources": {
        "bp_block:1:block_0": {"role": "document_self_report", "text": "项目介绍" * 100,
            "recorded_at": None, "bp_id": 12, "page": 30, "partial": True, "review_required": True}},
        "previous": None, "observed_at": "2026-10-11T00:00:00Z"}
    return value, {"source": "wecom_service", "wecom_service_customer_number": "123456"}


def test_bp_source_and_original_page_are_accepted() -> None:
    value, metadata = context()
    assert validate_context(value, metadata) == value
    assert "不能作为近期用户意愿确认" in CONVERGENCE_PROMPT


@pytest.mark.parametrize("field, replacement", [("page", 0), ("page", True), ("partial", "true"), ("review_required", 1), ("text", "x" * 601)])
def test_invalid_bp_metadata_is_rejected(field: str, replacement: object) -> None:
    value, metadata = context()
    value["sources"]["bp_block:1:block_0"][field] = replacement
    with pytest.raises(ValueError):
        validate_context(value, metadata)


def test_bp_metadata_cannot_be_disguised_as_user_confirmation() -> None:
    value, metadata = context()
    value["sources"]["bp_block:1:block_0"]["role"] = "user"
    with pytest.raises(ValueError):
        validate_context(value, metadata)
