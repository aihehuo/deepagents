"""Trusted service metadata must accept bounded manual parameters only."""
import pytest
from apps.group_agent_api.app.models import validate_group_agent_metadata


def test_trusted_manual_plan_and_execution_metadata():
    meta = {"source": "wecom_service", "wecom_service_skill": "get-ai-cofounders",
            "wecom_service_parameters": None, "wecom_service_excluded_recommendations": []}
    assert validate_group_agent_metadata(meta, allow_referral_context=True) == meta
    meta["wecom_service_parameters"] = {"q": "上海技术合伙人"}
    meta["wecom_service_excluded_recommendations"] = ["users:123"]
    assert validate_group_agent_metadata(meta, allow_referral_context=True) == meta


@pytest.mark.parametrize("trusted,source", [(False, "wecom_service"), (True, "group_agent_ui")])
def test_untrusted_manual_metadata_rejected(trusted, source):
    with pytest.raises(ValueError):
        validate_group_agent_metadata({"source": source, "wecom_service_parameters": {}}, allow_referral_context=trusted)


@pytest.mark.parametrize("value", [{}, ["x" * 101], [1], ["users:1"] * 2001])
def test_invalid_exclusions_rejected(value):
    with pytest.raises(ValueError):
        validate_group_agent_metadata({"source": "wecom_service", "wecom_service_excluded_recommendations": value}, allow_referral_context=True)


def test_parameters_cannot_inject_identity():
    with pytest.raises(ValueError):
        validate_group_agent_metadata({"source": "wecom_service", "wecom_service_skill": "get-ai-cofounders", "wecom_service_parameters": {"user_id": "123"}}, allow_referral_context=True)
