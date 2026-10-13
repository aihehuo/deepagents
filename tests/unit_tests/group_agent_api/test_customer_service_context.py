"""Customer-service policy composition across member capability modes."""

from collections.abc import Iterator

import pytest

from apps.group_agent_api.agent_factory.agent import _composed_member_prompt
from apps.group_agent_api.agent_factory.context import (
    CTX_SYSTEM_CUSTOMER_SERVICE,
    SYSTEM_FRAGMENT_IDS,
    build_system_prompt,
)
from apps.group_agent_api.agent_factory.module_config import (
    reset_modules_config_cache,
)


@pytest.fixture(autouse=True)
def reset_config() -> Iterator[None]:
    """Isolate the module switch cache from other suites."""
    reset_modules_config_cache()
    yield
    reset_modules_config_cache()


@pytest.mark.parametrize("can_search", [False, True])
@pytest.mark.parametrize("can_save", [False, True])
@pytest.mark.parametrize("public", [False, True])
def test_member_modes_include_customer_service_without_granting_tools(
    *, can_search: bool, can_save: bool, public: bool
) -> None:
    policy = build_system_prompt([CTX_SYSTEM_CUSTOMER_SERVICE])
    prompt = _composed_member_prompt(
        can_search=can_search, can_save=can_save, public=public
    )
    assert policy.strip()
    assert policy in prompt
    assert ("search_candidates" in prompt) == (can_search and not public)
    assert ("save_group_profile" in prompt) == can_save


def test_customer_service_policy_can_be_disabled_independently() -> None:
    policy = build_system_prompt([CTX_SYSTEM_CUSTOMER_SERVICE])
    without = build_system_prompt(
        [cid for cid in SYSTEM_FRAGMENT_IDS if cid != CTX_SYSTEM_CUSTOMER_SERVICE]
    )
    assert policy not in without
    assert "search_candidates" in without
    assert "save_group_profile" in without
    assert policy in build_system_prompt(SYSTEM_FRAGMENT_IDS)
