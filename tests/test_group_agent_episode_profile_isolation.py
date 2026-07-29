"""Episode isolation: new episode must not reuse prior-episode profile as「已更新」."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from apps.group_agent_api.agent_factory.profile_quality import (
    bind_profile_to_episode,
    profile_bound_to_episode,
)
from apps.group_agent_api.agent_factory.profile_schema import profile_from_flat
from apps.group_agent_api.agent_factory.profile_store import save_profile
from apps.group_agent_api.app.endpoints.chat import _profile_usable_this_turn
from apps.group_agent_api.app.utils import thread_id


def test_thread_id_includes_episode() -> None:
    assert thread_id(
        user_id="u1", group_id="g1", conversation_id="c1", episode_id="ep2"
    ) == "ga::u1::g1::c1::ep2"
    assert thread_id(
        user_id="u1", group_id="g1", conversation_id="c1"
    ) == "ga::u1::g1::c1"


def test_prior_episode_profile_not_usable_until_resave(tmp_path: Path) -> None:
    profile = profile_from_flat(
        user_id="u1",
        group_id="g1",
        doing="推进爱合伙群智能体产品推广",
        need="找AI agent拓客合伙人一起共建",
        offer="产品设计与技术协作能力",
    )
    save_profile(tmp_path, profile)
    bind_profile_to_episode(
        tmp_path, "u1", "g1", metadata={"episode_id": "ep_old"}
    )
    assert profile_bound_to_episode(
        tmp_path, "u1", "g1", metadata={"episode_id": "ep_old"}
    )
    assert not profile_bound_to_episode(
        tmp_path, "u1", "g1", metadata={"episode_id": "ep_new"}
    )

    usable, _ = _profile_usable_this_turn(
        base_dir=tmp_path,
        user_id="u1",
        group_id="g1",
        messages=[],
        msg_count_before=0,
        metadata={"episode_id": "ep_new"},
    )
    assert usable is False

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_group_profile",
                    "args": {},
                    "id": "tc1",
                }
            ],
        ),
        ToolMessage(
            content="ok: saved profile to /users/u1/groups/g1/profile.json",
            name="save_group_profile",
            tool_call_id="tc1",
        ),
    ]
    usable2, _ = _profile_usable_this_turn(
        base_dir=tmp_path,
        user_id="u1",
        group_id="g1",
        messages=messages,
        msg_count_before=0,
        metadata={"episode_id": "ep_new"},
    )
    assert usable2 is True
    assert profile_bound_to_episode(
        tmp_path, "u1", "g1", metadata={"episode_id": "ep_new"}
    )
