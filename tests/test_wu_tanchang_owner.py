from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from apps.wu_tanchang_api.agent_factory.agent import create_agent
from apps.wu_tanchang_api.agent_factory.owner_tools import (
    get_client_detail,
    get_consultation_stats,
    list_recent_clients,
)
from apps.wu_tanchang_api.config import WuAgentConfig


class _FakeCheckpointTuple:
    def __init__(
        self,
        thread_id: str,
        ts: str,
        messages: list[Any],
        checkpoint_id: str = "ckpt_1",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            }
        }
        self.checkpoint = {
            "v": 1,
            "ts": ts,
            "id": checkpoint_id,
            "channel_versions": {},
            "versions_seen": {},
            "updated_channels": [],
            "channel_values": {"messages": messages},
        }
        self.metadata = metadata or {"source": "loop", "step": 1, "parents": {}}


class _FakeCheckpointer:
    def __init__(self, checkpoint_tuples: list[_FakeCheckpointTuple]) -> None:
        self._tuples = checkpoint_tuples
        self.storage = {}
        for item in checkpoint_tuples:
            tid = item.config["configurable"]["thread_id"]
            ns = item.config["configurable"]["checkpoint_ns"]
            cid = item.config["configurable"]["checkpoint_id"]
            self.storage.setdefault(tid, {}).setdefault(ns, {})[cid] = item.checkpoint

    def list(self, config: dict[str, Any] | None = None) -> list[_FakeCheckpointTuple]:
        if (
            config
            and "configurable" in config
            and "thread_id" in config["configurable"]
        ):
            tid = config["configurable"]["thread_id"]
            return [
                x for x in self._tuples if x.config["configurable"]["thread_id"] == tid
            ]
        return self._tuples


class _FakeAgent:
    def __init__(self, checkpointer: _FakeCheckpointer) -> None:
        self.checkpointer = checkpointer


@pytest.fixture
def fake_workspace_setup(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
{
  "default_model_provider": "qwen",
  "providers": {
    "qwen": {
      "api_type": "openai-compatible",
      "base_url": "https://dashscope.test/v1",
      "api_key": "fake-key",
      "main_agent_model": "qwen-flash"
    }
  }
}
""",
        encoding="utf-8",
    )

    for name in ["IDENTITY.md", "SOUL.md", "AGENTS.md", "USER.md"]:
        (tmp_path / name).write_text(f"Client {name}", encoding="utf-8")

    owner_dir = tmp_path / "workspace_owner"
    owner_dir.mkdir(parents=True, exist_ok=True)
    for name in ["IDENTITY.md", "SOUL.md", "AGENTS.md", "USER.md"]:
        (owner_dir / name).write_text(f"Owner {name}", encoding="utf-8")

    return tmp_path


def test_owner_agent_creation(
    monkeypatch: pytest.MonkeyPatch, fake_workspace_setup: Path
) -> None:
    captured_kwargs: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return object()

    import deepagents._models

    monkeypatch.setattr(deepagents._models, "resolve_model", lambda m: m)
    monkeypatch.setattr(
        "apps.wu_tanchang_api.agent_factory.agent.create_deep_agent",
        fake_create_deep_agent,
    )
    monkeypatch.setattr(
        "apps.wu_tanchang_api.agent_factory.agent.create_model",
        lambda **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "apps.wu_tanchang_api.agent_factory.agent.default_runtime_dir",
        lambda: fake_workspace_setup / "runtime",
    )

    agent_config = WuAgentConfig(
        name="owner",
        provider="qwen",
        model="qwen-flash",
        max_tokens=800,
        workspace="workspace_owner",
    )

    agent, ckpt_path = create_agent(
        backend_root=fake_workspace_setup,
        agent_config=agent_config,
    )

    assert agent is not None
    system_prompt = captured_kwargs.get("system_prompt", "")
    assert "Owner IDENTITY.md" in system_prompt
    assert "Owner SOUL.md" in system_prompt
    assert "专属AI数据决策助理" in system_prompt

    tools = captured_kwargs.get("tools", [])
    tool_names = [t.name for t in tools]
    assert "get_consultation_stats" in tool_names
    assert "list_recent_clients" in tool_names
    assert "get_client_detail" in tool_names


@pytest.mark.anyio
async def test_get_consultation_stats() -> None:
    t1_msg = [
        HumanMessage(content="你好"),
        AIMessage(
            content="做个准备材料",
            tool_calls=[
                {
                    "name": "save_meeting_prep",
                    "args": {"body": "面馆准备材料"},
                    "id": "tc1",
                }
            ],
        ),
        ToolMessage(
            content="meeting_prep_saved", name="save_meeting_prep", tool_call_id="tc1"
        ),
        AIMessage(
            content="交付材料",
            tool_calls=[{"name": "mark_material_delivered", "args": {}, "id": "tc2"}],
        ),
        ToolMessage(
            content="material_delivered",
            name="mark_material_delivered",
            tool_call_id="tc2",
        ),
    ]
    t1 = _FakeCheckpointTuple(
        thread_id="wt::default::user_complete::conv1",
        ts="2026-06-14T10:00:00.000000+00:00",
        messages=t1_msg,
        metadata={"calendar_id": "o1"},
    )

    t2_msg = [
        HumanMessage(content="开奶茶店"),
        AIMessage(content="预算多少？"),
    ]
    t2 = _FakeCheckpointTuple(
        thread_id="wt::default::user_chatting::conv1",
        ts="2026-06-14T11:00:00.000000+00:00",
        messages=t2_msg,
        metadata={"calendar_id": "o1"},
    )

    checkpointer = _FakeCheckpointer([t1, t2])
    agent = _FakeAgent(checkpointer)

    config: RunnableConfig = {
        "configurable": {"thread_id": "wt::owner::o1::c1"},
        "metadata": {"agent_instance": agent},
    }

    stats_report = await get_consultation_stats.ainvoke(
        {"days": 7},
        config=config,
    )

    assert "**新增预聊客户数**: 2 人" in stats_report
    assert "**已生成准备资料数**: 1 份" in stats_report
    assert "**资料生成转化率**: 50.0%" in stats_report
    assert "2026-06-14 | 2 | 1 |" in stats_report


@pytest.mark.anyio
async def test_list_recent_clients() -> None:
    t1_msg = [HumanMessage(content="哈喽")]
    t1 = _FakeCheckpointTuple(
        thread_id="wt::default::u1::c1",
        ts="2026-06-14T10:00:00.000000+00:00",
        messages=t1_msg,
        metadata={"calendar_id": "o1"},
    )
    checkpointer = _FakeCheckpointer([t1])
    agent = _FakeAgent(checkpointer)

    config: RunnableConfig = {
        "configurable": {"thread_id": "wt::owner::o1::c1"},
        "metadata": {"agent_instance": agent},
    }

    clients_list = await list_recent_clients.ainvoke(
        {"days": 7, "status": "all"},
        config=config,
    )

    assert "u1" in clients_list
    assert "2026-06-14T10:00:00" in clients_list
    assert "💬 沟通中" in clients_list


@pytest.mark.anyio
async def test_get_client_detail() -> None:
    t1_msg = [
        HumanMessage(content="面馆"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_meeting_prep",
                    "args": {
                        "body": "### 上海面馆会议材料\n- 预算：50万\n- 主要挑战：写字楼客单低"
                    },
                    "id": "tc1",
                }
            ],
        ),
        ToolMessage(
            content="meeting_prep_saved", name="save_meeting_prep", tool_call_id="tc1"
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "mark_material_delivered", "args": {}, "id": "tc2"}],
        ),
        ToolMessage(
            content="material_delivered",
            name="mark_material_delivered",
            tool_call_id="tc2",
        ),
    ]
    t1 = _FakeCheckpointTuple(
        thread_id="wt::default::user_complete::conv1",
        ts="2026-06-14T10:00:00.000000+00:00",
        messages=t1_msg,
        metadata={"calendar_id": "o1"},
    )

    checkpointer = _FakeCheckpointer([t1])
    agent = _FakeAgent(checkpointer)

    config: RunnableConfig = {
        "configurable": {"thread_id": "wt::owner::o1::c1"},
        "metadata": {"agent_instance": agent},
    }

    detail = await get_client_detail.ainvoke(
        {"client_user_id": "user_complete"},
        config=config,
    )

    assert "user_complete" in detail
    assert "会议材料已交付" in detail
    assert "会议准备材料已生成" in detail
    assert "### 上海面馆会议材料" not in detail
    assert "expand_prep_details" in detail

    detail_expanded = await get_client_detail.ainvoke(
        {"client_user_id": "user_complete", "expand_prep_details": True},
        config=config,
    )
    assert "user_complete" in detail_expanded
    assert "### 上海面馆会议材料" in detail_expanded
    assert "主要挑战：写字楼客单低" in detail_expanded

    t2_msg = [
        HumanMessage(content="我想开奶茶店"),
        AIMessage(content="请问你的预算范围是多少？"),
        HumanMessage(content="20万"),
    ]
    t2 = _FakeCheckpointTuple(
        thread_id="wt::default::user_chatting::conv1",
        ts="2026-06-14T11:00:00.000000+00:00",
        messages=t2_msg,
        metadata={"calendar_id": "o1"},
    )

    checkpointer = _FakeCheckpointer([t2])
    agent = _FakeAgent(checkpointer)

    config = {
        "configurable": {"thread_id": "wt::owner::o1::c1"},
        "metadata": {"agent_instance": agent},
    }

    detail = await get_client_detail.ainvoke(
        {"client_user_id": "user_chatting"},
        config=config,
    )

    assert "user_chatting" in detail
    assert "仍在补充信息" in detail
    assert "该客户的会议准备材料尚未生成" in detail
    assert "我想开奶茶店" not in detail
    assert "expand_prep_details" in detail

    detail_expanded2 = await get_client_detail.ainvoke(
        {"client_user_id": "user_chatting", "expand_prep_details": True},
        config=config,
    )
    assert "user_chatting" in detail_expanded2
    assert "我想开奶茶店" in detail_expanded2
    assert "请问你的预算范围是多少？" in detail_expanded2
    assert "20万" in detail_expanded2


@pytest.mark.anyio
async def test_resolve_dynamic_agent(
    monkeypatch: pytest.MonkeyPatch, fake_workspace_setup: Path
) -> None:
    from apps.wu_tanchang_api.app.endpoints.chat import resolve_dynamic_agent
    from apps.wu_tanchang_api.app.state import AppState

    mock_agent_instance = MagicMock()
    mock_create_agent = MagicMock(return_value=(mock_agent_instance, Path("ckpt.pkl")))
    monkeypatch.setattr(
        "apps.wu_tanchang_api.agent_factory.agent.create_agent", mock_create_agent
    )

    state = AppState(
        agents={},
        agent_configs={},
        default_agent="default",
        checkpoints_path="ckpt.pkl",
        thread_locks={},
        backend_root=str(fake_workspace_setup),
    )

    # 1. Identical user_id and calendar_id -> owner mode, fallback to workspace_owner
    name, agent = await resolve_dynamic_agent(
        state,
        user_id="123",
        metadata={"calendar_id": "123"},
    )
    assert name == "owner"
    assert agent == mock_agent_instance
    mock_create_agent.assert_called_once()
    called_cfg = mock_create_agent.call_args[1]["agent_config"]
    assert called_cfg.workspace == "workspace_owner"

    mock_create_agent.reset_mock()

    # 2. If workspace_{calendar_id}_owner actually exists, route to it
    special_owner_dir = fake_workspace_setup / "workspace_456_owner"
    special_owner_dir.mkdir(parents=True, exist_ok=True)

    name, agent = await resolve_dynamic_agent(
        state,
        user_id="456",
        metadata={"calendar_id": "456"},
    )
    assert name == "owner"
    called_cfg = mock_create_agent.call_args[1]["agent_config"]
    assert called_cfg.workspace == "workspace_456_owner"

    mock_create_agent.reset_mock()

    # 3. Different user_id and calendar_id -> client mode, fallback to workspace
    name, agent = await resolve_dynamic_agent(
        state,
        user_id="client_user",
        metadata={"calendar_id": "789"},
    )
    assert name == "default"
    called_cfg = mock_create_agent.call_args[1]["agent_config"]
    assert called_cfg.workspace == "workspace"

    # 4. Invalid calendar_id (S3 path traversal prevention) -> raises HTTPException
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await resolve_dynamic_agent(
            state,
            user_id="123",
            metadata={"calendar_id": "../workspace_owner"},
        )
    assert exc_info.value.status_code == 400
    assert "Invalid calendar_id" in exc_info.value.detail


def test_mask_pii() -> None:
    from apps.wu_tanchang_api.agent_factory.owner_tools import mask_pii

    # Test emails
    assert mask_pii("my email is abc@example.com") == "my email is a***c@example.com"
    assert mask_pii("test.user+tag@domain.co.uk") == "t***g@domain.co.uk"

    # Test Chinese mobile phone numbers
    assert mask_pii("手机号是13812345678") == "手机号是138****5678"
    assert mask_pii("My phone is +86-18998765432") == "My phone is +86-189****5432"

    # Test landlines
    assert mask_pii("021-62345678") == "021****5678"
    assert mask_pii("010 8765 4321") == "010****4321"

    # Test 18-digit ID card
    assert mask_pii("身份证是110101199003072345") == "身份证是110101********2345"
    assert mask_pii("身份证X是11010119900307234X") == "身份证X是110101********234X"


@pytest.mark.anyio
async def test_get_client_detail_pii_masking() -> None:
    # Set up client messages and prep body containing PII
    t1_msg = [
        HumanMessage(content="我预留手机号是13911112222，邮箱是test@gmail.com"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_meeting_prep",
                    "args": {
                        "body": "### 客户预约材料\n- 电话：13911112222\n- 邮箱：test@gmail.com\n- 身份证号：110101199003072345"
                    },
                    "id": "tc1",
                }
            ],
        ),
    ]
    t1 = _FakeCheckpointTuple(
        thread_id="wt::default::user_pii::conv1",
        ts="2026-06-14T10:00:00.000000+00:00",
        messages=t1_msg,
        metadata={"calendar_id": "o1"},
    )

    checkpointer = _FakeCheckpointer([t1])
    agent = _FakeAgent(checkpointer)

    config: RunnableConfig = {
        "configurable": {"thread_id": "wt::owner::o1::c1"},
        "metadata": {"agent_instance": agent},
    }

    # Retrieve with expand_prep_details=True
    detail = await get_client_detail.ainvoke(
        {"client_user_id": "user_pii", "expand_prep_details": True},
        config=config,
    )

    assert "user_pii" in detail
    # Phone, email and ID should be masked
    assert "139****2222" in detail
    assert "13911112222" not in detail
    assert "t***t@gmail.com" in detail
    assert "test@gmail.com" not in detail
    assert "110101********2345" in detail
    assert "110101199003072345" not in detail


@pytest.mark.anyio
async def test_owner_tools_strict_access_denied() -> None:
    # Test that if owner's calendar_id is missing, the tools fail safely
    t1_msg = [HumanMessage(content="你好")]
    t1 = _FakeCheckpointTuple(
        thread_id="wt::default::u1::c1",
        ts="2026-06-14T10:00:00.000000+00:00",
        messages=t1_msg,
        metadata={"calendar_id": "o1"},
    )
    checkpointer = _FakeCheckpointer([t1])
    agent = _FakeAgent(checkpointer)

    # config has no calendar_id metadata and thread_id is not wt::owner::o1::c1
    config: RunnableConfig = {
        "configurable": {"thread_id": "some_other_thread"},
        "metadata": {"agent_instance": agent},
    }

    # stats should return empty/failure message
    stats = await get_consultation_stats.ainvoke({"days": 7}, config=config)
    assert "0" in stats

    # list recent clients should return empty list message
    recent = await list_recent_clients.ainvoke({"days": 7}, config=config)
    assert "未找到" in recent

    # get client detail should return not found message
    detail = await get_client_detail.ainvoke({"client_user_id": "u1"}, config=config)
    assert "未找到" in detail


def test_save_meeting_prep_tool_validation() -> None:
    from apps.wu_tanchang_api.agent_factory.agent import save_meeting_prep

    # 1. Test size limit (S4)
    huge_body = "x" * 50001
    config: RunnableConfig = {
        "configurable": {"thread_id": "wt::default::u1::c1"},
        "metadata": {
            "user_id": "1",
            "calendar_id": "2",
            "callback_url": "http://localhost:3001/wu_tanchang_callbacks/",
        },
    }
    res = save_meeting_prep.invoke({"body": huge_body}, config=config)
    assert "超长" in res

    # 2. Test unauthorized callback URL (S4)
    config_unauthorized: RunnableConfig = {
        "configurable": {"thread_id": "wt::default::u1::c1"},
        "metadata": {
            "user_id": "1",
            "calendar_id": "2",
            "callback_url": "http://malicious-attacker.com/wu_tanchang_callbacks/",
        },
    }
    res = save_meeting_prep.invoke({"body": "valid body"}, config=config_unauthorized)
    assert "未被授权" in res


def test_app_state_compilation_locks() -> None:
    from apps.wu_tanchang_api.app.state import AppState

    state = AppState(
        agents={},
        agent_configs={},
        default_agent="default",
        checkpoints_path="",
        thread_locks={},
        backend_root="",
    )

    lock1 = state.get_compilation_lock("workspace_1")
    lock2 = state.get_compilation_lock("workspace_1")
    lock3 = state.get_compilation_lock("workspace_2")

    assert lock1 is lock2
    assert lock1 is not lock3

    # Test backward compatibility compilation_lock property
    dep_lock = state.compilation_lock
    assert dep_lock is state.get_compilation_lock("default")


def test_extract_name_and_title() -> None:
    from apps.wu_tanchang_api.agent_factory.owner_tools import _extract_name_and_title
    from langchain_core.messages import HumanMessage, AIMessage

    # 1. Direct name + title in text
    msgs1 = [HumanMessage(content="张总好，我准备开个面馆")]
    assert _extract_name_and_title(msgs1) == "张总"

    # 2. Introduction with "我是...的老板"
    msgs2 = [HumanMessage(content="我是面馆的李老板，想咨询一下连锁加盟")]
    assert _extract_name_and_title(msgs2) == "李老板"

    # 3. "我叫..."
    msgs3 = [HumanMessage(content="我叫王二，打算在上海开店")]
    assert _extract_name_and_title(msgs3) == "王二"

    # 4. "我姓..."
    msgs4 = [HumanMessage(content="你好，我姓陈")]
    assert _extract_name_and_title(msgs4) == "陈先生/女士"

    # 5. "我姓..." with title in the same content
    msgs5 = [HumanMessage(content="我姓王，是这边的经理")]
    assert _extract_name_and_title(msgs5) == "王经理"

    # 6. Fallback to "未知客户" if no patterns matched
    msgs6 = [HumanMessage(content="哈喽"), AIMessage(content="你好")]
    assert _extract_name_and_title(msgs6) == "未知客户"

    # 7. No messages
    assert _extract_name_and_title([]) == "未知客户"
