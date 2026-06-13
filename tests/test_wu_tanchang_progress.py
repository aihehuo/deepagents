from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from apps.wu_tanchang_api.app.utils import get_progress_status


def test_get_progress_status_main_agent() -> None:
    # 1. Main agent start node
    status = get_progress_status(
        subgraph_path=None, stream_mode="updates", data={"__start__": {}}
    )
    assert status == "正在规划回复内容..."

    # 2. Main agent calling subagent task tool
    # Check both model tool_calls representation and msg tool_calls representation
    status = get_progress_status(
        subgraph_path=(),
        stream_mode="updates",
        data={
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "task", "args": {}, "id": "call_1"}],
                    )
                ]
            }
        },
    )
    assert status == "正在为您调用知识库分析师进行检索..."

    # 3. Main agent calling mark_material_delivered
    status = get_progress_status(
        subgraph_path=(),
        stream_mode="updates",
        data={
            "agent": {
                "tool_calls": [
                    {"name": "mark_material_delivered", "args": {}, "id": "call_2"}
                ]
            }
        },
    )
    assert status == "正在为您生成会议准备材料..."


def test_get_progress_status_subagent() -> None:
    subpath = ("task:123",)

    # 1. Subagent start node
    status = get_progress_status(
        subgraph_path=subpath, stream_mode="updates", data={"__start__": {}}
    )
    assert status == "正在启动知识库检索与分析..."

    # 2. Subagent calling kb_semantic_search
    status = get_progress_status(
        subgraph_path=subpath,
        stream_mode="updates",
        data={
            "agent": {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "kb_semantic_search", "args": {}, "id": "call_3"}
                        ],
                    )
                ]
            }
        },
    )
    assert status == "正在对餐饮案例库进行语义检索..."

    # 3. Subagent calling filesystem / skill tool
    status = get_progress_status(
        subgraph_path=subpath,
        stream_mode="updates",
        data={
            "agent": {"tool_calls": [{"name": "read_file", "args": {}, "id": "call_4"}]}
        },
    )
    assert status == "正在调阅餐饮案例详情并分析商业动作..."

    # 4. Subagent tool message response
    status = get_progress_status(
        subgraph_path=subpath,
        stream_mode="updates",
        data={
            "tools": {
                "messages": [
                    ToolMessage(
                        content="content", name="read_file", tool_call_id="call_4"
                    )
                ]
            }
        },
    )
    assert status == "正在调阅餐饮案例详情并分析商业动作..."

    # 5. Subagent agent node without tool calls (synthesis phase)
    status = get_progress_status(
        subgraph_path=subpath,
        stream_mode="updates",
        data={"agent": {"messages": [AIMessage(content="Here is the analysis.")]}},
    )
    assert status == "正在整理匹配的案例与商业建议..."


def test_get_progress_status_ignored_cases() -> None:
    # Ignored stream mode
    status = get_progress_status(
        subgraph_path=None, stream_mode="messages", data=("hello", {})
    )
    assert status is None

    # Invalid data format
    status = get_progress_status(
        subgraph_path=None, stream_mode="updates", data="invalid"
    )
    assert status is None
