"""Public query receipts survive history loss without crossing requests."""

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from apps.group_agent_api.agent_factory.checks.reply_grounding import apply_reply_grounding_gate
from apps.group_agent_api.agent_factory.public_evidence import extract_public_reads
from apps.group_agent_api.agent_factory.write_skills import profile_write_allowed_for_turn
from apps.group_agent_api.app.endpoints.chat import _should_force_profile_save
from skillpacks.aihehuo import integration
from skillpacks.aihehuo.evidence import collect_public_evidence, current_results
from skillpacks.aihehuo.manage import install


async def test_real_tool_receipts_survive_replaced_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install(["get-ai-cofounders"], target=tmp_path)
    document = {"data": [{"name": f"人才{index}", "bio": "实际简介"} for index in range(25)]}
    monkeypatch.setattr(integration, "fetch", lambda *_args, **_kwargs: json.dumps(document))
    tool = integration.create_aihehuo_tool(skills_dir=tmp_path)

    @collect_public_evidence
    async def request() -> None:
        content = await tool.ainvoke({"resource": "users"})
        reads = extract_public_reads([AIMessage(content="保存失败")], 0)
        assert reads[0].document == json.loads(content)
        assert reads[0].records[20]["name"] == "人才20"
        messages = [
            AIMessage(content="", tool_calls=[{"id": "q", "name": tool.name, "args": {"resource": "users"}}]),
            ToolMessage(content=content, name=tool.name, tool_call_id="q"),
        ]
        assert extract_public_reads(messages, 0) == reads

    await request()
    assert current_results() == []
    assert extract_public_reads([], 0) == []


async def test_concurrent_tool_threads_do_not_share_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install(["get-ai-cofounders"], target=tmp_path)

    def fetch(path: str, *, output_format: str) -> str:
        del output_format
        return json.dumps({"data": [{"name": path}]})

    monkeypatch.setattr(integration, "fetch", fetch)
    tool = integration.create_aihehuo_tool(skills_dir=tmp_path)

    @collect_public_evidence
    async def request(keyword: str) -> None:
        await asyncio.to_thread(tool.invoke, {"resource": "users", "keyword": keyword})
        await asyncio.sleep(0)
        reads = extract_public_reads([], 0)
        assert len(reads) == 1
        assert keyword in reads[0].records[0]["name"]

    await asyncio.gather(request("alpha"), request("beta"))
    assert current_results() == []


async def test_query_error_and_request_exception_leave_no_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install(["get-ai-cofounders"], target=tmp_path)

    def fail(path: str, *, output_format: str) -> str:
        del path, output_format
        msg = "HTTP 429"
        raise RuntimeError(msg)

    monkeypatch.setattr(integration, "fetch", fail)
    tool = integration.create_aihehuo_tool(skills_dir=tmp_path)

    @collect_public_evidence
    async def request() -> None:
        assert "query failed" in await tool.ainvoke({"resource": "users"})
        assert extract_public_reads([], 0) == []
        msg = "request failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="request failed"):
        await request()
    assert current_results() == []


@pytest.mark.parametrize("message", ["仅查询公开资料,不保存我的画像", "不要更新画像", "don't save my profile"])
def test_explicit_write_refusal_disables_force_save(message: str) -> None:
    assert not profile_write_allowed_for_turn(message)
    assert not _should_force_profile_save(
        profile_ok=False, profile_status="failed", persist_alert="missing_file", messages=[], msg_count_before=0, user_message=message
    )


def test_profile_update_remains_allowed() -> None:
    assert profile_write_allowed_for_turn("请保存我的画像,我在做人工智能产品")


async def test_unreturned_person_is_still_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    install(["get-ai-cofounders"], target=tmp_path)
    monkeypatch.setattr(integration, "fetch", lambda *_args, **_kwargs: json.dumps({"data": [{"name": "真实人才"}]}))
    tool = integration.create_aihehuo_tool(skills_dir=tmp_path)

    @collect_public_evidence
    async def request() -> None:
        await tool.ainvoke({"resource": "users"})
        result = apply_reply_grounding_gate(
            reply="找到一位合伙人【虚构人才】，他曾主导全国项目。",  # noqa: RUF001  # Chinese user-facing reply.
            reply_mode="dialogue",
            candidates=[],
            public_reads=extract_public_reads([], 0),
            enabled=True,
        )
        assert result.abandoned
        assert result.results[0].codes == ["invented_candidate"]
        assert "虚构人才" not in result.reply
        assert "先匹配" not in result.reply
        assert "公开资料核验" in result.reply

    await request()
