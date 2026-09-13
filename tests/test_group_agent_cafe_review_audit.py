"""Private original-text review evidence without provider calls."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from apps.group_agent_api.agent_factory.cafe_output import afinalize_host_output, finalize_host_output
from tests.test_group_agent_host_da001 import _host


def _review(verdict: str, text: str, reason: str = "通过") -> AIMessage:
    return AIMessage(content=json.dumps({"verdict": verdict, "reply": text, "option_indices": [], "reason": reason}, ensure_ascii=False))


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_original_text_repair_reasons_and_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    monkeypatch.setenv("GROUP_AGENT_CAFE_REVIEW_AUDIT", "1")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    host = _host(tmp_path)
    model = MagicMock()
    responses = [_review("fail", "聊音乐就好。", "删除虚构点单状态"), _review("fail", "慢慢聊。", "删除未获准客人故事"), _review("fail", "坐会儿就好。", "仍需修正")]
    model.invoke.side_effect = responses
    model.ainvoke = AsyncMock(side_effect=responses)
    kwargs = {"host": host, "messages": [HumanMessage(content="我只是  想聊聊。\n不想做计划。")], "model": model}
    output = await afinalize_host_output("模型原始正文。", **kwargs) if asynchronous else finalize_host_output("模型原始正文。", **kwargs)
    files = list((tmp_path / "cafe_review_audits").glob("review-*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    events = record["events"]
    assert events[0]["draft"] == "模型原始正文。"
    assert events[0]["conversation"][0]["text"] == "我只是  想聊聊。\n不想做计划。"
    reviews = [event for event in events if event["stage"] == "review"]
    assert "删除虚构点单状态" in reviews[0]["response"]
    assert reviews[1]["input_reply"] == "聊音乐就好。"
    fallback = next(event for event in events if event["stage"] == "fallback")
    assert fallback["reason"] == "review_budget_exhausted"
    assert fallback["discarded_reply"] == "坐会儿就好。"
    assert events[-1]["reply"] == output.reply
    assert events[-1]["violations"] == ["host_semantic_repair"] * 3 + ["host_review_unavailable"]
    assert files[0].stat().st_mode & 0o777 == 0o600
    assert files[0].parent.stat().st_mode & 0o777 == 0o700


def test_disabled_does_not_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROUP_AGENT_CAFE_REVIEW_AUDIT", raising=False)
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    finalize_host_output("安静坐坐。", host=_host(tmp_path), messages=[])
    assert not (tmp_path / "cafe_review_audits").exists()


def test_credentials_redacted_but_contact_and_format_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_CAFE_REVIEW_AUDIT", "1")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    content = '我  的邮箱 a@example.com\n{"api_key":"abcdefgh12345678"} bearer abcdefgh87654321 password=notpublic sk-123456789abc'
    finalize_host_output("安全正文。", host=_host(tmp_path), messages=[HumanMessage(content=content)])
    record = next((tmp_path / "cafe_review_audits").glob("review-*.json")).read_text()
    for secret in ["abcdefgh12345678", "abcdefgh87654321", "notpublic", "sk-123456789abc"]:
        assert secret not in record
    assert "a@example.com" in record
    assert "我  的邮箱" in record


def test_write_failure_does_not_change_reply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("GROUP_AGENT_CAFE_REVIEW_AUDIT", "1")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "cafe_review_audits").write_text("not a directory")
    output = finalize_host_output("安静坐坐。", host=_host(tmp_path), messages=[])
    assert output.reply == "安静坐坐。"
    assert "cafe_review_audit_write_failed" in caplog.text
    assert "安静坐坐" not in caplog.text


def test_old_records_pruned_not_other_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_CAFE_REVIEW_AUDIT", "1")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    directory = tmp_path / "cafe_review_audits"
    directory.mkdir()
    old = directory / "review-old.json"
    old.write_text("{}")
    unrelated = directory / "notes.json"
    unrelated.write_text("keep")
    os.utime(old, (1, 1))
    finalize_host_output("坐会儿。", host=_host(tmp_path), messages=[])
    assert not old.exists() and unrelated.exists()


async def test_concurrent_turns_do_not_mix_original_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_CAFE_REVIEW_AUDIT", "1")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    host = _host(tmp_path)
    async def run(text: str) -> None:
        model = MagicMock()
        async def invoke(*args: object, **kwargs: object) -> AIMessage:
            await asyncio.sleep(0)
            return _review("pass", text)
        model.ainvoke = invoke
        await afinalize_host_output(text, host=host, messages=[HumanMessage(content=text)], model=model)
    await asyncio.gather(run("第一位的消息"), run("第二位的消息"))
    records = [json.loads(path.read_text()) for path in (tmp_path / "cafe_review_audits").glob("review-*.json")]
    assert len(records) == 2
    for record in records:
        assert record["events"][0]["draft"] == record["events"][-1]["reply"]
        assert len(record["events"]) == 6


@pytest.mark.parametrize("failure", ["exception", "invalid"])
def test_failure_cause_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.setenv("GROUP_AGENT_CAFE_REVIEW_AUDIT", "1")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    model = MagicMock()
    if failure == "exception":
        model.invoke.side_effect = RuntimeError("private provider diagnostic")
    else:
        model.invoke.return_value = AIMessage(content="not JSON")
    finalize_host_output("安全正文。", host=_host(tmp_path), messages=[], model=model)
    record = json.loads(next((tmp_path / "cafe_review_audits").glob("review-*.json")).read_text())
    fallback = next(event for event in record["events"] if event["stage"] == "fallback")
    assert fallback["reason"] == ("review_exception" if failure == "exception" else "invalid_review")
    assert "private provider diagnostic" not in json.dumps(record)


def test_symlink_directory_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUP_AGENT_CAFE_REVIEW_AUDIT", "1")
    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "cafe_review_audits").symlink_to(other, target_is_directory=True)
    output = finalize_host_output("正常回复。", host=_host(tmp_path), messages=[])
    assert output.reply == "正常回复。" and not list(other.iterdir())
