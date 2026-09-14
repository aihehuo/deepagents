"""Offline behavioral tests for independently installed AIHEHUO skills."""

import http.client
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[4]


def load_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "skillpacks" / "aihehuo" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = load_module("client")
manager = load_module("manage")


def test_single_package_runs_without_repository(tmp_path: Path) -> None:
    target = tmp_path / "standalone"
    installed = manager.install(["get-ai-cofounders"], target=target)
    assert [path.name for path in target.iterdir()] == ["get-ai-cofounders"]
    result = subprocess.run(  # noqa: S603  # executes only the generated trusted package entrypoint
        [sys.executable, str(installed[0] / "run.py"), "--help"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "{users}" in result.stdout
    forbidden = subprocess.run(  # noqa: S603  # executes only the generated trusted package entrypoint
        [sys.executable, str(installed[0] / "run.py"), "--resource", "projects"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert forbidden.returncode == 2  # noqa: PLR2004  # argparse invalid argument exit status


def test_combinations_are_isolated_and_repeatable(tmp_path: Path) -> None:
    names = ["get-ai-cofounders", "get-ai-communities"]
    first = manager.install(names, target=tmp_path / "a")
    assert manager.install(names + names, target=tmp_path / "a") == first
    second = manager.install(["get-ai-platform-help"], target=tmp_path / "b")
    assert {path.name for path in first} == set(names)
    assert [path.name for path in second] == ["get-ai-platform-help"]


def test_all_skills_are_discoverable_by_sdk(tmp_path: Path) -> None:
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.middleware.skills import _list_skills

    names = list(manager._catalog())
    manager.install(names, target=tmp_path / "skills")
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    discovered = _list_skills(backend, "/skills/")
    assert {item["name"] for item in discovered} == set(names)


def test_preflight_preserves_existing_files(tmp_path: Path) -> None:
    conflict = tmp_path / "get-ai-communities"
    conflict.mkdir()
    (conflict / "SKILL.md").write_text("custom", encoding="utf-8")
    with pytest.raises(FileExistsError):
        manager.install(["get-ai-cofounders", "get-ai-communities"], target=tmp_path)
    assert not (tmp_path / "get-ai-cofounders").exists()
    assert (conflict / "SKILL.md").read_text() == "custom"


def test_selection_dry_run_and_symlink_conflict(tmp_path: Path) -> None:
    target = tmp_path / "absent"
    manager.install(["get-ai-cofounders"], target=target, dry_run=True)
    assert not target.exists()
    for names in ([], ["../../bad"]):
        with pytest.raises(ValueError, match="known catalog skills"):
            manager.install(names, target=target)
    target.mkdir()
    (target / "get-ai-cofounders").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        manager.install(["get-ai-cofounders"], target=target)


def test_paths_encode_search_and_member_pagination() -> None:
    path = client.build_path("users", params={"keyword": "杭州 AI,运营", "q": "技术合伙人", "page": 2, "per": 30})
    parsed = urlsplit(path)
    assert parsed.path == "/ai/users.json"
    assert parse_qs(parsed.query) == {"keyword": ["杭州 AI,运营"], "q": ["技术合伙人"], "page": ["2"], "per": ["30"]}
    assert client.build_path("group-members", params={"page": 2}, event_id="763", output_format="md") == "/ai/wechat_groups/763.md?page=2"
    assert client.build_path("weekly", params={}, output_format="md") == "/ai/blog/weekly_markdown"


@pytest.mark.parametrize(
    ("resource", "params", "event_id"),
    [
        ("users", {"page": 0}, None),
        ("users", {"per": 501}, None),
        ("users", {"per": True}, None),
        ("projects", {"q": "AI"}, None),
        ("blog", {"keyword": "AI"}, None),
        ("weekly", {"page": 1}, None),
        ("group-members", {}, "../1"),
        ("group-members", {}, None),
        ("users", {}, "123"),
        ("https://evil.test", {}, None),
    ],
)
def test_invalid_queries_fail_before_network(resource: str, params: dict[str, str | int], event_id: str | None) -> None:
    with pytest.raises(ValueError, match=r"Invalid|Unsupported|require|supported"):
        client.build_path(resource, params=params, event_id=event_id)


class Connection:
    """Small in-memory HTTP transport; no sockets or server required."""

    def __init__(self, *, status: int = 200, body: bytes = b'{"data": [], "meta": {"has_more": false}}') -> None:
        self.status = status
        self.body = body
        self.closed = False
        self.headers: dict[str, str] = {}

    def request(self, method: str, path: str, headers: dict[str, str]) -> None:
        assert method == "GET"
        assert path.startswith("/ai/")
        self.headers = headers

    def getresponse(self) -> "Connection":
        return self

    def getheader(self, name: str) -> str | None:
        return "60" if name == "Retry-After" else None

    def read(self, size: int) -> bytes:
        return self.body[:size]

    def close(self) -> None:
        self.closed = True


def test_fetch_preserves_empty_results_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = Connection()
    monkeypatch.setattr(http.client, "HTTPSConnection", lambda _host, **_kwargs: connection)
    assert json.loads(client.fetch("/ai/users.json"))["data"] == []
    assert connection.closed


def test_weekly_cli_reads_markdown_and_reports_rate_limits(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    connection = Connection(body=b"# Recent articles")
    monkeypatch.setattr(http.client, "HTTPSConnection", lambda _host, **_kwargs: connection)
    assert client.main(["weekly", "blog"], []) == 0
    assert capsys.readouterr().out == "# Recent articles\n"
    assert connection.headers["Accept"] == "text/markdown"
    limited = Connection(status=429)
    monkeypatch.setattr(http.client, "HTTPSConnection", lambda _host, **_kwargs: limited)
    assert client.main(["users"], ["--per", "1"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Retry-After=60" in captured.err
    assert limited.closed


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (429, b"", RuntimeError),
        (302, b"", RuntimeError),
        (500, b"", RuntimeError),
        (200, b"not json", ValueError),
        (200, b"x" * (client.MAX_BYTES + 1), RuntimeError),
    ],
)
def test_fetch_errors_close_connection(monkeypatch: pytest.MonkeyPatch, status: int, body: bytes, error: type[Exception]) -> None:
    connection = Connection(status=status, body=body)
    monkeypatch.setattr(http.client, "HTTPSConnection", lambda _host, **_kwargs: connection)
    with pytest.raises(error):
        client.fetch("/ai/users.json")
    assert connection.closed


def test_adapter_restricts_resources_to_installed_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Root namespace packages are available when pytest is launched with python -m.
    from skillpacks.aihehuo import integration

    manager.install(["get-ai-platform-help"], target=tmp_path)
    tool = integration.create_aihehuo_tool(skills_dir=tmp_path)
    assert "not installed" in tool.invoke({"resource": "users"})
    monkeypatch.setattr(integration, "fetch", lambda _path, **_kwargs: json.dumps({"data": [], "meta": {}}))
    result = json.loads(tool.invoke({"resource": "faqs", "keyword": "社群"}))
    assert result["_source_url"].startswith("https://www.aihehuo.com/ai/faqs.json?keyword=")
    assert "failed" in tool.invoke({"resource": "faqs", "q": "不支持"})


def test_adapter_attaches_source_without_replacing_null_item_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillpacks.aihehuo import integration

    manager.install(["get-ai-platform-help"], target=tmp_path)
    document = {"data": [{"question": "问题", "answer": "说明", "source": "static", "url": None}], "meta": {"has_more": False}}
    monkeypatch.setattr(integration, "fetch", lambda _path, **_kwargs: json.dumps(document))
    tool = integration.create_aihehuo_tool(skills_dir=tmp_path)
    result = json.loads(tool.invoke({"resource": "faqs", "page": 1, "per": 2}))
    assert result["data"] == document["data"]
    assert result["meta"] == document["meta"]
    assert result["_source_url"] == "https://www.aihehuo.com/ai/faqs.json?page=1&per=2"


def test_adapter_preserves_markdown_body_and_attaches_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from skillpacks.aihehuo import integration

    manager.install(["get-ai-startup-digest"], target=tmp_path)
    body = "# 公开周报\n\n实际文章内容。\n"
    monkeypatch.setattr(integration, "fetch", lambda _path, **_kwargs: body)
    result = integration.create_aihehuo_tool(skills_dir=tmp_path).invoke({"resource": "weekly"})
    assert "https://www.aihehuo.com/ai/blog/weekly_markdown" in result
    assert result.endswith(body)
