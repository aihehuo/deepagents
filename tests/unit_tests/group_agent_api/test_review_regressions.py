"""Regression coverage for permission isolation and public reply evidence."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from apps.group_agent_api.agent_factory import agent as factory, member_graphs
from apps.group_agent_api.agent_factory.capability import CapabilityTier, SessionCapabilities
from apps.group_agent_api.agent_factory.checks.reply_grounding.gate import ReplyGroundingGateResult, apply_reply_grounding_gate
from apps.group_agent_api.agent_factory.checks.reply_grounding.ground import build_ground_from_turn
from apps.group_agent_api.agent_factory.checks.reply_grounding.l0 import run_l0
from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import ReplyGroundingInput
from apps.group_agent_api.agent_factory.integrations.membership_client import MembershipResult
from apps.group_agent_api.agent_factory.integrations.principal import SessionPrincipal
from apps.group_agent_api.agent_factory.public_evidence import extract_public_reads
from apps.group_agent_api.agent_factory.skill_backend import SkillReadOnlyBackend
from apps.group_agent_api.app import async_manager
from apps.group_agent_api.app.endpoints import chat as chat_ep
from apps.group_agent_api.app.models import AsyncCallRequest, ChatRequest
from apps.group_agent_api.app.session import TrustedSession
from apps.group_agent_api.app.state import AppState
from skillpacks.aihehuo import integration
from skillpacks.aihehuo.manage import install

SOURCE = "https://www.aihehuo.com/ai/users.json?page=1&per=2"
REPLY = f"公开人才资料：伙伴【张测试】有 Python 项目经验。来源：{SOURCE}"  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.


class ToolModel(BaseChatModel):
    """Issue a controlled tool call through the actual member graph."""

    calls: list[dict[str, Any]]
    reply: str = "调用结束。"

    @property
    def _llm_type(self) -> str:
        return "review-regression"

    def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> "ToolModel":
        del kwargs, tools
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del kwargs, run_manager, stop
        message = AIMessage(content=self.reply) if isinstance(messages[-1], ToolMessage) else AIMessage(content="", tool_calls=self.calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_public_helper_cannot_hide_queries_from_main_turn_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class DelegatingModel(ToolModel):
        bound_tools: list[list[str]] = Field(default_factory=list)

        def bind_tools(self, tools: list[BaseTool], **kwargs: Any) -> "DelegatingModel":
            del kwargs
            self.bound_tools.append([tool.name for tool in tools])
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            del kwargs, run_manager, stop
            last = messages[-1]
            if last.content == "只给查询思路":
                message = AIMessage(content="请主图查询公开人才。")
            elif isinstance(last, ToolMessage) and last.name == "task":
                message = AIMessage(content="", tool_calls=self.calls)
            elif isinstance(last, ToolMessage):
                message = AIMessage(content=REPLY)
            else:
                message = AIMessage(
                    content="",
                    tool_calls=[{"name": "task", "id": "helper", "args": {"description": "只给查询思路", "subagent_type": "general-purpose"}}],
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    install(["get-ai-cofounders"], target=tmp_path / "skills/aihehuo")
    monkeypatch.setattr(integration, "fetch", lambda *_args, **_kwargs: json.dumps({"data": [{"name": "张测试", "bio": "Python 项目经验"}]}))
    model = DelegatingModel(calls=[{"name": "query_aihehuo_public_data", "id": "public", "args": {"resource": "users", "page": 1, "per": 2}}])
    graph = factory.create_member_agent(runtime=tmp_path, model=model, checkpointer=InMemorySaver(), include_save=False)
    result = await graph.ainvoke({"messages": [HumanMessage(content="找人才")]}, {"configurable": {"thread_id": "delegate"}, "recursion_limit": 16})
    assert extract_public_reads(result["messages"], 0)[0].records[0]["name"] == "张测试"
    assert any("query_aihehuo_public_data" in names for names in model.bound_tools)
    assert any("query_aihehuo_public_data" not in names and "save_group_profile" not in names for names in model.bound_tools)


@pytest.mark.parametrize("save", [False, True])
async def test_real_graph_generic_file_writes_cannot_bypass_profile_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    save: bool,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "legacy" if save else "disabled")
    profile = tmp_path / "users/other/groups/g1/profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text('{"original":true}')
    skill = tmp_path / "skills/example/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("original skill")
    calls = [
        {"name": "write_file", "id": "new", "args": {"file_path": "/users/victim/groups/g1/profile.json", "content": "stolen"}},
        {
            "name": "edit_file",
            "id": "edit",
            "args": {"file_path": "/users/other/groups/g1/profile.json", "old_string": "true", "new_string": "false"},
        },
        {"name": "write_file", "id": "skill", "args": {"file_path": "/skills/evil/SKILL.md", "content": "evil"}},
        {"name": "edit_file", "id": "skill-edit", "args": {"file_path": "/skills/example/SKILL.md", "old_string": "original", "new_string": "evil"}},
    ]
    graph = factory.create_member_agent(
        runtime=tmp_path,
        model=ToolModel(calls=calls),
        checkpointer=InMemorySaver(),
        include_search=False,
        include_save=save,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="write files")]},
        {
            "configurable": {"thread_id": "attack"},
            "recursion_limit": 12,
            "metadata": {"user_id": "owner", "group_id": "g1", "base_dir": str(tmp_path), "can_write_profile": "false"},
        },
    )
    results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(results) == 4  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
    assert all("permission_denied" in str(m.content) for m in results)  # Explicit expected contract value in this regression assertion.
    assert profile.read_text() == '{"original":true}'
    assert skill.read_text() == "original skill"
    assert not (tmp_path / "users/victim").exists()
    assert not (tmp_path / "skills/evil").exists()


async def test_backend_blocks_uploads_private_reads_and_symlink_escape(tmp_path: Path) -> None:
    backend = SkillReadOnlyBackend(root_dir=str(tmp_path), virtual_mode=True)
    private = tmp_path / "users/owner/profile.json"
    private.parent.mkdir(parents=True)
    private.write_text("private")
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "escape").symlink_to(private.parent, target_is_directory=True)
    assert backend.read("/users/owner/profile.json").error
    assert backend.read("/skills/escape/profile.json").error
    assert backend.glob("**/*.json").matches == []
    assert backend.glob("**/*.json", "/").matches == []
    assert not backend.grep("private").matches
    assert not backend.grep("private", "/skills").matches
    assert all(r.error == "permission_denied" for r in await backend.aupload_files([(str(private), b"evil")]))
    assert (await backend.awrite("/skills/new.md", "evil")).error
    assert (await backend.aedit("/users/owner/profile.json", "private", "evil")).error
    assert private.read_text() == "private"


@pytest.mark.parametrize(("search", "save"), [(True, False), (False, True), (False, False)])
def test_missing_permission_graph_fails_closed(tmp_path: Path, *, search: bool, save: bool) -> None:
    state = AppState(agent=object(), base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="permission graph"):
        state.member_agent_for_capability(can_search=search, can_write_profile=save)


def test_shared_bootstrap_builds_every_variant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def member(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    def full(**kwargs: Any) -> tuple[object, object, Path]:
        calls.append(kwargs)
        return object(), object(), tmp_path / "checkpoints.pkl"

    monkeypatch.setattr(member_graphs, "create_member_agent", member)
    monkeypatch.setattr(member_graphs, "create_agent", full)
    graphs = member_graphs.build_member_graphs(runtime=tmp_path, model=object())
    assert set(graphs) == {"agent", "admin_agent", "checkpoints_path", "search_only_agent", "write_only_agent", "no_tool_agent"}
    assert len({id(c["model"]) for c in calls}) == len({id(c["checkpointer"]) for c in calls}) == 1
    assert [(c["include_search"], c["include_save"]) for c in calls[1:]] == [(True, False), (False, True), (False, False)]


@pytest.mark.parametrize(
    ("search", "save", "mode"),
    [
        (True, False, "legacy"),
        (False, True, "legacy"),
        (False, False, "legacy"),
        (True, False, "public_skills"),
        (True, True, "public_skills"),
    ],
)
def test_prompt_only_instructs_available_business_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    search: bool,
    save: bool,
    mode: str,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", mode)
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "legacy")
    install(["get-ai-cofounders"], target=tmp_path / "skills/aihehuo")
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(factory, "create_deep_agent", capture)
    factory.create_member_agent(runtime=tmp_path, model=object(), checkpointer=object(), include_search=search, include_save=save)
    prompt = captured["system_prompt"]
    if not save:
        assert "save_group_profile" not in prompt
    if not search:
        assert "search_candidates" not in prompt
        assert "query_aihehuo_public_data" not in prompt
    if mode == "public_skills":
        assert "调用 `search_candidates`" not in prompt
    if search and mode == "legacy":
        assert "调用 search_candidates" in prompt


def _public_messages(*, resource: str = "users", source: str = SOURCE) -> list[BaseMessage]:
    return [
        AIMessage(content="", tool_calls=[{"name": "query_aihehuo_public_data", "id": "read", "args": {"resource": resource}}]),
        ToolMessage(
            name="query_aihehuo_public_data",
            tool_call_id="read",
            content=json.dumps({"_source_url": source, "data": [{"id": "p1", "name": "张测试", "bio": "Python 项目经验", "phone": "private"}]}),
        ),
    ]


def test_public_evidence_is_independent_and_requires_current_correlated_source() -> None:
    messages = _public_messages()
    reads = extract_public_reads(messages, 0)
    ground = build_ground_from_turn(candidates=[], public_reads=reads)
    assert ground.candidate_count == 0
    assert ground.candidates == []
    assert ground.receipts == []
    assert reads[0].source_url == SOURCE
    assert reads[0].document == json.loads(str(messages[1].content))
    assert extract_public_reads(messages, 1) == []
    assert extract_public_reads(_public_messages(source="https://evil.invalid/ai/users.json"), 0) == []
    payload = ReplyGroundingInput(reply=REPLY, ground=ground)
    assert run_l0(payload) is None
    payload.ground.public_reads = extract_public_reads(_public_messages(resource="faqs"), 0)
    assert run_l0(payload).codes == ["invented_candidate"]


@pytest.mark.parametrize("claim", ["本群成员张测试", "匹配到伙伴【张测试】", "@张测试"])
def test_public_people_do_not_grant_group_match_or_mentions(claim: str) -> None:
    ground = build_ground_from_turn(candidates=[], public_reads=extract_public_reads(_public_messages(), 0))
    denial = run_l0(ReplyGroundingInput(reply=f"{claim}，来源：{SOURCE}", ground=ground))  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
    assert denial.codes == ["unsupported_claim"]


def test_public_person_reply_can_disclaim_group_membership() -> None:
    ground = build_ground_from_turn(candidates=[], public_reads=extract_public_reads(_public_messages(), 0))
    reply = f"{REPLY}。公开资料不代表是本群成员。"
    assert run_l0(ReplyGroundingInput(reply=reply, ground=ground)) is None


@pytest.mark.parametrize(("read", "write"), [(True, True), (True, False), (False, True), (False, False)])
def test_worker_rebuild_preserves_trusted_admission_permissions(
    monkeypatch: pytest.MonkeyPatch,
    *,
    read: bool,
    write: bool,
) -> None:
    from apps.group_agent_api.execution.admission import build_trusted_envelope
    from apps.group_agent_worker.tasks import _rebuild_session

    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "legacy")
    req = AsyncCallRequest(
        run_id="permissions",
        idempotency_key="permissions",
        user_id="owner",
        unionid="union-owner",
        group_id="g1",
        message="hello",
        callback_url="http://example.invalid/group_agent_callbacks",
        metadata={"can_write_profile": "true"},
    )
    session = TrustedSession(
        principal=SessionPrincipal("owner", "union-owner", None, "header"),
        membership=MembershipResult(tier=CapabilityTier.in_group, source="token"),
        group_id="g1",
        group_token=None,
        capabilities=SessionCapabilities(CapabilityTier.in_group, read, write),
    )
    envelope = build_trusted_envelope(req=req, session=session, thread_id="trusted-thread")
    rebuilt = _rebuild_session(envelope)[1]
    assert rebuilt.capabilities == session.capabilities
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "disabled")
    assert _rebuild_session(envelope)[1].capabilities.can_write_profile is False
    del envelope["session"]["capabilities"]
    rebuilt = _rebuild_session(envelope)[1]
    assert rebuilt.capabilities.can_search is False
    assert rebuilt.capabilities.can_write_profile is False


def test_semantic_judge_receives_public_facts_and_catches_fabricated_resume() -> None:
    class Judge:
        def invoke(self, messages: list[BaseMessage]) -> AIMessage:
            text = str(messages[-1].content)
            assert '"public_reads"' in text
            assert SOURCE in text
            assert "Python 项目经验" in text
            return AIMessage(
                content='{"verdict":"fail","codes":["unsupported_claim"],"spans":["十年经验"],"repairable_by":"llm","message":"无来源年限"}'
            )

    result = apply_reply_grounding_gate(
        reply=f"伙伴【张测试】有十年经验，来源：{SOURCE}",  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
        reply_mode="dialogue",
        candidates=[],
        public_reads=extract_public_reads(_public_messages(), 0),
        model=Judge(),
        enabled=True,
        max_attempts=1,
    )
    assert result.results[0].codes == ["unsupported_claim"]
    assert "十年" not in result.reply


@pytest.mark.parametrize("per", [21, 500])
async def test_all_returned_users_survive_endpoint_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    per: int,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "disabled")
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    install(["get-ai-cofounders"], target=tmp_path / "skills/aihehuo")
    records = [{"id": str(index), "name": f"用户{index}", "bio": "真实项目经验"} for index in range(1, per + 1)]
    monkeypatch.setattr(integration, "fetch", lambda *_args, **_kwargs: json.dumps({"data": records}))
    source = f"https://www.aihehuo.com/ai/users.json?page=1&per={per}"
    reply = f"伙伴【用户{per}】有真实项目经验。来源：{source}"  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
    model = ToolModel(calls=[{"name": "query_aihehuo_public_data", "id": "read", "args": {"resource": "users", "page": 1, "per": per}}], reply=reply)
    graph = factory.create_member_agent(runtime=tmp_path, model=model, checkpointer=InMemorySaver(), include_save=False)
    state = AppState(agent=object(), base_dir=tmp_path, search_only_agent=graph)
    session = TrustedSession(
        principal=SessionPrincipal("owner", None, None, "body_stub"),
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
        group_id="g1",
        group_token=None,
        capabilities=SessionCapabilities(tier=CapabilityTier.in_group, can_search=True, can_write_profile=False),
    )

    async def resolve(*args: Any, **kwargs: Any) -> TrustedSession:
        del args, kwargs
        return session

    monkeypatch.setattr(chat_ep, "resolve_trusted_session", resolve)
    monkeypatch.setattr(chat_ep, "reply_grounding_enabled_for_user", lambda _: True)
    result = await chat_ep.chat(ChatRequest(user_id="owner", group_id="g1", message="介绍返回的最后一位人才"), state)
    assert result.reply == reply


@pytest.mark.parametrize(
    ("resource", "pack", "record"),
    [
        ("projects", "get-ai-project-partnerships", {"partner_model": "股权合作", "partner_geo": ["全国", "远程"], "hq_city": "杭州"}),
        ("blog", "get-ai-startup-digest", {"body_text": "实际正文" * 1500 + "正文末尾的真实结论", "nested": {"new_field": "保留新增字段"}}),
    ],
)
def test_judge_sees_exact_tool_document_including_all_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resource: str,
    pack: str,
    record: dict[str, Any],
) -> None:
    install([pack], target=tmp_path)
    document = {"data": [record], "meta": {"has_more": False}}
    monkeypatch.setattr(integration, "fetch", lambda *_args, **_kwargs: json.dumps(document))
    tool_content = integration.create_aihehuo_tool(skills_dir=tmp_path).invoke({"resource": resource})
    messages = [
        AIMessage(content="", tool_calls=[{"name": "query_aihehuo_public_data", "id": "read", "args": {"resource": resource}}]),
        ToolMessage(name="query_aihehuo_public_data", tool_call_id="read", content=tool_content),
    ]
    reads = extract_public_reads(messages, 0)
    assert reads[0].document == json.loads(tool_content)
    assert reads[0].records == [record]

    class Judge:
        def invoke(self, messages: list[BaseMessage]) -> AIMessage:
            assert json.dumps(reads[0].document, ensure_ascii=False) in str(messages[-1].content)
            return AIMessage(content='{"verdict":"pass","codes":[],"spans":[],"repairable_by":"llm","message":""}')

    reply = "股权合作，支持全国远程，总部杭州" if resource == "projects" else "正文末尾的真实结论"  # noqa: RUF001  # Chinese text intentionally uses Chinese punctuation.
    result = apply_reply_grounding_gate(reply=reply, reply_mode="dialogue", candidates=[], public_reads=reads, model=Judge(), enabled=True)
    assert result.reply == reply
    assert result.passed


def test_markdown_and_group_detail_evidence_have_no_independent_truncation() -> None:
    body = "实际正文" * 1500 + "末尾证据"
    messages = [
        AIMessage(content="", tool_calls=[{"name": "query_aihehuo_public_data", "id": "read", "args": {"resource": "weekly"}}]),
        ToolMessage(
            name="query_aihehuo_public_data", tool_call_id="read", content=f"Source API: https://www.aihehuo.com/ai/blog/weekly_markdown\n\n{body}"
        ),
    ]
    assert extract_public_reads(messages, 0)[0].text == body
    document = {"_source_url": "https://www.aihehuo.com/ai/groups/1.json", "data": {"group_title": "真实群标题", "members": [{"name": "成员"}]}}
    messages = [
        AIMessage(content="", tool_calls=[{"name": "query_aihehuo_public_data", "id": "read", "args": {"resource": "group-members"}}]),
        ToolMessage(name="query_aihehuo_public_data", tool_call_id="read", content=json.dumps(document)),
    ]
    reads = extract_public_reads(messages, 0)
    assert reads[0].document == document
    assert reads[0].records == document["data"]["members"]


async def test_api_and_worker_initialize_identical_permission_graphs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_startup = importlib.import_module("apps.group_agent_api.app.startup")
    from apps.group_agent_worker import runtime as worker

    monkeypatch.setenv("GROUP_AGENT_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    monkeypatch.setenv("GROUP_AGENT_MODEL_MODE", "stub")
    monkeypatch.setenv("GROUP_AGENT_ANONYMOUS_ENABLED", "0")
    monkeypatch.setenv("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "0")
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "legacy")
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "legacy")
    ref: dict[str, AppState | None] = {"state": None}
    await api_startup.startup(ref)
    config = SimpleNamespace(worker_instance_id="test-worker", redis_prefix="test")
    store = SimpleNamespace(ping=lambda: True, _r=object())
    monkeypatch.setattr(worker, "_RUNTIME", {})
    monkeypatch.setattr(worker, "load_durable_queue_config", lambda **_kwargs: config)
    monkeypatch.setattr(worker.ExecutionStore, "from_config", lambda _: store)
    monkeypatch.setattr(worker, "BackpressureController", lambda *_args: object())
    state = worker.get_worker_runtime()["state"]
    for current in (ref["state"], state):
        graphs = [
            current.member_agent_for_capability(can_search=read, can_write_profile=write)
            for read, write in [(True, True), (True, False), (False, True), (False, False)]
        ]
        assert len({id(graph) for graph in graphs}) == 4  # noqa: PLR2004  # Explicit expected contract value in this regression assertion.
        assert len({id(graph.checkpointer) for graph in graphs}) == 1
        for graph, expected in zip(
            graphs, [{"save_group_profile", "search_candidates"}, {"search_candidates"}, {"save_group_profile"}, set()], strict=True
        ):
            available = set(graph.get_graph().nodes["tools"].data.tools_by_name)
            assert available & {"save_group_profile", "search_candidates"} == expected


@pytest.mark.parametrize("async_entry", [False, True])
async def test_public_person_reply_survives_actual_endpoint_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    async_entry: bool,
) -> None:
    monkeypatch.setenv("GROUP_AGENT_READ_MODE", "public_skills")
    monkeypatch.setenv("GROUP_AGENT_WRITE_MODE", "disabled")
    monkeypatch.setenv("GROUP_AGENT_INTEGRATION", "stub")
    install(["get-ai-cofounders"], target=tmp_path / "skills/aihehuo")
    monkeypatch.setattr(
        integration, "fetch", lambda *_args, **_kwargs: json.dumps({"data": [{"id": "p1", "name": "张测试", "bio": "Python 项目经验"}]})
    )
    model = ToolModel(calls=[{"name": "query_aihehuo_public_data", "id": "public", "args": {"resource": "users", "page": 1, "per": 2}}], reply=REPLY)
    graph = factory.create_member_agent(runtime=tmp_path, model=model, checkpointer=InMemorySaver(), include_save=False)
    state = AppState(agent=object(), base_dir=tmp_path, search_only_agent=graph)
    session = TrustedSession(
        principal=SessionPrincipal("owner", None, None, "body_stub"),
        membership=MembershipResult(tier=CapabilityTier.in_group, source="stub"),
        group_id="g1",
        group_token=None,
        capabilities=SessionCapabilities(tier=CapabilityTier.in_group, can_search=True, can_write_profile=False),
    )
    captured: list[dict[str, Any]] = []
    actual_gate = apply_reply_grounding_gate

    def grounding(**kwargs: Any) -> ReplyGroundingGateResult:
        captured.append(kwargs)
        return actual_gate(**kwargs, enabled=True)

    import apps.group_agent_api.agent_factory.checks.reply_grounding as grounding_module

    monkeypatch.setattr(grounding_module, "apply_reply_grounding_gate", grounding)
    monkeypatch.setattr(chat_ep, "reply_grounding_enabled_for_user", lambda _: True)
    monkeypatch.setattr(async_manager, "reply_grounding_enabled_for_user", lambda _: True)
    if async_entry:
        events: list[tuple[str, dict[str, Any]]] = []

        async def emit(kind: str, payload: dict[str, Any]) -> bool:
            events.append((kind, payload))
            return True

        req = AsyncCallRequest(
            run_id="public-test",
            idempotency_key="public-test",
            user_id="owner",
            group_id="g1",
            conversation_id="c1",
            message="找公开的技术人才",
            callback_url="http://example.invalid/group_agent_callbacks",
            run_invite=False,
        )
        await async_manager._execute_core_agent(req=req, session=session, state=state, tid="async-public", emit_callback=emit)
        result = next(payload for kind, payload in events if kind == "final")
    else:

        async def resolve(*args: Any, **kwargs: Any) -> TrustedSession:
            del args, kwargs
            return session

        monkeypatch.setattr(chat_ep, "resolve_trusted_session", resolve)
        result = (await chat_ep.chat(ChatRequest(user_id="owner", group_id="g1", message="找公开的技术人才"), state)).model_dump()
    assert result["reply"] == REPLY
    assert result["candidates"] == []
    if async_entry:
        assert result["reply_mode"] == "dialogue"
    assert captured[0]["public_reads"][0].records[0]["name"] == "张测试"
    assert not (tmp_path / "users").exists()
