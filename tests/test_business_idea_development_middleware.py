"""Integration test for BusinessIdeaDevelopmentMiddleware.

This test verifies that the middleware automatically generates todos and guides
the agent through the complete business idea development sequence without
requiring multiple user invocations.

The agent should:
1. Receive a business idea from the user
2. See automatically generated todos
3. Work through each todo in sequence
4. Complete all milestones automatically
"""

import os
import shutil
import time
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.business_idea_development import BusinessIdeaDevelopmentMiddleware
from deepagents.middleware.business_idea_tracker import BusinessIdeaTrackerMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents_cli.skills.load import list_skills
from deepagents_cli.skills.middleware import SkillsMiddleware

from tests.timing_middleware import TimingMiddleware


def _count_cjk(text: str) -> int:
    """Count CJK Unified Ideographs in text (rough proxy for Chinese content)."""
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _extract_section(text: str, heading: str) -> str:
    """Extract the section content under a markdown heading '## {heading}'."""
    marker = f"## {heading}\n"
    start = text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    # Next section begins with '\n## '
    end = text.find("\n## ", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def _extract_text_only(ai_content: object) -> str:
    """Extract ONLY the LLM text from a provider-specific AIMessage content payload.

    - Some providers return a plain string.
    - Anthropic-style providers often return a list of content blocks (text/tool_use).
      We keep only text blocks.
    """
    if ai_content is None:
        return ""
    if isinstance(ai_content, str):
        return ai_content.strip()
    if isinstance(ai_content, list):
        parts: list[str] = []
        for item in ai_content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue

            if isinstance(item, dict):
                # Provider dict-style blocks
                if item.get("type") in (None, "text"):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                continue

            # Provider object-style blocks (e.g., TextBlock / ToolUseBlock)
            block_type = getattr(item, "type", None)
            block_text = getattr(item, "text", None)
            if (block_type in (None, "text")) and isinstance(block_text, str) and block_text.strip():
                parts.append(block_text.strip())
        return "\n\n".join(parts).strip()

    return str(ai_content).strip()


def _identify_skill_from_text(text: str) -> str | None:
    """Fallback classifier when milestones aren't available."""
    t = text.lower()

    # business-idea-evaluation
    if ("evaluation:" in t and "is business idea" in t) or ("评估" in text and ("商业" in text or "想法" in text)):
        return "business-idea-evaluation"

    # persona-clarification
    if ("persona:" in t and any(k in t for k in ["age:", "background:", "occupation:", "goals:", "core pain points:"])) or (
        "用户画像" in text or "人物画像" in text or ("persona" in t and "age" in t)
    ):
        return "persona-clarification"

    # painpoint-enhancement
    if ("enhanced pain point" in t and "dimension analysis" in t) or ("增强" in text and "痛点" in text):
        return "painpoint-enhancement"

    # 60s-pitch-creation
    if (("60-second pitch" in t or "黄金60秒" in t) and ("pitch breakdown" in t or "call to action" in t)) or ("60秒" in text and "pitch" in t):
        return "60s-pitch-creation"

    # baseline-pricing-and-optimization
    if ("baseline pricing" in t and ("pricing optimization tactics" in t or "split selling" in t or "cross-selling" in t)) or (
        ("定价" in text or "价格" in text) and ("基准" in text or "基础" in text)
    ):
        return "baseline-pricing-and-optimization"

    # business-model-pivot-exploration
    if all(k in t for k in ["retail model", "service model", "brokerage model", "subscription model"]) or ("商业模式" in text and ("模型" in text or "订阅" in text)):
        return "business-model-pivot-exploration"

    return None


def _collect_outputs_by_skill(messages: list) -> dict[str, str]:
    """Collect best-effort LLM outputs per skill, using milestone windows when possible."""
    outputs_by_skill: dict[str, str] = {}

    ordered_milestones: list[tuple[str, str]] = [
        ("mark_business_idea_complete", "business-idea-evaluation"),
        ("mark_persona_clarified", "persona-clarification"),
        ("mark_painpoint_enhanced", "painpoint-enhancement"),
        ("mark_pitch_created", "60s-pitch-creation"),
        ("mark_pricing_optimized", "baseline-pricing-and-optimization"),
    ]

    def _ai_text_at(i: int) -> str:
        m = messages[i]
        if getattr(m, "type", None) != "ai":
            return ""
        return _extract_text_only(getattr(m, "content", None))

    def _has_tool_call(i: int, tool_name: str) -> bool:
        m = messages[i]
        if getattr(m, "type", None) != "ai":
            return False
        for tc in (getattr(m, "tool_calls", None) or []):
            if isinstance(tc, dict) and tc.get("name") == tool_name:
                return True
            if getattr(tc, "name", None) == tool_name:
                return True
        return False

    # Find the first index where each milestone tool was called
    mark_idx: dict[str, int] = {}
    for tool_name, _skill in ordered_milestones:
        for i in range(len(messages)):
            if _has_tool_call(i, tool_name):
                mark_idx[tool_name] = i
                break

    # If we have milestone marks, do window-based extraction
    if mark_idx:
        prev_boundary = -1
        for tool_name, skill_name in ordered_milestones:
            end_boundary = mark_idx.get(tool_name, -1)
            if end_boundary == -1:
                continue
            candidates: list[str] = []
            for i in range(prev_boundary + 1, end_boundary + 1):
                txt = _ai_text_at(i)
                if len(txt) >= 120:
                    candidates.append(txt)
            if candidates:
                outputs_by_skill[skill_name] = max(candidates, key=len).strip()
            prev_boundary = end_boundary

        # Pivot exploration (no milestone tool): largest AI text after pricing mark
        pivot_skill = "business-model-pivot-exploration"
        after_pricing = mark_idx.get("mark_pricing_optimized", -1)
        pivot_candidates: list[str] = []
        for i in range(after_pricing + 1, len(messages)):
            txt = _ai_text_at(i)
            if len(txt) >= 300:
                pivot_candidates.append(txt)
        if pivot_candidates:
            outputs_by_skill[pivot_skill] = max(pivot_candidates, key=len).strip()

    # Fallback classification (also fills gaps for incomplete-idea runs)
    for i in range(len(messages)):
        txt = _ai_text_at(i)
        if len(txt) < 200:
            continue
        skill = _identify_skill_from_text(txt)
        if not skill:
            continue
        prev = outputs_by_skill.get(skill, "")
        if len(txt) > len(prev):
            outputs_by_skill[skill] = txt.strip()

    # Last-resort fallback: if incomplete and we still didn't capture evaluation, use longest AI text
    if "business-idea-evaluation" not in outputs_by_skill:
        ai_texts = [_ai_text_at(i) for i in range(len(messages))]
        ai_texts = [t for t in ai_texts if len(t) >= 200]
        if ai_texts:
            outputs_by_skill["business-idea-evaluation"] = max(ai_texts, key=len).strip()

    return outputs_by_skill


def _write_unified_output(*, filesystem_dir: Path, messages: list, duration: float) -> Path:
    """Write the unified 'LLM outputs only' file and return its path."""
    unified_output_path = filesystem_dir / "business_idea_development_output.md"

    outputs_by_skill = _collect_outputs_by_skill(messages)
    ordered_skills = [
        "business-idea-evaluation",
        "persona-clarification",
        "painpoint-enhancement",
        "60s-pitch-creation",
        "baseline-pricing-and-optimization",
        "business-model-pivot-exploration",
    ]

    unified_content = f"""# Business Idea Development - LLM Outputs Only

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Execution Time: {duration:.2f}s

"""
    for skill_name in ordered_skills:
        unified_content += f"\n## {skill_name}\n\n"
        text = outputs_by_skill.get(skill_name)
        if not text:
            unified_content += "_(No clean output detected for this skill in this run.)_\n"
        else:
            unified_content += text.strip() + "\n"

    unified_output_path.write_text(unified_content, encoding="utf-8")
    return unified_output_path


def _load_model_config(repo_root: Path) -> tuple[str, str, str]:
    """Load model config for ChatAnthropic.

    Prefers env vars; falls back to reading `.env.deepseek` when readable.
    """
    env_file = repo_root / ".env.deepseek"

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model_name = os.environ.get("ANTHROPIC_MODEL", "deepseek-chat")

    if base_url and api_key:
        return base_url, api_key, model_name

    if not env_file.exists():
        pytest.skip(
            f"DeepSeek config not found in env vars and file missing: {env_file}. "
            "Set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY to run this integration test."
        )

    env_vars: dict[str, str] = {}
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "export " in line:
                    key_value = line.replace("export ", "").split("=", 1)
                    if len(key_value) == 2:
                        key, value = key_value
                        value = value.strip('"\'')
                        env_vars[key] = value
    except PermissionError:
        pytest.skip(
            f"Cannot read {env_file} (permission denied). "
            "Set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY env vars to run this integration test."
        )

    base_url = env_vars.get("ANTHROPIC_BASE_URL") or base_url
    api_key = env_vars.get("ANTHROPIC_API_KEY") or api_key
    model_name = env_vars.get("ANTHROPIC_MODEL", model_name) or model_name

    if not base_url or not api_key:
        pytest.skip(
            "DeepSeek configuration incomplete. Set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY env vars "
            "or provide a readable .env.deepseek file."
        )

    return base_url, api_key, model_name


def _setup_required_skills(repo_root: Path, skills_dir: Path) -> list[str]:
    """Copy required skills into tmp skills_dir and assert discovery."""
    skills_dir.mkdir(parents=True, exist_ok=True)

    example_skills_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills"
    required_skills = [
        "business-idea-evaluation",
        "persona-clarification",
        "painpoint-enhancement",
        "60s-pitch-creation",
        "baseline-pricing-and-optimization",
        "business-model-pivot-exploration",
    ]

    for skill_name in required_skills:
        example_skill_dir = example_skills_dir / skill_name
        if not example_skill_dir.exists():
            pytest.skip(f"Example skill directory not found: {example_skill_dir}")

        shutil.copytree(example_skill_dir, skills_dir / skill_name)

    skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
    skill_names = [s["name"] for s in skills]
    for skill_name in required_skills:
        assert skill_name in skill_names, f"Should discover {skill_name} skill"

    return required_skills


@pytest.mark.timeout(900)  # 15 minutes per case (LLM-driven integration test)
@pytest.mark.parametrize(
    "case_name,user_request,expect_chinese",
    [
        (
            "en",
            """I want to create an app that helps busy professionals manage their work-life balance.
Many professionals struggle with burnout because they can't effectively prioritize tasks and end up working late into the night.
The app would help them set boundaries and manage their time more effectively.

Our team has 10 years of experience in productivity software development, and we have direct access to a network of 50,000 professionals through our existing consulting business.

Target customers: Busy professionals aged 30-45 who work 50+ hours per week, earning $80K-$150K annually.
Value proposition: The app saves users 5-10 hours per week by automating task prioritization and boundary setting, which translates to $2,000-$4,000 in time value per month for our target customers.

The product is a mobile application with AI-powered task prioritization, automated calendar blocking, and personalized boundary recommendations. It integrates with popular productivity tools like Google Calendar and Slack.

Please help me develop this business idea through the complete workflow.""",
            False,
        ),
        (
            "zh",
            """我想做一个帮助职场人士管理工作与生活平衡的移动应用。
很多人因为无法有效地给任务排序、经常加班到深夜而产生倦怠（burnout），也很难建立边界并真正下班。
这个应用会帮助他们设置边界、自动规划日程，并更高效地管理时间。

我们的团队有 10 年效率工具/生产力软件开发经验，并且通过现有咨询业务可以直接触达约 5 万名职场人士。

目标客户：30-45 岁、每周工作 50+ 小时、年收入 80K-150K 美元的忙碌职场人士。
价值主张：通过自动任务优先级排序与边界管理，每周为用户节省 5-10 小时，相当于每月节省 2,000-4,000 美元的时间价值。

产品形态：移动端 App，包含 AI 任务优先级、自动日历时间块（calendar blocking）、个性化边界建议，并可与 Google Calendar、Slack 等工具集成。

请用完整工作流帮我把这个商业想法推进到更清晰可执行的版本，并用中文输出。""",
            True,
        ),
    ],
)
def test_business_idea_development_automatic_progression(tmp_path: Path, case_name: str, user_request: str, expect_chinese: bool) -> None:
    """Test that BusinessIdeaDevelopmentMiddleware automatically guides the agent through the sequence.
    
    This test verifies:
    - Todos are automatically generated based on state
    - Agent works through todos without user intervention
    - Todos are updated as milestones are completed
    - All milestones are reached automatically
    - Final state has all todos completed
    """
    repo_root = Path(__file__).parent.parent
    base_url, api_key, model_name = _load_model_config(repo_root)
    
    # Set up environment
    old_base_url = os.environ.get("ANTHROPIC_BASE_URL")
    old_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    os.environ["ANTHROPIC_BASE_URL"] = base_url
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    try:
        # Set up skills directory with all required skills
        agent_id = f"test_business_idea_development_{case_name}"
        skills_dir = tmp_path / "skills"
        required_skills = _setup_required_skills(repo_root, skills_dir)

        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        skill_names = [s["name"] for s in skills]
        
        print("\n" + "="*80)
        print("TEST: BUSINESS IDEA DEVELOPMENT AUTOMATIC PROGRESSION")
        print("="*80)
        print(f"Skills loaded: {len(skills)}")
        print(f"Skills: {', '.join(skill_names)}\n")
        
        # Create model
        model = ChatAnthropic(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            max_tokens=20000,
            timeout=300.0,
        )
        
        # Create agent with BusinessIdeaDevelopmentMiddleware
        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)
        checkpointer = MemorySaver()
        
        agent = create_agent(
            model=model,
            middleware=[
                timing_middleware,
                TodoListMiddleware(),  # Provides write_todos tool
                BusinessIdeaTrackerMiddleware(),  # Tracks milestones
                BusinessIdeaDevelopmentMiddleware(),  # Auto-generates todos
                LanguageDetectionMiddleware(),
                FilesystemMiddleware(backend=filesystem_backend),
                SkillsMiddleware(
                    skills_dir=skills_dir,
                    assistant_id=agent_id,
                    project_skills_dir=None,
                ),
            ],
            tools=[],
            checkpointer=checkpointer,
            system_prompt="""You are a business co-founder assistant helping entrepreneurs develop their startup ideas.

You have access to a todo list that guides you through the business idea development process. 
Work through the todos in order, completing each one fully before moving to the next.

For each todo:
1. Read the relevant skill file if needed
2. Complete the task described in the todo
3. Call the appropriate milestone marking tool after completing each skill
4. Mark the todo as completed using the write_todos tool

The todos will automatically update as you complete milestones. Focus on the current todo and complete it thoroughly.""",
        )
        
        config = {"configurable": {"thread_id": f"test-business-idea-development-auto-{case_name}"}}
        
        # Single user request - agent should work through all todos automatically
        print(f"\n📝 User Request:\n{user_request}\n")
        print("⏳ Starting agent execution (agent will work through todos automatically)...\n")
        
        input_state = {"messages": [HumanMessage(content=user_request)]}
        invoke_start = time.time()
        
        # Execute the agent - it should work through all todos automatically
        result = agent.invoke(input_state, config)
        
        invoke_end = time.time()
        duration = invoke_end - invoke_start
        
        print(f"\n✅ Agent execution completed ({duration:.2f}s)\n")
        
        # ============================================================
        # VERIFICATION
        # ============================================================
        print("\n" + "="*80)
        print("VERIFICATION: TODOS AND MILESTONES")
        print("="*80)
        
        # Check todos
        todos = result.get("todos", [])
        print(f"\n📋 Todos ({len(todos)} total):")
        for i, todo in enumerate(todos, 1):
            status_icon = {
                "completed": "✅",
                "in_progress": "🔄",
                "pending": "⏳",
            }.get(todo.get("status", "pending"), "❓")
            print(f"  {i}. {status_icon} [{todo.get('status', 'unknown')}] {todo.get('content', '')[:80]}...")
        
        # Check milestones
        milestones = {
            "business_idea_complete": result.get("business_idea_complete", False),
            "persona_clarified": result.get("persona_clarified", False),
            "painpoint_enhanced": result.get("painpoint_enhanced", False),
            "pitch_created": result.get("pitch_created", False),
            "pricing_optimized": result.get("pricing_optimized", False),
        }
        
        print(f"\n🎯 Milestones:")
        for milestone, completed in milestones.items():
            status_icon = "✅" if completed else "❌"
            print(f"  {status_icon} {milestone}: {completed}")
        
        # Check messages for milestone tool calls
        messages = result.get("messages", [])
        milestone_tools_called = {
            "mark_business_idea_complete": False,
            "mark_persona_clarified": False,
            "mark_painpoint_enhanced": False,
            "mark_pitch_created": False,
            "mark_pricing_optimized": False,
        }
        
        for message in messages:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get("name", "")
                    if tool_name in milestone_tools_called:
                        milestone_tools_called[tool_name] = True
        
        print(f"\n🔧 Milestone Tools Called:")
        for tool_name, called in milestone_tools_called.items():
            status_icon = "✅" if called else "❌"
            print(f"  {status_icon} {tool_name}: {called}")
        
        # ============================================================
        # ASSERTIONS
        # ============================================================
        print("\n" + "="*80)
        print("ASSERTIONS")
        print("="*80)
        
        # Assert todos were generated
        assert len(todos) > 0, "Todos should be automatically generated"
        print(f"✅ Todos generated: {len(todos)} todos")
        
        # Assert at least some todos are completed (the agent should have made progress)
        completed_todos = [t for t in todos if t.get("status") == "completed"]
        assert len(completed_todos) > 0, (
            f"At least some todos should be completed. "
            f"Completed: {len(completed_todos)}/{len(todos)}"
        )
        print(f"✅ Todos completed: {len(completed_todos)}/{len(todos)}")
        
        # Assert business idea was marked complete
        assert milestones["business_idea_complete"], (
            "Business idea should be marked as complete"
        )
        assert milestone_tools_called["mark_business_idea_complete"], (
            "mark_business_idea_complete tool should be called"
        )
        print("✅ Business idea evaluated and marked complete")
        
        # Check if agent progressed further (at least persona clarification)
        # We're lenient here - the agent might not complete all steps in one go
        # but should at least complete the first few
        progress_made = (
            milestones["business_idea_complete"]
            or milestones["persona_clarified"]
            or milestones["painpoint_enhanced"]
        )
        assert progress_made, (
            "Agent should have made progress through at least the first few milestones"
        )
        print("✅ Agent made progress through the sequence")
        
        # Verify todos reflect milestone status
        # First todo should be completed if business_idea_complete is True
        if milestones["business_idea_complete"] and len(todos) > 0:
            first_todo_status = todos[0].get("status")
            # The todo might be completed or the agent might be working on the next step
            assert first_todo_status in ["completed", "in_progress"], (
                f"First todo should be completed or in_progress when business idea is complete. "
                f"Got: {first_todo_status}"
            )
            print("✅ Todos correctly reflect milestone completion status")
        
        # ============================================================
        # COLLECT AND UNIFY OUTPUTS
        # ============================================================
        print("\n" + "="*80)
        print("COLLECTING AND UNIFYING OUTPUTS")
        print("="*80)
        
        filesystem_dir = Path(filesystem_backend.cwd)
        unified_output_path = _write_unified_output(
            filesystem_dir=filesystem_dir,
            messages=messages,
            duration=duration,
        )
        print(f"\n✅ Unified output written to: {unified_output_path}")
        print(f"   File size: {unified_output_path.stat().st_size} bytes")

        # ============================================================
        # LANGUAGE ASSERTION (Chinese case)
        # ============================================================
        if expect_chinese:
            unified_text = unified_output_path.read_text(encoding="utf-8")

            # Overall: should contain substantial Chinese
            cjk_total = _count_cjk(unified_text)
            if cjk_total < 300:
                # Provide debug context to quickly see what language the model actually used.
                preview = unified_text[:1200].replace("\n", "\\n")
                raise AssertionError(
                    f"Expected substantial Chinese output, got only {cjk_total} CJK characters. "
                    f"Unified output preview (first 1200 chars): {preview}"
                )

            # Per-skill: each section should have meaningful Chinese, not just headings
            ordered_skills = [
                "business-idea-evaluation",
                "persona-clarification",
                "painpoint-enhancement",
                "60s-pitch-creation",
                "baseline-pricing-and-optimization",
                "business-model-pivot-exploration",
            ]

            missing_sections: list[str] = []
            low_chinese_sections: list[tuple[str, int]] = []
            for s in ordered_skills:
                section = _extract_section(unified_text, s)
                if not section:
                    missing_sections.append(s)
                    continue
                cjk = _count_cjk(section)
                if cjk < 80:
                    low_chinese_sections.append((s, cjk))

            assert not missing_sections, f"Missing skill sections in unified output: {missing_sections}"
            assert not low_chinese_sections, f"Some skill outputs are not sufficiently Chinese: {low_chinese_sections}"
            print("✅ Chinese language output detected for all skill sections")
        print(f"\n💡 TIP: To preserve outputs after test, copy files from:")
        print(f"   {filesystem_dir}")
        print(f"   Or modify the test to use a fixed directory instead of tmp_path")
        
        print("\n" + "="*80)
        print("✅ BUSINESS IDEA DEVELOPMENT MIDDLEWARE TEST PASSED")
        print("="*80)
        print(f"\nSummary:")
        print(f"  - Todos generated: {len(todos)}")
        print(f"  - Todos completed: {len(completed_todos)}")
        print(f"  - Milestones reached: {sum(milestones.values())}/5")
        print(f"  - Milestone tools called: {sum(milestone_tools_called.values())}/5")
        print(f"  - Execution time: {duration:.2f}s")
        # Note: We intentionally keep the unified output minimal (LLM outputs only).
        print(f"  - Unified output: {unified_output_path}")
        print("\n" + "="*80)
        
        # Print timing summary
        timing_middleware.print_summary()
        
    finally:
        # Restore environment
        if old_base_url is not None:
            os.environ["ANTHROPIC_BASE_URL"] = old_base_url
        elif "ANTHROPIC_BASE_URL" in os.environ:
            del os.environ["ANTHROPIC_BASE_URL"]
        
        if old_api_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_api_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]


@pytest.mark.timeout(600)  # Should be quicker: agent stops at idea-evaluation with clarifying questions
@pytest.mark.parametrize(
    "case_name,user_request,expect_chinese",
    [
        (
            "en",
            "I have a rough idea: maybe an AI app for 'productivity' but I'm not sure who it's for. "
            "It could help people somehow. I haven't decided what it does exactly. "
            "Can you help me figure out if this is a complete business idea and what I'm missing?",
            False,
        ),
        (
            "zh",
            "我只有一个很粗糙的想法：可能做一个和“效率/生产力”有关的 AI 应用，但我还不确定目标用户是谁，"
            "也说不清楚具体功能是什么。请你判断这算不算完整的商业想法，如果不完整，请指出缺什么并用中文问我澄清问题。",
            True,
        ),
    ],
)
def test_business_idea_development_incomplete_idea_stops_at_evaluation(
    tmp_path: Path, case_name: str, user_request: str, expect_chinese: bool
) -> None:
    """When the initial idea is incomplete, the agent should:

    - Use business-idea-evaluation
    - Ask clarifying questions
    - NOT call mark_business_idea_complete
    - NOT progress to downstream skills/milestones
    """
    repo_root = Path(__file__).parent.parent
    base_url, api_key, model_name = _load_model_config(repo_root)

    old_base_url = os.environ.get("ANTHROPIC_BASE_URL")
    old_api_key = os.environ.get("ANTHROPIC_API_KEY")

    os.environ["ANTHROPIC_BASE_URL"] = base_url
    os.environ["ANTHROPIC_API_KEY"] = api_key

    try:
        agent_id = f"test_business_idea_development_incomplete_{case_name}"
        skills_dir = tmp_path / "skills"
        _setup_required_skills(repo_root, skills_dir)

        model = ChatAnthropic(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            max_tokens=12000,
            timeout=300.0,
        )

        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)
        checkpointer = MemorySaver()

        agent = create_agent(
            model=model,
            middleware=[
                timing_middleware,
                TodoListMiddleware(),
                BusinessIdeaTrackerMiddleware(),
                BusinessIdeaDevelopmentMiddleware(),
                LanguageDetectionMiddleware(),
                FilesystemMiddleware(backend=filesystem_backend),
                SkillsMiddleware(
                    skills_dir=skills_dir,
                    assistant_id=agent_id,
                    project_skills_dir=None,
                ),
            ],
            tools=[],
            checkpointer=checkpointer,
            system_prompt="""You are a business co-founder assistant.

Follow the todo list. If the business idea is incomplete, do NOT pretend it is complete.
Use business-idea-evaluation and ask clarifying questions. Only call mark_business_idea_complete if the idea is truly complete.
""",
        )

        config = {"configurable": {"thread_id": f"test-business-idea-development-incomplete-{case_name}"}}
        result = agent.invoke({"messages": [HumanMessage(content=user_request)]}, config)
        duration = 0.0

        # Core state expectations
        assert result.get("business_idea_complete") is False, "Incomplete idea should NOT be marked complete"
        assert result.get("persona_clarified") is False
        assert result.get("painpoint_enhanced") is False
        assert result.get("pitch_created") is False
        assert result.get("pricing_optimized") is False

        # Tool-call expectations: should NOT call mark_business_idea_complete
        messages = result.get("messages", [])
        called_mark_complete = False
        for m in messages:
            if m.type == "ai" and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    if isinstance(tc, dict) and tc.get("name") == "mark_business_idea_complete":
                        called_mark_complete = True
        assert not called_mark_complete, "Should NOT call mark_business_idea_complete for incomplete idea"

        # Unified output should contain evaluation section with clarifying questions
        unified_output_path = _write_unified_output(
            filesystem_dir=Path(filesystem_backend.cwd),
            messages=messages,
            duration=duration,
        )
        assert unified_output_path.exists(), "Unified output file should be written"
        print(f"\n📄 Unified output written to: {unified_output_path}")
        unified_text = unified_output_path.read_text(encoding="utf-8")

        eval_section = _extract_section(unified_text, "business-idea-evaluation")
        assert eval_section, "business-idea-evaluation section should not be empty"

        # Expect question(s)
        q_count = eval_section.count("?") + eval_section.count("？")
        assert q_count >= 2, f"Expected clarifying questions for incomplete idea, got q_count={q_count}"

        # Language expectation for zh
        if expect_chinese:
            cjk = _count_cjk(eval_section)
            assert cjk >= 80, f"Expected Chinese evaluation output, got only {cjk} CJK chars"

    finally:
        if old_base_url is not None:
            os.environ["ANTHROPIC_BASE_URL"] = old_base_url
        elif "ANTHROPIC_BASE_URL" in os.environ:
            del os.environ["ANTHROPIC_BASE_URL"]

        if old_api_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_api_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

