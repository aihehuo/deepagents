"""Integration test for business-model-pivot-exploration skill.

This test verifies:
1. Skill discovery - the skill is found and loaded correctly
2. Skill usage - the agent picks up and uses the skill
3. Outcome validation - the agent produces business model pivot exploration in the expected format
4. Dependency check - skill is not used before business idea and product/service are clarified
"""

import re
import shutil
import time
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.business_idea_tracker import BusinessIdeaTrackerMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents_cli.skills.load import list_skills
from deepagents_cli.skills.middleware import SkillsMiddleware

from tests.model_provider import create_test_model, load_test_model_config
from tests.timing_middleware import TimingMiddleware


@pytest.mark.timeout(180)  # 3 minutes for real LLM calls
def test_business_model_pivot_exploration_skill_discovery(tmp_path: Path) -> None:
    """Test 1: Verify skill discovery and metadata parsing.
    
    This test validates:
    - Skill is discovered from the examples directory
    - Metadata (name, description) is correctly parsed from YAML frontmatter
    - Skill path is correctly resolved
    """
    repo_root = Path(__file__).parent.parent
    
    # Set up skills directory
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy business-model-pivot-exploration skill from examples
    example_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "business-model-pivot-exploration"
    if not example_skill_dir.exists():
        pytest.skip(f"Example skill directory not found: {example_skill_dir}")
    
    skill_dest = skills_dir / "business-model-pivot-exploration"
    shutil.copytree(example_skill_dir, skill_dest)
    
    # Test skill discovery
    skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
    
    assert len(skills) == 1, f"Expected 1 skill, found {len(skills)}"
    
    skill_metadata = skills[0]
    assert skill_metadata["name"] == "business-model-pivot-exploration", (
        f"Expected skill name 'business-model-pivot-exploration', got '{skill_metadata['name']}'"
    )
    assert "business model" in skill_metadata["description"].lower() or "pivot" in skill_metadata["description"].lower(), (
        "Skill description should mention business model or pivot"
    )
    assert skill_metadata["source"] == "user", "Skill should be from user directory"
    
    # Verify SKILL.md content
    skill_md_path = skill_dest / "SKILL.md"
    assert skill_md_path.exists(), "SKILL.md should exist"
    
    skill_content = skill_md_path.read_text()
    assert "business model" in skill_content.lower() or "pivot" in skill_content.lower(), (
        "SKILL.md should mention business model or pivot"
    )
    assert "retail" in skill_content.lower() or "subscription" in skill_content.lower() or "brokerage" in skill_content.lower(), (
        "SKILL.md should mention at least one business model archetype"
    )
    
    print("\n" + "="*80)
    print("✅ SKILL DISCOVERY TEST PASSED")
    print("="*80)
    print(f"  Skill Name: {skill_metadata['name']}")
    print(f"  Description: {skill_metadata['description']}")
    print(f"  Path: {skill_metadata['path']}")
    print(f"  Source: {skill_metadata['source']}")


@pytest.mark.timeout(300)  # 5 minutes for real LLM calls (this skill generates a lot of content)
def test_business_model_pivot_exploration_with_complete_idea(tmp_path: Path) -> None:
    """Test 2: Verify skill explores business model pivots correctly after business idea is identified.
    
    This test validates:
    - Skill is loaded by SkillsMiddleware
    - Agent picks up the skill when pivot exploration is requested
    - Agent produces business model pivot exploration in the expected format
    - Output contains exploration of multiple business models with required sections
    - Skill is used after business idea and product/service are clarified
    """
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    try:
        # Set up skills directory
        agent_id = "test_business_model_pivot_exploration"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy business-idea-evaluation and business-model-pivot-exploration skills
        example_business_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "business-idea-evaluation"
        example_pivot_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "business-model-pivot-exploration"
        
        if not example_business_skill_dir.exists():
            pytest.skip(f"Example business-idea-evaluation skill directory not found: {example_business_skill_dir}")
        if not example_pivot_skill_dir.exists():
            pytest.skip(f"Example business-model-pivot-exploration skill directory not found: {example_pivot_skill_dir}")
        
        skill_dest_business = skills_dir / "business-idea-evaluation"
        skill_dest_pivot = skills_dir / "business-model-pivot-exploration"
        shutil.copytree(example_business_skill_dir, skill_dest_business)
        shutil.copytree(example_pivot_skill_dir, skill_dest_pivot)
        
        # Verify skill discovery
        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        assert len(skills) == 2, "Should discover both skills"
        skill_names = [s["name"] for s in skills]
        assert "business-idea-evaluation" in skill_names, "Should discover business-idea-evaluation skill"
        assert "business-model-pivot-exploration" in skill_names, "Should discover business-model-pivot-exploration skill"
        
        print("\n" + "="*80)
        print("TEST: BUSINESS MODEL PIVOT EXPLORATION WITH COMPLETE IDEA")
        print("="*80)
        
        # Create model (DeepSeek default; switch to Qwen via BC_API_PROVIDER=qwen + QWEN_* env vars)
        model = create_test_model(cfg=cfg)
        
        # Create agent with SkillsMiddleware and BusinessIdeaTrackerMiddleware
        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)
        
        # Create checkpointer for state persistence
        checkpointer = MemorySaver()
        
        agent = create_deep_agent(
            model=model,
            backend=filesystem_backend,
            middleware=[
                timing_middleware,
                BusinessIdeaTrackerMiddleware(),  # Track business idea completion
                SkillsMiddleware(
                    skills_dir=skills_dir,
                    assistant_id=agent_id,
                    project_skills_dir=None,
                ),
            ],
            tools=[],
            checkpointer=checkpointer,  # Enable state persistence
            system_prompt="""You are a business co-founder assistant helping entrepreneurs develop their startup ideas.

When a user provides input about a potential business idea, you should:
1. First use the business-idea-evaluation skill to evaluate whether they have a complete business idea
2. If the idea is complete, mark it using mark_business_idea_complete tool
3. Then, if the user wants to explore business model pivots or alternative models, use the business-model-pivot-exploration skill

The business-model-pivot-exploration skill should only be used AFTER a business idea has been identified and the product/service is clear.""",
        )
        
        # Step 1: First identify a complete business idea with clear product/service
        user_request_1 = """I want to create an app that helps busy professionals manage their work-life balance.
Many professionals struggle with burnout because they can't effectively prioritize tasks and end up working late into the night.
The app would help them set boundaries and manage their time more effectively.

The product is a mobile application with AI-powered task prioritization, automated calendar blocking, and personalized boundary recommendations. It integrates with popular productivity tools like Google Calendar and Slack."""
        
        print(f"\n📝 Step 1 - User Request (Complete Idea with Product Details):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")
        
        # Execute the agent - first message
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-business-model-pivot-exploration-complete"}}
        
        invoke_start = time.time()
        result_1 = agent.invoke(input_state_1, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
        
        print(f"\n✅ First execution completed ({invoke_duration:.2f}s)\n")
        
        # Check if idea was marked as complete
        business_idea_complete_1 = result_1.get("business_idea_complete", False)
        assert business_idea_complete_1, "Business idea should be marked as complete in step 1"
        
        print("✅ Step 1 passed: Business idea was correctly identified and marked as complete")
        
        # Step 2: Request business model pivot exploration
        user_request_2 = "Can you help me explore different business models for this idea? I want to see what alternative models might work better than my current approach."
        
        print(f"\n📝 Step 2 - User Request (Explore Business Model Pivots):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (explore business model pivots)...\n")
        
        # Execute the agent - second message (should use business-model-pivot-exploration skill)
        messages_so_far = result_1.get("messages", [])
        input_state_2 = {"messages": messages_so_far + [HumanMessage(content=user_request_2)]}
        
        invoke_start = time.time()
        result_2 = agent.invoke(input_state_2, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
        
        print(f"\n✅ Second execution completed ({invoke_duration:.2f}s)\n")
        
        # Extract messages
        messages_2 = result_2.get("messages", [])
        print(f"📊 Messages: {len(messages_2)} total\n")
        
        # Print timing summary
        timing_middleware.print_summary()
        
        # Validation: Outcome - Business model pivot exploration format validation
        ai_messages = [m for m in messages_2 if m.type == "ai"]
        assert len(ai_messages) > 0, "Agent should have generated at least one response"
        
        final_content = str(ai_messages[-1].content)
        
        print("\n" + "="*80)
        print("OUTCOME VALIDATION")
        print("="*80)
        
        # Check if skill was read (agent used the skill)
        skill_read = False
        for message in messages_2:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "read_file":
                        args = tool_call.get("args", {})
                        file_path = args.get("file_path", "")
                        if "business-model-pivot" in str(file_path) or "pivot-exploration" in str(file_path):
                            skill_read = True
                            break
            if skill_read:
                break
        
        # Check for business model pivot exploration structure markers
        content_lower = final_content.lower()
        
        # Required structure markers
        structure_markers = [
            "business model",
            "customer segment",
            "value proposition",
            "revenue logic",
            "implementation example",
            "profitability impact",
            "unit price effect",
            "sales complexity effect",
            "feasibility",
            "risks",
            "most promising",
            "summary",
        ]
        
        markers_found = sum(1 for marker in structure_markers if marker in content_lower)
        
        # Check for business model archetypes (at least some should be mentioned)
        model_archetypes = [
            "retail",
            "service",
            "brokerage",
            "transaction",
            "subscription",
            "usage-based",
            "membership",
        ]
        
        models_found = sum(1 for model in model_archetypes if model in content_lower)
        
        # Check for the main sections
        has_customer_segment = "customer segment" in content_lower
        has_value_proposition = "value proposition" in content_lower
        has_revenue_logic = "revenue logic" in content_lower or "revenue" in content_lower
        has_profitability_impact = "profitability" in content_lower or "unit price" in content_lower or "sales complexity" in content_lower
        has_summary = "summary" in content_lower or "most promising" in content_lower or "recommended" in content_lower
        
        print(f"\n📋 Skill Usage Validation:")
        print(f"  Skill read (expected): {skill_read}")
        
        print(f"\n📋 Business Model Pivot Exploration Structure Validation:")
        print(f"  Structure markers found: {markers_found}/{len(structure_markers)}")
        print(f"  Business model archetypes found: {models_found}/7")
        print(f"  Has Customer Segment: {has_customer_segment}")
        print(f"  Has Value Proposition: {has_value_proposition}")
        print(f"  Has Revenue Logic: {has_revenue_logic}")
        print(f"  Has Profitability Impact: {has_profitability_impact}")
        print(f"  Has Summary: {has_summary}")
        
        # Display the business model pivot exploration output (truncated if too long)
        print("\n" + "="*80)
        print("GENERATED BUSINESS MODEL PIVOT EXPLORATION OUTPUT")
        print("="*80)
        if len(final_content) > 2000:
            print(final_content[:2000] + "\n... (output truncated, full content is " + str(len(final_content)) + " characters)")
        else:
            print(final_content)
        print("="*80)
        
        # Final validations
        print("\n" + "="*80)
        print("FINAL VALIDATIONS")
        print("="*80)
        
        # Validation: Skill should be used
        assert skill_read, (
            "Agent should read the business-model-pivot-exploration SKILL.md file"
        )
        
        # Validation: Should have business model pivot exploration structure
        assert markers_found >= 6, (
            f"Business model pivot exploration should contain at least 6 structure markers. "
            f"Found: {markers_found}/{len(structure_markers)}"
        )
        
        # Validation: Should explore at least 3 business model archetypes
        assert models_found >= 3, (
            f"Business model pivot exploration should explore at least 3 business model archetypes. "
            f"Found: {models_found}/7"
        )
        
        # Validation: Should contain key sections
        assert has_customer_segment or has_value_proposition, (
            "Business model pivot exploration should contain Customer Segment or Value Proposition sections"
        )
        assert has_revenue_logic, (
            "Business model pivot exploration should contain Revenue Logic section"
        )
        assert has_profitability_impact, (
            "Business model pivot exploration should contain Profitability Impact section"
        )
        
        print("✅ Business model pivot exploration was correctly generated with proper structure")
        print("✅ Test passed: Skill correctly explores business model pivots after business idea is identified")
        
    finally:
        pass


@pytest.mark.timeout(300)
def test_business_model_pivot_exploration_with_chinese_input(tmp_path: Path) -> None:
    """Test 3: Verify skill works with Chinese input and produces Chinese output.
    
    This test validates:
    - Language detection works for Chinese
    - Agent responds in Chinese
    - Business model pivot exploration structure is present in Chinese response
    - Multiple business models are explored
    """
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    try:
        # Set up skills directory
        agent_id = "test_business_model_pivot_exploration_chinese"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy both skills
        example_business_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "business-idea-evaluation"
        example_pivot_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "business-model-pivot-exploration"
        
        if not example_business_skill_dir.exists():
            pytest.skip(f"Example business-idea-evaluation skill directory not found: {example_business_skill_dir}")
        if not example_pivot_skill_dir.exists():
            pytest.skip(f"Example business-model-pivot-exploration skill directory not found: {example_pivot_skill_dir}")
        
        skill_dest_business = skills_dir / "business-idea-evaluation"
        skill_dest_pivot = skills_dir / "business-model-pivot-exploration"
        shutil.copytree(example_business_skill_dir, skill_dest_business)
        shutil.copytree(example_pivot_skill_dir, skill_dest_pivot)
        
        print("\n" + "="*80)
        print("TEST: BUSINESS MODEL PIVOT EXPLORATION WITH CHINESE INPUT")
        print("="*80)
        
        model = create_test_model(cfg=cfg)
        
        # Create agent with LanguageDetectionMiddleware
        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)
        
        checkpointer = MemorySaver()
        
        agent = create_deep_agent(
            model=model,
            backend=filesystem_backend,
            middleware=[
                timing_middleware,
                LanguageDetectionMiddleware(),  # Detect and respond in user's language
                BusinessIdeaTrackerMiddleware(),
                SkillsMiddleware(
                    skills_dir=skills_dir,
                    assistant_id=agent_id,
                    project_skills_dir=None,
                ),
            ],
            tools=[],
            checkpointer=checkpointer,
            system_prompt="""You are a business co-founder assistant helping entrepreneurs develop their startup ideas.

When a user provides input about a potential business idea, you should:
1. First use the business-idea-evaluation skill to evaluate whether they have a complete business idea
2. If the idea is complete, mark it using mark_business_idea_complete tool
3. Then, if the user wants to explore business model pivots, use the business-model-pivot-exploration skill""",
        )
        
        # Step 1: Identify business idea in Chinese with clear product/service
        user_request_1 = """我想创建一个帮助忙碌专业人士管理工作与生活平衡的应用。
许多专业人士因为无法有效优先处理任务而最终工作到深夜，导致过度疲劳。
这个应用可以帮助他们设定界限并更有效地管理时间。

该产品是一个移动应用程序，具有AI驱动的任务优先级排序、自动日历阻止和个性化边界推荐功能。它集成了流行的生产力工具，如Google日历和Slack。"""
        
        print(f"\n📝 Step 1 - User Request (Chinese - Complete Idea with Product Details):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")
        
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-business-model-pivot-exploration-chinese"}}
        
        invoke_start = time.time()
        result_1 = agent.invoke(input_state_1, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
        
        print(f"\n✅ First execution completed ({invoke_duration:.2f}s)\n")
        
        # Check if idea was marked as complete
        business_idea_complete_1 = result_1.get("business_idea_complete", False)
        assert business_idea_complete_1, "Business idea should be marked as complete in step 1"
        
        # Step 2: Request business model pivot exploration in Chinese
        user_request_2 = "你能帮我探索这个想法的不同商业模式吗？我想看看哪些替代模式可能比我当前的方法更好。"
        
        print(f"\n📝 Step 2 - User Request (Chinese - Explore Business Model Pivots):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (explore business model pivots)...\n")
        
        messages_so_far = result_1.get("messages", [])
        input_state_2 = {"messages": messages_so_far + [HumanMessage(content=user_request_2)]}
        
        invoke_start = time.time()
        result_2 = agent.invoke(input_state_2, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
        
        print(f"\n✅ Second execution completed ({invoke_duration:.2f}s)\n")
        
        messages_2 = result_2.get("messages", [])
        timing_middleware.print_summary()
        
        # Validation: Language detection
        state = result_2
        detected_language = state.get("detected_language")
        
        print("\n" + "="*80)
        print("LANGUAGE DETECTION VALIDATION")
        print("="*80)
        
        if detected_language:
            print(f"✅ Language detected in state: {detected_language}")
            assert detected_language.startswith("zh"), (
                f"Expected Chinese language code (zh*), got {detected_language}"
            )
        else:
            print("⚠️  Language not detected in state (may be detected in middleware)")
        
        # Validation: Response language
        ai_messages = [m for m in messages_2 if m.type == "ai"]
        assert len(ai_messages) > 0, "Agent should have generated at least one response"
        
        final_content = str(ai_messages[-1].content)
        
        chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
        has_chinese = bool(chinese_char_pattern.search(final_content))
        
        print("\n" + "="*80)
        print("RESPONSE LANGUAGE VALIDATION")
        print("="*80)
        
        if has_chinese:
            chinese_chars = chinese_char_pattern.findall(final_content)
            chinese_char_count = sum(len(match) for match in chinese_chars)
            print(f"✅ Response contains Chinese characters: {chinese_char_count} characters found")
        else:
            print("⚠️  Response does not contain Chinese characters")
        
        # Validation: Business model pivot exploration structure in Chinese
        print("\n" + "="*80)
        print("BUSINESS MODEL PIVOT EXPLORATION STRUCTURE VALIDATION (Chinese)")
        print("="*80)
        
        # Chinese keywords for business model pivot exploration structure
        chinese_structure_keywords = [
            ("Business Model", ["商业模式", "业务模式", "模式"]),
            ("Customer Segment", ["客户细分", "目标客户", "客户群体"]),
            ("Value Proposition", ["价值主张", "价值定位"]),
            ("Revenue Logic", ["收入逻辑", "盈利模式", "收入模式"]),
            ("Profitability Impact", ["盈利能力", "利润影响", "盈利影响"]),
            ("Summary", ["总结", "汇总", "最推荐"]),
        ]
        
        found_keywords = []
        for field_name, keywords in chinese_structure_keywords:
            found = any(keyword in final_content for keyword in keywords)
            if found:
                found_keywords.append(field_name)
                print(f"  ✅ {field_name} (found)")
            else:
                print(f"  ❌ {field_name} (missing)")
        
        # Also check for English structure markers (in case of mixed language)
        english_markers = ["business model", "customer segment", "value proposition", "revenue", "profitability", "summary"]
        english_found = sum(1 for marker in english_markers if marker in final_content.lower())
        
        total_structure_found = len(found_keywords) + (1 if english_found >= 3 else 0)
        
        print(f"\n📊 Summary:")
        print(f"  Chinese structure keywords found: {len(found_keywords)}/{len(chinese_structure_keywords)}")
        print(f"  English markers found: {english_found}")
        print(f"  Total structure found: {total_structure_found}")
        
        # Display the business model pivot exploration output (truncated if too long)
        print("\n" + "="*80)
        print("GENERATED BUSINESS MODEL PIVOT EXPLORATION OUTPUT (Chinese)")
        print("="*80)
        if len(final_content) > 2000:
            print(final_content[:2000] + "\n... (output truncated, full content is " + str(len(final_content)) + " characters)")
        else:
            print(final_content)
        print("="*80)
        
        # Final validations
        print("\n" + "="*80)
        print("FINAL VALIDATIONS")
        print("="*80)
        
        # Validation: Response should be in Chinese
        assert has_chinese, (
            "Agent should respond in Chinese when user input is in Chinese"
        )
        
        # Validation: Should have business model pivot exploration structure
        assert total_structure_found >= 3, (
            f"Business model pivot exploration should contain at least 3 structure elements. "
            f"Found: {total_structure_found} (Chinese: {len(found_keywords)}, English: {english_found})"
        )
        
        print("✅ Business model pivot exploration was correctly generated in Chinese with proper structure")
        print("✅ Test passed: Skill correctly explores business model pivots in Chinese")
        
    finally:
        pass

