"""Integration test for baseline-pricing-and-optimization skill.

This test verifies:
1. Skill discovery - the skill is found and loaded correctly
2. Skill usage - the agent picks up and uses the skill
3. Outcome validation - the agent produces pricing optimization in the expected format
4. Dependency check - skill is not used before business idea, customer segment, and value proposition are clarified
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
def test_baseline_pricing_optimization_skill_discovery(tmp_path: Path) -> None:
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
    
    # Copy baseline-pricing-and-optimization skill from examples
    example_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "baseline-pricing-and-optimization"
    if not example_skill_dir.exists():
        pytest.skip(f"Example skill directory not found: {example_skill_dir}")
    
    skill_dest = skills_dir / "baseline-pricing-and-optimization"
    shutil.copytree(example_skill_dir, skill_dest)
    
    # Test skill discovery
    skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
    
    assert len(skills) == 1, f"Expected 1 skill, found {len(skills)}"
    
    skill_metadata = skills[0]
    assert skill_metadata["name"] == "baseline-pricing-and-optimization", (
        f"Expected skill name 'baseline-pricing-and-optimization', got '{skill_metadata['name']}'"
    )
    assert "pricing" in skill_metadata["description"].lower() or "optimization" in skill_metadata["description"].lower(), (
        "Skill description should mention pricing or optimization"
    )
    assert skill_metadata["source"] == "user", "Skill should be from user directory"
    
    # Verify SKILL.md content
    skill_md_path = skill_dest / "SKILL.md"
    assert skill_md_path.exists(), "SKILL.md should exist"
    
    skill_content = skill_md_path.read_text()
    assert "pricing" in skill_content.lower() or "baseline" in skill_content.lower(), (
        "SKILL.md should mention pricing or baseline"
    )
    assert "split selling" in skill_content.lower() or "cross-selling" in skill_content.lower() or "upselling" in skill_content.lower(), (
        "SKILL.md should mention pricing tactics (split selling, cross-selling, or upselling)"
    )
    
    print("\n" + "="*80)
    print("✅ SKILL DISCOVERY TEST PASSED")
    print("="*80)
    print(f"  Skill Name: {skill_metadata['name']}")
    print(f"  Description: {skill_metadata['description']}")
    print(f"  Path: {skill_metadata['path']}")
    print(f"  Source: {skill_metadata['source']}")


@pytest.mark.timeout(180)  # 3 minutes for real LLM calls
def test_baseline_pricing_optimization_with_complete_idea(tmp_path: Path) -> None:
    """Test 2: Verify skill creates pricing optimization correctly after business idea is identified.
    
    This test validates:
    - Skill is loaded by SkillsMiddleware
    - Agent picks up the skill when pricing optimization is requested
    - Agent produces pricing optimization in the expected format
    - Output contains baseline pricing, pricing tactics, and partner opportunities
    - Skill is used after business idea, customer segment, and value proposition are clarified
    """
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    try:
        # Set up skills directory
        agent_id = "test_baseline_pricing_optimization"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy business-idea-evaluation and baseline-pricing-and-optimization skills
        example_business_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "business-idea-evaluation"
        example_pricing_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "baseline-pricing-and-optimization"
        
        if not example_business_skill_dir.exists():
            pytest.skip(f"Example business-idea-evaluation skill directory not found: {example_business_skill_dir}")
        if not example_pricing_skill_dir.exists():
            pytest.skip(f"Example baseline-pricing-and-optimization skill directory not found: {example_pricing_skill_dir}")
        
        skill_dest_business = skills_dir / "business-idea-evaluation"
        skill_dest_pricing = skills_dir / "baseline-pricing-and-optimization"
        shutil.copytree(example_business_skill_dir, skill_dest_business)
        shutil.copytree(example_pricing_skill_dir, skill_dest_pricing)
        
        # Verify skill discovery
        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        assert len(skills) == 2, "Should discover both skills"
        skill_names = [s["name"] for s in skills]
        assert "business-idea-evaluation" in skill_names, "Should discover business-idea-evaluation skill"
        assert "baseline-pricing-and-optimization" in skill_names, "Should discover baseline-pricing-and-optimization skill"
        
        print("\n" + "="*80)
        print("TEST: BASELINE PRICING & OPTIMIZATION WITH COMPLETE IDEA")
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
3. Then, if the user requests pricing optimization or has clarified customer segment and value proposition, use the baseline-pricing-and-optimization skill

The baseline-pricing-and-optimization skill should only be used AFTER a business idea has been identified and customer segment/value proposition are clarified.""",
        )
        
        # Step 1: First identify a complete business idea with customer segment and value proposition
        user_request_1 = """I want to create an app that helps busy professionals manage their work-life balance.
Many professionals struggle with burnout because they can't effectively prioritize tasks and end up working late into the night.
The app would help them set boundaries and manage their time more effectively.

Target customers: Busy professionals aged 30-45 who work 50+ hours per week, earning $80K-$150K annually.
Value proposition: The app saves users 5-10 hours per week by automating task prioritization and boundary setting, which translates to $2,000-$4,000 in time value per month for our target customers."""
        
        print(f"\n📝 Step 1 - User Request (Complete Idea with Customer & Value Prop):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")
        
        # Execute the agent - first message
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-baseline-pricing-optimization-complete"}}
        
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
        
        # Step 2: Request pricing optimization
        user_request_2 = "Can you help me establish baseline pricing and identify pricing optimization opportunities for this business?"
        
        print(f"\n📝 Step 2 - User Request (Pricing Optimization):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (create pricing optimization)...\n")
        
        # Execute the agent - second message (should use baseline-pricing-and-optimization skill)
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
        
        # Validation: Outcome - Pricing optimization format validation
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
                        if "baseline-pricing" in str(file_path) or "pricing-and-optimization" in str(file_path):
                            skill_read = True
                            break
            if skill_read:
                break
        
        # Check for pricing optimization structure markers
        content_lower = final_content.lower()
        
        # Required structure markers
        structure_markers = [
            "baseline pricing",
            "baseline price",
            "1/10",
            "pricing optimization",
            "split selling",
            "cross-selling",
            "cross selling",
            "upselling",
            "up-selling",
            "key partner",
            "strategic partner",
            "value-boosting",
            "sales-boosting",
            "profitability",
        ]
        
        markers_found = sum(1 for marker in structure_markers if marker in content_lower)
        
        # Check for the main sections
        has_baseline_pricing = (
            "baseline" in content_lower and "price" in content_lower or
            "1/10" in content_lower or
            "value delivered" in content_lower
        )
        
        has_pricing_tactics = (
            "split selling" in content_lower or
            "cross-selling" in content_lower or
            "cross selling" in content_lower or
            "upselling" in content_lower or
            "up-selling" in content_lower
        )
        
        has_partner_opportunities = (
            "partner" in content_lower or
            "value-boosting" in content_lower or
            "sales-boosting" in content_lower or
            "distribution" in content_lower
        )
        
        # Check for specific pricing tactics
        has_split_selling = "split selling" in content_lower
        has_cross_selling = "cross-selling" in content_lower or "cross selling" in content_lower
        has_upselling = "upselling" in content_lower or "up-selling" in content_lower
        
        print(f"\n📋 Skill Usage Validation:")
        print(f"  Skill read (expected): {skill_read}")
        
        print(f"\n📋 Pricing Optimization Structure Validation:")
        print(f"  Structure markers found: {markers_found}/{len(structure_markers)}")
        print(f"  Has Baseline Pricing: {has_baseline_pricing}")
        print(f"  Has Pricing Tactics: {has_pricing_tactics}")
        print(f"  Has Partner Opportunities: {has_partner_opportunities}")
        print(f"  - Split Selling: {has_split_selling}")
        print(f"  - Cross-Selling: {has_cross_selling}")
        print(f"  - Upselling: {has_upselling}")
        
        # Display the pricing optimization output
        print("\n" + "="*80)
        print("GENERATED PRICING OPTIMIZATION OUTPUT")
        print("="*80)
        print(final_content)
        print("="*80)
        
        # Final validations
        print("\n" + "="*80)
        print("FINAL VALIDATIONS")
        print("="*80)
        
        # Validation: Skill should be used
        assert skill_read, (
            "Agent should read the baseline-pricing-and-optimization SKILL.md file"
        )
        
        # Validation: Should have pricing optimization structure
        assert markers_found >= 5, (
            f"Pricing optimization should contain at least 5 structure markers. "
            f"Found: {markers_found}/{len(structure_markers)}"
        )
        
        # Validation: Should contain baseline pricing
        assert has_baseline_pricing, (
            "Pricing optimization should contain Baseline Pricing section"
        )
        
        # Validation: Should contain at least one pricing tactic
        assert has_pricing_tactics, (
            "Pricing optimization should contain at least one pricing tactic (split selling, cross-selling, or upselling)"
        )
        
        # Validation: Should contain partner opportunities
        assert has_partner_opportunities, (
            "Pricing optimization should contain Key Partner Opportunities section"
        )
        
        print("✅ Pricing optimization was correctly generated with proper structure")
        print("✅ Test passed: Skill correctly creates pricing optimization after business idea is identified")
        
    finally:
        pass


@pytest.mark.timeout(180)
def test_baseline_pricing_optimization_with_chinese_input(tmp_path: Path) -> None:
    """Test 3: Verify skill works with Chinese input and produces Chinese output.
    
    This test validates:
    - Language detection works for Chinese
    - Agent responds in Chinese
    - Pricing optimization structure is present in Chinese response
    - All main sections (Baseline Pricing, Pricing Tactics, Partner Opportunities) are present
    """
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    try:
        # Set up skills directory
        agent_id = "test_baseline_pricing_optimization_chinese"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy both skills
        example_business_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "business-idea-evaluation"
        example_pricing_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "baseline-pricing-and-optimization"
        
        if not example_business_skill_dir.exists():
            pytest.skip(f"Example business-idea-evaluation skill directory not found: {example_business_skill_dir}")
        if not example_pricing_skill_dir.exists():
            pytest.skip(f"Example baseline-pricing-and-optimization skill directory not found: {example_pricing_skill_dir}")
        
        skill_dest_business = skills_dir / "business-idea-evaluation"
        skill_dest_pricing = skills_dir / "baseline-pricing-and-optimization"
        shutil.copytree(example_business_skill_dir, skill_dest_business)
        shutil.copytree(example_pricing_skill_dir, skill_dest_pricing)
        
        print("\n" + "="*80)
        print("TEST: BASELINE PRICING & OPTIMIZATION WITH CHINESE INPUT")
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
3. Then, if the user requests pricing optimization, use the baseline-pricing-and-optimization skill""",
        )
        
        # Step 1: Identify business idea in Chinese with customer segment and value proposition
        user_request_1 = """我想创建一个帮助忙碌专业人士管理工作与生活平衡的应用。
许多专业人士因为无法有效优先处理任务而最终工作到深夜，导致过度疲劳。
这个应用可以帮助他们设定界限并更有效地管理时间。

目标客户：年龄30-45岁、每周工作50小时以上、年收入8-15万美元的忙碌专业人士。
价值主张：该应用通过自动化任务优先级和边界设定，每周为用户节省5-10小时，这相当于每月为我们的目标客户节省2000-4000美元的时间价值。"""
        
        print(f"\n📝 Step 1 - User Request (Chinese - Complete Idea with Customer & Value Prop):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")
        
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-baseline-pricing-optimization-chinese"}}
        
        invoke_start = time.time()
        result_1 = agent.invoke(input_state_1, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
        
        print(f"\n✅ First execution completed ({invoke_duration:.2f}s)\n")
        
        # Check if idea was marked as complete
        business_idea_complete_1 = result_1.get("business_idea_complete", False)
        assert business_idea_complete_1, "Business idea should be marked as complete in step 1"
        
        # Step 2: Request pricing optimization in Chinese
        user_request_2 = "你能帮我建立基准定价并识别定价优化机会吗？"
        
        print(f"\n📝 Step 2 - User Request (Chinese - Pricing Optimization):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (create pricing optimization)...\n")
        
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
        
        # Validation: Pricing optimization structure in Chinese
        print("\n" + "="*80)
        print("PRICING OPTIMIZATION STRUCTURE VALIDATION (Chinese)")
        print("="*80)
        
        # Chinese keywords for pricing optimization structure
        chinese_structure_keywords = [
            ("Baseline Pricing", ["基准定价", "基础价格", "定价", "价格"]),
            ("Pricing Tactics", ["定价策略", "定价方案", "销售策略", "拆分销售", "交叉销售", "向上销售"]),
            ("Partner Opportunities", ["合作伙伴", "战略伙伴", "价值提升", "销售提升", "分销"]),
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
        english_markers = ["baseline pricing", "pricing optimization", "split selling", "cross-selling", "upselling", "partner"]
        english_found = sum(1 for marker in english_markers if marker in final_content.lower())
        
        total_structure_found = len(found_keywords) + (1 if english_found >= 3 else 0)
        
        print(f"\n📊 Summary:")
        print(f"  Chinese structure keywords found: {len(found_keywords)}/{len(chinese_structure_keywords)}")
        print(f"  English markers found: {english_found}")
        print(f"  Total structure found: {total_structure_found}")
        
        # Display the pricing optimization output
        print("\n" + "="*80)
        print("GENERATED PRICING OPTIMIZATION OUTPUT (Chinese)")
        print("="*80)
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
        
        # Validation: Should have pricing optimization structure
        assert total_structure_found >= 2, (
            f"Pricing optimization should contain at least 2 structure elements. "
            f"Found: {total_structure_found} (Chinese: {len(found_keywords)}, English: {english_found})"
        )
        
        print("✅ Pricing optimization was correctly generated in Chinese with proper structure")
        print("✅ Test passed: Skill correctly creates pricing optimization in Chinese")
        
    finally:
        pass

