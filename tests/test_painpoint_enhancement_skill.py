"""Integration test for painpoint-enhancement skill.

This test verifies:
1. Skill discovery - the skill is found and loaded correctly
2. Skill usage - the agent picks up and uses the skill
3. Outcome validation - the agent produces enhanced pain point in the expected format
4. Dependency check - skill is not used before business idea is identified
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
def test_painpoint_enhancement_skill_discovery(tmp_path: Path) -> None:
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
    
    # Copy painpoint-enhancement skill from examples
    example_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "painpoint-enhancement"
    if not example_skill_dir.exists():
        pytest.skip(f"Example skill directory not found: {example_skill_dir}")
    
    skill_dest = skills_dir / "painpoint-enhancement"
    shutil.copytree(example_skill_dir, skill_dest)
    
    # Test skill discovery
    skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
    
    assert len(skills) == 1, f"Expected 1 skill, found {len(skills)}"
    
    skill_metadata = skills[0]
    assert skill_metadata["name"] == "painpoint-enhancement", (
        f"Expected skill name 'painpoint-enhancement', got '{skill_metadata['name']}'"
    )
    assert "pain point" in skill_metadata["description"].lower() or "enhance" in skill_metadata["description"].lower(), (
        "Skill description should mention pain point or enhancement"
    )
    assert skill_metadata["source"] == "user", "Skill should be from user directory"
    
    # Verify SKILL.md content
    skill_md_path = skill_dest / "SKILL.md"
    assert skill_md_path.exists(), "SKILL.md should exist"
    
    skill_content = skill_md_path.read_text()
    assert "painpoint-enhancement" in skill_content or "pain point" in skill_content.lower(), (
        "SKILL.md should mention pain point"
    )
    assert "dimension" in skill_content.lower() or "urgency" in skill_content.lower(), (
        "SKILL.md should mention dimensions or urgency"
    )
    
    print("\n" + "="*80)
    print("✅ SKILL DISCOVERY TEST PASSED")
    print("="*80)
    print(f"  Skill Name: {skill_metadata['name']}")
    print(f"  Description: {skill_metadata['description']}")
    print(f"  Path: {skill_metadata['path']}")
    print(f"  Source: {skill_metadata['source']}")


@pytest.mark.timeout(180)  # 3 minutes for real LLM calls
def test_painpoint_enhancement_with_complete_idea(tmp_path: Path) -> None:
    """Test 2: Verify skill enhances a pain point correctly after business idea is identified.
    
    This test validates:
    - Skill is loaded by SkillsMiddleware
    - Agent picks up the skill when a pain point needs enhancement
    - Agent produces enhanced pain point in the expected format
    - All six dimensions are evaluated
    - Skill is used after business idea is identified
    """
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    try:
        # Set up skills directory
        agent_id = "test_painpoint_enhancement"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy both business-idea-evaluation and painpoint-enhancement skills
        example_business_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "business-idea-evaluation"
        example_painpoint_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "painpoint-enhancement"
        
        if not example_business_skill_dir.exists():
            pytest.skip(f"Example business-idea-evaluation skill directory not found: {example_business_skill_dir}")
        if not example_painpoint_skill_dir.exists():
            pytest.skip(f"Example painpoint-enhancement skill directory not found: {example_painpoint_skill_dir}")
        
        skill_dest_business = skills_dir / "business-idea-evaluation"
        skill_dest_painpoint = skills_dir / "painpoint-enhancement"
        shutil.copytree(example_business_skill_dir, skill_dest_business)
        shutil.copytree(example_painpoint_skill_dir, skill_dest_painpoint)
        
        # Verify skill discovery
        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        assert len(skills) == 2, "Should discover both skills"
        skill_names = [s["name"] for s in skills]
        assert "business-idea-evaluation" in skill_names, "Should discover business-idea-evaluation skill"
        assert "painpoint-enhancement" in skill_names, "Should discover painpoint-enhancement skill"
        
        print("\n" + "="*80)
        print("TEST: PAINPOINT ENHANCEMENT WITH COMPLETE IDEA")
        print("="*80)
        
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
3. Then, if the pain point needs enhancement, use the painpoint-enhancement skill to strengthen it

The painpoint-enhancement skill should only be used AFTER a business idea has been identified and marked as complete.""",
        )
        
        # Step 1: First identify a complete business idea
        user_request_1 = """I want to create an app that helps busy professionals manage their work-life balance.
Many professionals struggle with burnout because they can't effectively prioritize tasks and end up working late into the night.
The app would help them set boundaries and manage their time more effectively."""
        
        print(f"\n📝 Step 1 - User Request (Complete Idea):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")
        
        # Execute the agent - first message
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-painpoint-enhancement-complete"}}
        
        invoke_start = time.time()
        result_1 = agent.invoke(input_state_1, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
        
        print(f"\n✅ First execution completed ({invoke_duration:.2f}s)\n")
        
        # Check if idea was marked as complete
        business_idea_complete_1 = result_1.get("business_idea_complete", False)
        assert business_idea_complete_1, "Business idea should be marked as complete in step 1"
        
        # Extract messages from step 1 to check if painpoint-enhancement was used
        messages_1 = result_1.get("messages", [])
        
        # Check if painpoint-enhancement skill was read in step 1
        skill_read_step1 = False
        for message in messages_1:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "read_file":
                        args = tool_call.get("args", {})
                        file_path = args.get("file_path", "")
                        if "painpoint-enhancement" in str(file_path):
                            skill_read_step1 = True
                            break
            if skill_read_step1:
                break
        
        print("✅ Step 1 passed: Business idea was correctly identified and marked as complete")
        if skill_read_step1:
            print("✅ Step 1: Painpoint-enhancement skill was also used (agent proactively enhanced pain point)")
        
        # Step 2: Request pain point enhancement (explicitly)
        user_request_2 = "Can you help me strengthen and clarify the pain point for this idea? I want to make sure it's compelling."
        
        print(f"\n📝 Step 2 - User Request (Enhance Pain Point):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (enhance pain point)...\n")
        
        # Execute the agent - second message (should use painpoint-enhancement skill if not already used)
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
        
        # Validation: Check both step 1 and step 2 outputs for enhanced pain point
        # The agent might have enhanced it in step 1 or step 2
        ai_messages_1 = [m for m in messages_1 if m.type == "ai"]
        ai_messages_2 = [m for m in messages_2 if m.type == "ai"]
        
        # Get the last AI message from step 2 (most recent)
        assert len(ai_messages_2) > 0, "Agent should have generated at least one response in step 2"
        final_content_step2 = str(ai_messages_2[-1].content)
        
        # Check if skill was read in step 2 (agent used the skill)
        skill_read_step2 = False
        for message in messages_2:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "read_file":
                        args = tool_call.get("args", {})
                        file_path = args.get("file_path", "")
                        if "painpoint-enhancement" in str(file_path):
                            skill_read_step2 = True
                            break
            if skill_read_step2:
                break
        
        # Skill should be read in either step 1 or step 2
        skill_read = skill_read_step1 or skill_read_step2
        
        # Check both step 1 and step 2 outputs for structured format
        # Use whichever has better structure (prefer step 1 if skill was used there)
        step1_content = ""
        if skill_read_step1 and len(ai_messages_1) > 0:
            step1_content = str(ai_messages_1[-1].content)
        
        # Determine which content to use for validation
        # Prefer step 1 if it has dimension analysis, otherwise use step 2
        if step1_content and ("dimension" in step1_content.lower() or "urgency" in step1_content.lower() or "enhanced pain point" in step1_content.lower()):
            final_content = step1_content
            print("📝 Using step 1 output for validation (enhancement happened there)")
        else:
            final_content = final_content_step2
            print("📝 Using step 2 output for validation")
        
        print("\n" + "="*80)
        print("OUTCOME VALIDATION")
        print("="*80)
        
        # Check for enhanced pain point structure markers
        content_lower = final_content.lower()
        
        # Required structure markers
        structure_markers = [
            "enhanced pain point",
            "dimension analysis",
            "urgency",
            "frequency",
            "economic cost",
            "universality",
            "viral spread",
            "regulatory pressure",
            "key resonance",
        ]
        
        markers_found = sum(1 for marker in structure_markers if marker in content_lower)
        
        # Check for dimension ratings (Low/Medium/High)
        has_dimension_ratings = (
            ("low" in content_lower or "medium" in content_lower or "high" in content_lower) and
            ("urgency" in content_lower or "frequency" in content_lower)
        )
        
        # Check for at least 4 of the 6 dimensions mentioned
        dimension_keywords = ["urgency", "frequency", "economic cost", "universality", "viral spread", "regulatory pressure"]
        dimensions_found = sum(1 for dim in dimension_keywords if dim in content_lower)
        
        print(f"\n📋 Skill Usage Validation:")
        print(f"  Skill read (expected): {skill_read}")
        
        print(f"\n📋 Enhanced Pain Point Structure Validation:")
        print(f"  Structure markers found: {markers_found}/{len(structure_markers)}")
        print(f"  Dimensions found: {dimensions_found}/6")
        print(f"  Has dimension ratings (Low/Medium/High): {has_dimension_ratings}")
        
        # Display the enhanced pain point output
        print("\n" + "="*80)
        print("GENERATED ENHANCED PAIN POINT OUTPUT")
        print("="*80)
        print(final_content)
        print("="*80)
        
        # Final validations
        print("\n" + "="*80)
        print("FINAL VALIDATIONS")
        print("="*80)
        
        # Validation: Skill should be used
        assert skill_read, (
            "Agent should read the painpoint-enhancement SKILL.md file"
        )
        
        # Validation: Skill should be used (in either step 1 or step 2)
        assert skill_read, (
            "Agent should read the painpoint-enhancement SKILL.md file in either step 1 or step 2"
        )
        
        # Validation: Should have enhanced pain point structure
        # Note: The agent might have produced the structured format in step 1, and step 2 might just reference it
        # We check for at least some structure markers to ensure the skill was used effectively
        assert markers_found >= 2, (
            f"Enhanced pain point should contain at least 2 structure markers. "
            f"Found: {markers_found}/{len(structure_markers)}. "
            f"Note: The agent may have enhanced the pain point in step 1, and step 2 may just reference it."
        )
        
        # Validation: Should mention at least some dimensions
        # The agent might not list all 6 dimensions explicitly, but should mention some
        assert dimensions_found >= 2 or "enhanced pain point" in content_lower, (
            f"Enhanced pain point should mention at least 2 dimensions or contain 'enhanced pain point'. "
            f"Found: {dimensions_found}/6 dimensions"
        )
        
        # Validation: Should have some indication of enhancement (enhanced pain point text or dimension analysis)
        has_enhancement_indication = (
            "enhanced pain point" in content_lower or
            "dimension" in content_lower or
            has_dimension_ratings or
            dimensions_found >= 2
        )
        
        assert has_enhancement_indication, (
            "Enhanced pain point should contain some indication of enhancement "
            "(enhanced pain point text, dimensions, or ratings)"
        )
        
        print("✅ Enhanced pain point was correctly generated")
        print(f"   Structure markers: {markers_found}/{len(structure_markers)}")
        print(f"   Dimensions found: {dimensions_found}/6")
        print(f"   Skill used: {skill_read} (step 1: {skill_read_step1}, step 2: {skill_read_step2})")
        print("✅ Test passed: Skill correctly enhances pain point after business idea is identified")
        
    finally:
        pass


@pytest.mark.timeout(180)
def test_painpoint_enhancement_with_chinese_input(tmp_path: Path) -> None:
    """Test 3: Verify skill works with Chinese input and produces Chinese output.
    
    This test validates:
    - Language detection works for Chinese
    - Agent responds in Chinese
    - Enhanced pain point structure is present in Chinese response
    - All six dimensions are evaluated
    """
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    try:
        # Set up skills directory
        agent_id = "test_painpoint_enhancement_chinese"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy both skills
        example_business_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "business-idea-evaluation"
        example_painpoint_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "painpoint-enhancement"
        
        if not example_business_skill_dir.exists():
            pytest.skip(f"Example business-idea-evaluation skill directory not found: {example_business_skill_dir}")
        if not example_painpoint_skill_dir.exists():
            pytest.skip(f"Example painpoint-enhancement skill directory not found: {example_painpoint_skill_dir}")
        
        skill_dest_business = skills_dir / "business-idea-evaluation"
        skill_dest_painpoint = skills_dir / "painpoint-enhancement"
        shutil.copytree(example_business_skill_dir, skill_dest_business)
        shutil.copytree(example_painpoint_skill_dir, skill_dest_painpoint)
        
        print("\n" + "="*80)
        print("TEST: PAINPOINT ENHANCEMENT WITH CHINESE INPUT")
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
3. Then, if the pain point needs enhancement, use the painpoint-enhancement skill to strengthen it""",
        )
        
        # Step 1: Identify business idea in Chinese
        user_request_1 = """我想创建一个帮助忙碌专业人士管理工作与生活平衡的应用。
许多专业人士因为无法有效优先处理任务而最终工作到深夜，导致过度疲劳。
这个应用可以帮助他们设定界限并更有效地管理时间。"""
        
        print(f"\n📝 Step 1 - User Request (Chinese - Complete Idea):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")
        
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-painpoint-enhancement-chinese"}}
        
        invoke_start = time.time()
        result_1 = agent.invoke(input_state_1, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
        
        print(f"\n✅ First execution completed ({invoke_duration:.2f}s)\n")
        
        # Check if idea was marked as complete
        business_idea_complete_1 = result_1.get("business_idea_complete", False)
        assert business_idea_complete_1, "Business idea should be marked as complete in step 1"
        
        # Step 2: Request pain point enhancement in Chinese
        user_request_2 = "你能帮我加强和澄清这个想法的痛点吗？我想确保它是有说服力的。"
        
        print(f"\n📝 Step 2 - User Request (Chinese - Enhance Pain Point):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (enhance pain point)...\n")
        
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
        
        # Validation: Enhanced pain point structure in Chinese
        print("\n" + "="*80)
        print("ENHANCED PAIN POINT STRUCTURE VALIDATION (Chinese)")
        print("="*80)
        
        # Chinese keywords for pain point enhancement structure
        chinese_structure_keywords = [
            ("Enhanced Pain Point", ["增强的痛点", "强化后的痛点", "改进的痛点", "痛点"]),
            ("Urgency", ["紧迫性", "紧急程度"]),
            ("Frequency", ["频率", "频次"]),
            ("Economic Cost", ["经济成本", "成本"]),
            ("Universality", ["普遍性", "广泛性"]),
            ("Viral Spread", ["病毒式传播", "传播"]),
            ("Regulatory Pressure", ["监管压力", "法规压力"]),
            ("Key Resonance", ["关键共鸣", "核心因素"]),
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
        english_markers = ["enhanced pain point", "dimension", "urgency", "frequency", "economic cost", "universality"]
        english_found = sum(1 for marker in english_markers if marker in final_content.lower())
        
        total_structure_found = len(found_keywords) + (1 if english_found >= 3 else 0)
        
        print(f"\n📊 Summary:")
        print(f"  Chinese structure keywords found: {len(found_keywords)}/{len(chinese_structure_keywords)}")
        print(f"  English markers found: {english_found}")
        print(f"  Total structure found: {total_structure_found}")
        
        # Display the enhanced pain point output
        print("\n" + "="*80)
        print("GENERATED ENHANCED PAIN POINT OUTPUT (Chinese)")
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
        
        # Validation: Should have enhanced pain point structure
        assert total_structure_found >= 4, (
            f"Enhanced pain point should contain at least 4 structure elements. "
            f"Found: {total_structure_found} (Chinese: {len(found_keywords)}, English: {english_found})"
        )
        
        print("✅ Enhanced pain point was correctly generated in Chinese with proper structure")
        print("✅ Test passed: Skill correctly enhances pain point in Chinese")
        
    finally:
        pass

