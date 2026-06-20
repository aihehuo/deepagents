"""Integration test for 60s-pitch-creation skill.

This test verifies:
1. Skill discovery - the skill is found and loaded correctly
2. Skill usage - the agent picks up and uses the skill
3. Outcome validation - the agent produces a 60-second pitch in the expected format
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
def test_60s_pitch_creation_skill_discovery(tmp_path: Path) -> None:
    """Test 1: Verify skill discovery and metadata parsing.
<<<<<<< HEAD
    
=======

>>>>>>> main
    This test validates:
    - Skill is discovered from the examples directory
    - Metadata (name, description) is correctly parsed from YAML frontmatter
    - Skill path is correctly resolved
    """
    repo_root = Path(__file__).parent.parent
<<<<<<< HEAD
    
    # Set up skills directory
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy 60s-pitch-creation skill from examples
    example_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "60s-pitch-creation"
    if not example_skill_dir.exists():
        pytest.skip(f"Example skill directory not found: {example_skill_dir}")
    
    skill_dest = skills_dir / "60s-pitch-creation"
    shutil.copytree(example_skill_dir, skill_dest)
    
    # Test skill discovery
    skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
    
    assert len(skills) == 1, f"Expected 1 skill, found {len(skills)}"
    
=======

    # Set up skills directory
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Copy 60s-pitch-creation skill from examples
    example_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "60s-pitch-creation"
    if not example_skill_dir.exists():
        pytest.skip(f"Example skill directory not found: {example_skill_dir}")

    skill_dest = skills_dir / "60s-pitch-creation"
    shutil.copytree(example_skill_dir, skill_dest)

    # Test skill discovery
    skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)

    assert len(skills) == 1, f"Expected 1 skill, found {len(skills)}"

>>>>>>> main
    skill_metadata = skills[0]
    assert skill_metadata["name"] == "60s-pitch-creation", (
        f"Expected skill name '60s-pitch-creation', got '{skill_metadata['name']}'"
    )
    assert "pitch" in skill_metadata["description"].lower() or "60" in skill_metadata["description"].lower(), (
        "Skill description should mention pitch or 60-second"
    )
    assert skill_metadata["source"] == "user", "Skill should be from user directory"
<<<<<<< HEAD
    
    # Verify SKILL.md content
    skill_md_path = skill_dest / "SKILL.md"
    assert skill_md_path.exists(), "SKILL.md should exist"
    
=======

    # Verify SKILL.md content
    skill_md_path = skill_dest / "SKILL.md"
    assert skill_md_path.exists(), "SKILL.md should exist"

>>>>>>> main
    skill_content = skill_md_path.read_text()
    assert "pitch" in skill_content.lower() or "60" in skill_content.lower(), (
        "SKILL.md should mention pitch or 60-second"
    )
    assert "pain point" in skill_content.lower() or "advantage" in skill_content.lower(), (
        "SKILL.md should mention pain point or advantage"
    )
<<<<<<< HEAD
    
=======

>>>>>>> main
    print("\n" + "="*80)
    print("✅ SKILL DISCOVERY TEST PASSED")
    print("="*80)
    print(f"  Skill Name: {skill_metadata['name']}")
    print(f"  Description: {skill_metadata['description']}")
    print(f"  Path: {skill_metadata['path']}")
    print(f"  Source: {skill_metadata['source']}")


@pytest.mark.timeout(180)  # 3 minutes for real LLM calls
def test_60s_pitch_creation_with_complete_idea(tmp_path: Path) -> None:
    """Test 2: Verify skill creates a 60-second pitch correctly after business idea is identified.
<<<<<<< HEAD
    
=======

>>>>>>> main
    This test validates:
    - Skill is loaded by SkillsMiddleware
    - Agent picks up the skill when a pitch is requested
    - Agent produces a 60-second pitch in the expected format
    - Pitch contains all three required parts (Pain Point Resonance, Team Advantage, CTA)
    - Skill is used after business idea is identified
    """
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    try:
        # Set up skills directory
        agent_id = "test_60s_pitch_creation"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD
        
        # Copy both business-idea-evaluation and 60s-pitch-creation skills
        example_business_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "business-idea-evaluation"
        example_pitch_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "60s-pitch-creation"
        
=======

        # Copy both business-idea-evaluation and 60s-pitch-creation skills
        example_business_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "business-idea-evaluation"
        example_pitch_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "60s-pitch-creation"

>>>>>>> main
        if not example_business_skill_dir.exists():
            pytest.skip(f"Example business-idea-evaluation skill directory not found: {example_business_skill_dir}")
        if not example_pitch_skill_dir.exists():
            pytest.skip(f"Example 60s-pitch-creation skill directory not found: {example_pitch_skill_dir}")
<<<<<<< HEAD
        
=======

>>>>>>> main
        skill_dest_business = skills_dir / "business-idea-evaluation"
        skill_dest_pitch = skills_dir / "60s-pitch-creation"
        shutil.copytree(example_business_skill_dir, skill_dest_business)
        shutil.copytree(example_pitch_skill_dir, skill_dest_pitch)
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Verify skill discovery
        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        assert len(skills) == 2, "Should discover both skills"
        skill_names = [s["name"] for s in skills]
        assert "business-idea-evaluation" in skill_names, "Should discover business-idea-evaluation skill"
        assert "60s-pitch-creation" in skill_names, "Should discover 60s-pitch-creation skill"
<<<<<<< HEAD
        
        print("\n" + "="*80)
        print("TEST: 60-SECOND PITCH CREATION WITH COMPLETE IDEA")
        print("="*80)
        
        # Create model (DeepSeek default; switch to Qwen via BC_API_PROVIDER=qwen + QWEN_* env vars)
        model = create_test_model(cfg=cfg)
        
        # Create agent with SkillsMiddleware and BusinessIdeaTrackerMiddleware
        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)
        
        # Create checkpointer for state persistence
        checkpointer = MemorySaver()
        
=======

        print("\n" + "="*80)
        print("TEST: 60-SECOND PITCH CREATION WITH COMPLETE IDEA")
        print("="*80)

        # Create model (DeepSeek default; switch to Qwen via BC_API_PROVIDER=qwen + QWEN_* env vars)
        model = create_test_model(cfg=cfg)

        # Create agent with SkillsMiddleware and BusinessIdeaTrackerMiddleware
        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)

        # Create checkpointer for state persistence
        checkpointer = MemorySaver()

>>>>>>> main
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
3. Then, if the user requests a pitch or needs to present their idea, use the 60s-pitch-creation skill

The 60s-pitch-creation skill should only be used AFTER a business idea has been identified and marked as complete.""",
        )
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Step 1: First identify a complete business idea with team advantages
        user_request_1 = """I want to create an app that helps busy professionals manage their work-life balance.
Many professionals struggle with burnout because they can't effectively prioritize tasks and end up working late into the night.
The app would help them set boundaries and manage their time more effectively.

Our team has 10 years of experience in productivity software development, and we have direct access to a network of 50,000 professionals through our existing consulting business."""
<<<<<<< HEAD
        
        print(f"\n📝 Step 1 - User Request (Complete Idea with Team Info):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")
        
        # Execute the agent - first message
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-60s-pitch-creation-complete"}}
        
=======

        print(f"\n📝 Step 1 - User Request (Complete Idea with Team Info):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")

        # Execute the agent - first message
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-60s-pitch-creation-complete"}}

>>>>>>> main
        invoke_start = time.time()
        result_1 = agent.invoke(input_state_1, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
<<<<<<< HEAD
        
        print(f"\n✅ First execution completed ({invoke_duration:.2f}s)\n")
        
        # Check if idea was marked as complete
        business_idea_complete_1 = result_1.get("business_idea_complete", False)
        assert business_idea_complete_1, "Business idea should be marked as complete in step 1"
        
        print("✅ Step 1 passed: Business idea was correctly identified and marked as complete")
        
        # Step 2: Request 60-second pitch creation
        user_request_2 = "Can you help me create a 60-second pitch for this idea? I need to present it to potential investors."
        
        print(f"\n📝 Step 2 - User Request (Create Pitch):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (create 60-second pitch)...\n")
        
        # Execute the agent - second message (should use 60s-pitch-creation skill)
        messages_so_far = result_1.get("messages", [])
        input_state_2 = {"messages": messages_so_far + [HumanMessage(content=user_request_2)]}
        
=======

        print(f"\n✅ First execution completed ({invoke_duration:.2f}s)\n")

        # Check if idea was marked as complete
        business_idea_complete_1 = result_1.get("business_idea_complete", False)
        assert business_idea_complete_1, "Business idea should be marked as complete in step 1"

        print("✅ Step 1 passed: Business idea was correctly identified and marked as complete")

        # Step 2: Request 60-second pitch creation
        user_request_2 = "Can you help me create a 60-second pitch for this idea? I need to present it to potential investors."

        print(f"\n📝 Step 2 - User Request (Create Pitch):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (create 60-second pitch)...\n")

        # Execute the agent - second message (should use 60s-pitch-creation skill)
        messages_so_far = result_1.get("messages", [])
        input_state_2 = {"messages": messages_so_far + [HumanMessage(content=user_request_2)]}

>>>>>>> main
        invoke_start = time.time()
        result_2 = agent.invoke(input_state_2, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
<<<<<<< HEAD
        
        print(f"\n✅ Second execution completed ({invoke_duration:.2f}s)\n")
        
        # Extract messages
        messages_2 = result_2.get("messages", [])
        print(f"📊 Messages: {len(messages_2)} total\n")
        
        # Print timing summary
        timing_middleware.print_summary()
        
        # Validation: Outcome - 60-second pitch format validation
        ai_messages = [m for m in messages_2 if m.type == "ai"]
        assert len(ai_messages) > 0, "Agent should have generated at least one response"
        
        final_content = str(ai_messages[-1].content)
        
        print("\n" + "="*80)
        print("OUTCOME VALIDATION")
        print("="*80)
        
=======

        print(f"\n✅ Second execution completed ({invoke_duration:.2f}s)\n")

        # Extract messages
        messages_2 = result_2.get("messages", [])
        print(f"📊 Messages: {len(messages_2)} total\n")

        # Print timing summary
        timing_middleware.print_summary()

        # Validation: Outcome - 60-second pitch format validation
        ai_messages = [m for m in messages_2 if m.type == "ai"]
        assert len(ai_messages) > 0, "Agent should have generated at least one response"

        final_content = str(ai_messages[-1].content)

        print("\n" + "="*80)
        print("OUTCOME VALIDATION")
        print("="*80)

>>>>>>> main
        # Check if skill was read (agent used the skill)
        skill_read = False
        for message in messages_2:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "read_file":
                        args = tool_call.get("args", {})
                        file_path = args.get("file_path", "")
                        if "60s-pitch-creation" in str(file_path) or "60-second" in str(file_path).lower():
                            skill_read = True
                            break
            if skill_read:
                break
<<<<<<< HEAD
        
        # Check for 60-second pitch structure markers
        content_lower = final_content.lower()
        
=======

        # Check for 60-second pitch structure markers
        content_lower = final_content.lower()

>>>>>>> main
        # Required structure markers
        structure_markers = [
            "60-second pitch",
            "60 second pitch",
            "pitch breakdown",
            "pain point resonance",
            "team advantage",
            "team advantages",
            "call to action",
            "cta",
        ]
<<<<<<< HEAD
        
        markers_found = sum(1 for marker in structure_markers if marker in content_lower)
        
=======

        markers_found = sum(1 for marker in structure_markers if marker in content_lower)

>>>>>>> main
        # Check for the three main parts of the pitch
        has_pain_point_resonance = (
            "pain point" in content_lower or
            "urgency" in content_lower or
            "frequency" in content_lower or
            "economic cost" in content_lower
        )
<<<<<<< HEAD
        
=======

>>>>>>> main
        has_team_advantage = (
            "team" in content_lower and ("advantage" in content_lower or "experience" in content_lower or "background" in content_lower) or
            "founder" in content_lower or
            "10x" in content_lower or
            "moat" in content_lower
        )
<<<<<<< HEAD
        
=======

>>>>>>> main
        has_cta = (
            "call to action" in content_lower or
            "cta" in content_lower or
            "join" in content_lower or
            "talk" in content_lower or
            "contact" in content_lower or
            "reach out" in content_lower
        )
<<<<<<< HEAD
        
        # Check pitch length (should be 120-200 words, approximately 600-1000 characters)
        word_count = len(final_content.split())
        char_count = len(final_content)
        
        print(f"\n📋 Skill Usage Validation:")
        print(f"  Skill read (expected): {skill_read}")
        
=======

        # Check pitch length (should be 120-200 words, approximately 600-1000 characters)
        word_count = len(final_content.split())
        char_count = len(final_content)

        print(f"\n📋 Skill Usage Validation:")
        print(f"  Skill read (expected): {skill_read}")

>>>>>>> main
        print(f"\n📋 60-Second Pitch Structure Validation:")
        print(f"  Structure markers found: {markers_found}/{len(structure_markers)}")
        print(f"  Has Pain Point Resonance: {has_pain_point_resonance}")
        print(f"  Has Team Advantage: {has_team_advantage}")
        print(f"  Has Call to Action (CTA): {has_cta}")
        print(f"  Pitch word count: {word_count} (target: 120-200 words)")
        print(f"  Pitch character count: {char_count} (target: ~600-1000 chars)")
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Display the pitch output
        print("\n" + "="*80)
        print("GENERATED 60-SECOND PITCH OUTPUT")
        print("="*80)
        print(final_content)
        print("="*80)
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Final validations
        print("\n" + "="*80)
        print("FINAL VALIDATIONS")
        print("="*80)
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Validation: Skill should be used
        assert skill_read, (
            "Agent should read the 60s-pitch-creation SKILL.md file"
        )
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Validation: Should have pitch structure
        assert markers_found >= 3, (
            f"60-second pitch should contain at least 3 structure markers. "
            f"Found: {markers_found}/{len(structure_markers)}"
        )
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Validation: Should contain all three main parts
        assert has_pain_point_resonance, (
            "Pitch should contain Pain Point Resonance section"
        )
        assert has_team_advantage, (
            "Pitch should contain Team Advantage section"
        )
        assert has_cta, (
            "Pitch should contain Call to Action (CTA)"
        )
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Validation: Pitch should be reasonable length (not too short, not too long)
        # Allow some flexibility - at least 50 words, not more than 500 words
        assert 50 <= word_count <= 500, (
            f"Pitch should be between 50-500 words. Found: {word_count} words"
        )
<<<<<<< HEAD
        
        print("✅ 60-second pitch was correctly generated with proper structure")
        print("✅ Test passed: Skill correctly creates 60-second pitch after business idea is identified")
        
=======

        print("✅ 60-second pitch was correctly generated with proper structure")
        print("✅ Test passed: Skill correctly creates 60-second pitch after business idea is identified")

>>>>>>> main
    finally:
        pass


@pytest.mark.timeout(180)
def test_60s_pitch_creation_with_chinese_input(tmp_path: Path) -> None:
    """Test 3: Verify skill works with Chinese input and produces Chinese output.
<<<<<<< HEAD
    
=======

>>>>>>> main
    This test validates:
    - Language detection works for Chinese
    - Agent responds in Chinese
    - 60-second pitch structure is present in Chinese response
    - All three parts (Pain Point Resonance, Team Advantage, CTA) are present
    """
    repo_root = Path(__file__).parent.parent
    cfg = load_test_model_config(repo_root=repo_root)
    try:
        # Set up skills directory
        agent_id = "test_60s_pitch_creation_chinese"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD
        
        # Copy both skills
        example_business_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "business-idea-evaluation"
        example_pitch_skill_dir = repo_root / "libs" / "deepagents-cli" / "examples" / "skills" / "60s-pitch-creation"
        
=======

        # Copy both skills
        example_business_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "business-idea-evaluation"
        example_pitch_skill_dir = repo_root / "libs" / "cli" / "examples" / "skills" / "60s-pitch-creation"

>>>>>>> main
        if not example_business_skill_dir.exists():
            pytest.skip(f"Example business-idea-evaluation skill directory not found: {example_business_skill_dir}")
        if not example_pitch_skill_dir.exists():
            pytest.skip(f"Example 60s-pitch-creation skill directory not found: {example_pitch_skill_dir}")
<<<<<<< HEAD
        
=======

>>>>>>> main
        skill_dest_business = skills_dir / "business-idea-evaluation"
        skill_dest_pitch = skills_dir / "60s-pitch-creation"
        shutil.copytree(example_business_skill_dir, skill_dest_business)
        shutil.copytree(example_pitch_skill_dir, skill_dest_pitch)
<<<<<<< HEAD
        
        print("\n" + "="*80)
        print("TEST: 60-SECOND PITCH CREATION WITH CHINESE INPUT")
        print("="*80)
        
        model = create_test_model(cfg=cfg)
        
        # Create agent with LanguageDetectionMiddleware
        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)
        
        checkpointer = MemorySaver()
        
=======

        print("\n" + "="*80)
        print("TEST: 60-SECOND PITCH CREATION WITH CHINESE INPUT")
        print("="*80)

        model = create_test_model(cfg=cfg)

        # Create agent with LanguageDetectionMiddleware
        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)

        checkpointer = MemorySaver()

>>>>>>> main
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
3. Then, if the user requests a pitch, use the 60s-pitch-creation skill""",
        )
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Step 1: Identify business idea in Chinese with team advantages
        user_request_1 = """我想创建一个帮助忙碌专业人士管理工作与生活平衡的应用。
许多专业人士因为无法有效优先处理任务而最终工作到深夜，导致过度疲劳。
这个应用可以帮助他们设定界限并更有效地管理时间。

我们的团队在生产力软件开发方面有10年的经验，并且通过我们现有的咨询业务可以直接接触到50,000名专业人士的网络。"""
<<<<<<< HEAD
        
        print(f"\n📝 Step 1 - User Request (Chinese - Complete Idea with Team Info):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")
        
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-60s-pitch-creation-chinese"}}
        
=======

        print(f"\n📝 Step 1 - User Request (Chinese - Complete Idea with Team Info):\n{user_request_1}\n")
        print("⏳ Starting first agent execution (identify business idea)...\n")

        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        config = {"configurable": {"thread_id": "test-60s-pitch-creation-chinese"}}

>>>>>>> main
        invoke_start = time.time()
        result_1 = agent.invoke(input_state_1, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
<<<<<<< HEAD
        
        print(f"\n✅ First execution completed ({invoke_duration:.2f}s)\n")
        
        # Check if idea was marked as complete
        business_idea_complete_1 = result_1.get("business_idea_complete", False)
        assert business_idea_complete_1, "Business idea should be marked as complete in step 1"
        
        # Step 2: Request 60-second pitch creation in Chinese
        user_request_2 = "你能帮我创建一个60秒的创业路演吗？我需要向潜在投资者展示这个想法。"
        
        print(f"\n📝 Step 2 - User Request (Chinese - Create Pitch):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (create 60-second pitch)...\n")
        
        messages_so_far = result_1.get("messages", [])
        input_state_2 = {"messages": messages_so_far + [HumanMessage(content=user_request_2)]}
        
=======

        print(f"\n✅ First execution completed ({invoke_duration:.2f}s)\n")

        # Check if idea was marked as complete
        business_idea_complete_1 = result_1.get("business_idea_complete", False)
        assert business_idea_complete_1, "Business idea should be marked as complete in step 1"

        # Step 2: Request 60-second pitch creation in Chinese
        user_request_2 = "你能帮我创建一个60秒的创业路演吗？我需要向潜在投资者展示这个想法。"

        print(f"\n📝 Step 2 - User Request (Chinese - Create Pitch):\n{user_request_2}\n")
        print("⏳ Starting second agent execution (create 60-second pitch)...\n")

        messages_so_far = result_1.get("messages", [])
        input_state_2 = {"messages": messages_so_far + [HumanMessage(content=user_request_2)]}

>>>>>>> main
        invoke_start = time.time()
        result_2 = agent.invoke(input_state_2, config)
        invoke_end = time.time()
        invoke_duration = invoke_end - invoke_start
        timing_middleware.total_time = invoke_duration
<<<<<<< HEAD
        
        print(f"\n✅ Second execution completed ({invoke_duration:.2f}s)\n")
        
        messages_2 = result_2.get("messages", [])
        timing_middleware.print_summary()
        
        # Validation: Language detection
        state = result_2
        detected_language = state.get("detected_language")
        
        print("\n" + "="*80)
        print("LANGUAGE DETECTION VALIDATION")
        print("="*80)
        
=======

        print(f"\n✅ Second execution completed ({invoke_duration:.2f}s)\n")

        messages_2 = result_2.get("messages", [])
        timing_middleware.print_summary()

        # Validation: Language detection
        state = result_2
        detected_language = state.get("detected_language")

        print("\n" + "="*80)
        print("LANGUAGE DETECTION VALIDATION")
        print("="*80)

>>>>>>> main
        if detected_language:
            print(f"✅ Language detected in state: {detected_language}")
            assert detected_language.startswith("zh"), (
                f"Expected Chinese language code (zh*), got {detected_language}"
            )
        else:
            print("⚠️  Language not detected in state (may be detected in middleware)")
<<<<<<< HEAD
        
        # Validation: Response language
        ai_messages = [m for m in messages_2 if m.type == "ai"]
        assert len(ai_messages) > 0, "Agent should have generated at least one response"
        
        final_content = str(ai_messages[-1].content)
        
        chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
        has_chinese = bool(chinese_char_pattern.search(final_content))
        
        print("\n" + "="*80)
        print("RESPONSE LANGUAGE VALIDATION")
        print("="*80)
        
=======

        # Validation: Response language
        ai_messages = [m for m in messages_2 if m.type == "ai"]
        assert len(ai_messages) > 0, "Agent should have generated at least one response"

        final_content = str(ai_messages[-1].content)

        chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
        has_chinese = bool(chinese_char_pattern.search(final_content))

        print("\n" + "="*80)
        print("RESPONSE LANGUAGE VALIDATION")
        print("="*80)

>>>>>>> main
        if has_chinese:
            chinese_chars = chinese_char_pattern.findall(final_content)
            chinese_char_count = sum(len(match) for match in chinese_chars)
            print(f"✅ Response contains Chinese characters: {chinese_char_count} characters found")
        else:
            print("⚠️  Response does not contain Chinese characters")
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Validation: 60-second pitch structure in Chinese
        print("\n" + "="*80)
        print("60-SECOND PITCH STRUCTURE VALIDATION (Chinese)")
        print("="*80)
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Chinese keywords for pitch structure
        chinese_structure_keywords = [
            ("60-Second Pitch", ["60秒", "六十秒", "路演", "演讲", "推介"]),
            ("Pain Point Resonance", ["痛点", "共鸣", "紧迫性", "频率", "经济成本"]),
            ("Team Advantage", ["团队优势", "团队", "创始人", "经验", "背景"]),
            ("Call to Action", ["行动号召", "联系我们", "加入", "合作", "联系"]),
        ]
<<<<<<< HEAD
        
=======

>>>>>>> main
        found_keywords = []
        for field_name, keywords in chinese_structure_keywords:
            found = any(keyword in final_content for keyword in keywords)
            if found:
                found_keywords.append(field_name)
                print(f"  ✅ {field_name} (found)")
            else:
                print(f"  ❌ {field_name} (missing)")
<<<<<<< HEAD
        
        # Also check for English structure markers (in case of mixed language)
        english_markers = ["pitch", "pain point", "team advantage", "call to action", "cta"]
        english_found = sum(1 for marker in english_markers if marker in final_content.lower())
        
        total_structure_found = len(found_keywords) + (1 if english_found >= 3 else 0)
        
=======

        # Also check for English structure markers (in case of mixed language)
        english_markers = ["pitch", "pain point", "team advantage", "call to action", "cta"]
        english_found = sum(1 for marker in english_markers if marker in final_content.lower())

        total_structure_found = len(found_keywords) + (1 if english_found >= 3 else 0)

>>>>>>> main
        print(f"\n📊 Summary:")
        print(f"  Chinese structure keywords found: {len(found_keywords)}/{len(chinese_structure_keywords)}")
        print(f"  English markers found: {english_found}")
        print(f"  Total structure found: {total_structure_found}")
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Display the pitch output
        print("\n" + "="*80)
        print("GENERATED 60-SECOND PITCH OUTPUT (Chinese)")
        print("="*80)
        print(final_content)
        print("="*80)
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Final validations
        print("\n" + "="*80)
        print("FINAL VALIDATIONS")
        print("="*80)
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Validation: Response should be in Chinese
        assert has_chinese, (
            "Agent should respond in Chinese when user input is in Chinese"
        )
<<<<<<< HEAD
        
=======

>>>>>>> main
        # Validation: Should have pitch structure
        assert total_structure_found >= 2, (
            f"60-second pitch should contain at least 2 structure elements. "
            f"Found: {total_structure_found} (Chinese: {len(found_keywords)}, English: {english_found})"
        )
<<<<<<< HEAD
        
        print("✅ 60-second pitch was correctly generated in Chinese with proper structure")
        print("✅ Test passed: Skill correctly creates 60-second pitch in Chinese")
        
=======

        print("✅ 60-second pitch was correctly generated in Chinese with proper structure")
        print("✅ Test passed: Skill correctly creates 60-second pitch in Chinese")

>>>>>>> main
    finally:
        pass

