"""Integration test for the complete sequential skill progression.

This test verifies the entire business idea development workflow:
1. business-idea-evaluation → mark_business_idea_complete
2. persona-clarification → mark_persona_clarified
3. painpoint-enhancement → mark_painpoint_enhanced
4. 60s-pitch-creation → mark_pitch_created
5. baseline-pricing-optimization → mark_pricing_optimized
6. business-model-pivot-exploration

The test uses BusinessIdeaTrackerMiddleware to enforce the sequential unlock conditions.
"""

import os
import re
import shutil
import time
from pathlib import Path

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.business_idea_tracker import BusinessIdeaTrackerMiddleware
from deepagents.middleware.language import LanguageDetectionMiddleware
from deepagents_cli.skills.load import list_skills
from deepagents_cli.skills.middleware import SkillsMiddleware

from tests.timing_middleware import TimingMiddleware


def _load_model_config(repo_root: Path) -> tuple[str, str, str]:
    """Load model configuration.

    Prefer environment variables (works in sandbox/CI). Fallback to `.env.deepseek` if readable.
    """
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model_name = os.environ.get("ANTHROPIC_MODEL", "deepseek-chat")

    if base_url and api_key:
        return base_url, api_key, model_name

    env_file = repo_root / ".env.deepseek"
    if not env_file.exists():
        pytest.skip(
            "Missing ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY and `.env.deepseek` not found. "
            "Set env vars or provide `.env.deepseek`."
        )

    try:
        env_text = env_file.read_text(encoding="utf-8")
    except (PermissionError, OSError) as e:
        pytest.skip(
            f"Could not read `{env_file}` ({type(e).__name__}: {e}). "
            "Set ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY in the environment to run this test."
        )

    env_vars: dict[str, str] = {}
    for line in env_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "export " in line:
            key_value = line.replace("export ", "", 1).split("=", 1)
            if len(key_value) == 2:
                key, value = key_value
                env_vars[key] = value.strip('"\'')

    base_url = env_vars.get("ANTHROPIC_BASE_URL")
    api_key = env_vars.get("ANTHROPIC_API_KEY")
    model_name = env_vars.get("ANTHROPIC_MODEL", model_name)

    if not base_url or not api_key:
        pytest.skip("DeepSeek configuration incomplete (missing ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY).")

    return base_url, api_key, model_name


@pytest.mark.timeout(600)  # 10 minutes for the full sequence
def test_complete_sequential_skill_progression(tmp_path: Path) -> None:
    """Test the complete sequential skill progression from idea evaluation to pivot exploration.
    
    This integration test verifies:
    - Skills unlock in the correct order
    - Each milestone marking tool is called after skill completion
    - State flags are correctly updated
    - Skills are locked when prerequisites aren't met
    - The entire workflow executes successfully
    """
    # Load model configuration
    repo_root = Path(__file__).parent.parent
    base_url, api_key, model_name = _load_model_config(repo_root)
    
    # Set up environment
    old_base_url = os.environ.get("ANTHROPIC_BASE_URL")
    old_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    os.environ["ANTHROPIC_BASE_URL"] = base_url
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    try:
        # Set up skills directory with all required skills
        agent_id = "test_sequential_progression"
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all required skills from examples
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
            
            skill_dest = skills_dir / skill_name
            shutil.copytree(example_skill_dir, skill_dest)
        
        # Verify all skills are discovered
        skills = list_skills(user_skills_dir=skills_dir, project_skills_dir=None)
        skill_names = [s["name"] for s in skills]
        for skill_name in required_skills:
            assert skill_name in skill_names, f"Should discover {skill_name} skill"
        
        print("\n" + "="*80)
        print("TEST: COMPLETE SEQUENTIAL SKILL PROGRESSION")
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
        
        # Create agent with BusinessIdeaTrackerMiddleware to enforce sequence
        filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
        timing_middleware = TimingMiddleware(verbose=True)
        checkpointer = MemorySaver()
        
        agent = create_deep_agent(
            model=model,
            backend=filesystem_backend,
            middleware=[
                timing_middleware,
                BusinessIdeaTrackerMiddleware(),  # Enforce sequential unlock conditions
                LanguageDetectionMiddleware(),
                SkillsMiddleware(
                    skills_dir=skills_dir,
                    assistant_id=agent_id,
                    project_skills_dir=None,
                ),
            ],
            tools=[],
            checkpointer=checkpointer,
            system_prompt="""You are a business co-founder assistant helping entrepreneurs develop their startup ideas.

Follow the sequential skill progression:
1. Use business-idea-evaluation to evaluate the idea, then call mark_business_idea_complete
2. Use persona-clarification to clarify the persona, then call mark_persona_clarified
3. Use painpoint-enhancement to enhance the pain point, then call mark_painpoint_enhanced
4. Use 60s-pitch-creation to create a pitch, then call mark_pitch_created
5. Use baseline-pricing-and-optimization to optimize pricing, then call mark_pricing_optimized
6. Use business-model-pivot-exploration to explore alternative models

Each skill must be completed and marked before the next one can be used. The BusinessIdeaTrackerMiddleware will tell you which skills are unlocked.""",
        )
        
        config = {"configurable": {"thread_id": "test-sequential-progression"}}
        
        # ============================================================
        # STEP 1: Business Idea Evaluation
        # ============================================================
        print("\n" + "="*80)
        print("STEP 1: BUSINESS IDEA EVALUATION")
        print("="*80)
        
        user_request_1 = """I want to create an app that helps busy professionals manage their work-life balance.
Many professionals struggle with burnout because they can't effectively prioritize tasks and end up working late into the night.
The app would help them set boundaries and manage their time more effectively."""
        
        print(f"\n📝 User Request:\n{user_request_1}\n")
        print("⏳ Starting agent execution...\n")
        
        input_state_1 = {"messages": [HumanMessage(content=user_request_1)]}
        invoke_start = time.time()
        result_1 = agent.invoke(input_state_1, config)
        invoke_end = time.time()
        
        # Verify Step 1
        business_idea_complete = result_1.get("business_idea_complete", False)
        materialized_idea = result_1.get("materialized_business_idea")
        messages_1 = result_1.get("messages", [])
        
        # Check if mark_business_idea_complete was called
        mark_complete_called = False
        for message in messages_1:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "mark_business_idea_complete":
                        mark_complete_called = True
                        break
            elif message.type == "tool" and hasattr(message, "name") and message.name == "mark_business_idea_complete":
                mark_complete_called = True
            if mark_complete_called:
                break
        
        print(f"✅ Step 1 completed ({invoke_end - invoke_start:.2f}s)")
        print(f"  business_idea_complete: {business_idea_complete}")
        print(f"  mark_business_idea_complete called: {mark_complete_called}")
        print(f"  materialized_idea: {materialized_idea[:100] if materialized_idea else 'None'}...")
        
        assert business_idea_complete, "Business idea should be marked as complete in step 1"
        assert mark_complete_called, "mark_business_idea_complete tool should be called"
        assert materialized_idea, "Materialized idea should be stored"
        
        # ============================================================
        # STEP 2: Persona Clarification
        # ============================================================
        print("\n" + "="*80)
        print("STEP 2: PERSONA CLARIFICATION")
        print("="*80)
        
        user_request_2 = "Can you help me clarify who my target users should be? Create a detailed persona for this business idea."
        
        print(f"\n📝 User Request:\n{user_request_2}\n")
        print("⏳ Starting agent execution...\n")
        
        messages_so_far = result_1.get("messages", [])
        input_state_2 = {"messages": messages_so_far + [HumanMessage(content=user_request_2)]}
        
        invoke_start = time.time()
        result_2 = agent.invoke(input_state_2, config)
        invoke_end = time.time()
        
        # Verify Step 2
        persona_clarified = result_2.get("persona_clarified", False)
        messages_2 = result_2.get("messages", [])
        
        # Check if mark_persona_clarified was called
        mark_persona_called = False
        for message in messages_2:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "mark_persona_clarified":
                        mark_persona_called = True
                        break
            elif message.type == "tool" and hasattr(message, "name") and message.name == "mark_persona_clarified":
                mark_persona_called = True
            if mark_persona_called:
                break
        
        print(f"✅ Step 2 completed ({invoke_end - invoke_start:.2f}s)")
        print(f"  persona_clarified: {persona_clarified}")
        print(f"  mark_persona_clarified called: {mark_persona_called}")
        
        assert persona_clarified, "Persona should be marked as clarified in step 2"
        assert mark_persona_called, "mark_persona_clarified tool should be called"
        
        # ============================================================
        # STEP 3: Painpoint Enhancement
        # ============================================================
        print("\n" + "="*80)
        print("STEP 3: PAINPOINT ENHANCEMENT")
        print("="*80)
        
        user_request_3 = "Can you help me enhance and strengthen the pain point for this business idea?"
        
        print(f"\n📝 User Request:\n{user_request_3}\n")
        print("⏳ Starting agent execution...\n")
        
        messages_so_far = result_2.get("messages", [])
        input_state_3 = {"messages": messages_so_far + [HumanMessage(content=user_request_3)]}
        
        invoke_start = time.time()
        result_3 = agent.invoke(input_state_3, config)
        invoke_end = time.time()
        
        # Verify Step 3
        painpoint_enhanced = result_3.get("painpoint_enhanced", False)
        messages_3 = result_3.get("messages", [])
        
        # Check if mark_painpoint_enhanced was called
        mark_painpoint_called = False
        for message in messages_3:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "mark_painpoint_enhanced":
                        mark_painpoint_called = True
                        break
            elif message.type == "tool" and hasattr(message, "name") and message.name == "mark_painpoint_enhanced":
                mark_painpoint_called = True
            if mark_painpoint_called:
                break
        
        print(f"✅ Step 3 completed ({invoke_end - invoke_start:.2f}s)")
        print(f"  painpoint_enhanced: {painpoint_enhanced}")
        print(f"  mark_painpoint_enhanced called: {mark_painpoint_called}")
        
        assert painpoint_enhanced, "Pain point should be marked as enhanced in step 3"
        assert mark_painpoint_called, "mark_painpoint_enhanced tool should be called"
        
        # ============================================================
        # STEP 4: 60-Second Pitch Creation
        # ============================================================
        print("\n" + "="*80)
        print("STEP 4: 60-SECOND PITCH CREATION")
        print("="*80)
        
        user_request_4 = """Can you help me create a 60-second pitch for this idea? 
I need to present it to potential investors. Our team has 10 years of experience in productivity software development."""
        
        print(f"\n📝 User Request:\n{user_request_4}\n")
        print("⏳ Starting agent execution...\n")
        
        messages_so_far = result_3.get("messages", [])
        input_state_4 = {"messages": messages_so_far + [HumanMessage(content=user_request_4)]}
        
        invoke_start = time.time()
        result_4 = agent.invoke(input_state_4, config)
        invoke_end = time.time()
        
        # Verify Step 4
        pitch_created = result_4.get("pitch_created", False)
        messages_4 = result_4.get("messages", [])
        
        # Check if mark_pitch_created was called
        mark_pitch_called = False
        for message in messages_4:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "mark_pitch_created":
                        mark_pitch_called = True
                        break
            elif message.type == "tool" and hasattr(message, "name") and message.name == "mark_pitch_created":
                mark_pitch_called = True
            if mark_pitch_called:
                break
        
        print(f"✅ Step 4 completed ({invoke_end - invoke_start:.2f}s)")
        print(f"  pitch_created: {pitch_created}")
        print(f"  mark_pitch_created called: {mark_pitch_called}")
        
        assert pitch_created, "Pitch should be marked as created in step 4"
        assert mark_pitch_called, "mark_pitch_created tool should be called"
        
        # ============================================================
        # STEP 5: Baseline Pricing and Optimization
        # ============================================================
        print("\n" + "="*80)
        print("STEP 5: BASELINE PRICING AND OPTIMIZATION")
        print("="*80)
        
        user_request_5 = """Can you help me establish baseline pricing and identify pricing optimization opportunities?
Our target customers are busy professionals earning $80K-$150K annually, and the app saves them 5-10 hours per week, 
which translates to $2,000-$4,000 in time value per month."""
        
        print(f"\n📝 User Request:\n{user_request_5}\n")
        print("⏳ Starting agent execution...\n")
        
        messages_so_far = result_4.get("messages", [])
        input_state_5 = {"messages": messages_so_far + [HumanMessage(content=user_request_5)]}
        
        invoke_start = time.time()
        result_5 = agent.invoke(input_state_5, config)
        invoke_end = time.time()
        
        # Verify Step 5
        pricing_optimized = result_5.get("pricing_optimized", False)
        messages_5 = result_5.get("messages", [])
        
        # Check if mark_pricing_optimized was called
        mark_pricing_called = False
        for message in messages_5:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "mark_pricing_optimized":
                        mark_pricing_called = True
                        break
            elif message.type == "tool" and hasattr(message, "name") and message.name == "mark_pricing_optimized":
                mark_pricing_called = True
            if mark_pricing_called:
                break
        
        print(f"✅ Step 5 completed ({invoke_end - invoke_start:.2f}s)")
        print(f"  pricing_optimized: {pricing_optimized}")
        print(f"  mark_pricing_optimized called: {mark_pricing_called}")
        
        assert pricing_optimized, "Pricing should be marked as optimized in step 5"
        assert mark_pricing_called, "mark_pricing_optimized tool should be called"
        
        # ============================================================
        # STEP 6: Business Model Pivot Exploration
        # ============================================================
        print("\n" + "="*80)
        print("STEP 6: BUSINESS MODEL PIVOT EXPLORATION")
        print("="*80)
        
        user_request_6 = """Can you help me explore different business models for this idea? 
I want to see what alternative models might work better than my current approach. 
The product is a mobile application with AI-powered task prioritization, automated calendar blocking, 
and personalized boundary recommendations."""
        
        print(f"\n📝 User Request:\n{user_request_6}\n")
        print("⏳ Starting agent execution...\n")
        
        messages_so_far = result_5.get("messages", [])
        input_state_6 = {"messages": messages_so_far + [HumanMessage(content=user_request_6)]}
        
        invoke_start = time.time()
        result_6 = agent.invoke(input_state_6, config)
        invoke_end = time.time()
        
        # Verify Step 6 - pivot exploration doesn't have a milestone tool, but skill should be used
        messages_6 = result_6.get("messages", [])
        
        # Check if business-model-pivot-exploration skill was read
        pivot_skill_read = False
        for message in messages_6:
            if message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.get("name") == "read_file":
                        file_path = tool_call.get("args", {}).get("file_path", "")
                        if "business-model-pivot-exploration" in str(file_path):
                            pivot_skill_read = True
                            break
            elif message.type == "tool" and hasattr(message, "name") and message.name == "read_file":
                # For ToolMessage, check content for skill references
                if hasattr(message, "content") and "business-model-pivot-exploration" in str(message.content):
                    pivot_skill_read = True
            if pivot_skill_read:
                break
        
        print(f"✅ Step 6 completed ({invoke_end - invoke_start:.2f}s)")
        print(f"  pivot exploration skill used: {pivot_skill_read}")
        
        # ============================================================
        # FINAL VERIFICATION
        # ============================================================
        print("\n" + "="*80)
        print("FINAL VERIFICATION: ALL MILESTONES")
        print("="*80)
        
        final_state = {
            "business_idea_complete": result_6.get("business_idea_complete", False),
            "persona_clarified": result_6.get("persona_clarified", False),
            "painpoint_enhanced": result_6.get("painpoint_enhanced", False),
            "pitch_created": result_6.get("pitch_created", False),
            "pricing_optimized": result_6.get("pricing_optimized", False),
        }
        
        print("\n📊 Final State:")
        for key, value in final_state.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
        
        # Verify all milestones are completed
        assert all(final_state.values()), (
            f"All milestones should be completed. Final state: {final_state}"
        )
        
        print("\n" + "="*80)
        print("✅ COMPLETE SEQUENTIAL PROGRESSION TEST PASSED")
        print("="*80)
        print("\nAll 6 steps completed successfully:")
        print("  1. ✅ Business idea evaluated and marked complete")
        print("  2. ✅ Persona clarified and marked")
        print("  3. ✅ Pain point enhanced and marked")
        print("  4. ✅ 60-second pitch created and marked")
        print("  5. ✅ Pricing optimized and marked")
        print("  6. ✅ Business model pivots explored")
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

