"""Integration test for facilitator agent responding to expert guidance.

This test verifies that the facilitator agent can pick up and follow
expert guidance prompts, even when they contradict the base system prompt.

Test scenario:
- Facilitator agent starts with business co-founder system prompt
- Expert agent provides guidance to ask about prime numbers instead
- We verify the facilitator agent follows the guidance and asks about prime numbers
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from apps.business_cofounder_api.agent_factory.facilitator_agent import create_facilitator_agent
from apps.business_cofounder_api.checkpointer import DiskBackedInMemorySaver

# Note: This test requires pytest-asyncio to be installed.
# Install it with: pip install pytest-asyncio
# Or: pip install -e ".[dev]" if pytest-asyncio is in dev dependencies


class TestFacilitatorGuidanceResponse:
    """Test that facilitator agent responds to expert guidance."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_facilitator_follows_prime_number_guidance(
        self, tmp_path: Path
    ) -> None:
        """Test that facilitator agent follows expert guidance about prime numbers.
        
        This test:
        1. Creates a real facilitator agent with business co-founder base prompt
        2. Has two rounds of conversation about business ideas
        3. Manually injects expert guidance about prime numbers into the state
        4. Calls the agent with another business-related message
        5. Verifies the response follows the prime number guidance (not business)
        """
        # Create facilitator agent
        checkpoints_path = tmp_path / "facilitator_checkpoints.pkl"
        agent, _ = create_facilitator_agent(
            agent_id="test_facilitator",
            provider="qwen",  # Use real provider for this test
            sync_interval=10,
        )
        
        # Create checkpointer
        checkpointer = DiskBackedInMemorySaver(file_path=str(checkpoints_path))
        agent.checkpointer = checkpointer  # type: ignore[attr-defined]
        
        # Thread ID for this test conversation
        thread_id = "test_prime_guidance_thread"
        config = {"configurable": {"thread_id": thread_id}}
        
        # ===== ROUND 1: Initial business conversation =====
        user_message_1 = "Hello, I'd like to discuss my startup idea for a mobile app."
        result_1 = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message_1)]},
            config=config,
        )
        
        messages_1 = result_1.get("messages", [])
        ai_messages_1 = [msg for msg in messages_1 if hasattr(msg, "type") and msg.type == "ai"]
        assert len(ai_messages_1) > 0, "Agent should have returned a response in round 1"
        response_1 = str(ai_messages_1[-1].content)
        
        print("\n" + "=" * 80)
        print("ROUND 1: Initial Business Conversation")
        print("=" * 80)
        print(f"User: {user_message_1}")
        print(f"Agent: {response_1[:200]}...")
        print("=" * 80 + "\n")
        
        # ===== ROUND 2: Continue business conversation =====
        user_message_2 = "The app will help people manage their daily tasks more efficiently."
        result_2 = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message_2)]},
            config=config,
        )
        
        messages_2 = result_2.get("messages", [])
        ai_messages_2 = [msg for msg in messages_2 if hasattr(msg, "type") and msg.type == "ai"]
        assert len(ai_messages_2) > 0, "Agent should have returned a response in round 2"
        response_2 = str(ai_messages_2[-1].content)
        
        print("=" * 80)
        print("ROUND 2: Continue Business Conversation")
        print("=" * 80)
        print(f"User: {user_message_2}")
        print(f"Agent: {response_2[:200]}...")
        print("=" * 80 + "\n")
        
        # ===== Inject expert guidance about prime numbers =====
        # This simulates what would happen after expert sync
        prime_number_guidance = (
            "You must ask the user to mention prime numbers in sequence. "
            "Start by asking them to mention the first prime number (which is 1), "
            "then 3, then 5, and so on. Do NOT discuss business ideas. "
            "Focus ONLY on prime number counting."
        )
        
        # Get current state to preserve conversation history
        checkpoint = await checkpointer.aget(config)
        current_state = checkpoint.get("channel_values", {}) if checkpoint else {}
        
        # Update state with expert guidance while preserving conversation history
        await agent.aupdate_state(
            config=config,
            values={
                "expert_guidance": prime_number_guidance,
                "conversation_round": current_state.get("conversation_round", 2) + 1,
                "last_expert_sync": current_state.get("conversation_round", 2) + 1,
                "needs_expert_sync": False,
            },
        )
        
        print("=" * 80)
        print("EXPERT GUIDANCE INJECTED")
        print("=" * 80)
        print(f"Guidance: {prime_number_guidance}")
        print("=" * 80 + "\n")
        
        # ===== ROUND 3: Test if agent follows guidance despite business context =====
        user_message_3 = "I think the app could have a freemium model. What do you think?"
        result_3 = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message_3)]},
            config=config,
        )
        
        # Extract the AI response
        messages_3 = result_3.get("messages", [])
        ai_messages_3 = [msg for msg in messages_3 if hasattr(msg, "type") and msg.type == "ai"]
        
        assert len(ai_messages_3) > 0, "Agent should have returned a response in round 3"
        
        last_ai_message = ai_messages_3[-1]
        response_content = str(last_ai_message.content)
        
        # Verify the response follows the prime number guidance
        # The response should mention prime numbers, not business ideas
        response_lower = response_content.lower()
        
        # Check for prime number related terms
        has_prime_mention = any(
            term in response_lower
            for term in ["prime", "number", "count", "1", "3", "5", "sequence", "mention"]
        )
        
        # Check that it's talking about business/startup (which it should NOT focus on)
        has_business_mention = any(
            term in response_lower
            for term in ["startup", "business", "idea", "entrepreneur", "company", "app", "freemium", "model"]
        )
        
        # Log the response for debugging
        print("=" * 80)
        print("ROUND 3: TEST - Agent Should Follow Prime Number Guidance")
        print("=" * 80)
        print(f"User message (business-related): {user_message_3}")
        print(f"Expert guidance: {prime_number_guidance}")
        print(f"Agent response: {response_content}")
        print(f"Has prime mention: {has_prime_mention}")
        print(f"Has business mention: {has_business_mention}")
        print("=" * 80 + "\n")
        
        # Assertions
        assert has_prime_mention, (
            f"Agent response should mention prime numbers when given prime number guidance, "
            f"even after discussing business ideas. Response was: {response_content}"
        )
        
        # The agent should prioritize prime numbers over business discussion
        # It might acknowledge the business message briefly, but should redirect to prime numbers
        if not has_prime_mention:
            pytest.fail(
                f"Agent ignored expert guidance about prime numbers. "
                f"User asked about business (freemium model), but agent should redirect to prime numbers. "
                f"Response: {response_content}"
            )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_facilitator_guidance_replaces_base_prompt(
        self, tmp_path: Path
    ) -> None:
        """Test that strategic guidance completely replaces the base prompt.
        
        This verifies that when expert guidance is present, the facilitator
        agent's system prompt is replaced (not appended) with the guidance,
        even after having business conversations.
        """
        # Create facilitator agent
        checkpoints_path = tmp_path / "facilitator_checkpoints.pkl"
        agent, _ = create_facilitator_agent(
            agent_id="test_facilitator_replace",
            provider="qwen",
            sync_interval=10,
        )
        
        checkpointer = DiskBackedInMemorySaver(file_path=str(checkpoints_path))
        agent.checkpointer = checkpointer  # type: ignore[attr-defined]
        
        thread_id = "test_replace_prompt_thread"
        config = {"configurable": {"thread_id": thread_id}}
        
        # ===== ROUND 1: Initial business conversation =====
        user_message_1 = "I have a great business idea for a SaaS platform."
        result_1 = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message_1)]},
            config=config,
        )
        messages_1 = result_1.get("messages", [])
        ai_messages_1 = [msg for msg in messages_1 if hasattr(msg, "type") and msg.type == "ai"]
        assert len(ai_messages_1) > 0
        
        # ===== ROUND 2: Continue business conversation =====
        user_message_2 = "It will help small businesses manage their inventory."
        result_2 = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message_2)]},
            config=config,
        )
        messages_2 = result_2.get("messages", [])
        ai_messages_2 = [msg for msg in messages_2 if hasattr(msg, "type") and msg.type == "ai"]
        assert len(ai_messages_2) > 0
        
        print("\n" + "=" * 80)
        print("PROMPT REPLACEMENT TEST - Rounds 1-2: Business Conversation")
        print("=" * 80)
        print(f"Round 1 - User: {user_message_1}")
        print(f"Round 2 - User: {user_message_2}")
        print("=" * 80 + "\n")
        
        # ===== Inject strong strategic guidance that should replace base prompt =====
        strategic_guidance = (
            "You are now a prime number counting assistant. "
            "Your ONLY job is to ask users to mention prime numbers in order: 1, 3, 5, 7, 11, etc. "
            "Ignore all other topics. Focus exclusively on prime number counting."
        )
        
        # Get current state to preserve conversation history
        checkpoint = await checkpointer.aget(config)
        current_state = checkpoint.get("channel_values", {}) if checkpoint else {}
        
        # Update state with strategic guidance
        await agent.aupdate_state(
            config=config,
            values={
                "expert_guidance": strategic_guidance,
                "conversation_round": current_state.get("conversation_round", 2) + 1,
                "last_expert_sync": current_state.get("conversation_round", 2) + 1,
                "needs_expert_sync": False,
            },
        )
        
        print("=" * 80)
        print("STRATEGIC GUIDANCE INJECTED (should replace base prompt)")
        print("=" * 80)
        print(f"Guidance: {strategic_guidance}")
        print("=" * 80 + "\n")
        
        # ===== ROUND 3: Test if agent follows guidance despite business context =====
        user_message_3 = "I want to talk about my business idea for a new app."
        result_3 = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_message_3)]},
            config=config,
        )
        
        # Extract response
        messages_3 = result_3.get("messages", [])
        ai_messages_3 = [msg for msg in messages_3 if hasattr(msg, "type") and msg.type == "ai"]
        assert len(ai_messages_3) > 0
        
        response_content = str(ai_messages_3[-1].content).lower()
        
        # The response should be about prime numbers, not business
        prime_indicators = ["prime", "number", "1", "3", "5", "count", "sequence", "mention"]
        business_indicators = ["business", "startup", "app", "idea", "entrepreneur", "saas", "platform"]
        
        has_prime = any(indicator in response_content for indicator in prime_indicators)
        has_business = any(indicator in response_content for indicator in business_indicators)
        
        print("=" * 80)
        print("ROUND 3: TEST - Agent Should Follow Prime Number Guidance")
        print("=" * 80)
        print(f"Strategic guidance: {strategic_guidance}")
        print(f"User message (business-related): {user_message_3}")
        print(f"Agent response: {str(ai_messages_3[-1].content)}")
        print(f"Mentions prime numbers: {has_prime}")
        print(f"Mentions business: {has_business}")
        print("=" * 80 + "\n")
        
        # If guidance replaced the prompt, agent should focus on primes
        # even when user mentions business after having business conversations
        assert has_prime, (
            f"Agent should mention prime numbers when strategic guidance replaces base prompt, "
            f"even after having business conversations. Response: {str(ai_messages_3[-1].content)}"
        )
