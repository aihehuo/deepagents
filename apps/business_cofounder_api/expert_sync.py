"""Expert synchronization logic for dual-agent architecture.

This module handles the coordination between facilitator (frontend) and expert (backend) agents:
- Detecting when expert sync is needed
- Extracting conversation history
- Triggering expert analysis
- Parsing expert responses (canvas + guidance)
- Updating shared state with analysis results

The canvas is a domain-agnostic JSON structure defined by the expert's prompt.
The backend treats it as an opaque blob and just stores/syncs it.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from deepagents.state.dual_agent_state import DualAgentState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# Use uvicorn's configured logger
_logger = logging.getLogger("uvicorn.error")


def should_trigger_expert(state: DualAgentState) -> bool:
    """Check if expert sync should be triggered based on conversation rounds.
    
    Args:
        state: Current shared state
        
    Returns:
        True if expert analysis should be triggered, False otherwise
    """
    # Check if explicitly flagged for sync
    if state.get("needs_expert_sync", False):
        return True
    
    # Check round-based trigger
    current_round = state.get("conversation_round", 0)
    last_sync = state.get("last_expert_sync", 0)
    _logger.info(f"[ExpertSync] should_trigger_expert: current_round: {current_round}, last_sync: {last_sync}")
    
    # Default sync interval is 3 rounds
    sync_interval = 3
    
    should_sync = current_round - last_sync >= sync_interval
    
    if should_sync:
        _logger.info(
            "[ExpertSync] Trigger condition met: round %d (last sync: %d, interval: %d)",
            current_round,
            last_sync,
            sync_interval,
        )
    
    return should_sync


def extract_recent_rounds(
    messages: list[BaseMessage],
    rounds: int = 10,
) -> list[BaseMessage]:
    """Extract the last N rounds of conversation (user-assistant pairs).
    
    A "round" is defined as a user message followed by assistant response(s).
    
    Args:
        messages: Full conversation history
        rounds: Number of rounds to extract (default: 10)
        
    Returns:
        List of messages from the last N rounds
    """
    if not messages:
        return []
    
    # Count rounds by counting user messages (each user message starts a round)
    user_message_indices = []
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            user_message_indices.append(i)
    
    # If we have fewer rounds than requested, return all messages
    if len(user_message_indices) <= rounds:
        return messages
    
    # Get the starting index for the last N rounds
    start_idx = user_message_indices[-rounds]
    
    recent_messages = messages[start_idx:]
    
    _logger.info(
        "[ExpertSync] Extracted %d messages from last %d rounds (out of %d total messages)",
        len(recent_messages),
        rounds,
        len(messages),
    )
    
    return recent_messages


def format_conversation_history(messages: list[BaseMessage]) -> str:
    """Format conversation messages into a readable text format for expert analysis.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Formatted conversation string
    """
    formatted_lines = []
    
    for msg in messages:
        # Determine role
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            role = "System"
        
        # Get content
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        
        # Skip empty messages
        if not content or not content.strip():
            continue
        
        # Format as role: content
        formatted_lines.append(f"{role}: {content}")
    
    conversation_text = "\n\n".join(formatted_lines)
    
    _logger.debug(
        "[ExpertSync] Formatted %d messages into %d characters of text",
        len(messages),
        len(conversation_text),
    )
    
    return conversation_text


async def trigger_expert_analysis(
    state: DualAgentState,
    expert_agent,
    conversation_history: list[BaseMessage],
    thread_id: str,
    expertise_dir: Path | None = None,
) -> dict[str, Any]:
    """Trigger expert agent to analyze recent conversation and generate guidance.
    
    Args:
        state: Current shared state
        expert_agent: The expert agent instance
        conversation_history: Recent messages to analyze
        thread_id: Thread ID for expert agent invocation
        expertise_dir: Directory containing expertise templates (optional)
        
    Returns:
        State updates from expert analysis (guidance and canvas)
    """
    _logger.info("[ExpertSync] Triggering expert analysis...")
    _logger.info("  Thread ID: %s", thread_id)
    _logger.info("  Messages to analyze: %d", len(conversation_history))
    _logger.info("  Current round: %d", state.get("conversation_round", 0))
    
    # Get expertise type from state (default to business_cofounder)
    expertise_type = state.get("expertise_type", "business_cofounder")
    _logger.info("  Expertise type: %s", expertise_type)
    
    # Load expertise template if expertise_dir provided
    canvas_template = "{}"
    if expertise_dir:
        from pathlib import Path
        from apps.business_cofounder_api.expertise_loader import load_expertise
        
        try:
            expertise = load_expertise(expertise_type, Path(expertise_dir))
            canvas_template = expertise["canvas_template"]
            _logger.info("  Loaded expertise template: %s", expertise["name"])
        except (FileNotFoundError, ValueError) as e:
            _logger.warning("  Failed to load expertise template: %s", str(e))
    
    # Format conversation history for analysis
    conversation_text = format_conversation_history(conversation_history)
    
    # Get current canvas and guidance
    current_guidance = state.get("expert_guidance", "None")
    current_canvas = state.get("canvas")
    
    # Build analysis prompt for expert
    analysis_prompt = f"""Analyze this conversation using your expertise.

## Conversation History (Last 10 Rounds)

{conversation_text}

## Current State

- Current guidance to facilitator: {current_guidance}
- Current canvas: {json.dumps(current_canvas, ensure_ascii=False, indent=2) if current_canvas else "None"}

## Your Task

Analyze this conversation and provide:

1. **Expert Guidance** (2-4 sentences):
   Strategic direction for the facilitator. What should they focus on in upcoming conversations?
   Be specific and actionable.

2. **Canvas Data** (structured JSON following the template):
   Use the canvas template structure below to assess the current state.

## Canvas Template

{canvas_template}

## Output Format

Provide your analysis as a JSON object with this exact structure:

```json
{{
  "expert_guidance": "Your 2-4 sentence strategic guidance here...",
  "canvas": {{
    ... follow the template structure above ...
  }}
}}
```

**Important**: 
- Return ONLY the JSON object, no additional text before or after
- Follow the canvas template structure
- Include only "expert_guidance" and "canvas" fields
"""
    
    # MOCK MODE: Set this to True to test if the issue is with async structure or the agent itself
    # Usage: EXPERT_SYNC_USE_MOCK=true python your_script.py
    USE_MOCK_EXPERT = os.getenv("EXPERT_SYNC_USE_MOCK", "false").lower() == "true"
    
    if USE_MOCK_EXPERT:
        _logger.info("[ExpertSync] ========================================")
        _logger.info("[ExpertSync] MOCK MODE ENABLED")
        _logger.info("[ExpertSync] This will help determine if the issue is with async structure or the agent")
        _logger.info("[ExpertSync] ========================================")
        
        # Test 1: Basic async operation
        _logger.info("[ExpertSync] MOCK: Testing basic async sleep...")
        start_test = datetime.utcnow()
        import asyncio
        await asyncio.sleep(0.5)
        elapsed_test = (datetime.utcnow() - start_test).total_seconds()
        _logger.info("[ExpertSync] MOCK: Async sleep completed in %.2fs", elapsed_test)
        
        # Test 2: Simulate a longer delay (like an agent call would take)
        _logger.info("[ExpertSync] MOCK: Simulating agent processing delay (2s)...")
        start_delay = datetime.utcnow()
        await asyncio.sleep(2.0)
        elapsed_delay = (datetime.utcnow() - start_delay).total_seconds()
        _logger.info("[ExpertSync] MOCK: Delay completed in %.2fs", elapsed_delay)
        
        # Test 3: Test timeout mechanism
        _logger.info("[ExpertSync] MOCK: Testing timeout mechanism with wait_for...")
        try:
            async def mock_slow_operation():
                await asyncio.sleep(0.1)  # Fast operation
                return "success"
            
            result = await asyncio.wait_for(mock_slow_operation(), timeout=1.0)
            _logger.info("[ExpertSync] MOCK: Timeout mechanism works! Result: %s", result)
        except Exception as e:
            _logger.error("[ExpertSync] MOCK: Timeout mechanism failed: %s", str(e))
        
        # Return a mock response
        mock_analysis = {
            "expert_guidance": "MOCK: This is a test guidance to verify async structure works correctly. If you see this, the async structure is working!",
            "canvas": {
                "status": "mock",
                "message": "This is a mock response for testing",
                "test_passed": True
            }
        }
        _logger.info("[ExpertSync] MOCK: Returning mock analysis")
        _logger.info("[ExpertSync] ========================================")
        return mock_analysis
    
    try:
        # Invoke expert agent with analysis prompt
        _logger.info("[ExpertSync] Invoking expert agent...")
        _logger.info("[ExpertSync] Analysis prompt length: %d characters", len(analysis_prompt))
        _logger.info("[ExpertSync] Expert agent type: %s", type(expert_agent).__name__)
        
        # Use an expert-specific thread ID to keep expert analysis separate
        expert_thread_id = f"expert_analysis_{thread_id}"
        
        # Use HumanMessage format (consistent with other agent invocations)
        import asyncio
        
        _logger.info("[ExpertSync] Starting expert agent invocation (thread_id=%s)...", expert_thread_id)
        _logger.info("[ExpertSync] About to call ainvoke at %s", datetime.utcnow().isoformat())
        
        # Create the input dict
        input_dict = {"messages": [HumanMessage(content=analysis_prompt)]}
        config_dict = {
            "configurable": {"thread_id": expert_thread_id},
        }
        _logger.info("[ExpertSync] Input prepared: messages=%d, config=%s", len(input_dict["messages"]), config_dict)
        
        # Use ainvoke with timeout
        _logger.info("[ExpertSync] Starting ainvoke with timeout (30s)...")
        start_time = datetime.utcnow()
        
        try:
            # Use wait_for with timeout
            response = await asyncio.wait_for(
                expert_agent.ainvoke(input_dict, config=config_dict),
                timeout=30.0,  # 30 second timeout
            )
            end_time = datetime.utcnow()
            elapsed = (end_time - start_time).total_seconds()
            _logger.info("[ExpertSync] Expert agent ainvoke completed at %s (elapsed: %.2fs)", end_time.isoformat(), elapsed)
        
        except asyncio.TimeoutError as e:
            end_time = datetime.utcnow()
            elapsed = (end_time - start_time).total_seconds()
            _logger.error("[ExpertSync] Expert agent ainvoke timed out after %.2fs at %s", elapsed, end_time.isoformat())
            _logger.error("[ExpertSync] TimeoutError details: %s", str(e))
            raise
        except Exception as e:
            end_time = datetime.utcnow()
            elapsed = (end_time - start_time).total_seconds()
            _logger.error("[ExpertSync] Expert agent ainvoke failed after %.2fs: %s", elapsed, str(e), exc_info=True)
            raise
        
        # Parse expert response
        analysis = parse_expert_response(response)
        
        # Add timestamp
        analysis["analysis_timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Update last sync round
        analysis["last_expert_sync"] = state.get("conversation_round", 0)
        analysis["needs_expert_sync"] = False
        
        _logger.info("[ExpertSync] Analysis parsed successfully")
        _logger.info("  Guidance: %s", analysis.get("expert_guidance", "")[:100])
        canvas = analysis.get("canvas", {})
        _logger.info("  Canvas keys: %s", list(canvas.keys()) if canvas else "[]")
        
        return analysis
        
    except Exception as e:
        _logger.error("[ExpertSync] Error during expert analysis: %s", str(e), exc_info=True)
        
        # Return minimal fallback analysis
        return {
            "expert_guidance": "Continue exploring the topic through natural conversation.",
            "canvas": {
                "status": "analysis_unavailable",
                "message": "Expert analysis temporarily unavailable - using fallback mode"
            },
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
            "last_expert_sync": state.get("conversation_round", 0),
            "needs_expert_sync": False,
        }


def parse_expert_response(response: dict[str, Any]) -> dict[str, Any]:
    """Parse expert agent's response and extract analysis results.
    
    The expert should return a JSON object with:
    - expert_guidance: Strategic guidance string for facilitator
    - canvas: Domain-agnostic JSON structure (opaque to backend)
    
    This function extracts and validates that structure.
    
    Args:
        response: Raw response from expert agent
        
    Returns:
        Parsed analysis dictionary with "expert_guidance" and "canvas" fields
        
    Raises:
        ValueError: If response format is invalid
    """
    _logger.debug("[ExpertSync] Parsing expert response...")
    
    # Get the last AI message content
    messages = response.get("messages", [])
    if not messages:
        raise ValueError("No messages in expert response")
    
    # Find the last AI message
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break
    
    if not last_ai_message:
        raise ValueError("No AI message in expert response")
    
    content = last_ai_message.content
    if not isinstance(content, str):
        content = str(content)
    
    _logger.debug("[ExpertSync] AI message content length: %d", len(content))
    
    # Try to extract JSON from the content
    # The expert should return JSON, possibly wrapped in markdown code blocks
    json_str = content.strip()
    
    # Remove markdown code blocks if present
    if json_str.startswith("```json"):
        json_str = json_str[7:]  # Remove ```json
    elif json_str.startswith("```"):
        json_str = json_str[3:]  # Remove ```
    
    if json_str.endswith("```"):
        json_str = json_str[:-3]  # Remove trailing ```
    
    json_str = json_str.strip()
    
    # Parse JSON
    try:
        analysis = json.loads(json_str)
    except json.JSONDecodeError as e:
        _logger.error("[ExpertSync] Failed to parse JSON: %s", str(e))
        _logger.error("[ExpertSync] Content: %s", json_str[:500])
        raise ValueError(f"Invalid JSON in expert response: {str(e)}")
    
    # Validate required fields
    required_fields = ["expert_guidance", "canvas"]
    missing_fields = [field for field in required_fields if field not in analysis]
    
    if missing_fields:
        _logger.warning("[ExpertSync] Missing fields in analysis: %s", missing_fields)
        # Fill in missing fields with defaults
        if "expert_guidance" not in analysis:
            analysis["expert_guidance"] = "Continue exploring the topic through conversation."
        if "canvas" not in analysis:
            analysis["canvas"] = {
                "status": "incomplete",
                "message": "Canvas data not provided by expert"
            }
    
    # Validate canvas is a dict (but don't validate its internal structure - it's opaque)
    canvas = analysis["canvas"]
    if not isinstance(canvas, dict):
        _logger.warning("[ExpertSync] Invalid canvas structure (not a dict), using empty canvas")
        analysis["canvas"] = {
            "status": "invalid",
            "message": "Canvas was not a valid JSON object"
        }
    
    _logger.debug("[ExpertSync] Response parsed successfully")
    
    return analysis


async def update_state_with_analysis(
    thread_id: str,
    analysis: dict[str, Any],
    checkpointer,
    agent=None,
) -> None:
    """Update shared state with expert analysis results.
    
    This function updates the state stored in the checkpointer with the
    analysis results from the expert agent.
    
    Args:
        thread_id: Thread ID for the conversation
        analysis: Analysis results from expert
        checkpointer: Checkpointer instance to update state
        agent: Optional agent instance to use aupdate_state (preferred method)
    """
    _logger.info("[ExpertSync] Updating state with analysis results...")
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # Try using agent.aupdate_state if agent is provided (preferred method)
        if agent is not None and hasattr(agent, 'aupdate_state'):
            try:
                _logger.info("[ExpertSync] Using agent.aupdate_state to update state")
                await agent.aupdate_state(
                    config=config,
                    values=analysis,
                )
                _logger.info("[ExpertSync] State updated successfully via agent.aupdate_state")
                _logger.info("  Updated fields: %s", list(analysis.keys()))
                return
            except Exception as e:
                _logger.warning("[ExpertSync] agent.aupdate_state failed, falling back to checkpointer.aput: %s", str(e))
        
        # Fallback: Use checkpointer directly
        # Get current state
        checkpoint = await checkpointer.aget(config)
        
        if checkpoint is None:
            _logger.warning("[ExpertSync] No checkpoint found for thread %s", thread_id)
            return
        
        # Update state with analysis results
        current_state = checkpoint.get("channel_values", {})
        
        # Merge analysis into state
        current_state.update(analysis)
        
        # Try to extract checkpoint_ns from the checkpoint if available
        # Otherwise use empty string as default
        checkpoint_ns = checkpoint.get("checkpoint_ns", "")
        
        # Prepare config for aput
        put_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
        
        # Save updated state
        await checkpointer.aput(
            put_config,
            {
                **checkpoint,
                "channel_values": current_state,
            },
            {},  # metadata
            {}   # new_versions
        )
        
        _logger.info("[ExpertSync] State updated successfully via checkpointer.aput")
        _logger.info("  Updated fields: %s", list(analysis.keys()))
        
    except Exception as e:
        _logger.error("[ExpertSync] Error updating state: %s", str(e), exc_info=True)


async def trigger_and_update_expert(
    thread_id: str,
    state: DualAgentState,
    expert_agent,
    checkpointer,
    expertise_dir: Path | None = None,
    facilitator_agent=None,
) -> None:
    """Trigger expert analysis and update state (convenience function).
    
    This combines trigger_expert_analysis and update_state_with_analysis
    into a single async operation.
    
    Args:
        thread_id: Thread ID for the conversation
        state: Current shared state
        expert_agent: Expert agent instance
        checkpointer: Checkpointer instance (from facilitator agent)
        expertise_dir: Directory containing expertise templates (optional)
        facilitator_agent: Optional facilitator agent to use for state updates (preferred)
    """
    _logger.info("[ExpertSync] Starting expert sync for thread %s", thread_id)
    
    # Extract recent conversation
    messages = state.get("messages", [])
    recent_messages = extract_recent_rounds(messages, rounds=10)
    
    # Trigger expert analysis
    analysis = await trigger_expert_analysis(
        state=state,
        expert_agent=expert_agent,
        conversation_history=recent_messages,
        thread_id=thread_id,
        expertise_dir=expertise_dir,
    )
    
    # Update state with results
    # Use facilitator_agent if provided (it has the correct checkpointer for the conversation)
    await update_state_with_analysis(
        thread_id=thread_id,
        analysis=analysis,
        checkpointer=checkpointer,
        agent=facilitator_agent,  # Pass facilitator agent to use aupdate_state
    )
    
    _logger.info("[ExpertSync] Expert sync completed for thread %s", thread_id)
