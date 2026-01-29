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

from langdetect import LangDetectException, detect


# Use uvicorn's configured logger
_logger = logging.getLogger("uvicorn.error")


# -----------------------------------------------------------------------------
# Language evaluation helpers (code-based, no LLM)
# -----------------------------------------------------------------------------


def extract_text_from_canvas(canvas: dict[str, Any]) -> str:
    """Recursively collect all string values from the canvas for language detection.

    Skips metadata keys (status, message) when they look like error/fallback content.
    Ignores very short strings (length < 3) to reduce noise.

    Args:
        canvas: Domain-agnostic canvas dict (nested dicts/lists of strings).

    Returns:
        Concatenated non-empty strings joined by spaces.
    """
    SKIP_KEYS = {"status", "message"}
    MIN_STR_LEN = 3
    chunks: list[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in SKIP_KEYS and isinstance(v, str):
                    s = (v or "").strip().lower()
                    if s in (
                        "analysis_unavailable",
                        "incomplete",
                        "invalid",
                        "mock",
                        "not a dict",
                    ) or "unavailable" in s or "error" in s:
                        continue
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        elif isinstance(obj, str):
            t = obj.strip()
            if len(t) >= MIN_STR_LEN:
                chunks.append(t)

    _walk(canvas)
    return " ".join(chunks)


def detect_canvas_language(canvas: dict[str, Any], min_length: int = 50) -> str | None:
    """Detect language of canvas text using langdetect (code-only, no LLM).

    Args:
        canvas: Canvas dict from expert analysis.
        min_length: Minimum total text length to run detection.

    Returns:
        Detected language code (e.g. 'en', 'zh') or None if too little text or detection fails.
    """
    text = extract_text_from_canvas(canvas)
    if len(text) < min_length:
        return None
    if detect is None:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def languages_match(user_lang: str, canvas_lang: str | None) -> bool:
    """Check if user language and canvas language match (base codes).

    Normalizes e.g. 'zh-cn' -> 'zh'. Treats canvas_lang None as 'no check' (match).

    Args:
        user_lang: User's detected language code.
        canvas_lang: Detected canvas language or None.

    Returns:
        True if same base language or canvas_lang is None.
    """
    if canvas_lang is None:
        return True
    base_user = (user_lang or "").split("-")[0].lower()
    base_canvas = (canvas_lang or "").split("-")[0].lower()
    return base_user == base_canvas


def _should_skip_language_eval(canvas: Any) -> bool:
    """Return True if we should skip language evaluation (missing, empty, or non-content blob)."""
    if canvas is None or not isinstance(canvas, dict): 
        return True
    keys = set(canvas.keys())
    if keys <= {"status", "message"}:
        return True
    return False


# Language names for facilitator/expert language-fix prompts (shared)
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
}


def detect_text_language(text: str, min_length: int = 20) -> str | None:
    """Detect language of a text string using langdetect (code-only, no LLM).

    Args:
        text: Plain text (e.g. facilitator reply).
        min_length: Minimum length to run detection.

    Returns:
        Detected language code (e.g. 'en', 'zh') or None if too short or detection fails.
    """
    if not text or len(text.strip()) < min_length:
        return None
    if detect is None:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def _should_skip_facilitator_language_eval(reply_text: str, min_length: int = 20) -> bool:
    """Return True if we should skip facilitator reply language evaluation."""
    return not reply_text or len(reply_text.strip()) < min_length


def facilitator_reply_needs_language_fix(
    state_values: dict[str, Any],
    user_lang: str,
    min_reply_length: int = 20,
) -> tuple[bool, str]:
    """Check if the facilitator's last reply is in a different language than the user.

    Args:
        state_values: Channel values from facilitator checkpoint (must include "messages").
        user_lang: User's detected language code (e.g. from detected_language).
        min_reply_length: Minimum reply length to run detection.

    Returns:
        (needs_fix, last_ai_content): True if reply language != user language and we should re-invoke;
        last_ai_content is the last AI message content (for logging or replacement).
    """
    messages = state_values.get("messages") or []
    if not messages:
        return False, ""
    last = messages[-1]
    if not isinstance(last, AIMessage) and (not hasattr(last, "type") or getattr(last, "type") != "ai"):
        return False, ""
    content = getattr(last, "content", None) or ""
    reply_text = content if isinstance(content, str) else (str(content) if content else "")
    if _should_skip_facilitator_language_eval(reply_text, min_reply_length):
        return False, reply_text
    reply_lang = detect_text_language(reply_text, min_reply_length)
    if languages_match(user_lang, reply_lang):
        return False, reply_text
    return True, reply_text


async def trigger_facilitator_language_fix(
    agent: Any,
    checkpointer: Any,
    thread_id: str,
    state_values: dict[str, Any],
    user_lang: str,
    config: dict[str, Any],
    facilitator_agent: Any | None = None,
) -> str | None:
    """Re-invoke facilitator to get a reply in the user's language; update checkpoint and return new reply.

    Call this when the facilitator's last reply was detected in a different language than the user.
    Appends a system instruction to rephrase in the target language, then replaces the wrong reply
    and the instruction message in the checkpoint with the corrected reply only.

    Args:
        agent: Facilitator agent (used for ainvoke).
        checkpointer: Checkpointer instance for the facilitator.
        thread_id: Thread ID.
        state_values: Channel values from checkpoint (with messages including the wrong AI reply).
        user_lang: User's language code.
        config: Config dict (configurable.thread_id, etc.).
        facilitator_agent: Optional facilitator agent for aupdate_state (preferred).

    Returns:
        Corrected reply text in user's language, or None on failure.
    """
    user_lang_base = (user_lang or "en").split("-")[0].lower()
    lang_name = LANGUAGE_NAMES.get(user_lang_base, LANGUAGE_NAMES.get(user_lang, "English"))
    instruction = (
        f"[Critical: You must respond only in {lang_name}. "
        f"Rephrase your previous response entirely in {lang_name}.]"
    )
    configurable = dict(config.get("configurable", {}))
    configurable["thread_id"] = thread_id
    fix_config = {"configurable": configurable, "metadata": config.get("metadata", {})}

    # Ensure detected_language stays the user's language for the fix invoke. Otherwise
    # LanguageDetectionMiddleware would re-detect from our English instruction and inject
    # "respond in English", overriding the Chinese rephrase request.
    try:
        await (facilitator_agent or agent).aupdate_state(
            config=fix_config,
            values={"detected_language": user_lang},
        )
    except Exception as e:
        _logger.debug(
            "[ExpertSync] Could not set detected_language before fix invoke: %s",
            str(e),
        )

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=instruction)]},
            config=fix_config,
        )
    except Exception as e:
        _logger.warning(
            "[ExpertSync] Facilitator language-fix re-invoke failed: %s: %s",
            type(e).__name__,
            str(e),
        )
        return None

    result_messages = result.get("messages", [])
    ai_in_result = [m for m in result_messages if isinstance(m, AIMessage) or (hasattr(m, "type") and getattr(m, "type") == "ai")]
    if not ai_in_result:
        _logger.warning("[ExpertSync] Facilitator language-fix re-invoke returned no AI message")
        return None

    new_ai = ai_in_result[-1]
    new_content = getattr(new_ai, "content", None) or ""
    corrected_reply = new_content if isinstance(new_content, str) else str(new_content)

    # Replace checkpoint messages: remove wrong AI and the instruction human message, keep corrected AI only
    # After ainvoke, result_messages are [... prev, user_msg, ai_wrong, human_instruction, ai_correct]
    # We want [... prev, user_msg, ai_correct]
    new_messages = result_messages[:-3] + [new_ai]

    await update_state_with_analysis(
        thread_id=thread_id,
        analysis={"messages": new_messages},
        checkpointer=checkpointer,
        agent=facilitator_agent or agent,
    )

    _logger.info(
        "[ExpertSync] Facilitator language-fix applied (thread_id=%s, user_lang=%s), reply length=%d",
        thread_id,
        user_lang,
        len(corrected_reply),
    )
    return corrected_reply


STATE_EXPERT_SYNC_INTERVAL = 1  # Default sync interval is 3 rounds
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
     
    sync_interval =  STATE_EXPERT_SYNC_INTERVAL
    
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


async def generate_proposal_statements(
    expert_agent,
    users: list[dict[str, Any]] | str,
    conversation_history: list[BaseMessage],
    partner_query: str,
    detected_language: str = "zh",
    thread_id: str = "proposal_generation",
) -> list[dict[str, Any]]:
    """Generate proposal statements for each user from partner search results.
    
    Args:
        expert_agent: The expert agent instance
        users: Either a list of user dictionaries OR a formatted string from search API
        conversation_history: Recent conversation messages for context
        partner_query: The partner query that was used for search
        detected_language: Language code (default: "zh" for Chinese)
        thread_id: Thread ID for expert agent invocation
        
    Returns:
        List of dictionaries, each containing user data and proposal_statement
    """
    # Handle string format (LLM-friendly formatted text)
    if isinstance(users, str):
        users_text = users
        # Count users by splitting on "---"
        user_blocks = users_text.split("\n---\n")
        user_blocks = [block.strip() for block in user_blocks if block.strip() and block.strip() != "---"]
        user_count = len(user_blocks)
        _logger.info("[ExpertSync] Generating proposal statements for %d users (from formatted string)", user_count)
    else:
        if not users:
            _logger.info("[ExpertSync] No users to generate proposals for")
            return []
        user_count = len(users)
        _logger.info("[ExpertSync] Generating proposal statements for %d users", user_count)
    
    # Language name mapping
    language_names = {
        "en": "English",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "it": "Italian",
    }
    language_name = language_names.get(detected_language, "Chinese")
    
    # Format conversation history for context
    conversation_text = format_conversation_history(conversation_history[-10:])  # Last 10 messages
    # Limit conversation text to avoid token limits
    if len(conversation_text) > 2000:
        conversation_text = conversation_text[:2000] + "..."
    
    # Format users for prompt
    if isinstance(users, str):
        # Already in formatted string format - use directly (limit length)
        users_text = users
        if len(users_text) > 10000:  # Limit to avoid token limits
            users_text = users_text[:10000] + "\n\n... (truncated)"
    else:
        # Convert list of dicts to JSON
        users_text = json.dumps(users[:10], ensure_ascii=False, indent=2)
    
    proposal_prompt = f"""Generate short proposal statements for connecting with potential partners.

## Language Requirement

**CRITICAL**: Generate proposals in {language_name} (language code: {detected_language}).
**YOU MUST write all proposals in {language_name}**.

## Context

Partner Search Query: {partner_query}

## Conversation History (for context)

{conversation_text}

## Users Found

{users_text}

## Your Task

Parse the user information above (which may be in formatted text or JSON format) and for each user, generate a short proposal statement (1-2 sentences in {language_name}) that:
1. Explains why this user might be a good fit as a partner
2. References the business idea/needs from the conversation
3. Is personalized based on the user's profile/background
4. Is friendly and professional

**If the users are in formatted text (separated by "---"):**
- Parse each user block (separated by "---")
- Extract key information: user ID, name, location, industry, background, etc.
- Create a structured user object for each

**If the users are in JSON format:**
- Use the existing structure directly

## Output Format

Return a JSON array where each element corresponds to a user (in the same order as they appear) and contains:
- A "user" object with ONLY these fields:
  - id: User ID (required)
  - avatar: Avatar/profile image URL (required - extract from user data, use empty string if not available)
- A "proposal_statement" field with the proposal text in {language_name}

Example structure:
```json
[
  {{
    "user": {{
      "id": "17081",
      "avatar": "https://example.com/avatar/17081.jpg"
    }},
    "proposal_statement": "基于您的AI技术背景，我们相信您可能是我们教育科技项目的理想合作伙伴..."
  }},
  ...
]
```

**Important**:
- Return ONLY the JSON array, no additional text
- Include ALL users from the input (same order)
- Each proposal must be in {language_name}
- Keep proposals concise (1-2 sentences each)
- **MUST include ONLY these fields in each user object**:
  - "id": User ID (required)
  - "avatar": extract from user data if available, use empty string "" if not found
- Do NOT include any other fields in the user object (no name, city, industry, work_experience, education_experience, etc.)
"""
    
    try:
        expert_thread_id = f"proposal_gen_{thread_id}"
        config_dict = {"configurable": {"thread_id": expert_thread_id}}
        
        input_dict = {
            "messages": [HumanMessage(content=proposal_prompt)],
            "detected_language": detected_language,
        }
        
        _logger.info("[ExpertSync] Invoking expert agent for proposal generation...")
        response = await asyncio.wait_for(
            expert_agent.ainvoke(input_dict, config=config_dict),
            timeout=60.0,
        )
        
        # Parse response
        messages = response.get("messages", [])
        if not messages:
            raise ValueError("No messages in proposal generation response")
        
        # Find the last AI message
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            raise ValueError("No AI message in proposal generation response")
        
        content = last_ai_message.content
        if not isinstance(content, str):
            content = str(content)
        
        # Extract JSON from content
        json_str = content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        proposals = json.loads(json_str)
        if not isinstance(proposals, list):
            raise ValueError("Proposal response is not a list")
        
        _logger.info("[ExpertSync] Generated %d proposal statements", len(proposals))
        return proposals
        
    except Exception as e:
        _logger.error("[ExpertSync] Error generating proposals: %s", str(e), exc_info=True)
        # Return users without proposals as fallback
        if isinstance(users, str):
            # Parse string format to create basic user objects
            user_blocks = users.split("\n---\n")
            user_blocks = [block.strip() for block in user_blocks if block.strip() and block.strip() != "---"]
            fallback_users = []
            for block in user_blocks:
                # Extract only id and avatar
                user_obj = {
                    "id": "N/A",
                    "avatar": ""
                }
                for line in block.split("\n"):
                    if line.startswith("用户ID:") or line.startswith("用户创业号:"):
                        user_obj["id"] = line.split(":", 1)[1].strip() if ":" in line else "N/A"
                    # Look for avatar/profile image fields (common patterns)
                    elif "avatar" in line.lower() or "头像" in line or "profile" in line.lower() and "image" in line.lower() or "photo" in line.lower():
                        if ":" in line:
                            user_obj["avatar"] = line.split(":", 1)[1].strip()
                fallback_users.append({"user": user_obj, "proposal_statement": ""})
            return fallback_users
        elif isinstance(users, list):
            # Extract only id and avatar from each user
            fallback_users = []
            for user in users:
                if isinstance(user, dict):
                    # Extract only id and avatar
                    user_simple = {
                        "id": str(user.get("id", user.get("user_id", "N/A"))),
                        "avatar": user.get("avatar") or user.get("avatar_url") or user.get("profile_image") or user.get("photo") or ""
                    }
                    fallback_users.append({"user": user_simple, "proposal_statement": ""})
                else:
                    fallback_users.append({
                        "user": {
                            "id": "N/A",
                            "avatar": ""
                        },
                        "proposal_statement": ""
                    })
            return fallback_users
        else:
            return []


def _parse_user_string(user_text: str) -> dict[str, Any]:
    """Parse a single user's text block into a structured dictionary.
    
    Args:
        user_text: Text block for a single user (separated by "---")
        
    Returns:
        Dictionary with user information
    """
    user_data = {
        "id": "N/A",
        "avatar": ""
    }
    
    # Extract only id and avatar using simple pattern matching
    lines = user_text.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line or line == "---":
            continue
        
        # Extract user ID
        if line.startswith("用户ID:") or line.startswith("用户创业号:"):
            user_data["id"] = line.split(":", 1)[1].strip() if ":" in line else "N/A"
        # Extract avatar (look for common patterns)
        elif "avatar" in line.lower() or "头像" in line or ("profile" in line.lower() and "image" in line.lower()) or "photo" in line.lower() or "image" in line.lower():
            if ":" in line:
                user_data["avatar"] = line.split(":", 1)[1].strip()
    
    return user_data


def extract_users_from_response(search_results: dict[str, Any]) -> list[dict[str, Any]] | str | None:
    """Extract users from search API response.
    
    Handles various response formats from the AI He Huo search API.
    The API may return:
    - A dictionary with "data" as a formatted string (LLM-friendly format) - returns the string
    - A dictionary with "users" as a list - returns the list
    - A dictionary with "data" containing a "users" list - returns the list
    - A list directly - returns the list
    
    Args:
        search_results: Response dictionary from _search_members_api
        
    Returns:
        - If "data" is a string: returns the string (for LLM parsing)
        - Otherwise: list of user dictionaries, or empty list if no users found
        - None if error or no data
    """
    # Check if "data" is a string (formatted text for LLM)
    if "data" in search_results and isinstance(search_results["data"], str):
        data_string = search_results["data"]
        # Return the string as-is for LLM to parse
        # But check if it's empty or just whitespace
        if data_string.strip():
            return data_string
        return []
    
    # Check for structured formats
    users = []
    if "users" in search_results:
        users = search_results["users"]
    elif "data" in search_results and isinstance(search_results["data"], dict):
        if "users" in search_results["data"]:
            users = search_results["data"]["users"]
    elif isinstance(search_results, list):
        users = search_results
    elif "results" in search_results:
        users = search_results["results"]
    
    return users if isinstance(users, list) else []


async def refine_partner_query(
    expert_agent,
    original_query: str,
    previous_queries: list[str],
    conversation_history: list[BaseMessage],
    detected_language: str = "zh",
    thread_id: str = "query_refinement",
) -> str | None:
    """Ask expert agent to refine a partner query that returned no results.
    
    When a partner search query returns no results, it typically means the query
    is too specific. This function asks the expert agent to generate a broader,
    less specific query.
    
    Args:
        expert_agent: The expert agent instance
        original_query: The query that returned no results
        previous_queries: List of all queries attempted so far (to avoid repetition)
        conversation_history: Recent conversation messages for context
        detected_language: Language code (default: "zh" for Chinese)
        thread_id: Thread ID for expert agent invocation
        
    Returns:
        New refined query string, or None if refinement failed
    """
    _logger.info("[ExpertSync] Refining partner query: %s", original_query)
    
    # Language name mapping
    language_names = {
        "en": "English",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "it": "Italian",
    }
    language_name = language_names.get(detected_language, "Chinese")
    
    # Format conversation history for context
    conversation_text = format_conversation_history(conversation_history[-10:])
    # Limit conversation text to avoid token limits
    if len(conversation_text) > 2000:
        conversation_text = conversation_text[:2000] + "..."
    
    # Format previous queries
    previous_queries_text = "\n".join(f"- {q}" for q in previous_queries)
    
    refinement_prompt = f"""The partner search query you generated returned zero results, which means the query is TOO SPECIFIC and needs to be SIGNIFICANTLY BROADENED.

## Original Query (No Results - Too Specific)

{original_query}

## Previous Queries Attempted (All Returned Zero Results)

{previous_queries_text}

## Conversation History (for context)

{conversation_text[:2000]}

## Your Task - SIGNIFICANTLY BROADEN THE QUERY

Generate a NEW, MUCH BROADER partner search query by AGGRESSIVELY REMOVING specific details:

**CRITICAL: You MUST remove or generalize:**
1. **Specific locations** (e.g., "上海" → remove entirely, "虹口" → remove entirely, "老城区" → remove or generalize to "城市")
2. **Very specific skill sets** (e.g., "合规验证与微改造" → "商业运营" or remove)
3. **Specific business models** (e.g., "非咖啡业态监管先例数据库" → "商业创新" or remove)
4. **Specific industries** (e.g., "社区商业" → "商业" or "创业")
5. **Overly detailed requirements** - keep only the core need

**Examples of aggressive simplification:**
- "寻找有上海老城区社区商业落地经验的创业者，熟悉虹口老公房沿街铺位合规验证与微改造"
  → "寻找有商业运营经验的创业者" (removed location, specific skills, specific area)
  
- "寻找有AI技术背景的创业者，希望合作开发教育科技产品"
  → "寻找有技术背景的创业者" (removed specific tech, specific industry)
  
- "寻找对教育科技领域感兴趣的投资人，有相关行业经验"
  → "寻找对创业项目感兴趣的投资人" (removed specific industry)

**Requirements:**
1. Make it MUCH simpler and broader than previous queries
2. Remove ALL specific locations if present
3. Remove very specific technical terms or niche skills
4. Keep only the most general core need (e.g., "创业者", "投资人", "合作伙伴")
5. Still 10-20 words in Chinese
6. Natural sentence/phrase suitable for semantic search
7. Must be meaningfully different from all previous queries

**If the query is still too specific after removing details, make it even more general.**
**The goal is to find ANY relevant users, not perfect matches.**

## Output Format

Return ONLY a JSON object with this structure:
```json
{{
  "refined_query": "Your new MUCH BROADER query in Chinese (10-20 words)"
}}
```

**Important**:
- Return ONLY the JSON object, no additional text
- The refined_query must be in Chinese
- It must be 10-20 words
- It must be SIGNIFICANTLY broader than previous queries
- Remove specific locations, specific skills, specific industries
- Make it as general as possible while still being relevant
"""
    
    try:
        expert_thread_id = f"query_refinement_{thread_id}"
        config_dict = {"configurable": {"thread_id": expert_thread_id}}
        
        input_dict = {
            "messages": [HumanMessage(content=refinement_prompt)],
            "detected_language": detected_language,
        }
        
        _logger.info("[ExpertSync] Invoking expert agent for query refinement...")
        response = await asyncio.wait_for(
            expert_agent.ainvoke(input_dict, config=config_dict),
            timeout=120.0,
        )
        
        # Parse response
        messages = response.get("messages", [])
        if not messages:
            raise ValueError("No messages in query refinement response")
        
        # Find the last AI message
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            raise ValueError("No AI message in query refinement response")
        
        content = last_ai_message.content
        if not isinstance(content, str):
            content = str(content)
        
        # Extract JSON from content
        json_str = content.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        elif json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        result = json.loads(json_str)
        if not isinstance(result, dict) or "refined_query" not in result:
            raise ValueError("Invalid response format: missing 'refined_query' field")
        
        refined_query = result["refined_query"]
        if not isinstance(refined_query, str):
            raise ValueError("Invalid response format: 'refined_query' is not a string")
        
        refined_query = refined_query.strip()
        
        # Validate query length
        if len(refined_query) < 6:
            _logger.warning("[ExpertSync] Refined query too short (%d chars), rejecting", len(refined_query))
            return None
        
        if len(refined_query) > 100:
            _logger.warning("[ExpertSync] Refined query too long (%d chars), truncating", len(refined_query))
            refined_query = refined_query[:100]
        
        _logger.info("[ExpertSync] Query refined successfully: %s", refined_query)
        return refined_query
        
    except Exception as e:
        _logger.error("[ExpertSync] Error refining partner query: %s", str(e), exc_info=True)
        return None


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
    
    # Get detected language from state (default to English)
    detected_language = state.get("detected_language", "en")
    _logger.info("  Detected language: %s", detected_language)
    
    # Language name mapping for prompt
    language_names = {
        "en": "English",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "it": "Italian",
    }
    language_name = language_names.get(detected_language, "English")
    
    # Build analysis prompt for expert
    analysis_prompt = f"""Analyze this conversation using your expertise.

## Language Requirement

**CRITICAL**: The user is communicating in {language_name} (language code: {detected_language}).
**YOU MUST respond in {language_name} for ALL output**, including:
- Expert guidance text
- Canvas data content (all text fields, descriptions, labels)
- Canvas update summary
- Any other text in your response

**DO NOT use English** - use {language_name} throughout your entire response.

## Conversation History (Last 10 Rounds)

{conversation_text}

## Current State

- Current guidance to facilitator: {current_guidance}
- Current canvas: {json.dumps(current_canvas, ensure_ascii=False, indent=2) if current_canvas else "None"}

## Your Task

Analyze this conversation and provide:

1. **Expert Guidance** (2-4 sentences in {language_name}):
   Strategic direction for the facilitator. What should they focus on in upcoming conversations?
   Be specific and actionable.
   **MUST be written in {language_name}**.

2. **Canvas Data** (structured JSON following the template):
   Use the canvas template structure below to assess the current state.
   **ALL text content in the canvas (array items, descriptions, etc.) MUST be in {language_name}**.
   For example, if the canvas has customer segments like ["Small business owners"], translate them to {language_name}.

3. **Canvas Update Summary** (2-3 sentences in {language_name}):
   A brief summary of what was updated in the canvas, written in {language_name} (language code: {detected_language}).
   This summary will be sent directly to the user, so make it clear, friendly, and informative.
   Focus on what new information was added or what changed in the Business Model Canvas.
   Examples in {language_name}:
   - "我们已更新了您的商业模式画布，添加了关于客户细分和价值主张的新信息。"
   - "您的画布现在包含了关于收入流和关键合作伙伴的详细信息。"
   - "画布已更新，包含了关于渠道和客户关系的信息。"

4. **Partner Query** (optional, string in Chinese, 10-20 words):
   If the business idea is sufficiently developed and you have enough information about the business needs,
   generate a partner search query. This should be a natural Chinese sentence/phrase describing the ideal partner
   based on the business idea, target market, and specific needs identified in the conversation.
   The query will be used to search for potential partners on the AI He Huo platform.
   Only include this field if you have enough information to create a meaningful search query.
   Example: "寻找有AI技术背景的创业者，希望合作开发教育科技产品"
   Example: "寻找对教育科技领域感兴趣的投资人，有相关行业经验"
   If the business idea is not yet developed enough, omit this field.

## Canvas Template

{canvas_template}

## Output Format

Provide your analysis as a JSON object with this exact structure:

```json
{{
  "expert_guidance": "Your 2-4 sentence strategic guidance in {language_name}...",
  "canvas": {{
    ... follow the template structure above, with ALL text content in {language_name} ...
  }},
  "canvas_update_summary": "A 2-3 sentence summary in {language_name} describing what was updated in the canvas.",
  "partner_query": "Optional: 10-20 word Chinese sentence describing ideal partner (only if business idea is sufficiently developed)"
}}
```

**Important**: 
- Return ONLY the JSON object, no additional text before or after
- Follow the canvas template structure
- **ALL text content MUST be in {language_name}** - expert_guidance, canvas content, and canvas_update_summary
- Include required fields: "expert_guidance", "canvas", and "canvas_update_summary"
- Include "partner_query" only if the business idea is sufficiently developed (10-20 words in Chinese)
- **DO NOT use English** - use {language_name} for everything (except partner_query which should always be in Chinese)
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
        
        _logger.info("[ExpertSync] Starting expert agent invocation (thread_id=%s)...", expert_thread_id)
        _logger.info("[ExpertSync] About to call ainvoke at %s", datetime.utcnow().isoformat())
        
        # Create the input dict with conversation history so LanguageDetectionMiddleware can detect language
        # Include conversation history in messages so the expert agent can detect language from user messages
        expert_messages = []
        # Add conversation history first (so language can be detected)
        for msg in conversation_history[-5:]:  # Include last 5 messages for language detection
            expert_messages.append(msg)
        # Then add the analysis prompt
        expert_messages.append(HumanMessage(content=analysis_prompt))
        
        input_dict = {"messages": expert_messages}
        # Explicitly set detected_language in initial state if we have it (helps ensure language is used)
        if detected_language:
            input_dict["detected_language"] = detected_language
            _logger.info("[ExpertSync] Setting detected_language=%s in initial state", detected_language)
        
        config_dict = {
            "configurable": {"thread_id": expert_thread_id},
        }
        _logger.info("[ExpertSync] Input prepared: messages=%d (including %d conversation messages for language detection), detected_language=%s, config=%s", 
                     len(input_dict["messages"]), len(conversation_history[-5:]), detected_language, config_dict)
        
        # Use ainvoke with timeout
        _logger.info("[ExpertSync] Starting ainvoke with timeout (120s)...")
        start_time = datetime.utcnow()
        
        try:
            # Use wait_for with timeout
            response = await asyncio.wait_for(
                expert_agent.ainvoke(input_dict, config=config_dict),
                timeout=120.0,  # 120 second timeout
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

        # Language evaluation: if canvas language != user language, ask expert to re-output in user's language
        canvas = analysis.get("canvas")
        # Preserve partner_query before language fix (it will be lost in re-invoke)
        preserved_partner_query = analysis.get("partner_query")
        if preserved_partner_query:
            _logger.info(
                "[ExpertSync] Preserving partner_query before language fix: %s",
                preserved_partner_query[:100] + "..." if len(preserved_partner_query) > 100 else preserved_partner_query,
            )
        
        if not _should_skip_language_eval(canvas):
            canvas_lang = detect_canvas_language(canvas)
            user_lang = detected_language
            if not languages_match(user_lang, canvas_lang):
                _logger.info(
                    "[ExpertSync] Canvas language %s != user language %s, requesting expert to re-output in user language.",
                    canvas_lang,
                    user_lang,
                )
                user_lang_base = (user_lang or "en").split("-")[0]
                lang_name = language_names.get(
                    user_lang_base, language_names.get(user_lang, "English")
                )
                fix_prompt = f"""The previous expert output (expert_guidance, canvas, canvas_update_summary) was in {canvas_lang} but the user uses {user_lang} ({lang_name}).
Re-output the exact same analysis as a single JSON object with the same structure, but with ALL text translated into {lang_name} ({user_lang}).
Return ONLY the JSON object, no markdown or extra text. Use the same keys: "expert_guidance", "canvas", "canvas_update_summary"."""

                fix_input: dict[str, Any] = {
                    "messages": [HumanMessage(content=fix_prompt)]
                }
                if detected_language:
                    fix_input["detected_language"] = detected_language

                try:
                    response2 = await asyncio.wait_for(
                        expert_agent.ainvoke(fix_input, config=config_dict),
                        timeout=120.0,
                    )
                    analysis = parse_expert_response(response2)
                    # Restore preserved partner_query after language fix
                    if preserved_partner_query:
                        analysis["partner_query"] = preserved_partner_query
                        _logger.info(
                            "[ExpertSync] Restored partner_query after language fix"
                        )
                    _logger.info(
                        "[ExpertSync] Language-fix re-invoke succeeded, using updated analysis"
                    )
                except (asyncio.TimeoutError, ValueError) as e:
                    _logger.warning(
                        "[ExpertSync] Language-fix re-invoke failed (%s), keeping original analysis",
                        str(e),
                    )

        # Partner search: if partner_query exists, search for partners with retry loop
        partner_query = analysis.get("partner_query")
        _logger.info(
            "[ExpertSync] Partner query check: exists=%s, value=%s, length=%s",
            partner_query is not None,
            partner_query[:100] + "..." if partner_query and len(partner_query) > 100 else partner_query,
            len(partner_query) if partner_query else 0,
        )
        if partner_query and len(partner_query.strip()) > 5:
            _logger.info("[ExpertSync] Partner query found, executing search with refinement loop...")
            
            # Import search API function
            from deepagents.middleware.aihehuo import _search_members_api
            
            # Initialize retry loop variables
            current_query = partner_query
            max_retries = int(os.getenv("PARTNER_SEARCH_MAX_RETRIES", "3"))
            retry_count = 0
            users = None  # Can be list, string, or None
            attempted_queries = [partner_query]
            
            def _has_users(users_data: list[dict[str, Any]] | str | None) -> bool:
                """Check if users_data contains any users."""
                if users_data is None:
                    return False
                if isinstance(users_data, str):
                    # Count users in string format
                    user_blocks = users_data.split("\n---\n")
                    user_blocks = [block.strip() for block in user_blocks if block.strip() and block.strip() != "---"]
                    return len(user_blocks) > 0
                if isinstance(users_data, list):
                    return len(users_data) > 0
                return False
            
            while not _has_users(users) and retry_count < max_retries:
                try:
                    _logger.info(
                        "[ExpertSync] Partner search attempt %d/%d with query: %s",
                        retry_count + 1, max_retries, current_query
                    )
                    
                    # Call search API
                    search_results = _search_members_api(
                        query=current_query,
                        max_results=10,
                        page=1,
                    )
                    
                    # Check for API errors
                    if "error" in search_results:
                        _logger.warning(
                            "[ExpertSync] Partner search API error: %s",
                            search_results.get("message", "Unknown error")
                        )
                        # Don't retry on API errors - they're not about query specificity
                        break
                    
                    # Extract users from response
                    users_data = extract_users_from_response(search_results)
                    
                    # Check if we got users (string format or list format)
                    if isinstance(users_data, str):
                        # String format - count users by splitting
                        user_blocks = users_data.split("\n---\n")
                        user_blocks = [block.strip() for block in user_blocks if block.strip() and block.strip() != "---"]
                        user_count = len(user_blocks)
                        if user_count > 0:
                            users = users_data  # Keep as string for LLM parsing
                            _logger.info(
                                "[ExpertSync] Found %d users (string format) after %d retry(ies) with query: %s",
                                user_count, retry_count, current_query
                            )
                            break
                    elif isinstance(users_data, list) and len(users_data) > 0:
                        users = users_data
                        _logger.info(
                            "[ExpertSync] Found %d users after %d retry(ies) with query: %s",
                            len(users), retry_count, current_query
                        )
                        break
                    
                    # No users found - refine query if we have retries left
                    if retry_count < max_retries - 1:
                        _logger.info(
                            "[ExpertSync] No users found with query '%s', refining query (attempt %d/%d)...",
                            current_query, retry_count + 1, max_retries
                        )
                        new_query = await refine_partner_query(
                            expert_agent=expert_agent,
                            original_query=current_query,
                            previous_queries=attempted_queries,
                            conversation_history=conversation_history,
                            detected_language=detected_language,
                            thread_id=thread_id,
                        )
                        
                        if new_query and new_query.strip() and new_query not in attempted_queries:
                            current_query = new_query
                            attempted_queries.append(new_query)
                            retry_count += 1
                        else:
                            _logger.warning(
                                "[ExpertSync] Query refinement failed or produced duplicate/invalid query, stopping retries"
                            )
                            break
                    else:
                        # Max retries reached
                        break
                        
                except Exception as e:
                    _logger.error(
                        "[ExpertSync] Error during partner search retry: %s",
                        str(e),
                        exc_info=True
                    )
                    break
            
            # Generate proposals if users found
            if _has_users(users):
                if isinstance(users, str):
                    # Count users in string format for logging
                    user_blocks = users.split("\n---\n")
                    user_blocks = [block.strip() for block in user_blocks if block.strip() and block.strip() != "---"]
                    user_count = len(user_blocks)
                    _logger.info("[ExpertSync] Found %d users (string format) from partner search", user_count)
                else:
                    user_count = len(users) if isinstance(users, list) else 0
                    _logger.info("[ExpertSync] Found %d users from partner search", user_count)
                
                # Generate proposal statements for each user
                proposals = await generate_proposal_statements(
                    expert_agent=expert_agent,
                    users=users,
                    conversation_history=conversation_history,
                    partner_query=current_query,  # Use the final query that succeeded
                    detected_language=detected_language,
                    thread_id=thread_id,
                )
                analysis["partner_search_results"] = proposals
                analysis["partner_query"] = current_query  # Update with final query used
                _logger.info(
                    "[ExpertSync] Partner search completed: %d proposals generated",
                    len(proposals) if isinstance(proposals, list) else 0
                )
                # Printout partner search results structure
                if proposals and isinstance(proposals, list) and len(proposals) > 0:
                    _logger.info(
                        "[ExpertSync] Partner search results structure (first result): user_id=%s, avatar=%s, has_proposal=%s",
                        proposals[0].get("user", {}).get("id", "N/A"),
                        proposals[0].get("user", {}).get("avatar", "N/A"),
                        bool(proposals[0].get("proposal_statement")),
                    )
                    import json
                    _logger.info(
                        "[ExpertSync] Full partner_search_results (first result): %s",
                        json.dumps(proposals[0], ensure_ascii=False, indent=2),
                    )
            else:
                _logger.warning(
                    "[ExpertSync] No users found after %d attempts with queries: %s",
                    retry_count + 1, attempted_queries
                )
                analysis["partner_search_results"] = []
                # Keep the last attempted query in partner_query field
                analysis["partner_query"] = current_query
                _logger.info(
                    "[ExpertSync] Partner search results set to empty list (no users found after retries)"
                )
        else:
            _logger.debug("[ExpertSync] No partner query or query too short, skipping search")
            # Don't set partner_search_results if no query

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
    
    # Handle optional canvas_update_summary field
    if "canvas_update_summary" not in analysis:
        _logger.debug("[ExpertSync] No canvas_update_summary provided, will use default")
        # Don't set a default - let it be None if not provided
    
    # Handle optional partner_query field
    if "partner_query" in analysis:
        partner_query = analysis["partner_query"]
        if isinstance(partner_query, str):
            partner_query = partner_query.strip()
            # Validate query length (should be 10-20 words approximately)
            # For Chinese, we can count characters (roughly 1-2 chars per word)
            # So 10-20 words ≈ 10-40 characters
            if len(partner_query) < 6:  # Minimum 6 chars (API requirement is 5, but we want meaningful queries)
                _logger.warning("[ExpertSync] Partner query too short (%d chars), ignoring", len(partner_query))
                analysis["partner_query"] = None
            elif len(partner_query) > 100:  # Too long, likely not 10-20 words
                _logger.warning("[ExpertSync] Partner query too long (%d chars), truncating", len(partner_query))
                analysis["partner_query"] = partner_query[:100]
            else:
                analysis["partner_query"] = partner_query
                _logger.info("[ExpertSync] Partner query extracted: %s", partner_query[:50])
        else:
            _logger.warning("[ExpertSync] Partner query is not a string, ignoring")
            analysis["partner_query"] = None
    else:
        _logger.debug("[ExpertSync] No partner_query provided")
        # Don't set a default - let it be None if not provided
    
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
) -> dict[str, Any]:
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
        
    Returns:
        The analysis dict containing canvas, expert_guidance, and related fields
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
    _logger.info("[ExpertSync] Returning analysis with canvas: %s", "canvas" in analysis)
    
    return analysis
