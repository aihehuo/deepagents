"""Unit tests for expert_sync helper functions."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from apps.business_cofounder_api.expert_sync import (
    _should_skip_language_eval,
    detect_canvas_language,
    extract_recent_rounds,
    extract_text_from_canvas,
    format_conversation_history,
    languages_match,
    parse_expert_response,
    should_trigger_expert,
)
from deepagents.state import DualAgentState


class TestShouldTriggerExpert:
    """Test should_trigger_expert function."""

    def test_should_trigger_true_at_5_rounds(self) -> None:
        """Should return True when the 5-round sync interval is reached."""
        state: DualAgentState = {
            "messages": [],
            "conversation_round": 5,
            "needs_expert_sync": True,
        }

        assert should_trigger_expert(state) is True

    def test_should_trigger_true_at_10_rounds(self) -> None:
        """Should return True at multiples of the 5-round sync interval."""
        state: DualAgentState = {
            "messages": [],
            "conversation_round": 10,
            "needs_expert_sync": True,
        }

        assert should_trigger_expert(state) is True

    def test_should_trigger_false_before_5_rounds(self) -> None:
        """Should return False before reaching the 5-round sync interval."""
        state: DualAgentState = {
            "messages": [],
            "conversation_round": 4,
            "needs_expert_sync": False,
        }

        assert should_trigger_expert(state) is False

    def test_should_trigger_true_with_explicit_flag(self) -> None:
        """Should return True if needs_expert_sync is explicitly True."""
        state: DualAgentState = {
            "messages": [],
            "conversation_round": 4,  # Before the interval
            "needs_expert_sync": True,  # But flag is set
        }

        # Should trigger because of explicit flag
        assert should_trigger_expert(state) is True

    def test_should_trigger_true_based_on_interval(self) -> None:
        """Should return True when interval is reached even without explicit flag."""
        # At round 5 with last_sync = 0, interval = 5, should trigger
        state: DualAgentState = {
            "messages": [],
            "conversation_round": 5,
            "last_expert_sync": 0,
            "needs_expert_sync": False,
        }

        assert should_trigger_expert(state) is True

    def test_should_trigger_handles_missing_fields(self) -> None:
        """Should handle state missing optional fields."""
        state: DualAgentState = {
            "messages": [],
        }

        # Should not raise error and return False
        assert should_trigger_expert(state) is False


class TestExtractRecentRounds:
    """Test extract_recent_rounds function."""

    def test_extract_exactly_10_messages(self) -> None:
        """Should extract last 10 messages (5 rounds)."""
        messages = []
        for i in range(20):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"User {i}"))
            else:
                messages.append(AIMessage(content=f"AI {i}"))

        result = extract_recent_rounds(messages, rounds=5)

        # 5 rounds = 10 messages
        assert len(result) == 10
        # Should get the last 10 messages
        assert result[0].content == "User 10"
        assert result[-1].content == "AI 19"

    def test_extract_fewer_than_requested_messages(self) -> None:
        """Should handle conversations with fewer than requested messages."""
        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            HumanMessage(content="User 2"),
            AIMessage(content="AI 2"),
        ]

        result = extract_recent_rounds(messages, rounds=10)

        # Should return all available messages
        assert len(result) == 4
        assert result[0].content == "User 1"
        assert result[-1].content == "AI 2"

    def test_extract_empty_messages_list(self) -> None:
        """Should handle empty messages list."""
        messages = []

        result = extract_recent_rounds(messages, rounds=10)

        assert result == []

    def test_extract_single_round(self) -> None:
        """Should extract single round (2 messages)."""
        messages = []
        for i in range(10):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"User {i}"))
            else:
                messages.append(AIMessage(content=f"AI {i}"))

        result = extract_recent_rounds(messages, rounds=1)

        assert len(result) == 2
        assert result[0].content == "User 8"
        assert result[1].content == "AI 9"

    def test_extract_odd_number_of_messages(self) -> None:
        """Should handle odd number of messages gracefully."""
        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            HumanMessage(content="User 2"),
        ]

        result = extract_recent_rounds(messages, rounds=5)

        # Should return all messages
        assert len(result) == 3


class TestFormatConversationHistory:
    """Test format_conversation_history function."""

    def test_format_basic_conversation(self) -> None:
        """Should format messages as 'User: ...' and 'Assistant: ...'"""
        messages = [
            HumanMessage(content="Hello, how are you?"),
            AIMessage(content="I'm doing well, thank you!"),
            HumanMessage(content="What's the weather?"),
            AIMessage(content="I don't have access to weather data."),
        ]

        result = format_conversation_history(messages)

        assert "User: Hello, how are you?" in result
        assert "Assistant: I'm doing well, thank you!" in result
        assert "User: What's the weather?" in result
        assert "Assistant: I don't have access to weather data." in result

    def test_format_empty_messages(self) -> None:
        """Should handle empty messages list."""
        messages = []

        result = format_conversation_history(messages)

        assert result == ""

    def test_format_preserves_order(self) -> None:
        """Should preserve message order."""
        messages = [
            HumanMessage(content="First"),
            AIMessage(content="Second"),
            HumanMessage(content="Third"),
        ]

        result = format_conversation_history(messages)

        # Messages are separated by \n\n, so split by that
        parts = result.strip().split("\n\n")
        assert "First" in parts[0]
        assert "Second" in parts[1]
        assert "Third" in parts[2]

    def test_format_multiline_content(self) -> None:
        """Should handle multiline message content."""
        messages = [
            HumanMessage(content="Line 1\nLine 2\nLine 3"),
            AIMessage(content="Response\nWith multiple\nLines"),
        ]

        result = format_conversation_history(messages)

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Response" in result
        assert "With multiple" in result


class TestParseExpertResponse:
    """Test parse_expert_response function."""

    def test_parse_valid_json_response(self) -> None:
        """Should extract expert_guidance and canvas from valid JSON."""
        response_content = json.dumps({
            "expert_guidance": "Focus on customer validation",
            "canvas": {
                "current_stage": "idea_exploration",
                "insights": ["Good technical background"],
            }
        })
        
        # parse_expert_response expects a dict with "messages" key
        response = {
            "messages": [AIMessage(content=response_content)]
        }

        result = parse_expert_response(response)

        assert result["expert_guidance"] == "Focus on customer validation"
        assert result["canvas"]["current_stage"] == "idea_exploration"
        assert "Good technical background" in result["canvas"]["insights"]

    def test_parse_json_with_extra_fields(self) -> None:
        """Should keep all fields from expert response."""
        response_content = json.dumps({
            "expert_guidance": "Test guidance",
            "canvas": {"field": "value"},
            "extra_field": "not_ignored",
        })
        
        response = {"messages": [AIMessage(content=response_content)]}
        result = parse_expert_response(response)

        assert result["expert_guidance"] == "Test guidance"
        assert result["canvas"]["field"] == "value"

    def test_parse_missing_expert_guidance_field(self) -> None:
        """Should add fallback if expert_guidance missing."""
        response_content = json.dumps({
            "canvas": {"field": "value"},
        })
        
        response = {"messages": [AIMessage(content=response_content)]}
        result = parse_expert_response(response)

        # Should have fallback guidance
        assert "expert_guidance" in result
        assert isinstance(result["expert_guidance"], str)
        assert len(result["expert_guidance"]) > 0

    def test_parse_missing_canvas_field(self) -> None:
        """Should add fallback canvas if missing."""
        response_content = json.dumps({
            "expert_guidance": "Test guidance",
        })
        
        response = {"messages": [AIMessage(content=response_content)]}
        result = parse_expert_response(response)

        # Should have fallback canvas
        assert "canvas" in result
        assert isinstance(result["canvas"], dict)

    def test_parse_malformed_json(self) -> None:
        """Should raise ValueError for malformed JSON."""
        response_content = "This is not JSON { invalid"
        response = {"messages": [AIMessage(content=response_content)]}

        with pytest.raises(ValueError):
            parse_expert_response(response)

    def test_parse_empty_response(self) -> None:
        """Should raise ValueError for empty response."""
        response = {"messages": []}

        with pytest.raises(ValueError):
            parse_expert_response(response)

    def test_parse_json_with_nested_structures(self) -> None:
        """Should handle complex nested JSON structures."""
        response_content = json.dumps({
            "expert_guidance": "Complex guidance",
            "canvas": {
                "level1": {
                    "level2": {
                        "level3": ["item1", "item2"],
                    }
                },
                "array": [{"key": "value"}, {"key2": "value2"}],
            }
        })
        
        response = {"messages": [AIMessage(content=response_content)]}
        result = parse_expert_response(response)

        assert result["expert_guidance"] == "Complex guidance"
        assert result["canvas"]["level1"]["level2"]["level3"] == ["item1", "item2"]
        assert len(result["canvas"]["array"]) == 2

    def test_parse_json_with_unicode(self) -> None:
        """Should handle Unicode in JSON response."""
        response_content = json.dumps({
            "expert_guidance": "Focus on 客户验证",
            "canvas": {
                "stage": "探索阶段",
                "insights": ["技术背景很好 🚀"],
            }
        }, ensure_ascii=False)
        
        response = {"messages": [AIMessage(content=response_content)]}
        result = parse_expert_response(response)

        assert "客户验证" in result["expert_guidance"]
        assert result["canvas"]["stage"] == "探索阶段"
        assert "🚀" in result["canvas"]["insights"][0]

    def test_parse_json_wrapped_in_markdown(self) -> None:
        """Should extract JSON even if wrapped in markdown code blocks."""
        response_content = """```json
{
  "expert_guidance": "Test guidance",
  "canvas": {"field": "value"}
}
```"""
        
        response = {"messages": [AIMessage(content=response_content)]}
        result = parse_expert_response(response)

        assert result["expert_guidance"] == "Test guidance"
        assert result["canvas"]["field"] == "value"

    def test_parse_json_with_null_expert_guidance(self) -> None:
        """Should add fallback for null expert_guidance."""
        response_content = json.dumps({
            "expert_guidance": None,
            "canvas": {
                "field": "value",
                "insights": [],
            }
        })
        
        response = {"messages": [AIMessage(content=response_content)]}
        result = parse_expert_response(response)

        # Should have fallback for None guidance (or allow None)
        assert "expert_guidance" in result


class TestLanguageEvaluationHelpers:
    """Unit tests for language evaluation helpers (extract_text_from_canvas, detect_canvas_language, languages_match)."""

    def test_extract_text_from_canvas_nested(self) -> None:
        """Should recursively collect string values from nested dict/list."""
        canvas: dict[str, Any] = {
            "key_partners": ["Partner A", "Partner B"],
            "value_propositions": ["Value X Y Z"],
            "nested": {"inner": ["item one", "item two"]},
        }
        result = extract_text_from_canvas(canvas)
        assert "Partner A" in result
        assert "Partner B" in result
        assert "Value X Y Z" in result
        assert "item one" in result
        assert "item two" in result

    def test_extract_text_from_canvas_skips_error_metadata(self) -> None:
        """Should skip status/message when they look like error/fallback."""
        canvas = {"status": "analysis_unavailable", "message": "error"}
        assert extract_text_from_canvas(canvas) == ""

    def test_extract_text_from_canvas_empty(self) -> None:
        """Should return empty string for empty canvas."""
        assert extract_text_from_canvas({}) == ""

    def test_extract_text_from_canvas_ignores_short_strings(self) -> None:
        """Should ignore strings shorter than 3 chars."""
        canvas = {"a": "x", "b": "ab", "c": "abc"}
        result = extract_text_from_canvas(canvas)
        assert result == "abc"

    def test_detect_canvas_language_too_short(self) -> None:
        """Should return None when text below min_length."""
        canvas = {"x": "hi"}
        assert detect_canvas_language(canvas, min_length=50) is None

    def test_detect_canvas_language_empty(self) -> None:
        """Should return None for empty canvas."""
        assert detect_canvas_language({}, min_length=50) is None

    def test_languages_match_none(self) -> None:
        """None canvas_lang means no check -> match."""
        assert languages_match("en", None) is True
        assert languages_match("zh", None) is True

    def test_languages_match_same_base(self) -> None:
        """Same base code should match."""
        assert languages_match("zh", "zh") is True
        assert languages_match("zh-cn", "zh") is True
        assert languages_match("zh", "zh-cn") is True
        assert languages_match("en", "en") is True

    def test_languages_match_different(self) -> None:
        """Different languages should not match."""
        assert languages_match("en", "zh") is False
        assert languages_match("zh", "ja") is False

    def test_should_skip_language_eval(self) -> None:
        """Should skip when missing, empty, or non-content blob."""
        assert _should_skip_language_eval(None) is True
        assert _should_skip_language_eval({}) is True
        assert _should_skip_language_eval({"status": "x", "message": "y"}) is True
        assert _should_skip_language_eval({"key_partners": []}) is False
        assert _should_skip_language_eval({"value_propositions": ["a", "b"]}) is False
