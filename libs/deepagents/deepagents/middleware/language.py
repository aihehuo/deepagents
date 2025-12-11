"""Middleware for detecting user language and instructing the LLM to respond in the same language."""

from collections.abc import Awaitable, Callable
from typing import NotRequired, TypedDict, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage

try:
    from langdetect import detect, DetectorFactory, LangDetectException

    # Set seed for consistent results
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    # Create dummy functions for type checking
    def detect(text: str) -> str:  # type: ignore[misc]
        """Dummy function when langdetect is not available."""
        return "en"

    class LangDetectException(Exception):  # type: ignore[misc]
        """Dummy exception when langdetect is not available."""

        pass


class LanguageDetectionState(AgentState):
    """State for the language detection middleware."""

    detected_language: NotRequired[str]
    """The detected language code (e.g., 'en', 'zh', 'fr')."""


class LanguageDetectionStateUpdate(TypedDict):
    """A state update for the language detection middleware."""

    detected_language: NotRequired[str]
    """The detected language code (e.g., 'en', 'zh', 'fr')."""


# Language code to language name mapping for system prompt
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
}


def _get_language_name(lang_code: str) -> str:
    """Get the human-readable language name from a language code.

    Args:
        lang_code: The language code (e.g., 'en', 'zh-cn').

    Returns:
        The language name (e.g., 'English', 'Chinese (Simplified)').
    """
    # Try exact match first
    if lang_code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[lang_code]

    # Try base language code (e.g., 'zh' for 'zh-cn')
    base_code = lang_code.split("-")[0]
    if base_code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[base_code]

    # Fallback to capitalized code
    return lang_code.upper()


LANGUAGE_SYSTEM_PROMPT_TEMPLATE = """## Language Preference

The user is communicating in {language_name} ({language_code}).
**IMPORTANT**: You must respond in the same language that the user is using.
Match the user's language for all your responses, including explanations, code comments, and any text output."""


class LanguageDetectionMiddleware(AgentMiddleware):
    """Middleware for detecting user language and instructing the LLM to respond in the same language.

    This middleware:
    1. Detects the language from user messages in the conversation
    2. Stores the detected language in state
    3. Injects a system prompt instruction to respond in that language

    Example:
        ```python
        from deepagents.middleware.language import LanguageDetectionMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            model="anthropic:claude-sonnet-4-20250514",
            middleware=[LanguageDetectionMiddleware()],
        )
        ```

    Note:
        Requires the `langdetect` package to be installed:
        ```bash
        pip install langdetect
        ```

        If `langdetect` is not available, the middleware will gracefully degrade
        and assume English as the default language.
    """

    state_schema = LanguageDetectionState

    def __init__(
        self,
        *,
        min_text_length: int = 10,
        default_language: str = "en",
        system_prompt_template: str | None = None,
    ) -> None:
        """Initialize the LanguageDetectionMiddleware.

        Args:
            min_text_length: Minimum text length required for language detection.
                Shorter texts may not be reliably detected. Defaults to 10.
            default_language: Default language code to use if detection fails or
                langdetect is not available. Defaults to 'en' (English).
            system_prompt_template: Optional custom system prompt template.
                Should contain {language_name} and {language_code} placeholders.
                If None, uses the default template.
        """
        if not LANGDETECT_AVAILABLE:
            import warnings

            warnings.warn(
                "langdetect package not installed. Language detection will be disabled. "
                "Install it with: pip install langdetect",
                UserWarning,
                stacklevel=2,
            )

        self.min_text_length = min_text_length
        self.default_language = default_language
        self.system_prompt_template = (
            system_prompt_template or LANGUAGE_SYSTEM_PROMPT_TEMPLATE
        )

    def _detect_language_from_messages(
        self, messages: list, state: LanguageDetectionState
    ) -> str | None:
        """Detect language from user messages.

        Args:
            messages: List of messages from the conversation.
            state: Current agent state.

        Returns:
            Detected language code, or None if detection failed or not enough text.
        """
        # If language already detected and stored in state, reuse it
        if "detected_language" in state and state["detected_language"]:
            return state["detected_language"]

        if not LANGDETECT_AVAILABLE:
            return None

        # Collect text from recent user messages
        user_texts: list[str] = []
        for message in reversed(messages):  # Start from most recent
            if isinstance(message, HumanMessage):
                content = message.content
                if isinstance(content, str) and len(content.strip()) >= self.min_text_length:
                    user_texts.append(content)
                    # Collect enough text for reliable detection (at least 50 chars)
                    if sum(len(text) for text in user_texts) >= 50:
                        break

        if not user_texts:
            return None

        # Combine user texts for detection
        combined_text = " ".join(user_texts)

        try:
            detected = detect(combined_text)
            return detected
        except LangDetectException:
            return None

    def before_agent(
        self,
        state: LanguageDetectionState,
        runtime,  # Runtime type from langgraph
    ) -> LanguageDetectionStateUpdate | None:
        """Detect language from messages before agent execution.

        This runs at the start of each agent interaction to detect the user's language
        from the conversation history.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            Updated state with detected_language populated, or None if no update needed.
        """
        # Get messages from state
        messages = state.get("messages", [])

        if not messages:
            return None

        detected_lang = self._detect_language_from_messages(messages, state)

        if detected_lang:
            return LanguageDetectionStateUpdate(detected_language=detected_lang)

        return None

    def _build_language_prompt(self, language_code: str) -> str:
        """Build the language instruction system prompt.

        Args:
            language_code: The detected language code.

        Returns:
            The formatted system prompt instruction.
        """
        language_name = _get_language_name(language_code)
        return self.system_prompt_template.format(
            language_name=language_name, language_code=language_code
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject language instruction into the system prompt.

        This runs on every model call to ensure the language instruction is always present.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Get detected language from state
        state = cast("LanguageDetectionState", request.state)
        detected_language = state.get("detected_language")

        # If no language detected yet, try to detect from messages in the request
        if not detected_language:
            # Try to get messages from request state or request itself
            messages = state.get("messages", [])
            if hasattr(request, "messages") and request.messages:
                messages = request.messages

            detected_language = self._detect_language_from_messages(messages, state)

        # Use detected language or default
        language_code = detected_language or self.default_language

        # Only inject prompt if language is not English (to avoid unnecessary prompts)
        # or if explicitly detected
        if language_code != "en" or detected_language:
            language_prompt = self._build_language_prompt(language_code)

            if request.system_prompt:
                new_system_prompt = request.system_prompt + "\n\n" + language_prompt
            else:
                new_system_prompt = language_prompt

            return handler(request.override(system_prompt=new_system_prompt))

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Inject language instruction into the system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Get detected language from state
        state = cast("LanguageDetectionState", request.state)
        detected_language = state.get("detected_language")

        # If no language detected yet, try to detect from messages in the request
        if not detected_language:
            # Try to get messages from request state or request itself
            messages = state.get("messages", [])
            if hasattr(request, "messages") and request.messages:
                messages = request.messages

            detected_language = self._detect_language_from_messages(messages, state)

        # Use detected language or default
        language_code = detected_language or self.default_language

        # Only inject prompt if language is not English (to avoid unnecessary prompts)
        # or if explicitly detected
        if language_code != "en" or detected_language:
            language_prompt = self._build_language_prompt(language_code)

            if request.system_prompt:
                new_system_prompt = request.system_prompt + "\n\n" + language_prompt
            else:
                new_system_prompt = language_prompt

            return await handler(request.override(system_prompt=new_system_prompt))

        return await handler(request)

