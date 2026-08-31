"""Stream chunk buffering and private tag stripping for Group Agent (TSD-15)."""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable


class StreamChunkBuffer:
    """Aggregates fine-grained LLM tokens into throttled chunks and strips private tags without locking on normal '<'."""

    TARGET_TAG_PREFIX = "<suggested_replies"

    def __init__(
        self,
        emit_callback: Callable[[str, dict[str, Any]], Awaitable[bool]],
        min_interval_s: float = 0.1,
        max_buffer_chars: int = 40,
    ) -> None:
        self.emit_callback = emit_callback
        self.min_interval_s = min_interval_s
        self.max_buffer_chars = max_buffer_chars
        self._buffer: list[str] = []
        self._last_emit_t = time.monotonic()

        # Tag detection state
        self._in_marker_zone = False
        self._held_marker_chars: list[str] = []
        self._confirmed_marker = False

    async def feed_token(self, delta: str) -> None:
        if not delta:
            return

        for char in delta:
            if self._confirmed_marker:
                # Once we are in a confirmed <suggested_replies> tag, hold all rest of marker
                self._held_marker_chars.append(char)
                continue

            if not self._in_marker_zone:
                if char == "<":
                    self._in_marker_zone = True
                    self._held_marker_chars.append(char)
                else:
                    self._buffer.append(char)
            else:
                self._held_marker_chars.append(char)
                held_str = "".join(self._held_marker_chars)
                normalized = held_str.replace(" ", "").lower()

                if self.TARGET_TAG_PREFIX.startswith(normalized):
                    # Prefix matches so far
                    if normalized == self.TARGET_TAG_PREFIX:
                        self._confirmed_marker = True
                else:
                    # Mismatch (e.g. "< 5000" or "< 10")
                    # Release the held characters back into buffer safely and exit marker zone
                    self._buffer.extend(self._held_marker_chars)
                    self._held_marker_chars.clear()
                    self._in_marker_zone = False

        now = time.monotonic()
        has_punctuation = any(p in delta for p in ("，", "。", "！", "？", "\n", ",", ".", "!", "?"))

        if (
            (now - self._last_emit_t >= self.min_interval_s and len(self._buffer) >= 2)
            or has_punctuation
            or sum(len(s) for s in self._buffer) >= self.max_buffer_chars
        ):
            await self.flush()

    async def flush(self) -> None:
        if not self._buffer:
            return
        chunk_text = "".join(self._buffer)
        self._buffer.clear()
        self._last_emit_t = time.monotonic()
        await self.emit_callback("chunk", {"text": chunk_text, "is_streaming": True})

    def get_held_marker_content(self) -> str:
        """Return any held marker content (e.g., the raw <suggested_replies> tag)."""
        return "".join(self._held_marker_chars)
