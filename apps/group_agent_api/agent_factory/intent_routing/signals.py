"""Text normalization helpers retained after L0 semantic routing removal."""

from __future__ import annotations

import re
import unicodedata

_SPACE = re.compile(r"\s+")


def normalize_text(message: str) -> str:
    """Normalize width, case, whitespace, and punctuation without regex backtracking."""
    normalized = unicodedata.normalize("NFKC", message or "").lower()
    return _SPACE.sub(" ", normalized).strip()
