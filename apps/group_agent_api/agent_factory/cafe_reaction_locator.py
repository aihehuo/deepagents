"""Locate model-selected reaction excerpts in the immutable public reply."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_HIGH_RISK = re.compile(r"\d|候选|人选|姓名|联系方式|地址|搜索结果|公开查询|(?:省|市|区|县|镇|街道)")
_PROTECTED = frozenset("不没无未非只别勿必须拒否")


def _normalized(text: str) -> tuple[str, list[int]]:
    """Discard only typography while retaining offsets into the original text."""
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        for folded in unicodedata.normalize("NFKC", char).lower():
            if folded.isspace() or unicodedata.category(folded).startswith(("P", "Z")) or folded in "*_~":
                continue
            chars.append(folded)
            offsets.append(index)
    return "".join(chars), offsets


def _paragraphs(reply: str) -> list[tuple[int, str]]:
    """Keep paragraph boundaries so two anchors cannot bridge separate ideas."""
    breaks = list(_PARAGRAPH_BREAK.finditer(reply))
    starts = [0, *(match.end() for match in breaks)]
    ends = [*(match.start() for match in breaks), len(reply)]
    return [(start, reply[start:end]) for start, end in zip(starts, ends)]


def _all_occurrences(text: str, fragment: str) -> list[int]:
    """Find every candidate position, including overlapping occurrences."""
    positions: list[int] = []
    position = text.find(fragment)
    while position >= 0:
        positions.append(position)
        position = text.find(fragment, position + 1)
    return positions


def _typographic_match(reply: str, normalized_quote: str) -> tuple[int, int] | None:
    """Map one punctuation/whitespace-insensitive hit back to exact source offsets."""
    candidates: list[tuple[int, int]] = []
    for base, paragraph in _paragraphs(reply):
        normalized, offsets = _normalized(paragraph)
        for position in _all_occurrences(normalized, normalized_quote):
            candidates.append((base + offsets[position], base + offsets[position + len(normalized_quote) - 1] + 1))
    return candidates[0] if len(candidates) == 1 else None


def _safe_edit(quote: str, source: str) -> bool:
    """Allow short omissions/additions, never changed words or protected facts."""
    edits = 0
    for operation, left, right, source_left, source_right in SequenceMatcher(
        None, quote, source, autojunk=False
    ).get_opcodes():
        if operation == "equal":
            continue
        changed = quote[left:right] + source[source_left:source_right]
        if operation == "replace" or len(changed) > 3:
            return False
        if any(char in _PROTECTED or char.isdigit() or char.isascii() and char.isalnum() for char in changed):
            return False
        edits += len(changed)
    return edits <= max(3, len(quote) // 8)


def _anchor_match(reply: str, normalized_quote: str) -> tuple[int, int] | None:
    """Use two unique ordered source anchors to recover a short omitted phrase."""
    if len(normalized_quote) < 12 or _HIGH_RISK.search(normalized_quote):
        return None
    normalized_reply, _ = _normalized(reply)
    minimum_anchor = max(4, min(8, len(normalized_quote) // 4))
    candidates: list[tuple[int, int]] = []
    for base, paragraph in _paragraphs(reply):
        normalized, offsets = _normalized(paragraph)
        blocks = [
            block
            for block in SequenceMatcher(None, normalized_quote, normalized, autojunk=False).get_matching_blocks()
            if block.size >= minimum_anchor
        ]
        if len(blocks) < 2:
            continue
        first, last = blocks[0], blocks[-1]
        if first.a > 2 or len(normalized_quote) - last.a - last.size > 2:
            continue
        if sum(block.size for block in blocks) / len(normalized_quote) < 0.85:
            continue
        first_anchor = normalized[first.b:first.b + first.size]
        last_anchor = normalized[last.b:last.b + last.size]
        if normalized_reply.count(first_anchor) != 1 or normalized_reply.count(last_anchor) != 1:
            continue
        source = normalized[first.b:last.b + last.size]
        if abs(len(source) - len(normalized_quote)) > max(3, len(normalized_quote) // 8):
            continue
        if not _safe_edit(normalized_quote, source):
            continue
        start = base + offsets[first.b]
        end = base + offsets[last.b + last.size - 1] + 1
        if _HIGH_RISK.search(reply[start:end]):
            continue
        candidates.append((start, end))
    return candidates[0] if len(candidates) == 1 else None


def locate_reaction_quote(reply: str, quote: str) -> tuple[int, int] | None:
    """Resolve a model quote to one unique original span without another LLM call.

    Exact matching wins. Typography-only normalization is next. Two-anchor
    recovery is limited to short non-semantic omissions/additions in one
    paragraph; ambiguous or meaning-changing matches fail closed.
    """
    if reply.count(quote) == 1:
        start = reply.index(quote)
        return start, start + len(quote)
    normalized_quote, _ = _normalized(quote)
    if not normalized_quote:
        return None
    return _typographic_match(reply, normalized_quote) or _anchor_match(reply, normalized_quote)
