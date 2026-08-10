"""Model-authored suggested replies for Group Agent clients."""

from __future__ import annotations

import json
import re
from typing import Any


SUGGESTED_REPLIES_PROMPT = """

## 可点击建议回复（客户端能力）
- 当本轮存在 2～4 个自然且有价值的后续方向时，可在正常回复末尾追加机器标记：
  `<suggested_replies>[{"label":"按钮短文案","reply":"用户可直接发送的完整回复"}]</suggested_replies>`
- 选项必须结合当前对话动态生成，禁止套用固定业务选项；`label` 要简短，`reply` 要像用户本人会说的话。
- 标记内只能放合法 JSON，不要使用 Markdown 代码块。没有明确分支时不要输出标记。
- 正常回复必须独立完整；机器标记不计入正常回复字数，也不要在正文解释这个标记。
"""

_MARKER_RE = re.compile(
    r"<suggested_replies>\s*(.*?)\s*</suggested_replies>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _compact_label(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.replace("\x00", "").split())[:40]


def _compact_reply(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\x00", "").strip()[:300]


def extract_suggested_replies(text: str) -> tuple[str, list[dict[str, str]]]:
    """Strip the private marker and return a validated 2–4 item payload."""
    raw = str(text or "")
    matches = list(_MARKER_RE.finditer(raw))
    visible = _MARKER_RE.sub("", raw).strip()

    for match in reversed(matches):
        try:
            decoded = json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(decoded, list):
            continue

        suggestions: list[dict[str, str]] = []
        seen_replies: set[str] = set()
        for item in decoded:
            if not isinstance(item, dict):
                continue
            label = _compact_label(item.get("label"))
            reply = _compact_reply(item.get("reply"))
            if not label or not reply or reply in seen_replies:
                continue
            seen_replies.add(reply)
            suggestions.append(
                {
                    "id": f"suggested_{len(suggestions) + 1}",
                    "label": label,
                    "reply": reply,
                }
            )
            if len(suggestions) >= 4:
                break

        if len(suggestions) >= 2:
            return visible, suggestions

    return visible, []
