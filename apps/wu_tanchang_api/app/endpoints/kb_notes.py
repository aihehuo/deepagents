"""Public KB note lookup by Xiaohongshu URL (calendar drawer)."""

from __future__ import annotations

from fastapi import HTTPException, Query

from apps.wu_tanchang_api.agent_factory.kb_note_lookup import lookup_note_by_xhs_url
from apps.wu_tanchang_api.app.endpoints.extract_intake import _verify_agent_key
from apps.wu_tanchang_api.app.models import KbNoteResponse


async def get_kb_note(
    url: str = Query(..., min_length=8, description="xhslink.cn URL or share口令"),
    x_agent_key: str | None = None,
) -> KbNoteResponse:
    _verify_agent_key(x_agent_key)
    note = lookup_note_by_xhs_url(url)
    if not note:
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    return KbNoteResponse(**note)
