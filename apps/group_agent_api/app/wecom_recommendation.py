"""Select exactly one unseen public record; identity and links never come from the LLM."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any
from urllib.parse import urlsplit

from skillpacks.aihehuo import client

MAX_PAGES = 5
PAGE_SIZE = 20
PUBLIC_HOSTS = {"www.aihehuo.com", "aihehuo.com", "m.aihehuo.com", "member.aihehuo.com"}


def public_url(value: Any) -> str:
    """Return only ordinary public AIHEHUO links, without credentials or ports."""
    if not isinstance(value, str) or len(value) > 512:
        return ""
    try:
        url = urlsplit(value)
        if url.scheme not in {"https", "http"} or url.hostname not in PUBLIC_HOSTS or url.username or url.password or url.port:
            return ""
    except ValueError:
        return ""
    return value


def public_record(resource: str, row: Any) -> dict[str, str] | None:
    """Project one public row to safe display fields and a stable resource identity."""
    if not isinstance(row, dict):
        return None
    identity = str(row.get("number") if resource == "users" and row.get("number") else row.get("id") or "")
    if resource == "faqs":
        question = str(row.get("question") or "").strip()
        if not question:
            return None
        identity = hashlib.sha256(json.dumps([row.get("source"), row.get("category"), question], ensure_ascii=False).encode()).hexdigest()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identity):
        return None
    title = str(row.get("title") or row.get("name") or row.get("question") or "").strip()[:120]
    if not title:
        return None
    summary = str(row.get("summary") or row.get("description") or row.get("bio") or row.get("answer") or "").strip()[:800]
    # Nested group members and related projects are deliberately excluded.
    return {"resource": resource, "id": identity, "key": f"{resource}:{identity}",
            "title": title, "summary": summary, "url": public_url(row.get("url"))}


async def find_one(*, resource: str, parameters: dict[str, str], excluded: Any,
                   side_effect_gate: Any = None, customer_number: str | None = None) -> tuple[dict[str, str] | None, dict[str, int]]:
    """Scan a bounded number of pages and return at most one unseen identity."""
    if not isinstance(excluded, list) or len(excluded) > 2000 or any(not isinstance(key, str) or len(key) > 100 for key in excluded):
        raise ValueError("invalid_recommendation_exclusions")
    seen = set(excluded)
    scope = {"pages": 0, "examined": 0, "max_pages": MAX_PAGES, "returned": 0}
    for page in range(1, MAX_PAGES + 1):
        if side_effect_gate is not None:
            side_effect_gate("wecom_search_page")
        query = {key: value for key, value in parameters.items() if value}
        query.update(page=page, per=PAGE_SIZE)
        document = json.loads(await asyncio.to_thread(client.fetch, client.build_path(resource, params=query), output_format="json"))
        if not isinstance(document, dict) or not isinstance(document.get("data"), list):
            raise ValueError("invalid_public_search_response")
        rows = document["data"]
        scope["pages"] += 1
        for row in rows:
            scope["examined"] += 1
            if customer_number and resource in {"users", "projects", "ideas"}:
                owner = (str(row.get("number") or row.get("id") or "") if resource == "users" else
                         str(row.get("user_number") or (row.get("user") if isinstance(row.get("user"), dict) else {}).get("number") or "")) if isinstance(row, dict) else ""
                # Without publisher identity we cannot prove a project is not the customer's.
                if owner == str(customer_number) or (resource != "users" and not owner):
                    continue
            card = public_record(resource, row)
            if card and card["key"] not in seen:
                scope["returned"] = 1
                return card, scope
        meta = document.get("meta") or {}
        if meta.get("has_more") is False or not rows or ("has_more" not in meta and len(rows) < PAGE_SIZE):
            break
    return None, scope


def recommendation_text(raw: Any, card: dict[str, str]) -> str:
    """Bound the customer phrase and append only the source's validated link."""
    text = raw.strip() if isinstance(raw, str) else ""
    if not text or len(text) > 200 or re.search(r"https?://|www\.|```", text, re.IGNORECASE):
        text = f"这条「{card['title'][:40]}」资料可以先参考一下，看看是否符合你的需求。"
    return text + ("\n" + card["url"] if card["url"] else "")
