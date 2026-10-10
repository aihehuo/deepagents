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


def public_record(resource: str, row: Any) -> dict[str, Any] | None:
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
                   side_effect_gate: Any = None, customer_number: str | None = None,
                   reviewer: Any = None) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """Scan a bounded number of pages and return at most one unseen identity."""
    if not isinstance(excluded, list) or len(excluded) > 2000 or any(not isinstance(key, str) or len(key) > 100 for key in excluded):
        raise ValueError("invalid_recommendation_exclusions")
    seen = set(excluded)
    scope = {"pages": 0, "examined": 0, "max_pages": MAX_PAGES, "returned": 0}
    from apps.group_agent_api.app.investor_client import fetch_investors
    from apps.group_agent_api.app.wecom_match_review import MAX_REVIEWED, BATCH_SIZE
    investment_search = resource == "users" and bool(parameters.get("investor_status"))
    reviewed = 0
    pending = None
    scope.update({"reviewed": 0, "rejected": 0, "needs_verification": 0})
    phases = (["verified", "pending"] if investment_search and parameters.get("investor_status") == "verified_or_pending"
              else [parameters.get("investor_status", "")])
    for phase in phases:
        phase_start = reviewed
        for page in range(1, (2 if len(phases) == 2 else MAX_PAGES) + 1):
            if side_effect_gate is not None:
                side_effect_gate("wecom_search_page")
            query = {key: value for key, value in parameters.items() if value and key != "wechat_reachable"}
            if resource in {"users", "ideas"} and parameters.get("wechat_reachable", "true") == "true":
                query["wechat_reachable"] = "true"
            if investment_search:
                query["investor_status"] = phase
            query.update(page=page, per=10 if len(phases) == 2 else PAGE_SIZE)
            if investment_search:
                document = json.loads(await asyncio.to_thread(fetch_investors, query))
            else:
                document = json.loads(await asyncio.to_thread(client.fetch, client.build_path(resource, params=query), output_format="json"))
            if not isinstance(document, dict) or not isinstance(document.get("data"), list):
                raise ValueError("invalid_public_search_response")
            rows = document["data"]
            scope["pages"] += 1
            batch = []
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
                    publisher = row.get("user") if isinstance(row.get("user"), dict) else {}
                    number = str(row.get("number") or row.get("id") or "") if resource == "users" else str(row.get("user_number") or publisher.get("number") or "")
                    if resource in {"users", "ideas"} and re.fullmatch(r"[1-9][0-9]{0,19}", number):
                        card["publisher_number"] = number
                    flag = row.get("wechat_reachable") if resource == "users" else publisher.get("wechat_reachable", row.get("wechat_reachable"))
                    if resource in {"users", "ideas"} and type(flag) is bool:
                        card["wechat_reachable"] = flag
                    if investment_search:
                        investment = row.get("investment")
                        if not isinstance(investment, dict) or investment.get("status") not in {"verified", "pending"}:
                            scope["rejected"] += 1
                            continue
                        card["investment"] = investment
                    if reviewer is None:
                        if investment_search:
                            raise ValueError("investor_review_required")
                        scope["returned"] = 1
                        return card, scope
                    batch.append(card)
                    seen.add(card["key"])
                    if len(batch) >= min(MAX_REVIEWED - reviewed, (10 - (reviewed - phase_start)) if len(phases) == 2 else MAX_REVIEWED):
                        break
            for start in range(0, len(batch), BATCH_SIZE):
                portion = batch[start:start + BATCH_SIZE]
                if side_effect_gate is not None:
                    side_effect_gate("wecom_match_review")
                reviews = await reviewer(portion)
                reviewed += len(portion)
                scope["reviewed"] = reviewed
                by_id = {review["id"]: review for review in reviews}
                matched = None
                for card in portion:
                    review = by_id[card["id"]]
                    if review["status"] == "rejected":
                        scope["rejected"] += 1
                        continue
                    card["matching_review"] = {key: review[key] for key in ("status", "reason", "evidence", "missing")}
                    if review["status"] == "matched":
                        if matched is None:
                            matched = card
                    else:
                        scope["needs_verification"] += 1
                        if pending is None:
                            pending = card
                if matched is not None:
                    scope["returned"] = 1
                    return matched, scope
            if reviewed >= MAX_REVIEWED or (len(phases) == 2 and reviewed - phase_start >= 10):
                break
            meta = document.get("meta") or {}
            if meta.get("has_more") is False or not rows or ("has_more" not in meta and len(rows) < PAGE_SIZE):
                break
        # A relevant certified investor awaiting current-intent confirmation remains the preferred candidate.
        if pending is not None or reviewed >= MAX_REVIEWED:
            break
    if pending is not None:
        scope["returned"] = 1
    return pending, scope


def recommendation_text(raw: Any, card: dict[str, str]) -> str:
    """Bound the customer phrase and append only the source's validated link."""
    text = raw.strip() if isinstance(raw, str) else ""
    if not text or len(text) > 200 or re.search(r"https?://|www\.|```", text, re.IGNORECASE):
        text = ("目前未能整理出可靠的平台答复，可以补充具体问题后再查询。" if card.get("resource") == "faqs" else
                f"这条「{card['title'][:40]}」资料可以先参考一下，看看是否符合你的需求。")
    return text + ("\n" + card["url"] if card["url"] else "")


async def enrich_reachability(card: dict[str, Any], *, side_effect_gate: Any = None) -> None:
    """Read actual reachability for the selected publisher only; never infer from filters."""
    if card.get("resource") not in {"users", "ideas"} or type(card.get("wechat_reachable")) is bool:
        return
    number = card.get("publisher_number")
    if not isinstance(number, str) or not re.fullmatch(r"[1-9][0-9]{0,19}", number):
        return
    if side_effect_gate is not None:
        side_effect_gate("wecom_publisher_detail")
    try:
        raw = await asyncio.to_thread(client.fetch, f"/ai/users/{number}.json", output_format="json")
        detail = json.loads(raw).get("data")
        if isinstance(detail, dict) and type(detail.get("wechat_reachable")) is bool:
            card["wechat_reachable"] = detail["wechat_reachable"]
    except (ValueError, OSError, TimeoutError, RuntimeError):
        # Missing public evidence means no promise of WeChat introduction.
        return
