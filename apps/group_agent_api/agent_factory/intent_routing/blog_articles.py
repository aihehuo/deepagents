"""Parse the public weekly blog document into bounded article facts."""

from __future__ import annotations

import re

from apps.group_agent_api.agent_factory.contact_scrub import scrub_contact_text

_ARTICLE_HEADING = re.compile(r"^##\s+\d+\.\s+(.+?)\s*$", re.MULTILINE)
_ARTICLE_URL = re.compile(
    r"^-\s+\*\*链接\*\*：\s*(https://www\.aihehuo\.com/blog/([A-Za-z0-9_-]+))",
    re.MULTILINE,
)
_ARTICLE_SUMMARY = re.compile(r"^-\s+\*\*摘要\*\*：\s*(.+)$", re.MULTILINE)


def weekly_articles(markdown: str, *, limit: int = 20) -> list[dict[str, str]]:
    """Extract public titles, summaries and detail links in API order.

    The fixed weekly endpoint returns Markdown, not individual JSON records.
    Only headings with official blog links become client-visible cards; article
    body text is deliberately excluded from the bounded search result event.
    """
    headings = list(_ARTICLE_HEADING.finditer(markdown or ""))
    articles: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, heading in enumerate(headings):
        section_end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        section = markdown[heading.end() : section_end]
        link = _ARTICLE_URL.search(section)
        if link is None or link.group(1) in seen:
            continue
        title = scrub_contact_text(re.sub(r"\s+", " ", heading.group(1)).strip())[:80]
        if len(title) < 2:
            continue
        summary_match = _ARTICLE_SUMMARY.search(section)
        summary = (
            scrub_contact_text(re.sub(r"\s+", " ", summary_match.group(1)).strip())[:200]
            if summary_match is not None
            else ""
        )
        seen.add(link.group(1))
        articles.append({
            "id": link.group(2)[:64],
            "url": link.group(1),
            "name": title,
            "summary": summary,
        })
        if len(articles) >= limit:
            break
    return articles
