"""Harvest successful, correlated public tool responses from this turn only."""

import json
from urllib.parse import urlsplit

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from apps.group_agent_api.agent_factory.checks.reply_grounding.protocol import PublicReadGround

PUBLIC_TOOL = "query_aihehuo_public_data"


def extract_public_reads(messages: list[BaseMessage], start: int) -> list[PublicReadGround]:
    """Collect public facts from matched model calls and successful tool results.

    Args:
        messages: Graph messages containing actual model and tool messages.
        start: First message index belonging to the current turn.

    Returns:
        Bounded evidence with official sources and no group membership claims.
    """
    calls: dict[str, str] = {}
    evidence: list[PublicReadGround] = []
    for message in messages[max(0, start) :]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call.get("name") == PUBLIC_TOOL:
                    calls[str(call.get("id"))] = str(call.get("args", {}).get("resource", ""))
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        resource = calls.pop(message.tool_call_id, None)
        if resource is None or message.name != PUBLIC_TOOL or not isinstance(message.content, str):
            continue
        item = _parse_response(resource, message.content)
        if item is not None:
            evidence.append(item)
    return evidence


def _parse_response(resource: str, content: str) -> PublicReadGround | None:
    try:
        document = json.loads(content)
    except (ValueError, TypeError):
        prefix = "Source API: "
        if not content.startswith(prefix):
            return None
        source, _, text = content[len(prefix) :].partition("\n\n")
        if not _official_source(source):
            return None
        return PublicReadGround(resource=resource, source_url=source, text=text)
    if not isinstance(document, dict) or "data" not in document:
        return None
    source = document.get("_source_url")
    if not isinstance(source, str) or not _official_source(source):
        return None
    return PublicReadGround(resource=resource, source_url=source, document=document)


def _official_source(source: str) -> bool:
    try:
        url = urlsplit(source)
        return url.scheme == "https" and url.netloc == "www.aihehuo.com" and url.path.startswith("/ai/")
    except ValueError:
        return False
