"""Authenticated, read-only investor lookup; credentials are never sent to public search."""
from __future__ import annotations

import http.client
import json
import os
from typing import Any
from urllib.parse import urlencode, urlsplit

from skillpacks.aihehuo import client


def fetch_investors(parameters: dict[str, Any]) -> str:
    """Read the fixed protected endpoint using a server-owned Agent Key."""
    key = os.environ.get("AIHEHUO_AGENT_KEY", "").strip()
    if not key:
        raise RuntimeError("investor_search_auth_unconfigured")
    base = os.environ.get(client.INTERNAL_API_BASE_URL_ENV) or "https://www.aihehuo.com"
    if base not in {client.INTERNAL_API_BASE_URL, "https://www.aihehuo.com"}:
        raise ValueError("invalid_investor_search_origin")
    allowed = {"investor_status", "investment_amount", "investment_sector", "keyword", "q", "wechat_reachable", "page", "per"}
    if parameters.keys() - allowed:
        raise ValueError("invalid_investor_search_parameters")
    url = urlsplit(base)
    connection = (http.client.HTTPConnection(url.hostname, url.port, timeout=client.TIMEOUT) if url.scheme == "http"
                  else http.client.HTTPSConnection(url.hostname, timeout=client.TIMEOUT))
    try:
        connection.request("GET", "/backend/agent/investors?" + urlencode(parameters),
                           headers={"Accept": "application/json", "X-Agent-Key": key})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError("investor_search_http_" + str(response.status))
        body = response.read(client.MAX_BYTES + 1)
        if len(body) > client.MAX_BYTES:
            raise ValueError("investor_search_response_too_large")
        text = body.decode("utf-8")
        document = json.loads(text)
        if document.get("success") is not True:
            raise ValueError("invalid_investor_search_response")
        return text
    finally:
        connection.close()
