"""Dependency-free, read-only client bundled into each installed skill."""

import argparse
import http.client
import json
import sys
from collections.abc import Sequence
from urllib.parse import urlencode

HOST = "www.aihehuo.com"
TIMEOUT = 20
MAX_BYTES = 4 * 1024 * 1024
MAX_PER = 500
ENDPOINTS = {
    "users": {"keyword", "q"},
    "ideas": {"keyword", "q"},
    "projects": {"keyword"},
    "groups": {"keyword"},
    "group-members": set(),
    "blog": set(),
    "weekly": set(),
    "faqs": {"keyword"},
}


def build_path(endpoint: str, *, params: dict[str, str | int], output_format: str = "json", event_id: str | None = None) -> str:
    """Build an allowlisted API path without accepting arbitrary URLs.

    Args:
        endpoint: Resource alias from `ENDPOINTS`.
        params: Query parameters supported by the resource.
        output_format: JSON or Markdown output.
        event_id: Public event identifier for group member queries.

    Returns:
        Encoded path and query string on the official host.

    Raises:
        ValueError: A resource, output_format, identifier, or parameter is invalid.
    """
    if endpoint not in ENDPOINTS or output_format not in {"json", "md"}:
        msg = "Unsupported endpoint or output_format"
        raise ValueError(msg)
    supported = ENDPOINTS[endpoint] | {"page", "per", "created_since", "updated_since"}
    if endpoint == "weekly":
        supported = set()
    if params.keys() - supported:
        msg = f"Unsupported parameters for {endpoint}: {sorted(params.keys() - supported)}"
        raise ValueError(msg)
    for name in ("page", "per"):
        value = params.get(name, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or (name == "per" and value > MAX_PER):
            msg = f"Invalid {name}; use a positive integer (per <= {MAX_PER})"
            raise ValueError(msg)
    if endpoint == "group-members":
        if not event_id or not event_id.isascii() or not event_id.isdecimal():
            msg = "Group members require a numeric event_id, not a WeChat identifier"
            raise ValueError(msg)
        resource = f"wechat_groups/{event_id}"
    else:
        if event_id is not None:
            msg = "event_id is supported only for group-members"
            raise ValueError(msg)
        resource = {"groups": "wechat_groups", "weekly": "blog/weekly_markdown"}.get(endpoint, endpoint)
    path = f"/ai/{resource}" if endpoint == "weekly" else f"/ai/{resource}.{output_format}"
    query = urlencode(params)
    return f"{path}?{query}" if query else path


def fetch(
    path: str, *, output_format: str = "json", timeout: float = TIMEOUT
) -> str:
    """Read a bounded response from the official HTTPS host.

    Args:
        path: Relative API path produced by `build_path`.
        output_format: Expected JSON or Markdown response.

    Returns:
        Response text, preserving source fields and pagination metadata.

    Raises:
        ValueError: The path or output_format is invalid, or JSON decoding fails.
        RuntimeError: HTTP failure or oversized response, including rate limits.
        OSError: Connection or timeout failure.
    """
    if not path.startswith("/ai/") or output_format not in {"json", "md"}:
        msg = "Only official public API paths and json/md formats are supported"
        raise ValueError(msg)
    if timeout <= 0 or timeout > TIMEOUT:
        raise ValueError("Invalid public query timeout")
    connection = http.client.HTTPSConnection(HOST, timeout=timeout)
    try:
        connection.request("GET", path, headers={"Accept": "application/json" if output_format == "json" else "text/markdown"})
        response = connection.getresponse()
        if response.status != http.client.OK:
            retry = response.getheader("Retry-After")
            msg = f"HTTP {response.status}; Retry-After={retry or 'unspecified'}. Reduce requests; do not retry immediately."
            raise RuntimeError(msg)
        body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            msg = "Response exceeds size limit; reduce per or paginate"
            raise RuntimeError(msg)
        text = body.decode("utf-8")
        if output_format == "json":
            json.loads(text)
        return text
    finally:
        connection.close()


def main(allowed: Sequence[str], argv: Sequence[str] | None = None) -> int:
    """Run the standalone query command for one skill.

    Args:
        allowed: Resource aliases this skill exposes.
        argv: Explicit arguments for testing or embedding; otherwise process arguments.

    Returns:
        Zero on success, one on retrieval failure. Argument errors exit with two.
    """
    parser = argparse.ArgumentParser(description="Read AIHEHUO public data")
    parser.add_argument("--resource", choices=allowed, default=allowed[0])
    parser.add_argument("--keyword")
    parser.add_argument("--q")
    parser.add_argument("--page", type=int)
    parser.add_argument("--per", type=int)
    parser.add_argument("--created-since")
    parser.add_argument("--updated-since")
    parser.add_argument("--event-id")
    parser.add_argument("--format", choices=("json", "md"), default="json")
    args = parser.parse_args(argv)
    params = {key: value for key in ("keyword", "q", "page", "per", "created_since", "updated_since") if (value := getattr(args, key)) is not None}
    # The weekly endpoint returns Markdown even without a .md extension.
    output_format = "md" if args.resource == "weekly" else args.format
    try:
        path = build_path(args.resource, params=params, output_format=output_format, event_id=args.event_id)
    except ValueError as error:
        parser.error(str(error))
    try:
        result = fetch(path, output_format=output_format)
    except (OSError, http.client.HTTPException, ValueError, RuntimeError) as error:
        sys.stderr.write(f"AIHEHUO request failed: {error}\n")
        return 1
    sys.stdout.write(result + "\n")
    return 0
