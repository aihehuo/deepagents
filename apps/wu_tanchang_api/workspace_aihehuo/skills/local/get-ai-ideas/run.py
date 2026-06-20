#!/usr/bin/env python3
# ruff: noqa: T201, BLE001, D103
"""Script to call the get-ai-ideas API."""

import argparse
import json
import sys

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch aihehuo idea list.")
    parser.add_argument("--keyword", help="Keywords (OR combined)")
    parser.add_argument("--q", help="Semantic query text")
    parser.add_argument("--page", type=int, default=1, help="Page number")
    parser.add_argument("--per", type=int, default=20, help="Per page count")
    parser.add_argument("--format", default="json", choices=["json", "md"], help="Output format")
    args = parser.parse_args()

    url = f"https://www.aihehuo.com/ai/ideas.{args.format}"
    params = {
        "page": args.page,
        "per": args.per,
    }
    if args.keyword:
        params["keyword"] = args.keyword
    if args.q:
        params["q"] = args.q

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        if args.format == "json":
            print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
        else:
            print(resp.text)
    except Exception as e:
        print(f"Error calling get-ai-ideas: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
