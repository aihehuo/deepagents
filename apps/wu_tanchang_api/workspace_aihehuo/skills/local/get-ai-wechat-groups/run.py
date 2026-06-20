#!/usr/bin/env python3
# ruff: noqa: T201, BLE001, D103
"""Script to call the get-ai-wechat-groups API."""

import argparse
import json
import sys

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch aihehuo active WeChat groups.")
    parser.add_argument("--keyword", help="Keywords to filter groups")
    parser.add_argument("--format", default="json", choices=["json", "md"], help="Output format")
    args = parser.parse_args()

    url = f"https://www.aihehuo.com/ai/wechat_groups.{args.format}"
    params = {}
    if args.keyword:
        params["keyword"] = args.keyword

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        if args.format == "json":
            print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
        else:
            print(resp.text)
    except Exception as e:
        print(f"Error calling get-ai-wechat-groups: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
