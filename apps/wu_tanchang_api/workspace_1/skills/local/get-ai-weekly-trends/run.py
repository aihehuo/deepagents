#!/usr/bin/env python3
# ruff: noqa: T201, BLE001, D103
"""Script to call the get-ai-weekly-trends API."""

import sys

import requests


def main() -> None:
    url = "https://www.aihehuo.com/ai/blog/weekly_markdown"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        print(resp.text)
    except Exception as e:
        print(f"Error calling get-ai-weekly-trends: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
