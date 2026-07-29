#!/usr/bin/env python3
"""Export recent group_agent debug traces or a LangGraph checkpoint thread.

Usage (on the Deep Agents host / container):

  GROUP_AGENT_RUNTIME_DIR=/path/to/runtime \\
    python apps/group_agent_api/scripts/export_debug_trace.py --list

  python apps/group_agent_api/scripts/export_debug_trace.py --run-id <run_id>

  python apps/group_agent_api/scripts/export_debug_trace.py \\
    --thread-id 'ga::uid::gid::conv::ep' --from-checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path


def _runtime_dir() -> Path:
    env = os.environ.get("GROUP_AGENT_RUNTIME_DIR", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".deepagents" / "group_agent_api"


def _traces_dir(runtime: Path) -> Path:
    return runtime / "debug_traces"


def cmd_list(runtime: Path, limit: int) -> int:
    traces = _traces_dir(runtime)
    if not traces.is_dir():
        print(f"No traces dir: {traces}", file=sys.stderr)
        return 1
    files = sorted(traces.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files[:limit]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"{path.name}\t(unreadable)")
            continue
        print(
            f"{path.name}\trun={data.get('run_id')}\t"
            f"ep={data.get('episode_id')}\t"
            f"msg={str(data.get('user_message') or '')[:60]!r}"
        )
    return 0


def cmd_run(runtime: Path, run_id: str) -> int:
    traces = _traces_dir(runtime)
    matches = sorted(traces.glob(f"*_{run_id}.json"), key=lambda p: p.stat().st_mtime)
    if not matches:
        # fuzzy: filename contains run_id
        matches = sorted(
            [p for p in traces.glob("*.json") if run_id in p.name],
            key=lambda p: p.stat().st_mtime,
        )
    if not matches:
        print(f"No trace for run_id={run_id} under {traces}", file=sys.stderr)
        return 1
    for path in matches:
        print(path.read_text(encoding="utf-8"))
        print("\n---\n")
    return 0


def cmd_checkpoint(runtime: Path, thread_id: str) -> int:
    ckpt_path = runtime / "checkpoints.pkl"
    if not ckpt_path.is_file():
        print(f"Missing checkpoint file: {ckpt_path}", file=sys.stderr)
        return 1
    with ckpt_path.open("rb") as fh:
        blob = pickle.load(fh)
    storage = blob.get("storage") if isinstance(blob, dict) else None
    if not isinstance(storage, dict) or thread_id not in storage:
        print(f"thread_id not found: {thread_id}", file=sys.stderr)
        print("Available thread prefixes:", file=sys.stderr)
        for key in list(storage or {})[:30]:
            print(f"  {key}", file=sys.stderr)
        return 1
    # Best-effort dump: structure varies by LangGraph version
    print(
        json.dumps(
            {
                "thread_id": thread_id,
                "checkpoint_file": str(ckpt_path),
                "namespaces": list((storage.get(thread_id) or {}).keys()),
                "raw_type": str(type(storage.get(thread_id))),
                "note": (
                    "Full checkpoint blob is large; prefer GROUP_AGENT_DEBUG_TRACE "
                    "turn dumps for readable AI/Tool/Human chains."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default="", help="Override GROUP_AGENT_RUNTIME_DIR")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--thread-id", default="")
    parser.add_argument("--from-checkpoint", action="store_true")
    args = parser.parse_args()

    runtime = Path(args.runtime_dir) if args.runtime_dir else _runtime_dir()
    if args.list:
        return cmd_list(runtime, args.limit)
    if args.run_id:
        return cmd_run(runtime, args.run_id)
    if args.from_checkpoint and args.thread_id:
        return cmd_checkpoint(runtime, args.thread_id)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
