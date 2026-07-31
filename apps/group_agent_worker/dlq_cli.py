"""Local DLQ admin CLI (REQ-032) — not exposed as a public HTTP endpoint.

Usage::

    python -m apps.group_agent_worker.dlq_cli list
    python -m apps.group_agent_worker.dlq_cli inspect <run_id>
    python -m apps.group_agent_worker.dlq_cli replay <run_id> --operator op1 --reason manual
    python -m apps.group_agent_worker.dlq_cli cancel <run_id> --operator op1 --reason abandon
"""

from __future__ import annotations

import argparse
import os
import sys

from apps.group_agent_api.execution.broker import enqueue_delivery
from apps.group_agent_api.execution.config import load_durable_queue_config
from apps.group_agent_api.execution.dlq import DlqAdmin, format_safe_json
from apps.group_agent_api.execution.redis_store import ExecutionStore


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("GROUP_AGENT_DURABLE_QUEUE_ENABLED", "1")
    parser = argparse.ArgumentParser(description="Group Agent DLQ local admin")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    p_ins = sub.add_parser("inspect")
    p_ins.add_argument("run_id")
    p_rep = sub.add_parser("replay")
    p_rep.add_argument("run_id")
    p_rep.add_argument("--operator", required=True)
    p_rep.add_argument("--reason", required=True)
    p_rep.add_argument("--replay-id", default=None)
    p_can = sub.add_parser("cancel")
    p_can.add_argument("run_id")
    p_can.add_argument("--operator", required=True)
    p_can.add_argument("--reason", required=True)

    args = parser.parse_args(argv)
    cfg = load_durable_queue_config(require_enabled=True)
    assert cfg is not None
    store = ExecutionStore.from_config(cfg)
    admin = DlqAdmin(store, cfg, enqueue=lambda d: enqueue_delivery(cfg, d))

    if args.cmd == "list":
        print(format_safe_json(admin.list_dead_lettered()))
        return 0
    if args.cmd == "inspect":
        print(format_safe_json([admin.inspect(args.run_id)]))
        return 0
    if args.cmd == "replay":
        view = admin.replay(
            args.run_id,
            operator_id=args.operator,
            reason=args.reason,
            replay_id=args.replay_id,
        )
        print(format_safe_json([view]))
        return 0
    if args.cmd == "cancel":
        view = admin.cancel(args.run_id, operator_id=args.operator, reason=args.reason)
        print(format_safe_json([view]))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
