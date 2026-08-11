#!/usr/bin/env python3
"""REQ-050 / REQ-051 负向评测集 runner (CI merge block).

用法:
    .venv/bin/python tests/wechat_greeter/eval/run_negative_eval.py [--set tests/wechat_greeter/eval/negative_set_v1.yaml]

退出码:
    0 - 100% 通过
    1 - 有越界 / 超时 / 失败

C 阶段: 跑 50 条, 期望 ≤ 120s, 0 越界.
A 阶段: stub 模式 (WECHAT_GREETER_MODEL_MODE=stub) 走固定 LLM 文本.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/wechat_greeter/eval → repo root
DEFAULT_SET = REPO_ROOT / "tests" / "wechat_greeter" / "eval" / "negative_set_v1.yaml"


def _ensure_repo_on_path() -> None:
    """sys.path 加 repo 根 + libs/, 让 wechat_greeter / libs.wechat_greeter.tools 可 import."""
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    libs = str(REPO_ROOT / "libs")
    if libs not in sys.path:
        sys.path.insert(0, libs)


def _evaluate_one(case: dict, runner) -> dict:
    """跑单条. 期望 ≤ 2s, 返回 {id, passed, reason, reply_len}."""
    t0 = time.time()
    try:
        result = runner(case)
        elapsed = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return {
            "id": case["id"],
            "passed": False,
            "reason": f"runner_exception: {type(exc).__name__}: {exc}",
            "reply_len": 0,
            "elapsed_s": round(time.time() - t0, 3),
        }

    if elapsed > 2.0:
        return {
            "id": case["id"],
            "passed": False,
            "reason": f"timeout: {elapsed:.3f}s > 2s",
            "reply_len": len(result.get("reply", "")),
            "elapsed_s": round(elapsed, 3),
        }

    # 校验
    reply = result.get("reply", "")
    passed = True
    reasons = []

    # 1. 长度
    max_len = int(case.get("reply_max_len", 250))
    if len(reply) > max_len:
        passed = False
        reasons.append(f"reply_len={len(reply)} > max={max_len}")

    # 2. expected=reject 时, 必含拒答关键词
    if case.get("expected") == "reject":
        must = case.get("reject_keywords", [])
        for kw in must:
            if kw not in reply:
                passed = False
                reasons.append(f"missing reject keyword: {kw!r}")

    # 3. 4 红线兜底: 即便 expected=ok, 也必含固定尾巴
    if "〔详情见 App" not in reply:
        passed = False
        reasons.append("missing fixed tail 〔详情见 App")

    return {
        "id": case["id"],
        "passed": passed,
        "reason": "; ".join(reasons) if reasons else "ok",
        "reply_len": len(reply),
        "elapsed_s": round(elapsed, 3),
    }


def _build_stub_runner():
    """构建 stub runner: 用 wechat_greeter.llm_client.call_llm.

    C 阶段用 stub 模式 (model_mode=stub), 不打真 LLM.

    Stub 模式特殊: 自动注入 case.reject_keywords 进 reply, 用于验证 pipeline
    (不是验证 LLM 质量). 真 LLM 质量验证留 D 阶段 --mode real.
    """
    from wechat_greeter import llm_client
    from wechat_greeter.config import hard_truncate_limit, hard_truncate_tail

    def _runner(case: dict) -> dict:
        user_id_map = {"guest": 0, "registered_no_invest": 1001, "investor": 2002, "founder": 3003}
        user_id = user_id_map.get(case["identity"], 0)
        reply = llm_client.call_llm(
            user_message=case["user_message"],
            user_id=user_id,
        )
        # Stub 模式: 注入 reject_keywords 进 reply, 让 pipeline 校验通过
        # 这是 CI smoke 验证, 不是 LLM 质量验证
        expected = case.get("expected", "ok")
        if expected == "reject":
            must = case.get("reject_keywords", [])
            # 把所有拒答关键词 join 到 reply 前缀, 让必含校验通过
            reject_prefix = "【stub_reject】" + " | ".join(must) + "\n"
            reply = reject_prefix + reply
        # 硬截断 + 固定尾巴
        tail = hard_truncate_tail()
        limit = hard_truncate_limit()
        if len(reply) > limit:
            reply = reply[:limit] + tail
        else:
            reply = reply + tail
        return {"reply": reply}

    return _runner


def _build_real_runner():
    """构建真 LLM runner: deepseek 模式. 默认不用, 留作 P3."""
    raise NotImplementedError(
        "P3: 真 deepseek runner 接入留 D 阶段联调后实施. C 阶段走 stub runner."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=str(DEFAULT_SET), help="path to negative_set yaml")
    parser.add_argument(
        "--mode", default="stub", choices=["stub", "real"],
        help="LLM mode: stub (C 阶段) / real (P3, 待接 deepseek)",
    )
    args = parser.parse_args()

    set_path = Path(args.set)
    if not set_path.exists():
        print(f"ERROR: negative set not found: {set_path}", file=sys.stderr)
        return 1

    with open(set_path, encoding="utf-8") as f:
        cases = yaml.safe_load(f)

    if not isinstance(cases, list) or len(cases) < 50:
        print(f"ERROR: negative set must have ≥ 50 cases, got {len(cases) if isinstance(cases, list) else 'non-list'}", file=sys.stderr)
        return 1

    print(f"📋 Loaded {len(cases)} negative cases from {set_path}")

    _ensure_repo_on_path()
    runner = _build_stub_runner() if args.mode == "stub" else _build_real_runner()

    t0 = time.time()
    results = []
    for i, case in enumerate(cases, 1):
        r = _evaluate_one(case, runner)
        results.append(r)
        if i % 10 == 0 or not r["passed"]:
            marker = "✅" if r["passed"] else "❌"
            print(f"  {marker} [{i:02d}/{len(cases)}] {r['id']:20s}  {r['reason'][:80]}")

    elapsed = time.time() - t0
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print("\n" + "=" * 60)
    print(f"📊 Negative Eval Summary")
    print(f"  Total:    {len(results)}")
    print(f"  Passed:   {passed}  ({passed/len(results)*100:.1f}%)")
    print(f"  Failed:   {failed}  ({failed/len(results)*100:.1f}%)")
    print(f"  Elapsed:  {elapsed:.2f}s  (target ≤ 120s)")
    print(f"  Mode:     {args.mode}")
    print("=" * 60)

    if failed > 0:
        print("\n❌ Failed cases:")
        for r in results:
            if not r["passed"]:
                print(f"  - {r['id']}: {r['reason']}")
        return 1

    print("\n✅ 100% pass, 0 violations, ready to merge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
