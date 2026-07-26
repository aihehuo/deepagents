#!/usr/bin/env bash
# =============================================================================
# REQ-012 · Group Agent 独立 Real-LLM + L1 Mock Scenario 验收 Runner
#
# ⚠️  消耗真实 Qwen/DashScope LLM 额度，请确认额度后再运行。
# ⚠️  不启动 Docker / Micro / New API / callback simulator。
# ⚠️  不部署、不 push、不构建镜像。
#
# 前置条件（从当前进程环境取得，禁止从 .env 文件读取）：
#   GROUP_AGENT_REAL_LLM_TEST=1          # 必须显式设置为 1
#   GROUP_AGENT_PROVIDER=qwen            # 当前仅支持 qwen
#   GROUP_AGENT_MODEL=...                # 例如 qwen-turbo-0624
#   GROUP_AGENT_BASE_URL=...             # 例如 https://dashscope.aliyuncs.com/compatible-mode/v1
#   DASHSCOPE_API_KEY=...                # 或 GROUP_AGENT_API_KEY
#
# 可选：
#   GROUP_AGENT_MAX_TOKENS=800           # 默认 800
#   GROUP_AGENT_TIMEOUT_S=60             # 默认 60
#   GROUP_AGENT_HUMAN_AUDIT_REPORT=1     # 生成 REQ-013 人工内容审计报告
#   GROUP_AGENT_HUMAN_AUDIT_OUTPUT_DIR=  # 可选；默认 .local-artifacts/group-agent-audit
#
# Usage:
#   export GROUP_AGENT_REAL_LLM_TEST=1
#   export DASHSCOPE_API_KEY="sk-..."
#   export GROUP_AGENT_MODEL="qwen-turbo-0624"
#   export GROUP_AGENT_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
#   bash apps/group_agent_api/scripts/run_req012_real_llm.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${APP_DIR}/../.." && pwd)"

# ── 1. Opt-in gate ──────────────────────────────────────────────────────────
if [ "${GROUP_AGENT_REAL_LLM_TEST:-}" != "1" ]; then
    echo "[SKIP] REQ-012: set GROUP_AGENT_REAL_LLM_TEST=1 to run real-LLM acceptance"
    exit 0
fi

# ── 2. Pre-flight checks (existence only, never echo values) ─────────────────
echo "[REQ-012] Verifying real-LLM configuration..."

PROVIDER="${GROUP_AGENT_PROVIDER:-qwen}"
MODEL="${GROUP_AGENT_MODEL:-}"
BASE_URL="${GROUP_AGENT_BASE_URL:-}"
API_KEY="${GROUP_AGENT_API_KEY:-${DASHSCOPE_API_KEY:-}}"

FAILED=0
if [ -z "$MODEL" ]; then
    echo "[FAIL] GROUP_AGENT_MODEL is not set"
    FAILED=1
fi
if [ -z "$BASE_URL" ]; then
    echo "[FAIL] GROUP_AGENT_BASE_URL is not set"
    FAILED=1
fi
if [ -z "$API_KEY" ]; then
    echo "[FAIL] GROUP_AGENT_API_KEY / DASHSCOPE_API_KEY is not set"
    FAILED=1
fi

if [ "$FAILED" -eq 1 ]; then
    echo "[ERROR] Missing required real-LLM configuration. Aborting."
    echo "[INFO]  Required: GROUP_AGENT_MODEL, GROUP_AGENT_BASE_URL, and DASHSCOPE_API_KEY (or GROUP_AGENT_API_KEY)"
    exit 1
fi

echo "[OK]   provider = ${PROVIDER}"
echo "[OK]   model   = ${MODEL}"
echo "[OK]   base_url = $( ([ -n "$BASE_URL" ] && echo 'configured') || echo 'not set')"
echo "[OK]   api_key = $( ([ -n "$API_KEY" ] && echo 'configured') || echo 'not set')"

# ── 3. Create isolated runtime ──────────────────────────────────────────────
RUNTIME_DIR="$(mktemp -d)"
echo "[REQ-012] Using isolated runtime: ${RUNTIME_DIR}"

cleanup() {
    echo "[REQ-012] Cleaning up runtime: ${RUNTIME_DIR}"
    rm -rf "${RUNTIME_DIR}"
    if [ -n "${AUDIT_META_FILE:-}" ]; then
        rm -f "${AUDIT_META_FILE}"
    fi
}
trap cleanup EXIT

# ── 4. Set explicit test env vars (no .env reading) ─────────────────────────
export GROUP_AGENT_INTEGRATION=stub
export GROUP_AGENT_ENV=test
export GROUP_AGENT_MODEL_MODE=real
export GROUP_AGENT_PROVIDER="${PROVIDER}"
export GROUP_AGENT_MODEL="${MODEL}"
export GROUP_AGENT_BASE_URL="${BASE_URL}"
export GROUP_AGENT_TEST_LEVEL=L1
export GROUP_AGENT_RUNTIME_DIR="${RUNTIME_DIR}"
export GROUP_AGENT_MAX_TOKENS="${GROUP_AGENT_MAX_TOKENS:-800}"
export GROUP_AGENT_TIMEOUT_S="${GROUP_AGENT_TIMEOUT_S:-60}"
export GROUP_AGENT_LLM_POLISH=0
# Minimal HMAC secret for stub session resolution (test-only, never real)
export GROUP_AGENT_PRINCIPAL_HMAC_SECRET="test_32byte_secret_for_req012!!"

# ── 5. Run pytest ───────────────────────────────────────────────────────────
echo "[REQ-012] Running real-LLM acceptance test..."
echo "[REQ-012] pytest -v tests/test_group_agent_req012_real_llm.py -m real_llm"

# The test writes a single structured outcome line to this file; the runner
# reads ONLY this file for its verdict (never parses full pytest output — FIX2 §3).
OUTCOME_FILE="$(mktemp /tmp/req012_outcome.XXXXXX)"
export GROUP_AGENT_REQ012_OUTCOME_FILE="${OUTCOME_FILE}"
AUDIT_META_FILE="$(mktemp /tmp/req013_audit_meta.XXXXXX)"
export GROUP_AGENT_REQ013_REPORT_META_FILE="${AUDIT_META_FILE}"

set +e
"${REPO_DIR}/.venv/bin/pytest" \
    "${REPO_DIR}/tests/test_group_agent_req012_real_llm.py" \
    -v \
    -s \
    -m real_llm \
    -p no:cacheprovider
EXIT_CODE=$?
set -e

# ── 6. Structured outcome via allow-matrix (REQ-012-FIX3 §2) ───────────────
# The verdict is decided by decide_runner_result(exit_code, outcome_text),
# which enforces: exit=0 ⇒ only PASSED; exit!=0 ⇒ PASSED forbidden;
# unknown/multiline/oversized ⇒ FAILED:INTERNAL. The runner never trusts a
# raw outcome file on its own, and never parses full pytest stdout.
OUTCOME=""
if [ -s "${OUTCOME_FILE}" ]; then
    OUTCOME="$(head -c 4096 "${OUTCOME_FILE}")"
fi
rm -f "${OUTCOME_FILE}"

VERDICT="$(GROUP_AGENT_REQ012_EXIT="${EXIT_CODE}" GROUP_AGENT_REQ012_OUTCOME="${OUTCOME}" \
    PYTHONPATH="${REPO_DIR}" "${REPO_DIR}/.venv/bin/python" - <<'PYEOF'
import os
from tests.support.req012_llm_budget import decide_runner_result
exit_code = int(os.environ.get("GROUP_AGENT_REQ012_EXIT", "1") or "1")
outcome = os.environ.get("GROUP_AGENT_REQ012_OUTCOME", "")
print(decide_runner_result(exit_code, outcome))
PYEOF
)"

RESULT_LINE="REQ-012 RESULT: ${VERDICT}"

echo ""
echo "=============================================="
echo "${RESULT_LINE}"
echo "=============================================="
echo ""

# Exit code reflects the reconciled verdict: 0 only when it is exactly PASSED.
if [ "${VERDICT}" = "PASSED" ]; then
    if [ "${GROUP_AGENT_HUMAN_AUDIT_REPORT:-}" = "1" ]; then
        # Validate the structured metadata AND the final files before printing
        # only absolute path, byte size and SHA-256. Never print report content.
        set +e
        AUDIT_LINES="$(GROUP_AGENT_REQ013_META="${AUDIT_META_FILE}" \
            "${REPO_DIR}/.venv/bin/python" - <<'PYEOF'
import hashlib
import json
import os
import re
from pathlib import Path

meta_path = Path(os.environ["GROUP_AGENT_REQ013_META"])
try:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    lines = []
    validated = {}
    for kind in ("markdown", "json"):
        item = payload[kind]
        path = Path(item["path"])
        size = item["size"]
        expected = item["sha256"]
        if not path.is_absolute() or any(ord(ch) < 32 for ch in str(path)):
            raise ValueError("unsafe path")
        if not isinstance(size, int) or size < 1:
            raise ValueError("invalid size")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid sha256")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != size or actual != expected:
            raise ValueError("metadata mismatch")
        validated[kind] = {
            "path": path,
            "size": size,
            "sha256": expected,
        }
        lines.append(f"REQ-013 AUDIT {kind}: path={path} size={size} sha256={expected}")
    ready_item = payload["ready"]
    ready_path = Path(ready_item["path"])
    ready_expected = ready_item["sha256"]
    if (
        not ready_path.is_absolute()
        or ready_path.name != "READY.json"
        or any(ord(ch) < 32 for ch in str(ready_path))
        or not isinstance(ready_expected, str)
        or not re.fullmatch(r"[0-9a-f]{64}", ready_expected)
    ):
        raise ValueError("invalid READY metadata")
    ready_bytes = ready_path.read_bytes()
    if hashlib.sha256(ready_bytes).hexdigest() != ready_expected:
        raise ValueError("READY hash mismatch")
    ready = json.loads(ready_bytes)
    if (
        ready.get("schema_version") != "GA-HUMAN-AUDIT-V1"
        or ready.get("run_id") not in ready_path.parent.name
    ):
        raise ValueError("READY identity mismatch")
    for kind in ("markdown", "json"):
        item = ready["files"][kind]
        actual = validated[kind]
        if (
            actual["path"].parent != ready_path.parent
            or item["name"] != actual["path"].name
            or item["size"] != actual["size"]
            or item["sha256"] != actual["sha256"]
        ):
            raise ValueError("READY pair mismatch")
except Exception:
    raise SystemExit(1)
print("\n".join(lines))
PYEOF
        )"
        AUDIT_META_STATUS=$?
        set -e
        if [ "${AUDIT_META_STATUS}" -ne 0 ]; then
            echo "[ERROR] REQ-013 audit metadata validation failed"
            exit 1
        fi
        echo "${AUDIT_LINES}"
    fi
    echo "[INFO] 非 real-LLM 回归测试请运行:"
    echo "  pytest tests/test_group_agent_req010.py -v -m 'not real_llm'"
    exit 0
fi

# Non-pass verdict: preserve a non-zero exit (use pytest's if it was non-zero).
if [ "${EXIT_CODE}" -eq 0 ]; then
    exit 1
fi
exit "${EXIT_CODE}"
