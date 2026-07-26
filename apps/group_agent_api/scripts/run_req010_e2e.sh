#!/usr/bin/env bash
set -euo pipefail

echo "========================================================"
echo "REQ-010 Local Container Dialog Runner & E2E Acceptance"
echo "========================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${APP_DIR}/../.." && pwd)"

TAG="group-agent-api:req010-head"
NET_NAME="req010-net-$$"
SIMULATOR_CONTAINER="req010-sim-$$"
API_CONTAINER="req010-api-$$"

SIMULATOR_PORT=3009
API_PORT=3008

TEST_PRINCIPAL_SECRET="test_principal_secret_32bytes_long!"
TEST_CALLBACK_SECRET="test_callback_secret_32bytes_long!"

cleanup() {
  echo "[Runner] Cleaning up containers and networks..."
  docker rm -f "${SIMULATOR_CONTAINER}" "${API_CONTAINER}" 2>/dev/null || true
  docker network rm "${NET_NAME}" 2>/dev/null || true
  echo "[Runner] Cleanup complete."
}
trap cleanup EXIT

echo "[1/6] Building HEAD image '${TAG}'..."
docker build -t "${TAG}" -f "${APP_DIR}/Dockerfile" "${REPO_DIR}"

echo "[2/6] Creating isolated Docker network '${NET_NAME}'..."
docker network create "${NET_NAME}"

echo "[3/6] Starting Callback Simulator container..."
docker run -d \
  --name "${SIMULATOR_CONTAINER}" \
  --network "${NET_NAME}" \
  -e GROUP_AGENT_CALLBACK_HMAC_SECRET="${TEST_CALLBACK_SECRET}" \
  -p "127.0.0.1:${SIMULATOR_PORT}:3009" \
  "${TAG}" \
  python -m uvicorn apps.group_agent_api.fixtures.callback_simulator:app --host 0.0.0.0 --port 3009

echo "[4/6] Starting group_agent_api container..."
docker run -d \
  --name "${API_CONTAINER}" \
  --network "${NET_NAME}" \
  -e GROUP_AGENT_INTEGRATION=stub \
  -e GROUP_AGENT_ENV=test \
  -e GROUP_AGENT_MODEL_MODE=stub \
  -e GROUP_AGENT_PRINCIPAL_HMAC_SECRET="${TEST_PRINCIPAL_SECRET}" \
  -e GROUP_AGENT_CALLBACK_HMAC_SECRET="${TEST_CALLBACK_SECRET}" \
  -e GROUP_AGENT_CALLBACK_ALLOWED_BASE_URLS="http://${SIMULATOR_CONTAINER}:3009/group_agent_callbacks,http://127.0.0.1:${SIMULATOR_PORT}/group_agent_callbacks" \
  -e GROUP_AGENT_TEST_LEVEL=L1 \
  -p "127.0.0.1:${API_PORT}:8001" \
  "${TAG}"

echo "[5/6] Waiting for services to be ready..."
READY=0
for i in {1..30}; do
  if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null && curl -sf "http://127.0.0.1:${SIMULATOR_PORT}/health" >/dev/null; then
    echo "[Runner] Services ready!"
    READY=1
    break
  fi
  sleep 1
done

if [ "${READY}" -ne 1 ]; then
  echo "[ERROR] Readiness check timed out after 30s!"
  echo "=== API Container Logs ==="
  docker logs "${API_CONTAINER}" || true
  echo "=== Simulator Container Logs ==="
  docker logs "${SIMULATOR_CONTAINER}" || true
  exit 1
fi

echo "[6/6] Executing REQ-010 Scenario Integration Tests..."
export REQ010_CONTAINER_API_BASE="http://127.0.0.1:${API_PORT}"
export REQ010_CONTAINER_SIMULATOR_BASE="http://127.0.0.1:${SIMULATOR_PORT}"
export REQ010_CONTAINER_CALLBACK_URL="http://${SIMULATOR_CONTAINER}:3009/group_agent_callbacks"
export GROUP_AGENT_PRINCIPAL_HMAC_SECRET="${TEST_PRINCIPAL_SECRET}"
export GROUP_AGENT_CALLBACK_HMAC_SECRET="${TEST_CALLBACK_SECRET}"

if ! PYTHONPATH="${REPO_DIR}" "${REPO_DIR}/.venv/bin/pytest" "${REPO_DIR}/tests/test_group_agent_req010.py" -v -m "container_e2e or l1 or l2 or l3"; then
  echo "=== API Container Logs on Test Failure ==="
  docker logs "${API_CONTAINER}" || true
  exit 1
fi

echo "========================================================"
echo "[Runner Stats Report]"
STATS_JSON=$(curl -sf "http://127.0.0.1:${SIMULATOR_PORT}/stats")
echo "${STATS_JSON}"
echo ""

RECORDS_COUNT=$(echo "${STATS_JSON}" | grep -o '"records_count":[0-9]*' | cut -d':' -f2 || echo "0")
TERMINAL_RUNS=$(echo "${STATS_JSON}" | grep -o '"terminal_runs":[0-9]*' | cut -d':' -f2 || echo "0")
ACTIVE_RUNS=$(echo "${STATS_JSON}" | grep -o '"active_runs":[0-9]*' | cut -d':' -f2 || echo "0")
HMAC_FAILURES=$(echo "${STATS_JSON}" | grep -o '"hmac_failures":[0-9]*' | cut -d':' -f2 || echo "0")
SEQ_FAILURES=$(echo "${STATS_JSON}" | grep -o '"seq_failures":[0-9]*' | cut -d':' -f2 || echo "0")
TERMINAL_FAILURES=$(echo "${STATS_JSON}" | grep -o '"terminal_failures":[0-9]*' | cut -d':' -f2 || echo "0")

if [ "${RECORDS_COUNT}" -eq 0 ]; then
  echo "[ERROR] Callback records count is 0! Container E2E did not receive any callbacks!"
  exit 1
fi

if [ "${TERMINAL_RUNS}" -ne 4 ]; then
  echo "[ERROR] Expected terminal_runs == 4 for L1 container scenarios, got ${TERMINAL_RUNS}!"
  exit 1
fi

if [ "${ACTIVE_RUNS}" -ne 0 ]; then
  echo "[ERROR] Expected active_runs == 0 after all runs completed, got ${ACTIVE_RUNS}!"
  exit 1
fi

if [ "${HMAC_FAILURES}" -ne 0 ] || [ "${SEQ_FAILURES}" -ne 0 ] || [ "${TERMINAL_FAILURES}" -ne 0 ]; then
  echo "[ERROR] Callback failures detected: hmac=${HMAC_FAILURES}, seq=${SEQ_FAILURES}, terminal=${TERMINAL_FAILURES}!"
  exit 1
fi

echo "REQ-010 Container E2E Acceptance PASSED successfully! (records_count=${RECORDS_COUNT}, terminal_runs=${TERMINAL_RUNS}, active_runs=${ACTIVE_RUNS})"
echo "========================================================"
