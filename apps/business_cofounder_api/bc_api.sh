#!/usr/bin/env bash
set -euo pipefail

# Simple CLI for manually testing the Business Co-Founder API (local or remote).
#
# Defaults:
#   BASE_URL=http://127.0.0.1:8001
#   USER_ID=u1
#   CONV_ID=default
#
# Override via env:
#   BC_API_BASE_URL, BC_API_USER_ID, BC_API_CONV_ID
#
# Examples:
#   chmod +x apps/business_cofounder_api/bc_api.sh
#   ./apps/business_cofounder_api/bc_api.sh health
#   ./apps/business_cofounder_api/bc_api.sh reset
#   ./apps/business_cofounder_api/bc_api.sh chat "Write a complete business plan."
#   ./apps/business_cofounder_api/bc_api.sh stream "Convert the plan into a single HTML page. Output ONLY HTML."
#   BC_API_ENABLE_STATE_ENDPOINT=1 ./apps/business_cofounder_api/bc_api.sh state

BASE_URL="${BC_API_BASE_URL:-http://127.0.0.1:8001}"
USER_ID="${BC_API_USER_ID:-u1}"
CONV_ID="${BC_API_CONV_ID:-default}"

json_escape() {
  python - <<'PY' "$1"
import json,sys
print(json.dumps(sys.argv[1]))
PY
}

health() {
  curl -sS "${BASE_URL}/health"
  echo
}

reset_thread() {
  curl -sS -X POST "${BASE_URL}/reset" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"${USER_ID}\",\"conversation_id\":\"${CONV_ID}\"}"
  echo
}

chat() {
  local msg="${1:?message required}"
  curl -sS -X POST "${BASE_URL}/chat" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"${USER_ID}\",\"conversation_id\":\"${CONV_ID}\",\"message\":$(json_escape "${msg}")}"
  echo
}

stream() {
  local msg="${1:?message required}"
  curl -N -sS -X POST "${BASE_URL}/chat/stream" \
    -H "Accept: text/event-stream" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"${USER_ID}\",\"conversation_id\":\"${CONV_ID}\",\"message\":$(json_escape "${msg}")}"
  echo
}

state() {
  # Requires server to be started with BC_API_ENABLE_STATE_ENDPOINT=1
  local resp
  resp="$(curl -sS "${BASE_URL}/state?user_id=${USER_ID}&conversation_id=${CONV_ID}")"
  if echo "${resp}" | grep -q '"detail"[[:space:]]*:[[:space:]]*"Not found"'; then
    echo "ERROR: /state is disabled on the server." >&2
    echo "Hint: restart the API server with: export BC_API_ENABLE_STATE_ENDPOINT=1" >&2
    echo "" >&2
    echo "${resp}" | python -m json.tool
    echo
    return 1
  fi
  echo "${resp}" | python -m json.tool
  echo
}

usage() {
  cat <<EOF
Usage:
  BC_API_BASE_URL=http://127.0.0.1:8001 BC_API_USER_ID=u1 BC_API_CONV_ID=default \\
    ./apps/business_cofounder_api/bc_api.sh <command> [args...]

Commands:
  health                  GET /health
  reset                   POST /reset
  state                   GET /state (milestones + todos; enable server with BC_API_ENABLE_STATE_ENDPOINT=1)
  chat   "<message>"      POST /chat (non-stream)
  stream "<message>"      POST /chat/stream (SSE streaming)
EOF
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  health) health ;;
  reset) reset_thread ;;
  state) state ;;
  chat) chat "${*}" ;;
  stream) stream "${*}" ;;
  ""|-h|--help|help) usage ;;
  *) echo "Unknown command: ${cmd}" >&2; usage; exit 2 ;;
esac


