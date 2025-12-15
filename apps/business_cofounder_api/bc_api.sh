#!/usr/bin/env bash
set -euo pipefail

# CLI for testing Business Co-Founder API and BP Generation Agent API (local or remote).
#
# Supports two APIs:
#   - BC API (Business Co-Founder): chat-based, uses user_id/conversation_id
#   - BP API (BP Generation Agent): async generation with callbacks, uses session_id
#
# Defaults:
#   API_TYPE=bc (or bp)
#   BC API: BASE_URL=http://127.0.0.1:8001, USER_ID=u1, CONV_ID=default
#   BP API: BP_BASE_URL=http://127.0.0.1:8000
#
# Override via env:
#   API_TYPE=bc|bp
#   BC_API_BASE_URL, BC_API_USER_ID, BC_API_CONV_ID, BC_API_PORT
#   BP_API_BASE_URL, BP_API_PORT
#
# Examples:
#   # BC API
#   ./apps/business_cofounder_api/bc_api.sh health
#   ./apps/business_cofounder_api/bc_api.sh chat "Write a complete business plan."
#   ./apps/business_cofounder_api/bc_api.sh stream "Convert the plan into a single HTML page."
#
#   # BP API
#   API_TYPE=bp ./apps/business_cofounder_api/bc_api.sh health
#   API_TYPE=bp ./apps/business_cofounder_api/bc_api.sh generate "我想创建一个AI教育平台"
#   API_TYPE=bp ./apps/business_cofounder_api/bc_api.sh list-files <session_id>
#
#   # Test both (comparison)
#   ./apps/business_cofounder_api/bc_api.sh compare-health
#   ./apps/business_cofounder_api/bc_api.sh compare-generate "我想创建一个AI教育平台"

API_TYPE="${API_TYPE:-bc}"
BC_PORT="${BC_API_PORT:-8001}"
BC_BASE_URL="${BC_API_BASE_URL:-http://127.0.0.1:${BC_PORT}}"
USER_ID="${BC_API_USER_ID:-u1}"
CONV_ID="${BC_API_CONV_ID:-default}"

BP_PORT="${BP_API_PORT:-8000}"
BP_BASE_URL="${BP_API_BASE_URL:-http://127.0.0.1:${BP_PORT}}"

json_escape() {
  python - <<'PY' "$1"
import json,sys
print(json.dumps(sys.argv[1]))
PY
}

# ============================================================================
# BC API Functions (Business Co-Founder)
# ============================================================================

bc_health() {
  local base_url="${1:-${BC_BASE_URL}}"
  curl -sS "${base_url}/health"
  echo
}

bc_reset() {
  local base_url="${1:-${BC_BASE_URL}}"
  curl -sS -X POST "${base_url}/reset" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"${USER_ID}\",\"conversation_id\":\"${CONV_ID}\"}"
  echo
}

bc_chat() {
  local base_url="${1:-${BC_BASE_URL}}"
  local msg="${2:?message required}"
  curl -sS -X POST "${base_url}/chat" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"${USER_ID}\",\"conversation_id\":\"${CONV_ID}\",\"message\":$(json_escape "${msg}")}"
  echo
}

bc_stream() {
  local base_url="${1:-${BC_BASE_URL}}"
  local msg="${2:?message required}"
  curl -N -sS -X POST "${base_url}/chat/stream" \
    -H "Accept: text/event-stream" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"${USER_ID}\",\"conversation_id\":\"${CONV_ID}\",\"message\":$(json_escape "${msg}")}"
  echo
}

bc_state() {
  local base_url="${1:-${BC_BASE_URL}}"
  local resp
  resp="$(curl -sS "${base_url}/state?user_id=${USER_ID}&conversation_id=${CONV_ID}")"
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

# ============================================================================
# BP API Functions (BP Generation Agent)
# ============================================================================

bp_health() {
  local base_url="${1:-${BP_BASE_URL}}"
  curl -sS "${base_url}/health"
  echo
}

bp_config() {
  local base_url="${1:-${BP_BASE_URL}}"
  curl -sS "${base_url}/api/v1/config" | python -m json.tool
  echo
}

bp_generate_async() {
  local base_url="${1:-${BP_BASE_URL}}"
  local business_idea="${2:?business_idea required}"
  local session_id="${3:-}"  # optional
  local callback_url="${4:-}"  # optional
  
  local payload="{\"business_idea\":$(json_escape "${business_idea}"),\"save_report\":true"
  if [[ -n "${session_id}" ]]; then
    payload="${payload},\"session_id\":$(json_escape "${session_id}")"
  fi
  if [[ -n "${callback_url}" ]]; then
    payload="${payload},\"callback_url\":$(json_escape "${callback_url}")"
  fi
  payload="${payload}}"
  
  curl -sS -X POST "${base_url}/api/v1/generate-async" \
    -H "Content-Type: application/json" \
    -d "${payload}" | python -m json.tool
  echo
}

bp_list_files() {
  local base_url="${1:-${BP_BASE_URL}}"
  local session_id="${2:?session_id required}"
  curl -sS "${base_url}/api/v1/files/${session_id}" | python -m json.tool
  echo
}

bp_get_file() {
  local base_url="${1:-${BP_BASE_URL}}"
  local session_id="${2:?session_id required}"
  local filename="${3:?filename required}"
  curl -sS "${base_url}/api/v1/files/${session_id}/${filename}"
  echo
}

# ============================================================================
# Wrapper functions that route to BC or BP based on API_TYPE
# ============================================================================

health() {
  if [[ "${API_TYPE}" == "bp" ]]; then
    bp_health
  else
    bc_health
  fi
}

reset() {
  if [[ "${API_TYPE}" == "bp" ]]; then
    echo "ERROR: BP API does not have a reset endpoint" >&2
    echo "Hint: Use a new session_id for each generation" >&2
    return 1
  else
    bc_reset
  fi
}

chat() {
  local msg="${1:?message required}"
  if [[ "${API_TYPE}" == "bp" ]]; then
    echo "ERROR: BP API uses 'generate' command, not 'chat'" >&2
    echo "Hint: Use: API_TYPE=bp $0 generate \"${msg}\"" >&2
    return 1
  else
    bc_chat "" "${msg}"
  fi
}

stream() {
  local msg="${1:?message required}"
  if [[ "${API_TYPE}" == "bp" ]]; then
    echo "ERROR: BP API uses async generation with callbacks, not streaming" >&2
    echo "Hint: Use: API_TYPE=bp $0 generate \"${msg}\"" >&2
    return 1
  else
    bc_stream "" "${msg}"
  fi
}

state() {
  if [[ "${API_TYPE}" == "bp" ]]; then
    echo "ERROR: BP API does not have a state endpoint" >&2
    echo "Hint: Use 'list-files <session_id>' to check generation status" >&2
    return 1
  else
    bc_state
  fi
}

# ============================================================================
# Comparison functions (test both APIs)
# ============================================================================

compare_health() {
  echo "=== BC API Health ==="
  bc_health
  echo
  echo "=== BP API Health ==="
  bp_health
  echo
}

compare_generate() {
  local business_idea="${1:?business_idea required}"
  
  echo "=== Testing BC API ==="
  echo "Business Idea: ${business_idea}"
  echo "Command: API_TYPE=bc $0 chat \"${business_idea}\""
  echo
  bc_chat "" "${business_idea}"
  echo
  echo "=== Testing BP API ==="
  echo "Business Idea: ${business_idea}"
  echo "Command: API_TYPE=bp $0 generate \"${business_idea}\""
  echo
  bp_generate_async "" "${business_idea}"
  echo
}

usage() {
  cat <<EOF
Usage:
  API_TYPE=bc|bp [BC_API_BASE_URL=...] [BP_API_BASE_URL=...] \\
    ./apps/business_cofounder_api/bc_api.sh [-p PORT] <command> [args...]

API Selection:
  API_TYPE=bc  - Business Co-Founder API (default)
  API_TYPE=bp  - BP Generation Agent API

BC API Commands (API_TYPE=bc):
  health                  GET /health
  reset                   POST /reset
  state                   GET /state (milestones + todos; enable with BC_API_ENABLE_STATE_ENDPOINT=1)
  chat   "<message>"      POST /chat (non-stream)
  stream "<message>"      POST /chat/stream (SSE streaming)

BP API Commands (API_TYPE=bp):
  health                  GET /health
  config                  GET /api/v1/config
  generate "<idea>"       POST /api/v1/generate-async
  list-files <session_id> GET /api/v1/files/<session_id>
  get-file <session_id> <filename> GET /api/v1/files/<session_id>/<filename>

Comparison Commands (tests both APIs):
  compare-health          Test health endpoint on both APIs
  compare-generate "<idea>" Test generation on both APIs

Examples:
  # BC API
  ./apps/business_cofounder_api/bc_api.sh health
  ./apps/business_cofounder_api/bc_api.sh chat "Write a business plan"
  
  # BP API
  API_TYPE=bp ./apps/business_cofounder_api/bc_api.sh health
  API_TYPE=bp ./apps/business_cofounder_api/bc_api.sh generate "我想创建一个AI教育平台"
  
  # Compare both
  ./apps/business_cofounder_api/bc_api.sh compare-health
EOF
}

while getopts ":p:" opt; do
  case "${opt}" in
    p)
      # Set port based on API type
      if [[ "${API_TYPE}" == "bp" ]]; then
        BP_PORT="${OPTARG}"
        if [[ -z "${BP_API_BASE_URL:-}" ]]; then
          BP_BASE_URL="http://127.0.0.1:${BP_PORT}"
        fi
      else
        BC_PORT="${OPTARG}"
        if [[ -z "${BC_API_BASE_URL:-}" ]]; then
          BC_BASE_URL="http://127.0.0.1:${BC_PORT}"
        fi
      fi
      ;;
    \?)
      echo "Unknown option: -${OPTARG}" >&2
      usage
      exit 2
      ;;
    :)
      echo "Missing value for -${OPTARG}" >&2
      usage
      exit 2
      ;;
  esac
done
shift $((OPTIND - 1))

# If the user didn't override base URLs, rebuild using the (possibly overridden) ports.
if [[ -z "${BC_API_BASE_URL:-}" ]]; then
  BC_BASE_URL="http://127.0.0.1:${BC_PORT}"
fi
if [[ -z "${BP_API_BASE_URL:-}" ]]; then
  BP_BASE_URL="http://127.0.0.1:${BP_PORT}"
fi

cmd="${1:-}"
shift || true

case "${cmd}" in
  # BC API commands
  health) health ;;
  reset) reset ;;
  state) state ;;
  chat) chat "${*}" ;;
  stream) stream "${*}" ;;
  
  # BP API commands (work with API_TYPE=bp or can be called directly)
  config) 
    if [[ "${API_TYPE}" == "bp" ]]; then
      bp_config
    else
      echo "ERROR: 'config' is a BP API command. Use API_TYPE=bp" >&2
      exit 2
    fi
    ;;
  generate)
    if [[ "${API_TYPE}" == "bp" ]]; then
      bp_generate_async "" "${*}"
    else
      echo "ERROR: 'generate' is a BP API command. Use API_TYPE=bp" >&2
      exit 2
    fi
    ;;
  list-files)
    if [[ "${API_TYPE}" == "bp" ]]; then
      bp_list_files "" "${1:?session_id required}"
    else
      echo "ERROR: 'list-files' is a BP API command. Use API_TYPE=bp" >&2
      exit 2
    fi
    ;;
  get-file)
    if [[ "${API_TYPE}" == "bp" ]]; then
      bp_get_file "" "${1:?session_id required}" "${2:?filename required}"
    else
      echo "ERROR: 'get-file' is a BP API command. Use API_TYPE=bp" >&2
      exit 2
    fi
    ;;
  
  # Comparison commands
  compare-health) compare_health ;;
  compare-generate) compare_generate "${*}" ;;
  
  ""|-h|--help|help) usage ;;
  *) echo "Unknown command: ${cmd}" >&2; usage; exit 2 ;;
esac


