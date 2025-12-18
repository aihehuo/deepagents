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
#   # Complete workflow (builds business plan from vague idea)
#   ./apps/business_cofounder_api/bc_api.sh demo "I want to make an app"
#   ./apps/business_cofounder_api/bc_api.sh demo  # uses default vague idea
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
# Demo/Workflow function (complete end-to-end business plan build-up)
# ============================================================================

demo_workflow() {
  local vague_idea="${1:-I want to make an app for businesses}"
  
  if [[ "${API_TYPE}" == "bp" ]]; then
    echo "ERROR: Demo workflow only works with BC API (API_TYPE=bc)" >&2
    echo "Hint: The workflow demonstrates progressive conversation-based idea development" >&2
    return 1
  fi

  echo "=================================================================================="
  echo "Business Plan Development Workflow"
  echo "=================================================================================="
  echo "Starting with a vague, incomplete idea:"
  echo "  \"${vague_idea}\""
  echo ""
  echo "This workflow will progressively build this into a complete business plan."
  echo "We'll check progress at each milestone by calling the state API."
  echo "=================================================================================="
  echo ""

  # Step 0: Reset the conversation
  echo ">>> Step 0: Resetting conversation..."
  bc_reset > /dev/null 2>&1
  echo "✓ Conversation reset"
  echo ""

  # Helper function to check state and extract milestone status
  get_milestone_status() {
    local milestone="$1"
    local state_output
    state_output="$(bc_state 2>&1 | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    milestone_name = '${milestone}'
    status = data.get('milestones', {}).get(milestone_name, False)
    print(str(status).lower())
except Exception:
    print('false')
" 2>/dev/null || echo "false")"
    echo "${state_output}"
  }

  # Helper function to check state and display milestones
  check_state() {
    local step_name="$1"
    echo ">>> Checking state after: ${step_name}..."
    local state_output
    if state_output="$(bc_state 2>&1)"; then
      echo "${state_output}"
    else
      echo "⚠ State endpoint not available (this is okay, continuing anyway)"
    fi
    echo ""
  }

  # Helper function to make a streaming chat call and show response
  make_chat() {
    local step_name="$1"
    local message="$2"
    echo ">>> ${step_name}..."
    echo "Sending: ${message}"
    echo ""
    local response
    response="$(bc_stream "" "${message}" 2>&1)"
    echo "${response}"
    echo ""
    
    # Check if response contains an error
    if echo "${response}" | grep -q '"error_type"'; then
      echo "⚠ API error encountered, but continuing workflow..."
      echo ""
    fi
    
    echo "---"
    echo ""
  }

  # Helper function to wait for milestone completion (with timeout)
  wait_for_milestone() {
    local milestone="$1"
    local max_attempts="${2:-5}"
    local attempt=0
    
    while [[ ${attempt} -lt ${max_attempts} ]]; do
      sleep 2
      local status
      status="$(get_milestone_status "${milestone}")"
      if [[ "${status}" == "true" ]]; then
        return 0
      fi
      attempt=$((attempt + 1))
    done
    return 1
  }

  # Step 1: Start with vague idea - let agent evaluate and guide
  make_chat "Step 1: Initial idea submission" "${vague_idea}"
  check_state "Initial idea submission"

  # Step 2: Provide a more complete idea when agent asks for clarification
  # Build a complete idea based on common patterns
  local refined_idea="I want to build a B2B SaaS platform for small to medium-sized restaurants (20-100 employees) to manage inventory and reduce food waste. The problem: Restaurants lose money due to poor inventory tracking, leading to over-ordering, spoilage, and waste. The solution: An AI-powered inventory management app that integrates with POS systems, tracks inventory in real-time, predicts ordering needs based on historical data and weather patterns, and sends automated alerts for items approaching expiration. Target users are restaurant managers and owners who struggle with manual inventory processes."
  
  echo ">>> Step 2: Providing refined business idea based on agent's guidance..."
  echo "The agent requested more details (WHO, WHAT, HOW). Providing a complete idea:"
  echo ""
  make_chat "Step 2: Refined idea submission" "${refined_idea}"
  
  # Wait for business_idea_complete milestone
  echo ">>> Waiting for business idea to be marked as complete..."
  if wait_for_milestone "business_idea_complete" 5; then
    echo "✓ Business idea marked as complete"
  else
    echo "⚠ Business idea not yet marked as complete, but continuing..."
  fi
  check_state "Business idea completion"

  # Step 3: Persona clarification (skip if already done)
  local persona_status
  persona_status="$(get_milestone_status "persona_clarified")"
  if [[ "${persona_status}" != "true" ]]; then
    make_chat "Step 3: Persona clarification" "Now please help me clarify the target user persona. Create a detailed persona profile with demographics, goals, pain points, and behaviors. When done, mark the persona as clarified."
    
    # Wait for persona_clarified milestone
    echo ">>> Waiting for persona to be marked as clarified..."
    if wait_for_milestone "persona_clarified" 5; then
      echo "✓ Persona marked as clarified"
    else
      echo "⚠ Persona not yet marked as clarified, but continuing..."
    fi
  else
    echo ">>> Step 3: Persona clarification (already completed, skipping)"
  fi
  check_state "Persona clarification"

  # Step 4: Pain point enhancement (skip if already done)
  local painpoint_status
  painpoint_status="$(get_milestone_status "painpoint_enhanced")"
  if [[ "${painpoint_status}" != "true" ]]; then
    make_chat "Step 4: Pain point enhancement" "Please enhance the pain point using the six emotional-resonance dimensions (urgency, frequency, economic cost, universality, viral spread, regulatory pressure). When done, mark the pain point as enhanced."
    
    # Wait for painpoint_enhanced milestone
    echo ">>> Waiting for pain point to be marked as enhanced..."
    if wait_for_milestone "painpoint_enhanced" 5; then
      echo "✓ Pain point marked as enhanced"
    else
      echo "⚠ Pain point not yet marked as enhanced, but continuing..."
    fi
  else
    echo ">>> Step 4: Pain point enhancement (already completed, skipping)"
  fi
  check_state "Pain point enhancement"

  # Step 5: 60-second pitch creation (skip if already done)
  local pitch_status
  pitch_status="$(get_milestone_status "pitch_created")"
  if [[ "${pitch_status}" != "true" ]]; then
    make_chat "Step 5: 60-second pitch creation" "Create a structured 60-second pitch for this business idea. Include pain point resonance, team advantage statement, and call to action. When done, mark the pitch as created."
    
    # Wait for pitch_created milestone
    echo ">>> Waiting for pitch to be marked as created..."
    if wait_for_milestone "pitch_created" 8; then
      echo "✓ Pitch marked as created"
    else
      echo "⚠ Pitch not yet marked as created, but continuing..."
    fi
  else
    echo ">>> Step 5: 60-second pitch creation (already completed, skipping)"
  fi
  check_state "Pitch creation"

  # Step 6: Pricing optimization (skip if already done)
  local pricing_status
  pricing_status="$(get_milestone_status "pricing_optimized")"
  if [[ "${pricing_status}" != "true" ]]; then
    make_chat "Step 6: Pricing optimization" "Establish baseline pricing and optimization using the 1/10 value rule. Generate pricing tactics and identify key partners. When done, mark pricing as optimized."
    
    # Wait for pricing_optimized milestone
    echo ">>> Waiting for pricing to be marked as optimized..."
    if wait_for_milestone "pricing_optimized" 8; then
      echo "✓ Pricing marked as optimized"
    else
      echo "⚠ Pricing not yet marked as optimized, but continuing..."
    fi
  else
    echo ">>> Step 6: Pricing optimization (already completed, skipping)"
  fi
  check_state "Pricing optimization"

  # Step 7: Business model exploration (this step doesn't have a milestone, always do it)
  echo ">>> Step 7: Business model exploration..."
  make_chat "Step 7: Business model exploration" "Explore alternative business model archetypes (Retail, Service, Brokerage, Subscription, Usage-based, Membership, Transaction). Test-fit our product/service into these models and identify the most promising alternatives."
  sleep 2
  check_state "Business model exploration"

  # Step 8: Complete business plan generation (no milestone for this, always do it)
  echo ">>> Step 8: Complete business plan generation..."
  make_chat "Step 8: Complete business plan generation" "Now write a complete, comprehensive business plan in English for this startup. Include sections: Executive Summary, Problem Statement, Solution, Target Customers & Persona, Market Size, Competitive Landscape, Business Model & Pricing, Go-To-Market Strategy, Product Roadmap (next 90 days), Operations Plan, Team Requirements, Financial Projections, Risks & Mitigations, and Next Steps. Use clear formatting with bullets and short paragraphs."
  sleep 2
  check_state "Business plan generation"

  # Step 9: Convert to HTML
  echo ">>> Step 9: Converting business plan to HTML..."
  echo "This will generate a complete HTML page with the business plan."
  echo ""
  local html_output
  html_output="$(bc_stream "" "Convert the complete business plan in our conversation into a SINGLE, complete, modern HTML page. Requirements: Return a complete HTML document (<html>...</html>) with embedded CSS. Use a clean, professional layout with readable typography. Include all sections from the business plan with proper headings and formatting. Do not explain or summarize - output ONLY the HTML code." 2>&1)"
  echo "${html_output}"
  echo ""
  
  # Check if HTML generation had errors
  if echo "${html_output}" | grep -q '"error_type"'; then
    echo "⚠ HTML generation encountered an error, but workflow completed milestones."
    echo "You may need to fix the conversation state or regenerate HTML separately."
    echo ""
  fi
  
  # Final state check
  echo ">>> Final state check..."
  check_state "HTML generation"

  echo "=================================================================================="
  echo "Workflow Complete!"
  echo "=================================================================================="
  echo "The business plan has been developed from a vague idea to a complete HTML document."
  echo "All milestones should be marked as complete in the state."
  echo ""
  echo "To review the final state, run:"
  echo "  $0 state"
  echo ""
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
  demo   "[<vague_idea>]" Complete workflow: build business plan from vague idea (BC API only)

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
  ./apps/business_cofounder_api/bc_api.sh demo "I want to make an app for businesses"
  
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
  chat) chat "${*:-}" ;;
  stream) stream "${*:-}" ;;
  demo)
    if [[ $# -eq 0 ]]; then
      demo_workflow
    else
      demo_workflow "$*"
    fi
    ;;
  
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


