# Business Co-Founder API — Architecture

This document describes the architecture of the Business Co-Founder API, including the dual-agent design, context engineering, and current dataflow. It is intended to be maintained as the system evolves.

---

## 1. Overview

The Business Co-Founder API is a FastAPI service that provides conversational AI for entrepreneurial guidance. It supports two operational modes:

- **Dual-Agent Mode** (default): Facilitator (frontend) + Expert (backend) agents; natural conversation plus structured analysis and guidance.
- **Single-Agent Mode** (legacy): One combined Business Co-Founder agent with full middleware, skills, and workflows.

Both modes use the same HTTP API surface. Primary entrypoints are `/chat`, `/chat/stream`, and `/deep_agent/call_async`. Thread identity is `thread_id = bc::{user_id}::{conversation_id}`.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Application                                  │
│  /health  /chat  /chat/stream  /canvas  /reset  /state  /deep_agent/call_async   │
│                         /simulated_user/chat                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
            │   /chat       │   │ /chat/stream  │   │ /deep_agent/  │
            │   (sync)      │   │ (SSE)         │   │ call_async    │
            └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │         AppState (startup)             │
                    │  agent, fallback_agent, thread_locks   │
                    │  facilitator_agent, expert_agent       │
                    │  use_dual_agent, expertise_dir, …      │
                    └───────────────────┬───────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐             ┌─────────────────┐             ┌─────────────────┐
│  Facilitator  │             │     Expert      │             │  Simulated User │
│  (frontend)   │             │   (analyzer)    │             │  (testing)      │
│  create_deep_ │             │  create_expert_ │             │  create_user_   │
│  agent        │             │  agent          │             │  agent          │
└───────────────┘             └─────────────────┘             └─────────────────┘
        │                               │
        │  shared state                 │  analysis → state
        │  (checkpointer)               │  canvas, expert_guidance,
        │                               │  partner_query, partner_search_results
        └───────────────────────────────┘
```

- **Startup** (`app/startup.py`): Reads `BC_API_USE_DUAL_AGENT`; builds facilitator + expert (and optional simulated user) in dual-agent mode, or a single business-cofounder agent otherwise. All agents use `create_deep_agent` (from `deepagents`) with app-specific middleware.
- **Request handling**: Chat and deep-agent endpoints resolve `thread_id`, acquire a per-thread lock, and invoke the primary agent (and optionally run expert sync in dual-agent mode). Async deep-agent runs the same flow in a background thread and POSTs updates to a callback URL.

---

## 3. Dual-Agent Design

### 3.1 Roles

| Agent | Role | Focus |
|-------|------|--------|
| **Facilitator** | Frontend conversation | Idea bouncer: acknowledge, one question per reply, pass expert guidance verbatim. Reply cap ~500 chars. |
| **Expert** | Backend analysis | Structured analysis, methodology, canvas updates, strategic guidance, optional partner search. |

- The **facilitator** holds conversation state (checkpoints). The **expert** runs in separate “expert” threads (e.g. `expert_analysis_{thread_id}`) and never owns the main conversation checkpoint.
- Expert output (guidance, canvas, partner query, etc.) is written into the **shared state** stored in the facilitator’s checkpointer. The facilitator sees this via middleware (e.g. injected guidance in the system prompt).

### 3.2 Facilitator Agent

- **Factory**: `agent_factory/facilitator_agent.py` → `create_facilitator_agent()`
- **Graph**: `create_deep_agent` with `subagents=[]`, light middleware, strict system prompt.
- **Middleware** (order matters):
  1. `AccountantMiddleware` (max 25 tool calls)
  2. `LanguageDetectionMiddleware`
  3. `ApiMemoryMiddleware` (long-term memory under `~/.deepagents/business_cofounder_api`)
  4. `ExpertGuidanceMiddleware` (round tracking, sync flagging, **injects expert guidance into system prompt**)
  5. `PromptLoggingMiddleware`
- **Checkpointer**: `~/.deepagents/business_cofounder_api/facilitator_checkpoints.pkl`
- **System prompt**: Idea bouncer only; no expansion. Expert guidance is passed through verbatim and explicitly attributed.

### 3.3 Expert Agent

- **Factory**: `agent_factory/expert_agent.py` → `create_expert_agent(expertise_type=...)`
- **Graph**: `create_deep_agent` with full middleware stack, **no** `ExpertGuidanceMiddleware`.
- **Middleware**:
  1. `AccountantMiddleware` (max 50 tool calls)
  2. `LanguageDetectionMiddleware`
  3. `BusinessIdeaTrackerMiddleware`
  4. `BusinessIdeaDevelopmentMiddleware`
  5. `VirtualPathSkillsMiddleware` (wraps `SkillsMiddleware`); skills under `~/.deepagents/business_cofounder_api/skills`
  6. `AihehuoMiddleware` (partner/market search)
  7. `AssetUploadMiddleware`
  8. `ArtifactsMiddleware`
- **Checkpointer**: `~/.deepagents/business_cofounder_api/expert_checkpoints.pkl` (expert-specific threads only).
- **System prompt**: Generic expert instructions plus **expertise-specific** content loaded from `expertise/{expertise_type}.md` (see §5.2).

### 3.4 Expert Sync (When and How)

- **When**: Before the facilitator handles a user message, the chat/stream and deep-agent flows check “should we run expert?” via `should_trigger_expert(state)` in `expert_sync.py`.
  - Triggers if `needs_expert_sync` is True (set by `ExpertGuidanceMiddleware` when `round - last_expert_sync >= sync_interval`), **or**
  - If `conversation_round - last_expert_sync >= STATE_EXPERT_SYNC_INTERVAL` (round-based).
- **Note**: `ExpertGuidanceMiddleware.sync_interval` is 10. `STATE_EXPERT_SYNC_INTERVAL` in `expert_sync.py` is 1. The round-based branch can therefore cause sync every round; the middleware branch every 10. This is a known configuration inconsistency to unify (e.g. single `EXPERT_SYNC_INTERVAL` env var).
- **How**:
  1. Load checkpoint for `thread_id`, get `messages` and state.
  2. `extract_recent_rounds(messages, rounds=10)`.
  3. `trigger_expert_analysis(...)`: build analysis prompt (including expertise template, canvas template, language), invoke expert in `expert_analysis_{thread_id}`.
  4. Parse JSON output → `expert_guidance`, `canvas`, `canvas_update_summary`, optional `partner_query`.
  5. Optional **language fix**: if canvas language ≠ user language, re-invoke expert to re-output in user’s language.
  6. Optional **partner search**: if `partner_query` present, call AI He Huo search API; on empty results, optionally `refine_partner_query` and retry (env `PARTNER_SEARCH_MAX_RETRIES`); then `generate_proposal_statements` for each user. Store `partner_query` and `partner_search_results` in analysis.
  7. `update_state_with_analysis(thread_id, analysis, checkpointer, facilitator_agent)`: merge analysis into facilitator checkpoint (via `aupdate_state` or `checkpointer.aput`). This updates `last_expert_sync`, `needs_expert_sync`, and all analysis fields.

Expert sync runs **synchronously** before the facilitator call (with a 60s timeout in chat flow). It blocks that request until done.

---

## 4. Single-Agent Mode (Legacy)

- **Factory**: `agent_factory/business_agent.py` → `create_business_cofounder_agent()`
- **Graph**: `create_deep_agent` with full middleware (business idea tracker, development, skills, Aihehuo, asset upload, artifacts, etc.) and optional subagents (e.g. coder, aihehuo).
- **Checkpointer**: `~/.deepagents/business_cofounder_api/checkpoints.pkl`
- **Use**: Set `BC_API_USE_DUAL_AGENT=0`. `/canvas` is disabled in single-agent mode.

---

## 5. Context Engineering

### 5.1 Shared State (`DualAgentState`)

Defined in `libs/deepagents/deepagents/state/dual_agent_state.py`. Used by both facilitator and expert-related logic (expert sync, canvas endpoint, etc.).

| Field | Purpose |
|-------|---------|
| `messages` | Conversation history (LangChain message types). |
| `conversation_round` | Incremented per user message (`ExpertGuidanceMiddleware`). |
| `last_expert_sync` | Round at which expert last ran. |
| `needs_expert_sync` | Flag to run expert; set by middleware when interval exceeded. |
| `expertise_type` | e.g. `business_cofounder`; selects expertise template and canvas schema. |
| `expert_guidance` | Strategic guidance from expert → injected into facilitator system prompt. |
| `canvas` | Domain-agnostic JSON (structure from expertise template). |
| `analysis_timestamp` | ISO 8601 of last expert run. |
| `partner_query` | Optional Chinese partner-search query from expert. |
| `partner_search_results` | Optional list of `{user: {id, avatar}, proposal_statement}` from partner search. |
| `user_id`, `conversation_id` | For memory paths and API wiring. |
| `detected_language` | From `LanguageDetectionMiddleware`. |
| `artifacts`, `tool_call_count`, token counts, etc. | Used by other middlewares. |

Canvas is treated as an opaque blob by the backend; the frontend interprets it per `expertise_type`.

### 5.2 Expert Guidance Injection

- **Middleware**: `ExpertGuidanceMiddleware` (in `libs/deepagents/deepagents/middleware/expert_guidance.py`).
- **Hooks**:
  - `before_agent`: Increment `conversation_round`; set `needs_expert_sync` when `round - last_expert_sync >= sync_interval`.
  - `wrap_model_call` / `awrap_model_call`: Read `expert_guidance` from state. If non-empty, format `EXPERT_GUIDANCE_SYSTEM_PROMPT_TEMPLATE` with `{guidance_content}` and **append** to the base system prompt before each model call.

The template stresses that the guidance overrides other instructions and must be followed immediately.

### 5.3 Expertise Templates and Canvas

- **Location**: `~/.deepagents/business_cofounder_api/expertise/` (and app-level `expertise/`). Files like `business_cofounder.md`, `prime_number.md`.
- **Format**: Markdown with YAML frontmatter:
  - `name`, `description`
  - `canvas_template`: JSON schema for the canvas (e.g. BMC blocks).
- **Loader**: `expertise_loader.load_expertise(expertise_type, dir)` returns `{name, description, system_prompt, canvas_template}`.
- **Usage**: Expert’s system prompt is built from generic expert text + expertise `system_prompt`. The analysis prompt includes `canvas_template` and instructs the expert to output JSON with `expert_guidance`, `canvas`, `canvas_update_summary`, and optional `partner_query`.

`expertise_type` can be set per request (`ChatRequest.expertise_type`, or `metadata.expertise_type` for deep-agent). Default `DEFAULT_EXPERTISE_TYPE` env.

### 5.4 Memory, Language, and Other Context

- **Memory**: `ApiMemoryMiddleware` uses `user_id` and `conversation_id` from metadata to scope persistent memory under the base dir.
- **Language**: `LanguageDetectionMiddleware` sets `detected_language`. Expert sync uses it for analysis prompt and partner search (e.g. proposal language). Canvas language is checked and optionally corrected via a second expert call.
- **Metadata**: Request-level `metadata` (and `expertise_type`) is passed into agent config and used by expert-sync logic (e.g. in callbacks) to resolve `expertise_type` when updating state.

---

## 6. Dataflow

### 6.1 `/chat` (Synchronous)

1. Parse `ChatRequest` → `user_id`, `conversation_id`, `message`, `expertise_type`, `metadata`.
2. `thread_id = bc::{user_id}::{conversation_id}`; get or create per-thread lock; acquire it.
3. **Dual-agent**: Load facilitator checkpoint; build state dict (`messages`, `conversation_round`, `expertise_type`, …). If `should_trigger_expert(state)`:
   - Call `trigger_and_update_expert(thread_id, state, expert_agent, checkpointer, expertise_dir, facilitator_agent)` (wait up to 60s).
   - Expert runs, state updated with analysis.
4. `agent.ainvoke({messages: [HumanMessage(content=message)], expertise_type}, {configurable: {thread_id}, metadata: {user_id, expertise_type, ...}})`. Primary agent is facilitator in dual-agent mode.
5. On failure, optional fallback to `fallback_agent` (e.g. deepseek).
6. Extract last AI message content → `reply`; log I/O and optional state debug; return `ChatResponse`.

### 6.2 `/chat/stream` (SSE)

- Same lock and expert-sync check as `/chat` (inside the async generator).
- Use `agent.astream_events` (or equivalent) to stream chunks. Emit SSE: `delta`, `progress`, `final`, and optionally `error`.
- Progress events can include tool-call activity. Token usage is parsed from message metadata when available.

### 6.3 `/deep_agent/call_async`

1. Parse `CallDeepAgentAsyncRequest` (includes `callback` URL and `metadata`).
2. `thread_id` as above; ensure thread lock exists (not held across the async call).
3. Start a **background thread** that runs `run_async_stream_with_callback(agent, message, thread_id, user_id, metadata, callback_url, fallback_agent, docs_dir, backend_root, expert_agent, use_dual_agent, expertise_dir)`.
4. Return immediately with `CallDeepAgentAsyncResponse(session_id=thread_id, ...)`.

**Callback flow** (inside the background thread):

- Uses same thread lock when performing agent/expert work for that `thread_id`.
- Expert sync logic mirrors chat: load checkpoint, `should_trigger_expert`, then `trigger_and_update_expert` if needed. `expertise_type` is taken from `metadata` (or state) and persisted when updating state.
- Streams agent output and POSTs updates to `callback_url`. Can include progress, deltas, and final reply.

### 6.4 Expert Sync Dataflow (Dual-Agent)

```
Request (chat/stream/call_async)
    │
    ▼
should_trigger_expert(state)  ← state from facilitator checkpoint
    │
    ├─ no  → skip expert sync
    │
    └─ yes
          │
          ▼
extract_recent_rounds(messages, 10)
          │
          ▼
trigger_expert_analysis(state, expert_agent, conversation_history, thread_id, expertise_dir)
          │
          │  • Load expertise template (canvas + instructions)
          │  • Build analysis prompt (conversation, current canvas/guidance, language)
          │  • expert_agent.ainvoke(…, expert_analysis_{thread_id})
          │  • parse_expert_response → expert_guidance, canvas, canvas_update_summary, partner_query
          │  • Optional: language fix re-invoke; partner search + proposals
          │  • Set last_expert_sync, needs_expert_sync=False in analysis
          ▼
update_state_with_analysis(thread_id, analysis, checkpointer, facilitator_agent)
          │
          │  • aupdate_state or checkpointer.aput into facilitator checkpoint
          ▼
Facilitator sees updated expert_guidance (and canvas, etc.) on next turn via ExpertGuidanceMiddleware
```

### 6.5 `/canvas` and `/state`

- **`/canvas`**: Dual-agent only. Reads facilitator checkpoint for `thread_id`, returns `CanvasResponse`: `canvas`, `expert_guidance`, `current_round`, `last_sync_round`, `analysis_timestamp`.
- **`/state`**: Returns full state dict for `user_id` + `conversation_id` (from checkpoint).

---

## 7. API Endpoints Summary

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness/readiness. |
| POST | `/chat` | Sync chat; returns `ChatResponse.reply`. |
| POST | `/chat/stream` | Streaming chat (SSE). |
| POST | `/canvas` | Get canvas + guidance (dual-agent only). |
| GET | `/state` | Get state for user/conversation. |
| POST | `/reset` | Reset conversation state for user/conversation. |
| POST | `/deep_agent/call_async` | Start async run; callback URL receives updates. |
| POST | `/simulated_user/chat` | Talk to simulated user agent (e.g. testing). |

---

## 8. Key Files and Directories

| Path | Purpose |
|------|---------|
| `app/__init__.py` | FastAPI app, routes, startup. |
| `app/startup.py` | Agent creation, dual vs single mode, `AppState` init. |
| `app/state.py` | `AppState` dataclass. |
| `app/endpoints/chat.py` | `/chat`, `/chat/stream`. |
| `app/endpoints/canvas.py` | `/canvas`. |
| `app/endpoints/deep_agent.py` | `/deep_agent/call_async`. |
| `app/callbacks.py` | `run_async_stream_with_callback`, expert-sync wiring for async. |
| `app/models.py` | Pydantic models for requests/responses. |
| `agent_factory/facilitator_agent.py` | Facilitator agent factory. |
| `agent_factory/expert_agent.py` | Expert agent factory. |
| `agent_factory/business_agent.py` | Single-agent (legacy) factory. |
| `expert_sync.py` | `should_trigger_expert`, `trigger_expert_analysis`, `trigger_and_update_expert`, `update_state_with_analysis`, partner search, proposal generation. |
| `expertise_loader.py` | Load expertise templates. |
| `expertise/*.md` | Expertise definitions and canvas templates. |
| `libs/deepagents/.../state/dual_agent_state.py` | `DualAgentState` schema. |
| `libs/deepagents/.../middleware/expert_guidance.py` | Round tracking, guidance injection. |

---

## 9. Configuration and Environment

Relevant env vars (see also `ENVIRONMENT_VARIABLES.md`, `deploy.env.example`):

- `BC_API_USE_DUAL_AGENT`: Enable dual-agent mode (default true).
- `DEFAULT_EXPERTISE_TYPE`: Default expertise (e.g. `business_cofounder`).
- `EXPERT_SYNC_INTERVAL`: Not yet unified; middleware uses 10, `expert_sync` uses `STATE_EXPERT_SYNC_INTERVAL` (1). Prefer single source (e.g. env) for future cleanup.
- `EXPERT_SYNC_USE_MOCK`: If true, expert sync returns mock analysis (for debugging).
- `PARTNER_SEARCH_MAX_RETRIES`: Retries for partner search when result set is empty.
- `BC_API_LOG_CHAT_IO`, `BC_API_LOG_STATE`: Extra logging.
- Model and API keys: e.g. `QWEN_*`, `DASHSCOPE_*`, etc.

---

## 10. Maintenance and Future Improvements

- **Sync interval**: Unify expert sync interval (middleware vs `expert_sync`) and make it configurable via one env var.
- **Expert sync timing**: Option to run expert sync asynchronously (e.g. after sending reply) to avoid blocking the request.
- **Context engineering**: Consider token budgets for analysis prompt, conversation truncation, and summarization when history grows.
- **Partner search**: Clarify retry vs refinement policy and error handling; optionally make partner search opt-in per expertise or request.
- **Canvas contract**: Document canvas schema per expertise type for frontend consumers.
- **Testing**: Broaden integration tests for expert sync, partner search, and callback flow under dual-agent mode.

When making changes, update this document and any referenced guides (`DUAL_AGENT_GUIDE.md`, etc.) so the architecture and dataflow stay accurate.
