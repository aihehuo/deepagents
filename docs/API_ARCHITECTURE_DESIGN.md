# DeepAgents API Architecture Design

## Overview

This document outlines the architectural design for transforming DeepAgents CLI into an API interface, maximizing code reuse from the existing CLI implementation.

---

## 1. Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (New)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ REST API     │  │ WebSocket    │  │ Session Mgmt │      │
│  │ (FastAPI)    │  │ (Streaming)  │  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                 │              │
│         └──────────────────┼─────────────────┘              │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Service Layer (Reuse + Adapt)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Agent Service│  │ Task Service │  │ HITL Service │      │
│  │ (Reuse)      │  │ (Adapt)      │  │ (Adapt)      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                 │              │
└─────────┼──────────────────┼─────────────────┼──────────────┘
          │                  │                 │
          ▼                  ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│           Core CLI Components (Reuse Directly)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │create_cli_   │  │execute_task()│  │parse_file_   │      │
│  │agent()       │  │              │  │mentions()    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │AgentMemory   │  │TokenTracker  │  │FileOpTracker │      │
│  │Middleware    │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Reuse Strategy

### 2.1 Direct Reuse (No Changes)

These components can be used directly:

| Component | Location | Usage |
|-----------|----------|-------|
| `create_cli_agent()` | `agent.py:326` | Agent creation |
| `parse_file_mentions()` | `input.py` | File mention parsing |
| `TokenTracker` | `ui.py` | Token usage tracking |
| `FileOpTracker` | `file_ops.py` | File operation tracking |
| `AgentMemoryMiddleware` | `agent_memory.py` | Memory management |
| `SkillsMiddleware` | `skills/middleware.py` | Skills management |
| `ShellMiddleware` | `shell.py` | Shell execution |
| `get_system_prompt()` | `agent.py:94` | System prompt generation |
| `list_agents()` | `agent.py:29` | Agent listing |
| `reset_agent()` | `agent.py:60` | Agent reset |

### 2.2 Adaptation Layer (Wrapper Functions)

These need wrapper functions to adapt CLI-specific behavior:

| Component | Adaptation Needed |
|-----------|-------------------|
| `execute_task()` | Remove console output, return structured data |
| `prompt_for_tool_approval()` | Replace console prompt with API response |
| `SessionState` | Convert to API session model |
| Streaming logic | Replace console streaming with WebSocket/SSE |

### 2.3 New Components

| Component | Purpose |
|-----------|---------|
| API Router | FastAPI routes for REST endpoints |
| WebSocket Handler | Real-time streaming for agent responses |
| Session Manager | API session lifecycle management |
| HITL Handler | Human-in-the-loop approval via API |
| Event Emitter | Convert agent events to API events |

---

## 3. API Design

### 3.1 REST Endpoints

#### Agent Management

```
GET    /api/v1/agents                    # List all agents
GET    /api/v1/agents/{agent_id}         # Get agent info
POST   /api/v1/agents/{agent_id}/reset   # Reset agent
DELETE /api/v1/agents/{agent_id}         # Delete agent (optional)
```

#### Session Management

```
POST   /api/v1/sessions                  # Create session
GET    /api/v1/sessions/{session_id}     # Get session info
DELETE /api/v1/sessions/{session_id}     # Delete session
```

#### Task Execution

```
POST   /api/v1/sessions/{session_id}/messages    # Send message (non-streaming)
POST   /api/v1/sessions/{session_id}/tasks        # Execute task (with options)
```

#### Human-in-the-Loop

```
GET    /api/v1/sessions/{session_id}/interrupts           # List pending interrupts
POST   /api/v1/sessions/{session_id}/interrupts/{id}/respond  # Respond to interrupt
```

#### File Operations (if needed)

```
POST   /api/v1/sessions/{session_id}/files/upload        # Upload files
GET    /api/v1/sessions/{session_id}/files/{path}        # Get file
```

### 3.2 WebSocket Endpoints

```
WS     /api/v1/sessions/{session_id}/stream              # Real-time streaming
```

**WebSocket Message Format**:
```json
{
  "type": "message" | "tool_call" | "tool_result" | "interrupt" | "todo" | "error",
  "data": { ... },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 3.3 Server-Sent Events (Alternative)

```
GET    /api/v1/sessions/{session_id}/events             # SSE stream
```

---

## 4. Service Layer Design

### 4.1 Agent Service

**Purpose**: Manage agent lifecycle and configuration

**Reuses**: `create_cli_agent()`, `list_agents()`, `reset_agent()`

```python
class AgentService:
    def create_agent(
        self,
        agent_id: str,
        model: str | None = None,
        sandbox_type: str | None = None,
        auto_approve: bool = False,
        **kwargs
    ) -> tuple[Pregel, CompositeBackend]:
        """Create agent - directly calls create_cli_agent()"""
        return create_cli_agent(...)
    
    def list_agents(self) -> list[AgentInfo]:
        """List agents - wraps list_agents()"""
        ...
    
    def get_agent_info(self, agent_id: str) -> AgentInfo:
        """Get agent information"""
        ...
```

### 4.2 Task Service

**Purpose**: Execute tasks and handle streaming

**Reuses**: `execute_task()` core logic, `parse_file_mentions()`

**Adapts**: Console output → Structured events

```python
class TaskService:
    async def execute_task(
        self,
        session_id: str,
        user_input: str,
        agent: Pregel,
        config: dict,
        backend: CompositeBackend,
        stream_handler: Callable[[Event], None] | None = None
    ) -> TaskResult:
        """
        Execute task - adapted from execute_task()
        
        Changes:
        - Remove console.print() calls
        - Emit structured events via stream_handler
        - Return TaskResult instead of printing
        """
        # Reuse: parse_file_mentions()
        prompt_text, mentioned_files = parse_file_mentions(user_input)
        
        # Reuse: Core streaming logic from execute_task()
        async for chunk in agent.astream(...):
            # Convert to Event objects
            event = self._chunk_to_event(chunk)
            if stream_handler:
                stream_handler(event)
        
        return TaskResult(...)
    
    def _chunk_to_event(self, chunk) -> Event:
        """Convert agent stream chunk to Event"""
        ...
```

### 4.3 Session Service

**Purpose**: Manage API sessions

**Reuses**: `SessionState` concept, thread_id management

**Adapts**: CLI session → API session

```python
class SessionService:
    def create_session(
        self,
        agent_id: str,
        auto_approve: bool = False,
        **kwargs
    ) -> Session:
        """Create new session"""
        session = Session(
            session_id=uuid.uuid4(),
            agent_id=agent_id,
            thread_id=uuid.uuid4(),  # Reuse SessionState.thread_id concept
            auto_approve=auto_approve,
            created_at=datetime.now(),
        )
        # Create agent for this session
        agent, backend = agent_service.create_agent(agent_id, ...)
        session.agent = agent
        session.backend = backend
        return session
    
    def get_session(self, session_id: str) -> Session:
        """Get session"""
        ...
    
    def delete_session(self, session_id: str) -> None:
        """Clean up session"""
        ...
```

### 4.4 HITL Service

**Purpose**: Handle human-in-the-loop approvals

**Reuses**: `prompt_for_tool_approval()` logic, `HITLRequest`/`HITLResponse` types

**Adapts**: Console prompt → API endpoint

```python
class HITLService:
    def get_pending_interrupts(
        self,
        session_id: str
    ) -> list[InterruptInfo]:
        """Get pending interrupts for session"""
        ...
    
    def respond_to_interrupt(
        self,
        session_id: str,
        interrupt_id: str,
        decision: Decision
    ) -> None:
        """
        Respond to interrupt - adapted from prompt_for_tool_approval()
        
        Changes:
        - Instead of console prompt, store decision
        - Resume agent execution with decision
        """
        # Reuse: HITLRequest/HITLResponse types
        # Reuse: Decision validation logic
        ...
```

---

## 5. Data Models

### 5.1 Request/Response Models

```python
# Request Models
class CreateSessionRequest(BaseModel):
    agent_id: str
    auto_approve: bool = False
    sandbox_type: str | None = None
    model: str | None = None

class SendMessageRequest(BaseModel):
    content: str
    file_mentions: list[str] | None = None  # Optional explicit file paths

class InterruptResponseRequest(BaseModel):
    decision: Literal["approve", "reject", "edit"]
    edited_args: dict | None = None  # If decision is "edit"

# Response Models
class SessionResponse(BaseModel):
    session_id: str
    agent_id: str
    thread_id: str
    created_at: datetime
    status: Literal["active", "paused", "completed"]

class TaskResult(BaseModel):
    session_id: str
    messages: list[Message]
    tool_calls: list[ToolCall]
    todos: list[Todo] | None
    token_usage: TokenUsage
    completed_at: datetime

class InterruptInfo(BaseModel):
    interrupt_id: str
    tool_name: str
    description: str
    args: dict
    created_at: datetime
```

### 5.2 Event Models (for WebSocket/SSE)

```python
class Event(BaseModel):
    type: Literal["message", "tool_call", "tool_result", "interrupt", "todo", "error", "done"]
    data: dict
    timestamp: datetime

class MessageEvent(Event):
    type: Literal["message"] = "message"
    data: Message

class ToolCallEvent(Event):
    type: Literal["tool_call"] = "tool_call"
    data: ToolCallInfo

class InterruptEvent(Event):
    type: Literal["interrupt"] = "interrupt"
    data: InterruptInfo
```

---

## 6. Design Challenges and Solutions

### Challenge 1: Streaming Output

**Problem**: CLI uses `console.print()` for real-time output. API needs structured streaming.

**Solution**:
- **Extract streaming logic**: Create `stream_task()` function that yields events instead of printing
- **Event-based architecture**: Convert all console output to structured events
- **Multiple transport options**: Support WebSocket (real-time) and SSE (HTTP-based)

```python
# Reuse core logic, adapt output
async def stream_task(...) -> AsyncIterator[Event]:
    # Reuse: execute_task() streaming logic
    async for chunk in agent.astream(...):
        # Convert chunk to Event
        event = _chunk_to_event(chunk)
        yield event  # Instead of console.print()
```

**Reuse**: 90% of `execute_task()` streaming logic
**New**: Event conversion layer, WebSocket/SSE handlers

---

### Challenge 2: Human-in-the-Loop (HITL)

**Problem**: CLI uses interactive console prompts. API needs async approval mechanism.

**Solution**:
- **Interrupt queue**: Store interrupts in session state
- **Polling endpoint**: `GET /sessions/{id}/interrupts` to check for pending interrupts
- **Response endpoint**: `POST /sessions/{id}/interrupts/{id}/respond` to provide decision
- **Resume execution**: After response, resume agent with decision

```python
# Reuse: HITLRequest/HITLResponse types, validation logic
# Adapt: Store interrupt instead of prompting
class HITLService:
    def store_interrupt(self, session_id: str, interrupt: HITLRequest):
        # Store in session state
        ...
    
    def respond_and_resume(self, session_id: str, interrupt_id: str, decision: Decision):
        # Reuse: Decision validation from prompt_for_tool_approval()
        # Resume agent execution
        ...
```

**Reuse**: `HITLRequest`, `HITLResponse`, `Decision` types, validation logic
**New**: Interrupt storage, resume mechanism, API endpoints

---

### Challenge 3: Session Management

**Problem**: CLI uses single long-lived session. API needs multi-session support.

**Solution**:
- **Session registry**: In-memory or database-backed session storage
- **Session lifecycle**: Create, retrieve, delete sessions
- **Agent caching**: Cache agent instances per session (or per agent_id)

```python
# Reuse: SessionState concept (thread_id, auto_approve)
# New: Session registry, lifecycle management
class SessionManager:
    _sessions: dict[str, Session] = {}
    
    def create_session(self, agent_id: str, ...) -> Session:
        # Reuse: create_cli_agent()
        agent, backend = create_cli_agent(...)
        session = Session(agent=agent, backend=backend, ...)
        self._sessions[session.id] = session
        return session
```

**Reuse**: `SessionState` concept, `create_cli_agent()`
**New**: Session registry, lifecycle management

---

### Challenge 4: File Operations

**Problem**: CLI uses local filesystem. API may need file upload/download.

**Solution**:
- **File upload endpoint**: Accept files via multipart/form-data
- **Temporary storage**: Store uploaded files in session-scoped directory
- **File mention resolution**: Adapt `parse_file_mentions()` to resolve uploaded files

```python
# Reuse: parse_file_mentions() logic
# Adapt: Resolve file paths from uploads
class FileService:
    def upload_file(self, session_id: str, file: UploadFile) -> str:
        # Store in session directory
        path = f"/tmp/sessions/{session_id}/uploads/{file.filename}"
        ...
        return path
    
    def resolve_file_mentions(self, session_id: str, content: str) -> tuple[str, list[Path]]:
        # Reuse: parse_file_mentions() logic
        # Adapt: Resolve @filename to uploaded file path
        ...
```

**Reuse**: `parse_file_mentions()` core logic
**New**: File upload handling, temporary storage

---

### Challenge 5: Error Handling

**Problem**: CLI prints errors to console. API needs structured error responses.

**Solution**:
- **Exception mapping**: Map exceptions to HTTP status codes
- **Error events**: Include errors in event stream
- **Graceful degradation**: Handle cancellations, timeouts gracefully

```python
# Reuse: Error handling logic from execute_task()
# Adapt: Return error events instead of printing
async def stream_task(...):
    try:
        async for chunk in agent.astream(...):
            yield _chunk_to_event(chunk)
    except asyncio.CancelledError:
        yield ErrorEvent(type="cancelled", message="Task cancelled")
    except Exception as e:
        yield ErrorEvent(type="error", message=str(e))
```

**Reuse**: Error handling patterns from CLI
**New**: Error event types, HTTP status mapping

---

### Challenge 6: Token Tracking

**Problem**: CLI tracks tokens for display. API needs token usage in responses.

**Solution**:
- **Reuse TokenTracker**: Use existing `TokenTracker` class
- **Include in response**: Add token usage to `TaskResult`
- **Per-session tracking**: Track tokens per session

```python
# Reuse: TokenTracker class directly
class TaskService:
    def __init__(self):
        self.token_trackers: dict[str, TokenTracker] = {}
    
    async def execute_task(self, session_id: str, ...):
        tracker = self.token_trackers.get(session_id) or TokenTracker()
        # Reuse: tracker.add(input_tokens, output_tokens)
        ...
        return TaskResult(token_usage=tracker.get_usage())
```

**Reuse**: `TokenTracker` class 100%
**New**: Per-session tracking, response inclusion

---

### Challenge 7: Sandbox Management

**Problem**: CLI manages sandbox lifecycle. API needs sandbox per session or shared.

**Solution**:
- **Reuse sandbox creation**: Use existing sandbox factory
- **Session-scoped sandboxes**: Create sandbox per session (or reuse)
- **Cleanup on session end**: Ensure sandbox cleanup

```python
# Reuse: create_sandbox() from integrations/sandbox_factory.py
class SessionService:
    def create_session(self, agent_id: str, sandbox_type: str, ...):
        if sandbox_type != "none":
            # Reuse: create_sandbox()
            with create_sandbox(sandbox_type, ...) as sandbox:
                agent, backend = create_cli_agent(..., sandbox=sandbox)
                # Store sandbox in session for cleanup
                ...
```

**Reuse**: `create_sandbox()` function
**New**: Session-scoped sandbox management, cleanup hooks

---

### Challenge 8: Configuration Management

**Problem**: CLI uses environment variables and settings. API needs config per request/session.

**Solution**:
- **Reuse Settings class**: Use existing `Settings` for defaults
- **Override per session**: Allow config overrides per session
- **API config model**: Request-level configuration

```python
# Reuse: Settings class from config.py
class SessionService:
    def create_session(self, agent_id: str, config: SessionConfig):
        # Reuse: settings for defaults
        model = config.model or settings.get_default_model()
        # Override with request config
        ...
```

**Reuse**: `Settings` class, default model creation
**New**: Per-session config overrides

---

## 7. Implementation Strategy

### Phase 1: Core API (Minimal Viable)

1. **REST endpoints**:
   - `POST /sessions` - Create session
   - `POST /sessions/{id}/messages` - Send message (non-streaming)
   - `GET /sessions/{id}` - Get session status

2. **Reuse**:
   - `create_cli_agent()` directly
   - `execute_task()` core logic (adapted)
   - `parse_file_mentions()` directly

3. **New**:
   - FastAPI router
   - Session manager
   - Response models

### Phase 2: Streaming

1. **WebSocket support**:
   - `WS /sessions/{id}/stream`
   - Event conversion layer
   - Real-time streaming

2. **Reuse**:
   - `execute_task()` streaming logic (adapted)

3. **New**:
   - WebSocket handler
   - Event models

### Phase 3: HITL

1. **Interrupt handling**:
   - `GET /sessions/{id}/interrupts`
   - `POST /sessions/{id}/interrupts/{id}/respond`

2. **Reuse**:
   - `HITLRequest`/`HITLResponse` types
   - Decision validation

3. **New**:
   - Interrupt storage
   - Resume mechanism

### Phase 4: Advanced Features

1. **File uploads**
2. **Agent management endpoints**
3. **Token usage tracking**
4. **Sandbox management**

---

## 8. Code Organization

```
libs/deepagents-api/
├── __init__.py
├── main.py                    # FastAPI app entry point
├── config.py                  # API-specific config
│
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── agents.py         # Agent management routes
│   │   ├── sessions.py       # Session management routes
│   │   ├── tasks.py          # Task execution routes
│   │   └── interrupts.py    # HITL routes
│   │
│   ├── models/
│   │   ├── requests.py       # Request models
│   │   ├── responses.py      # Response models
│   │   └── events.py         # Event models
│   │
│   └── websocket.py          # WebSocket handler
│
├── services/
│   ├── __init__.py
│   ├── agent_service.py      # Wraps create_cli_agent()
│   ├── task_service.py       # Adapted execute_task()
│   ├── session_service.py    # Session management
│   └── hitl_service.py       # HITL handling
│
└── adapters/
    ├── __init__.py
    ├── streaming.py          # Stream chunk → Event conversion
    └── file_upload.py        # File upload handling
```

---

## 9. Key Design Principles

### 9.1 Maximize Reuse

- **Direct reuse**: Use CLI functions directly where possible
- **Minimal wrappers**: Only wrap when necessary for API adaptation
- **Shared utilities**: Keep shared logic in common modules

### 9.2 Separation of Concerns

- **API layer**: HTTP/WebSocket handling only
- **Service layer**: Business logic, reuses CLI components
- **Core layer**: CLI components (unchanged)

### 9.3 Backward Compatibility

- **CLI unchanged**: Don't modify CLI code for API
- **Shared utilities**: Extract common logic if needed
- **Independent deployment**: API can be deployed separately

### 9.4 Extensibility

- **Plugin architecture**: Allow custom services
- **Event system**: Extensible event types
- **Middleware support**: API-level middleware

---

## 10. Summary

### Reuse Statistics

| Component | Reuse Level | Notes |
|-----------|-------------|-------|
| `create_cli_agent()` | 100% | Direct reuse |
| `parse_file_mentions()` | 100% | Direct reuse |
| `TokenTracker` | 100% | Direct reuse |
| `execute_task()` logic | ~80% | Core logic reused, output adapted |
| `prompt_for_tool_approval()` | ~60% | Types reused, prompt logic adapted |
| `SessionState` | ~70% | Concept reused, implementation adapted |

### New Components

- API router (FastAPI)
- WebSocket handler
- Session manager
- Event conversion layer
- HITL storage/resume
- Request/response models

### Estimated Effort

- **Phase 1 (Core API)**: ~2-3 days
- **Phase 2 (Streaming)**: ~2-3 days
- **Phase 3 (HITL)**: ~2-3 days
- **Phase 4 (Advanced)**: ~3-5 days

**Total**: ~9-14 days for full implementation

---

## Next Steps

1. Review and approve architecture
2. Set up project structure (`libs/deepagents-api/`)
3. Implement Phase 1 (Core API)
4. Test with CLI components
5. Iterate on design based on feedback

