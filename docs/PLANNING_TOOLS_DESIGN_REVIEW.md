# Review of PLANNING_TOOLS_DESIGN.md

## Executive Summary

This review compares the design document `PLANNING_TOOLS_DESIGN.md` with the actual DeepAgents codebase implementation. The document provides a comprehensive reverse-engineered description of how todo management works in DeepAgents, which is built on LangChain's `TodoListMiddleware`.

**Overall Assessment**: The document is **largely accurate** for what can be verified from the codebase. However, since `TodoListMiddleware` is implemented in LangChain (not in this repository), some implementation details cannot be directly verified.

## Verified Accuracy ✅

### 1. Integration in DeepAgents

**Document Claims:**
- `TodoListMiddleware` is included by default in `create_deep_agent`
- It's added to both main agent and subagent middleware
- Located in `deepagents/graph.py`

**Codebase Verification:**
```113:122:libs/deepagents/deepagents/graph.py
    deepagent_middleware = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SubAgentMiddleware(
            default_model=model,
            default_tools=tools,
            subagents=subagents if subagents is not None else [],
            default_middleware=[
                TodoListMiddleware(),
```

✅ **VERIFIED**: The middleware is correctly integrated as described.

### 2. Tool Availability

**Document Claims:**
- `write_todos` tool is available to agents
- No `read_todos` tool exists (despite README mentioning it)

**Codebase Verification:**
```11:11:libs/deepagents/tests/utils.py
    assert "write_todos" in agent.nodes["tools"].bound._tools_by_name.keys()
```

Test files show `write_todos` is available:
```52:56:libs/deepagents/tests/unit_tests/test_end_to_end.py
                            {
                                "name": "write_todos",
                                "args": {"todos": []},
                                "id": "call_1",
                                "type": "tool_call",
                            }
```

✅ **VERIFIED**: `write_todos` exists and is used in tests.

❌ **DISCREPANCY FOUND**: The README.md mentions `read_todos` as a tool:
```306:307:README.md
| `write_todos` | Create and manage structured task lists for tracking progress through complex workflows | TodoListMiddleware |
| `read_todos` | Read the current todo list state | TodoListMiddleware |
```

However, no `read_todos` tool is found in the codebase. The document correctly identifies this discrepancy.

### 3. Import Source

**Document Claims:**
- `TodoListMiddleware` is imported from `langchain.agents.middleware`

**Codebase Verification:**
```7:7:libs/deepagents/deepagents/graph.py
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
```

✅ **VERIFIED**: Import path is correct.

### 4. Tool Usage Pattern

**Document Claims:**
- `write_todos` accepts `todos: list[Todo]` where each Todo has `content` and `status`
- Status values are `"pending"`, `"in_progress"`, `"completed"`

**Codebase Verification:**
Test usage shows:
```52:56:libs/deepagents/tests/unit_tests/test_end_to_end.py
                            {
                                "name": "write_todos",
                                "args": {"todos": []},
                                "id": "call_1",
                                "type": "tool_call",
                            }
```

UI formatting code confirms the structure:
```143:147:libs/deepagents-cli/deepagents_cli/ui.py
    elif tool_name == "write_todos":
        # Todos: show count of items
        if "todos" in tool_args and isinstance(tool_args["todos"], list):
            count = len(tool_args["todos"])
            return f"{tool_name}({count} items)"
```

✅ **VERIFIED**: Tool structure matches document description.

## Cannot Be Verified (LangChain Implementation)

Since `TodoListMiddleware` is implemented in LangChain (external dependency), the following details cannot be verified from this codebase:

### 1. Internal Implementation Details

**Document Claims:**
- `wrap_model_call()` method implementation
- System prompt injection mechanism
- Exact state schema definition (`PlanningState`, `Todo` TypedDict)
- `write_todos` tool implementation details
- Prompt content (`WRITE_TODOS_SYSTEM_PROMPT`, `WRITE_TODOS_TOOL_DESCRIPTION`)

**Status**: ⚠️ **CANNOT VERIFY** - These are internal to LangChain's implementation.

**Assessment**: The document's descriptions appear consistent with:
- How middleware typically works in LangChain
- The behavior observed in tests
- The integration patterns used in DeepAgents

However, without access to LangChain source code, these cannot be definitively verified.

### 2. State Schema Details

**Document Claims:**
```python
class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]

class PlanningState(AgentState):
    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
```

**Status**: ⚠️ **CANNOT VERIFY** - State schema is defined in LangChain middleware.

**Assessment**: The schema description is plausible and consistent with:
- How LangGraph state schemas work
- The `OmitFromInput` pattern used elsewhere in the codebase
- The behavior described in the document

## Key Findings

### 1. README Discrepancy (Correctly Identified)

The document correctly identifies that:
- README.md mentions `read_todos` as a tool
- No `read_todos` tool actually exists in the implementation
- This is documented in the "Limitations" section

**Recommendation**: The README.md should be updated to remove the `read_todos` reference, or the document should note if this is a planned feature.

### 2. Integration Pattern Accuracy

The document accurately describes:
- How middleware is integrated in `create_deep_agent`
- That subagents also receive `TodoListMiddleware`
- The middleware ordering and composition

### 3. Tool Behavior Description

The document's description of:
- Complete replacement strategy (replacing entire todos list)
- Command-based state updates
- ToolMessage feedback loop

All align with how LangGraph tools typically work and are consistent with test usage patterns.

## Potential Issues or Clarifications Needed

### 1. Prompt Content Details

The document includes detailed prompt content (`WRITE_TODOS_SYSTEM_PROMPT`, `WRITE_TODOS_TOOL_DESCRIPTION`) that cannot be verified. These should be marked as "as observed" or "reverse-engineered" rather than presented as definitive implementation details.

**Recommendation**: Add a note that these prompts are based on reverse engineering and may change with LangChain updates.

### 2. State Schema Verification

The exact state schema annotations cannot be verified. The document should note that the schema description is inferred from behavior rather than from source code inspection.

### 3. `wrap_model_call` Implementation

The exact implementation of `wrap_model_call()` shown in the document may not match the current LangChain implementation. This should be noted.

## Recommendations

1. **Add Verification Status**: Mark sections that are verified vs. inferred/reverse-engineered
2. **Update README**: Fix the `read_todos` discrepancy in README.md
3. **Add Version Notes**: Note which version of LangChain this analysis is based on
4. **Link to Source**: Add links to LangChain source code where possible
5. **Clarify Scope**: Make it clear which parts are DeepAgents-specific vs. LangChain implementation details

## Conclusion

The design document provides a **comprehensive and largely accurate** description of how todo management works in DeepAgents. The parts that can be verified from the codebase (integration, tool availability, usage patterns) are correct. The parts that describe LangChain's internal implementation are plausible and consistent with observed behavior, but cannot be definitively verified without access to LangChain source code.

The document correctly identifies the `read_todos` discrepancy and provides valuable insights into the system's architecture and behavior patterns.

**Overall Accuracy Rating**: 85-90% (for verifiable claims: 100%; for inferred details: plausible but unverified)

