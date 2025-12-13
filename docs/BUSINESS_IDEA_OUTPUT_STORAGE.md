# Business Idea Development Output Storage

This document explains where the outcomes from each skill execution are stored and how to access and unify them.

## Where Outputs Are Stored

### 1. Filesystem Backend (Primary Storage)

When using `FilesystemBackend` (as in the test), files are written to the **filesystem root directory**:

```python
filesystem_backend = FilesystemBackend(root_dir=str(tmp_path))
```

**Location**: All files written via `write_file` tool are stored in `filesystem_backend.cwd`

**In the test**: This is `tmp_path` (a pytest temporary directory)

**Path Resolution**:
- If agent writes to `/output.md` → stored at `{tmp_path}/output.md`
- If agent writes to `/results/persona.md` → stored at `{tmp_path}/results/persona.md`
- Relative paths are resolved relative to `root_dir`

### 2. Agent State (Ephemeral Storage)

If using `StateBackend` (default for `FilesystemMiddleware`), files are stored in agent state:

```python
result.get("files", {})  # Dictionary of file paths to FileData
```

**Location**: In-memory, only accessible via `result.get("files")`

**Persistence**: Only persists within the same conversation thread (if checkpointer is used)

### 3. Large Tool Results

Large tool results (>20k tokens by default) are automatically evicted to filesystem:

**Location**: `/large_tool_results/{tool_call_id}`

**Example**: If a skill execution returns a very large output, it's saved to:
```
{filesystem_backend.cwd}/large_tool_results/{sanitized_tool_call_id}
```

## How to Access Outputs

### In the Test

The test automatically:
1. **Scans the filesystem directory** for all files
2. **Checks agent state** for files stored there
3. **Creates a unified output document** at `{filesystem_dir}/business_idea_development_output.md`

### After Test Execution

The test prints the filesystem root directory:
```
📂 Filesystem root directory: /tmp/pytest-of-user/pytest-123/test_business_idea_development_automatic_progression0
```

**Note**: pytest temporary directories are cleaned up after tests. To preserve files:

1. **Use a fixed directory** instead of `tmp_path`:
   ```python
   output_dir = Path("/tmp/business_idea_outputs")
   output_dir.mkdir(exist_ok=True)
   filesystem_backend = FilesystemBackend(root_dir=str(output_dir))
   ```

2. **Copy files before test ends**:
   ```python
   import shutil
   preserved_dir = Path("/tmp/preserved_outputs")
   shutil.copytree(filesystem_dir, preserved_dir)
   ```

## Unified Output Document

The test automatically creates a unified markdown document containing:

1. **Summary**: Todos, milestones, execution time
2. **Milestones**: Completion status of each milestone
3. **Todos**: All todos with their status
4. **Files Generated**: All files written during execution with their content
5. **Milestone Tool Calls**: Which milestone marking tools were called

**Location**: `{filesystem_dir}/business_idea_development_output.md`

## Recommended Output Organization

For better organization, instruct the agent to save outputs in a structured way:

### Option 1: Single Unified Document

Instruct the agent to write all outputs to a single file:

```python
system_prompt = """... After completing each skill, append the output to /business_idea_development.md
Use the edit_file tool to append content to this file."""
```

### Option 2: Organized Directory Structure

Instruct the agent to organize outputs by step:

```
/business_idea_development/
  ├── 01_business_idea_evaluation.md
  ├── 02_persona_clarification.md
  ├── 03_painpoint_enhancement.md
  ├── 04_60s_pitch_creation.md
  ├── 05_pricing_optimization.md
  ├── 06_business_model_pivots.md
  └── summary.md
```

### Option 3: Use Persistent Storage

Use `StoreBackend` or `CompositeBackend` for persistent storage:

```python
from deepagents.backends import CompositeBackend, StoreBackend
from langgraph.checkpoint.postgres import PostgresSaver

# Persistent storage for business idea outputs
store = PostgresSaver.from_conn_string("postgresql://...")
backend = CompositeBackend(
    default=FilesystemBackend(root_dir=str(tmp_path)),  # Temporary files
    routes={
        "/business_idea/": StoreBackend(store=store),  # Persistent outputs
    }
)
```

Then instruct the agent to save outputs to `/business_idea/` prefix for persistence.

## Current Test Implementation

The test (`test_business_idea_development_middleware.py`) automatically:

1. ✅ Scans filesystem for all files
2. ✅ Lists files in agent state
3. ✅ Extracts skill outputs from messages
4. ✅ Creates unified output document
5. ✅ Prints filesystem root directory location

**To access outputs after test**:
- Check the test output for the filesystem root directory path
- Read `business_idea_development_output.md` in that directory
- Or modify the test to use a fixed directory instead of `tmp_path`

