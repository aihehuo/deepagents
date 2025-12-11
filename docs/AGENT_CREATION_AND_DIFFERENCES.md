# Agent Creation and Differences in DeepAgents CLI

## Overview

In DeepAgents CLI, **agents are separate instances with isolated memory, skills, and configurations**. Each agent has its own directory and persistent memory file.

---

## 1. How to Create an Agent

### Automatic Creation (No Explicit Command)

**Agents are created automatically when you first use them!** There's no explicit "create agent" command.

#### Method 1: Using `--agent` Flag

```bash
# Create and use an agent named "python-dev"
deepagents --agent python-dev

# Create and use an agent named "data-scientist"
deepagents --agent data-scientist
```

#### Method 2: Default Agent

```bash
# Uses default agent named "agent"
deepagents
# Equivalent to: deepagents --agent agent
```

#### What Happens on First Use

When you run `deepagents --agent my-agent` for the first time:

1. **Directory Creation**: `~/.deepagents/my-agent/` is created
2. **Memory File Creation**: `~/.deepagents/my-agent/agent.md` is created with default content
3. **Skills Directory**: `~/.deepagents/my-agent/skills/` is created (if skills are enabled)

**Code Location**: `libs/deepagents-cli/deepagents_cli/agent.py:368-374`

```368:374:libs/deepagents-cli/deepagents_cli/agent.py
    # Setup agent directory for persistent memory (if enabled)
    if enable_memory or enable_skills:
        agent_dir = settings.ensure_agent_dir(assistant_id)
        agent_md = agent_dir / "agent.md"
        if not agent_md.exists():
            source_content = get_default_coding_instructions()
            agent_md.write_text(source_content)
```

---

## 2. What Different Agents Mean

### Agent = Isolated Instance

Each agent is a **separate instance** with:

- **Separate Memory**: Each agent has its own `agent.md` file
- **Separate Skills**: Each agent has its own `skills/` directory
- **Separate Conversation History**: Each agent has its own thread_id/checkpointer
- **Separate Configuration**: Each agent can have different system prompts

### Directory Structure

```
~/.deepagents/
├── agent/              # Default agent
│   ├── agent.md       # Memory/personality
│   └── skills/        # Custom skills
│
├── python-dev/        # Python development agent
│   ├── agent.md       # Python-specific memory
│   └── skills/        # Python-specific skills
│
└── data-scientist/    # Data science agent
    ├── agent.md       # Data science memory
    └── skills/        # Data science skills
```

---

## 3. How Agents Differ From Each Other

### A. Persistent Memory (`agent.md`)

**Location**: `~/.deepagents/{agent_name}/agent.md`

Each agent has its own memory file that contains:
- **Personality and style**: How the agent communicates
- **Coding preferences**: Formatting, conventions, patterns
- **Learned patterns**: Things the agent has learned from feedback
- **Role definitions**: What the agent specializes in

**Example - Python Dev Agent** (`~/.deepagents/python-dev/agent.md`):
```markdown
# Python Development Agent

I specialize in Python development with a focus on:
- Type hints and modern Python (3.10+)
- FastAPI for APIs
- pytest for testing
- Clean architecture patterns

I prefer functional programming over OOP when possible.
```

**Example - Data Scientist Agent** (`~/.deepagents/data-scientist/agent.md`):
```markdown
# Data Science Agent

I specialize in data analysis and machine learning:
- pandas and numpy for data manipulation
- scikit-learn for ML models
- matplotlib/seaborn for visualization
- Jupyter notebooks for exploration

I always include data validation and error handling.
```

### B. Skills Directory

**Location**: `~/.deepagents/{agent_name}/skills/`

Each agent can have different custom skills:

```bash
# Python dev agent might have:
~/.deepagents/python-dev/skills/
├── api-design/
│   └── SKILL.md
└── testing-patterns/
    └── SKILL.md

# Data scientist agent might have:
~/.deepagents/data-scientist/skills/
├── data-cleaning/
│   └── SKILL.md
└── model-evaluation/
    └── SKILL.md
```

### C. Conversation Thread Isolation

Each agent maintains separate conversation threads:

```python
# Agent "python-dev" has its own thread
config = {
    "configurable": {"thread_id": "uuid-for-python-dev"},
    "metadata": {"assistant_id": "python-dev"}
}

# Agent "data-scientist" has a different thread
config = {
    "configurable": {"thread_id": "uuid-for-data-scientist"},
    "metadata": {"assistant_id": "data-scientist"}
}
```

**Code Location**: `libs/deepagents-cli/deepagents_cli/execution.py:210-213`

```210:213:libs/deepagents-cli/deepagents_cli/execution.py
    config = {
        "configurable": {"thread_id": session_state.thread_id},
        "metadata": {"assistant_id": assistant_id} if assistant_id else {},
    }
```

### D. System Prompt Customization

While the base system prompt is the same, each agent's `agent.md` is injected into the system prompt, making each agent unique.

**Code Location**: `libs/deepagents-cli/deepagents_cli/agent_memory.py`

The `AgentMemoryMiddleware` loads:
1. User agent.md: `~/.deepagents/{assistant_id}/agent.md`
2. Project agent.md: `[project-root]/.deepagents/agent.md` (if in a project)

Both are combined and injected into the system prompt.

---

## Listing Agents

### Command

```bash
deepagents list
```

### What It Shows

**Code Location**: `libs/deepagents-cli/deepagents_cli/agent.py:29-57`

```29:57:libs/deepagents-cli/deepagents_cli/agent.py
def list_agents() -> None:
    """List all available agents."""
    agents_dir = settings.user_deepagents_dir

    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]No agents found.[/yellow]")
        console.print(
            "[dim]Agents will be created in ~/.deepagents/ when you first use them.[/dim]",
            style=COLORS["dim"],
        )
        return

    console.print("\n[bold]Available Agents:[/bold]\n", style=COLORS["primary"])

    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "agent.md"

            if agent_md.exists():
                console.print(f"  • [bold]{agent_name}[/bold]", style=COLORS["primary"])
                console.print(f"    {agent_path}", style=COLORS["dim"])
            else:
                console.print(
                    f"  • [bold]{agent_name}[/bold] [dim](incomplete)[/dim]", style=COLORS["tool"]
                )
                console.print(f"    {agent_path}", style=COLORS["dim"])

    console.print()
```

**Output Example**:
```
Available Agents:

  • agent
    /Users/yc/.deepagents/agent

  • python-dev
    /Users/yc/.deepagents/python-dev

  • data-scientist
    /Users/yc/.deepagents/data-scientist
```

---

## Resetting an Agent

### Command

```bash
# Reset to default
deepagents reset --agent python-dev

# Copy from another agent
deepagents reset --agent python-dev --target data-scientist
```

**Code Location**: `libs/deepagents-cli/deepagents_cli/agent.py:60-91`

```60:91:libs/deepagents-cli/deepagents_cli/agent.py
def reset_agent(agent_name: str, source_agent: str | None = None) -> None:
    """Reset an agent to default or copy from another agent."""
    agents_dir = settings.user_deepagents_dir
    agent_dir = agents_dir / agent_name

    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "agent.md"

        if not source_md.exists():
            console.print(
                f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found "
                "or has no agent.md"
            )
            return

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(f"Removed existing agent directory: {agent_dir}", style=COLORS["tool"])

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "agent.md"
    agent_md.write_text(source_content)

    console.print(f"✓ Agent '{agent_name}' reset to {action_desc}", style=COLORS["primary"])
    console.print(f"Location: {agent_dir}\n", style=COLORS["dim"])
```

---

## Use Cases for Multiple Agents

### 1. **Specialization**

Different agents for different domains:

```bash
# Python backend development
deepagents --agent python-backend

# Frontend development
deepagents --agent frontend

# DevOps/infrastructure
deepagents --agent devops
```

### 2. **Different Personalities**

```bash
# Verbose, detailed agent
deepagents --agent detailed

# Concise, minimal agent
deepagents --agent concise
```

### 3. **Project-Specific Agents**

```bash
# Agent for specific project
deepagents --agent project-alpha

# Another agent for different project
deepagents --agent project-beta
```

### 4. **Learning and Experimentation**

```bash
# Experimental agent for trying new things
deepagents --agent experimental

# Stable, production agent
deepagents --agent stable
```

---

## How Agent Identity Works

### The `assistant_id` Parameter

The `assistant_id` parameter (from `--agent` flag) is used throughout the system:

1. **Directory Path**: `~/.deepagents/{assistant_id}/`
2. **Memory Loading**: Loads `agent.md` from agent directory
3. **Skills Loading**: Loads skills from `{assistant_id}/skills/`
4. **Thread Isolation**: Each `assistant_id` has separate conversation threads
5. **Metadata**: Passed in config metadata for tracking

**Code Location**: `libs/deepagents-cli/deepagents_cli/main.py:100-104`

```100:104:libs/deepagents-cli/deepagents_cli/main.py
    parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for separate memory stores (default: agent).",
    )
```

**Code Location**: `libs/deepagents-cli/deepagents_cli/agent.py:326-338`

```326:338:libs/deepagents-cli/deepagents_cli/agent.py
def create_cli_agent(
    model: str | BaseChatModel,
    assistant_id: str,
    *,
    tools: list[BaseTool] | None = None,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
    system_prompt: str | None = None,
    auto_approve: bool = False,
    enable_memory: bool = True,
    enable_skills: bool = True,
    enable_shell: bool = True,
) -> tuple[Pregel, CompositeBackend]:
    """Create a CLI-configured agent with flexible options.
```

---

## Summary

### 1. How to Create an Agent

- **Automatic**: Just use `--agent <name>` flag
- **First use**: Creates directory and `agent.md` file automatically
- **No explicit command**: No `deepagents create-agent` needed

### 2. What Different Agents Mean

- **Separate instances**: Each agent is isolated
- **Separate memory**: Each has its own `agent.md`
- **Separate skills**: Each has its own `skills/` directory
- **Separate conversations**: Each has its own thread_id

### 3. How Agents Differ

| Aspect | How They Differ |
|--------|----------------|
| **Memory** | Different `agent.md` files with different personalities/preferences |
| **Skills** | Different `skills/` directories with different custom tools |
| **Conversations** | Separate thread_ids, no shared conversation history |
| **System Prompt** | Same base, but `agent.md` content differs |
| **Configuration** | Can have different tools, models, etc. (if configured) |

### Key Takeaway

**Agents are like separate "personas" or "instances"** - each with its own memory, skills, and conversation history. They're perfect for:
- Specialization (Python dev vs Data scientist)
- Different personalities (verbose vs concise)
- Project isolation (different projects, different agents)
- Experimentation (try new things without affecting stable agent)

The `assistant_id` parameter is the key that ties everything together - it determines which directory, memory file, and skills to use.

