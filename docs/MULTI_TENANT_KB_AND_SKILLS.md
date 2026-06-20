# Multi-Tenant Dynamic KB and Skills Loading Architecture

This document describes how the `wu_tanchang_api` implements dynamic, isolated loading of tenant-level Knowledge Bases (KB) and local skills under a multi-tenant workspace architecture. This design ensures strict data isolation while optimizing startup and runtime execution performance.

---

## 1. Context and Problem Statement

In a multi-tenant AI application, each tenant (or user/team) has their own workspace directory (e.g., `workspace_aihehuo`, `workspace_custom`, `workspace_1`). These workspaces contain:
1. **Tenant-specific KB**: SQLite database (`kb.db`) and ChromaDB vector search files.
2. **Tenant-specific Local Skills**: Custom API integrations or specialized scripts that only apply to that specific tenant (e.g., `skills/local/get-ai-blog`).

When a single FastAPI server hosts the agent runner, it must:
* **Ensure strict data isolation**: Tenant A must never query Tenant B's database, trigger Tenant B's local skills, or leak default fallback data.
* **Optimize startup times**: Symlinking large vector databases prevents startup latency, while skills are copied into tenant runtime directories so their path templates can be rewritten safely.
* **Prevent repository pollution**: Read-only symlinks inside the sandbox must be protected from subagent write tools.
* **Address stateless tools**: LangGraph tools are typically functions without direct reference to the active agent or session. They must dynamically locate the active tenant's databases at runtime and **fail closed** if routing is ambiguous.

---

## 2. Architecture Overview

Below is the execution flow from the source workspace to the runtime sandbox and dynamic tool execution:

```mermaid
graph TD
    subgraph Source Workspace Directory
        src_ws_aihehuo["workspace_aihehuo/ (Source)"]
        src_kb["kb/ (SQLite + Chroma)"]
        src_default_skills["skills/default/ (App-wide templates)"]
        src_local_skills["skills/local/ (Tenant templates)"]
        src_ws_aihehuo --> src_kb
        src_ws_aihehuo --> src_local_skills
    end

    subgraph Runtime Sandbox (Isolated)
        rt_dir["runtime_dir/ (Session-specific)"]
        rt_ws_aihehuo["workspace_aihehuo/ (Sandbox)"]
        rt_identity["IDENTITY.md (Copied for writes)"]
        rt_kb_link["kb/ (Symlink)"]
        rt_default_skills["skills/default/ (Copied + Rewritten)"]
        rt_local_skills["skills/local/ (Copied + Rewritten)"]

        rt_dir --> rt_ws_aihehuo
        rt_ws_aihehuo --> rt_identity
        rt_ws_aihehuo -->|Fast Deploy| rt_kb_link
        rt_ws_aihehuo -->|Isolated Copy| rt_default_skills
        rt_ws_aihehuo -->|Isolated Copy| rt_local_skills
    end

    src_kb -.->|Symlinked| rt_kb_link
    src_default_skills -.->|Copied + Path-Rewritten| rt_default_skills
    src_local_skills -.->|Copied + Path-Rewritten| rt_local_skills

    subgraph Dynamic Execution Routing
        config["RunnableConfig (thread_id)"] -->|Lookup| active_agent["Active Agent Object"]
        active_agent -->|Resolves| ws_name["workspace_name = 'workspace_aihehuo'"]
        ws_name -->|Locates KB| db_path["runtime_dir / workspace_aihehuo / kb / kb.db"]
        ws_name -->|Appends Default Skills| default_skill_path["/workspace_aihehuo/skills/default/kb_analyst/"]
        ws_name -->|Appends Skills| skill_path["/workspace_aihehuo/skills/local/"]
    end
```

---

## 3. Sandboxing and Fast Deployment (`utils.py`)

To achieve write isolation without copying heavy files, `ensure_runtime_workspace` separates writable configuration from read-only databases/skills:

> [!NOTE]
> * **Copied**: Files like `IDENTITY.md`, `MEMORY.md`, and `SOUL.md` are copied (`symlink=False`) so the agent can write memory changes or personalize them without polluting the source template.
> * **Symlinked**: The `kb/` directory is symlinked (`symlink=True`). This prevents copying gigabytes of vector data, making session initialization near-instant.
> * **Copied + Rewritten**: App-wide default skills are copied from `skills/default/` to `/{workspace_name}/skills/default/`, and tenant local skills are copied from `workspace_x/skills/local/` to `/{workspace_name}/skills/local/`. After copying, all `SKILL.md` files are recursively scanned and `kb/` path references are rewritten to `/{workspace_name}/kb/` so the agent reads from the correct tenant directory. This prevents cross-tenant path leakage in skill instructions while avoiding incorrect nested paths such as `default/default` or `local/local`.

---

## 4. Strict Connection Routing & Fail-Closed Behavior (`kb_search.py`)

Tools invoked by LangGraph agents are stateless. To dynamically route queries to the correct database:

1. The tool function takes an implicit `config: RunnableConfig` argument.
2. The `thread_id` inside `config` matches the session.
3. We query the global active agent map to find the agent matching that thread.
4. We extract the agent's `workspace_name` and the runtime root directory.

### Fail-Closed Execution Check
To prevent any cross-tenant data leak or silent fallback to the default workspace:
* If the tool is invoked in a production context (config contains a `configurable` key) but the active agent session cannot be resolved (due to missing, expired, or unregistered thread_ids), the connection helper immediately raises a `ValueError` (Fail Closed).
* Fallbacks to `WU_KB_DB_PATH` env vars are restricted strictly to offline scripts or direct API tests without LangGraph runner context.

### Fine-Grained Connection Caching
The SQLite/Chroma connections are cached in `_clients_cache` using a specific path tuple as the key, rather than just the workspace name:
```python
cache_key = (str(db_path.resolve()), str(vector_dir.resolve()))
```
This guarantees that changing directories, test setups pointing to different DB files, or duplicate workspace names in different parent deployment directories will never crosstalk or share stale client connections.

---

## 5. Write-Access & Symlink Protection (`agent.py`)

Because sandboxed tenant `kb/` directories are symlinks pointing back to the repository's source directories, they present a risk of workspace contamination if the agent invokes write/edit tools. Runtime `skills/` directories are copied, but write access is still denied so tenant sessions cannot mutate copied skill instructions or create divergent runtime skill state.

To enforce the read-only nature of these assets, the subagent is explicitly configured with `FilesystemPermission` rules:

```python
kb_permissions = [
    # 1. Deny all read and write operations on root /kb/ to prevent fallback crosstalk
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/kb/**"],
        mode="deny",
    ),
    # 2. Deny all write/edit operations on any tenant kb/skills and app-wide skills directories
    FilesystemPermission(
        operations=["write"],
        paths=[
            "/workspace*/kb/**",
            "/workspace*/skills/**",
            "/skills/**",
        ],
        mode="deny",
    ),
]
```

These rules are intercepted by `FilesystemMiddleware` at the tool call boundary, forcing write operations on symlinked files to fail closed with a permission error.

---

## 6. Dynamic Path Formatting (Prompt + SKILL.md)

To prevent the agent from attempting to read the root `/kb/index.json` or `/kb/METHOD.md` files (which would trigger permission errors due to the rules above), paths are formatted dynamically at two levels:

### 6.1 System Prompt Rewriting

```python
# apps/wu_tanchang_api/agent_factory/agent.py
formatted_kb_prompt = KB_ANALYST_PROMPT.replace("kb/", f"/{effective_workspace}/kb/")
```

### 6.2 SKILL.md Path Rewriting

During `ensure_runtime_workspace`, after skills are copied into each tenant's runtime directory, all `SKILL.md` files are recursively scanned and `kb/` references are replaced with the tenant-scoped path:

```python
# apps/wu_tanchang_api/agent_factory/utils.py
for skill_md in tenant_skills_dir.glob("**/SKILL.md"):
    content = skill_md.read_text(encoding="utf-8")
    updated = content.replace("kb/", f"/{folder.name}/kb/")
    skill_md.write_text(updated, encoding="utf-8")
```

This ensures both the system prompt and the `SkillsMiddleware`-loaded skill instructions use `/{effective_workspace}/kb/METHOD.md`, `/{effective_workspace}/kb/PLAYBOOK.md`, etc., preventing any cross-tenant path leakage.

> [!WARNING]
> Source SKILL.md files must use `kb/` as a template placeholder. Do **not** hard-code workspace names in source skill files — the runtime rewriting will produce double-prefixed paths.

---

## 7. Review Checklist for New Agents

When reviewing or extending this codebase, ensure the following principles are maintained:

* [ ] **Always pass `config` down**: When calling search or retrieval tools from agents, ensure the `RunnableConfig` is passed down to tool functions, so `_get_kb_connection` can perform correct routing.
* [ ] **Use the Cache Clear Helper in Tests**: When writing unit tests that modify active agents or databases, always clear `_clients_cache` in the test setup or teardown block.
* [ ] **Keep permissions intact**: Never remove `kb_permissions` from subagent specs, as this is the primary firewall protecting the codebase's source directory from LLM writes.
* [ ] **Use `kb/` as template paths in source SKILL.md**: Source skill files must reference `kb/index.json`, `kb/METHOD.md`, etc. without workspace prefixes. The runtime deployment rewrites these to tenant-scoped paths automatically.
