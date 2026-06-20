```markdown
# deepagents Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the `deepagents` Python codebase. You'll learn how to structure files, write imports and exports, follow commit message conventions, and understand the testing patterns present in the repository. This guide is ideal for contributors who want to maintain consistency and quality in their code contributions.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `agent_core.py`, `data_loader.py`

### Import Style
- Prefer **relative imports** within the package.
  - Example:
    ```python
    from .utils import load_config
    from .models.agent import Agent
    ```

### Export Style
- Use **named exports** (explicitly specify what is exported).
  - Example:
    ```python
    __all__ = ["Agent", "load_config"]
    ```

### Commit Messages
- Commit messages are generally **freeform**, with occasional use of the `release` prefix.
- Keep commit messages concise (average length: ~31 characters).
  - Example:
    ```
    release: v1.2.0
    Fix agent initialization bug
    ```

## Workflows

### Release Management
**Trigger:** When preparing a new release version.
**Command:** `/release`

1. Update version numbers as needed.
2. Ensure all features and bug fixes are merged.
3. Write a commit with the `release` prefix, e.g., `release: v1.2.0`.
4. Tag the release in version control.
5. Push changes and tags to the remote repository.

### Adding a New Module
**Trigger:** When introducing new functionality.
**Command:** `/add-module`

1. Create a new Python file using snake_case naming.
2. Use relative imports to access shared utilities or models.
3. Add named exports to the module via `__all__`.
4. Write or update corresponding test files (see Testing Patterns).
5. Commit changes with a descriptive message.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `agent.test.py`).
- The specific testing framework is **unknown**; check existing test files for structure.
- To add a test:
  - Create a test file named with `.test.` in the filename.
  - Place test functions/classes inside this file.
  - Example:
    ```python
    # agent.test.py
    def test_agent_initialization():
        agent = Agent()
        assert agent.is_initialized
    ```

## Commands
| Command      | Purpose                                  |
|--------------|------------------------------------------|
| /release     | Prepare and tag a new release            |
| /add-module  | Add a new module following conventions   |
```
