# Subagent Prompts

This directory contains system prompt files for subagents. These prompts are loaded at runtime and can be edited without modifying the code.

## File Structure

- `coder.md` - System prompt for the coder subagent (specialized for coding tasks)
- `aihehuo.md` - System prompt for the AI He Huo subagent (specialized for partner/investor searches)

## How It Works

1. When a subagent is created via `build_coder_subagent_from_env()` or `build_aihehuo_subagent_from_env()`, the code attempts to load the corresponding markdown file from this directory.

2. If the file exists and can be read, its content (excluding markdown headers) is used as the system prompt.

3. If the file doesn't exist or cannot be read, the code falls back to the default hardcoded prompt.

## Editing Prompts

To modify a subagent's behavior:

1. Edit the corresponding `.md` file in this directory
2. The first line can be a markdown header (e.g., `# Coder Subagent System Prompt`) - it will be automatically stripped
3. The rest of the content will be used as the system prompt
4. No code changes or restarts are needed - the prompt is loaded each time a subagent is created

## File Format

The markdown files should contain:
- An optional header line (will be stripped automatically)
- The actual system prompt content

Example:
```markdown
# Coder Subagent System Prompt

You are a coding-focused subagent.

You specialize in:
- Writing and editing code
- ...
```

## Adding New Subagents

To add a new subagent prompt:

1. Create a new `.md` file in this directory (e.g., `new_subagent.md`)
2. Add the prompt content
3. In `subagent_presets.py`, modify the corresponding builder function to call:
   ```python
   system_prompt = _load_subagent_prompt("new_subagent", default_prompt)
   ```
