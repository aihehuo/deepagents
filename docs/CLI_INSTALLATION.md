# Installing and Using the DeepAgents CLI

## Current Status

The `deepagents-cli` executable is already installed in your conda environment! ✅

You can verify it's available:
```bash
which deepagents
# Output: /Users/yc/miniconda3/bin/deepagents
```

## How It Works

The CLI is defined in `libs/deepagents-cli/pyproject.toml`:

```toml
[project.scripts]
deepagents = "deepagents_cli:cli_main"
deepagents-cli = "deepagents_cli:cli_main"
```

When you install the package, Python's setuptools creates console scripts that:
- Are placed in your Python environment's `bin/` directory
- Point to the `cli_main()` function in `deepagents_cli.main`
- Can be run from anywhere in your terminal

## Installation Methods

### Method 1: Install in Editable Mode (For Development)

If you want to make changes to the CLI and have them immediately available:

```bash
cd libs/deepagents-cli
pip install -e .
```

This installs the package in "editable" mode, meaning:
- Changes to source code are immediately available
- The `deepagents` command will use your local code
- No need to reinstall after making changes

### Method 2: Install from PyPI (For Production Use)

```bash
pip install deepagents-cli
```

### Method 3: Install Using uv

```bash
cd libs/deepagents-cli
uv pip install -e .
```

## Verifying Installation

After installation, verify it works:

```bash
# Check if the command is available
which deepagents

# Test the CLI
deepagents --help

# Or use the alternative name
deepagents-cli --help
```

## Using the CLI

Once installed, you can run:

```bash
# Start the interactive CLI
deepagents

# Use a specific agent
deepagents --agent mybot

# Auto-approve tool usage
deepagents --auto-approve

# Use a remote sandbox
deepagents --sandbox modal

# Get help
deepagents help

# List available agents
deepagents list
```

## How Console Scripts Work

When you install a Python package with `[project.scripts]` in `pyproject.toml`:

1. **setuptools reads the configuration**:
   ```toml
   [project.scripts]
   deepagents = "deepagents_cli:cli_main"
   ```

2. **Creates executable scripts** in your Python's `bin/` directory:
   - On macOS/Linux: `~/.local/bin/deepagents` or `{conda_env}/bin/deepagents`
   - On Windows: `{env}/Scripts/deepagents.exe`

3. **The script is a wrapper** that:
   - Activates the Python environment
   - Imports `deepagents_cli.cli_main`
   - Calls the function

4. **You can run it from anywhere** because:
   - The `bin/` directory is in your PATH
   - The script knows which Python to use

## Troubleshooting

### Command Not Found

If `deepagents` is not found:

1. **Check if it's installed**:
   ```bash
   pip show deepagents-cli
   ```

2. **Check your PATH**:
   ```bash
   echo $PATH
   # Should include your Python's bin directory
   ```

3. **Reinstall in editable mode**:
   ```bash
   cd libs/deepagents-cli
   pip install -e .
   ```

### Using a Different Python Environment

If you have multiple Python environments (conda, venv, etc.):

1. **Activate the environment first**:
   ```bash
   conda activate deepagents  # or your environment name
   ```

2. **Then install**:
   ```bash
   pip install -e libs/deepagents-cli
   ```

3. **Verify it's using the right Python**:
   ```bash
   which deepagents
   which python3
   # Should be in the same environment
   ```

## Development Workflow

For development, install in editable mode:

```bash
# From the root directory
pip install -e libs/deepagents-cli

# Or from the package directory
cd libs/deepagents-cli
pip install -e .
```

Now you can:
- Edit code in `libs/deepagents-cli/deepagents_cli/`
- Run `deepagents` and see changes immediately
- No need to reinstall after each change

## Summary

- ✅ **Already installed**: The CLI is available as `deepagents` command
- 📦 **Installation**: `pip install -e libs/deepagents-cli` (editable mode for dev)
- 🚀 **Usage**: Just run `deepagents` from anywhere
- 🔧 **Development**: Install in editable mode to see changes immediately

The CLI doesn't need to be "compiled" - it's a Python script that gets installed as a console command!

