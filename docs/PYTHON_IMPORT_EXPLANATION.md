# Why You Need to Install the Package to Run Tests

## The Problem

When you try to run tests from the root directory, you get:
```
ModuleNotFoundError: No module named 'deepagents.middleware.datetime'
```

This happens even though the code exists at `libs/deepagents/deepagents/middleware/datetime.py`.

## How Python Imports Work

When Python executes `from deepagents.middleware.datetime import DateTimeMiddleware`, it searches for the `deepagents` package in these locations (in order):

1. **Current directory** - Python looks in the directory where you run the script
2. **PYTHONPATH** - Environment variable listing additional directories
3. **Installed packages** - Packages installed via `pip install` (in `site-packages`)
4. **Standard library** - Built-in Python modules

### What Python is Looking For

Python expects to find a package structure like this:
```
deepagents/          # Package directory
  __init__.py       # Makes it a package
  middleware/
    __init__.py
    datetime.py
```

## The Repository Structure

Your repository is structured like this:
```
deepagents/                    # Root directory
  libs/
    deepagents/                # Package source code
      deepagents/              # The actual package
        __init__.py
        middleware/
          __init__.py
          datetime.py
      pyproject.toml           # Package configuration
  tests/
    test_datetime_middleware.py
```

### The Issue

When you run `pytest tests/test_datetime_middleware.py` from the root:

1. Python starts in `/Users/yc/workspace/deepagents/`
2. The test file tries to import: `from deepagents.middleware.datetime import DateTimeMiddleware`
3. Python looks for `deepagents/` in the current directory → **Not found** ❌
4. Python looks in PYTHONPATH → **Not found** ❌
5. Python looks in installed packages → **Not found** ❌ (unless you installed it)

The actual code is at `libs/deepagents/deepagents/`, but Python doesn't know to look there!

## Why It Works in `libs/deepagents/.venv`

When you run tests from `libs/deepagents/` using `uv run pytest`:

1. The `uv` tool activates a virtual environment
2. That environment has `deepagents` **installed** (via `uv pip install -e .`)
3. When installed, Python knows where to find it
4. The import works! ✅

## Solutions

### Solution 1: Install the Package (Recommended)

Install the package in editable mode in your Python environment:

```bash
pip install -e libs/deepagents
```

**What this does:**
- Creates a link in your Python's `site-packages` pointing to `libs/deepagents/`
- Python can now find `deepagents` when importing
- Changes to source code are immediately reflected (editable mode)
- You can run tests from anywhere

**After installation, Python's import path includes:**
```
/Users/yc/workspace/deepagents/libs/deepagents/  ← Now Python knows about this!
```

### Solution 2: Modify PYTHONPATH

You could add the package directory to PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/Users/yc/workspace/deepagents/libs/deepagents"
pytest tests/test_datetime_middleware.py
```

**Why this works:**
- Tells Python to also look in `libs/deepagents/` for packages
- Python finds `deepagents/` there
- But you have to set this every time (or add to shell config)

**Why Solution 1 is better:**
- More standard Python practice
- Works across different shells/terminals
- Matches how packages are used in production
- No need to remember to set environment variables

### Solution 3: Use the uv Environment

Run tests using the uv environment that already has the package:

```bash
cd libs/deepagents
uv run --group test pytest ../../tests/test_datetime_middleware.py
```

**Why this works:**
- The uv environment has the package installed
- But you have to be in the `libs/deepagents/` directory

## Why Packages Need to Be "Installed"

### The Difference: Code vs Package

- **Code in a directory**: Just files sitting there
- **Installed package**: Python knows about it and can import it

Think of it like:
- **Code**: A book sitting on your desk
- **Installed package**: A book registered in the library catalog

Python's import system is like a library catalog - it only knows about "registered" (installed) packages.

### What `pip install -e` Does

When you run `pip install -e libs/deepagents`:

1. Reads `pyproject.toml` to understand the package structure
2. Creates a link in `site-packages` pointing to `libs/deepagents/`
3. Python's import system now knows: "When someone imports `deepagents`, look in `libs/deepagents/deepagents/`"
4. The `-e` (editable) flag means changes to source code are immediately available

## Visual Example

### Before Installation

```
Python's import search path:
  /current/directory/          ← Looks here first
  /site-packages/              ← Looks here (empty for deepagents)
  
Your code:
  /Users/yc/workspace/deepagents/libs/deepagents/deepagents/  ← Python doesn't know about this!
  
Result: ModuleNotFoundError ❌
```

### After `pip install -e libs/deepagents`

```
Python's import search path:
  /current/directory/
  /site-packages/
    deepagents → /Users/yc/workspace/deepagents/libs/deepagents/  ← Link created!
  
Your code:
  /Users/yc/workspace/deepagents/libs/deepagents/deepagents/  ← Python can find it!
  
Result: Import successful ✅
```

## Why This Design?

This might seem inconvenient, but it's actually a feature:

1. **Isolation**: Different projects can have different versions of the same package
2. **Clarity**: You know exactly which packages your code depends on
3. **Reproducibility**: Others can install the same packages and get the same behavior
4. **Standard Practice**: This is how Python packages work everywhere

## Summary

- **The Problem**: Python can't find `deepagents` because it's not in Python's search path
- **The Solution**: Install it with `pip install -e libs/deepagents`
- **Why**: Python only imports from installed packages (or current directory/PYTHONPATH)
- **Editable Mode (`-e`)**: Changes to source code are immediately available, no reinstall needed

This is standard Python package management - even for your own code, you need to "install" it (even in editable mode) for Python to find it!

