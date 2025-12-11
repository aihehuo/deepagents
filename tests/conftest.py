"""Pytest configuration for tests."""

import sys
from pathlib import Path

# Add libs/deepagents to Python path so we can import deepagents modules
# This is needed because the package might not be installed in editable mode
repo_root = Path(__file__).parent.parent
deepagents_lib = repo_root / "libs" / "deepagents"
if deepagents_lib.exists() and str(deepagents_lib) not in sys.path:
    sys.path.insert(0, str(deepagents_lib))

