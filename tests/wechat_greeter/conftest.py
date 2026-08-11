"""Pytest configuration for wechat_greeter tests.

Adds `libs/` (parent) to sys.path so `from wechat_greeter import X` works
(`wechat_greeter/__init__.py` is at `libs/wechat_greeter/__init__.py`,
so Python needs `libs/` in sys.path to find it as a top-level package).

The existing root conftest.py adds `libs/deepagents` so `from deepagents import X`
works; both paths coexist because `import deepagents` resolves to the real package
at `libs/deepagents/deepagents/__init__.py` (first match wins, and `libs/deepagents/`
is earlier in sys.path).
"""

from __future__ import annotations

import sys
from pathlib import Path

# deepagents root = tests/wechat_greeter/conftest.py → 3 levels up
repo_root = Path(__file__).parent.parent.parent
libs_dir = repo_root / "libs"
if libs_dir.exists() and str(libs_dir) not in sys.path:
    sys.path.insert(0, str(libs_dir))
