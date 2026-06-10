"""Utility functions for Wu Tanchang agent factory."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

_logger = logging.getLogger("uvicorn.error")


def default_workspace_path() -> Path:
    """Return the default workspace directory for Wu Tanchang."""
    return Path(__file__).resolve().parent.parent / "workspace"


def default_runtime_dir() -> Path:
    """Return the runtime data directory."""
    return Path.home() / ".deepagents" / "wu_tanchang_api"


def ensure_runtime_workspace(*, workspace_src: Path, runtime_dir: Path) -> Path:
    """Deploy workspace assets (kb, skills, intake) into runtime backend root.

    Copies kb/, skills/, and intake/ from workspace into runtime_dir so the
    FilesystemBackend can access them at virtual paths /kb/, /skills/, etc.

    Args:
        workspace_src: Source workspace directory in the repo.
        runtime_dir: Target runtime backend root.

    Returns:
        The runtime_dir path (created if needed).
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # Also copy persona .md files from workspace root to runtime
    for md_file in workspace_src.glob("*.md"):
        if not md_file.name.startswith("kb") and not md_file.name.startswith("memory"):
            dest = runtime_dir / md_file.name
            shutil.copy2(md_file, dest)
            _logger.info("[WuTanchang] Deployed persona: %s -> %s", md_file, dest)
    for name in ("kb", "skills", "intake"):
        src = workspace_src / name
        if not src.exists():
            continue
        dest = runtime_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        _logger.info("[WuTanchang] Deployed %s -> %s", src, dest)
    return runtime_dir


def copy_default_expertise(*, dest_expertise_dir: Path) -> None:
    """Copy expertise templates from app source to runtime directory."""
    dest_expertise_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent.parent / "expertise"
    if not source.exists():
        _logger.warning("[WuTanchang] Expertise source not found: %s", source)
        return
    for source_file in source.glob("*.md"):
        shutil.copy2(source_file, dest_expertise_dir / source_file.name)
        _logger.info("[WuTanchang] Copied expertise: %s", source_file.name)


def mask_sensitive_value(value: str | None, show_chars: int = 8) -> str:
    """Mask a sensitive value for logging."""
    if not value:
        return "(not set)"
    if len(value) <= show_chars + 4:
        return "***"
    return f"{value[:show_chars]}...{value[-4:]}"
