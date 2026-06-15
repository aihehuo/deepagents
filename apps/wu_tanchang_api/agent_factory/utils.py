"""Utility functions for Wu Tanchang agent factory."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path


_logger = logging.getLogger("uvicorn.error")


def default_workspace_path() -> Path:
    """Return the default workspace directory for Wu Tanchang."""
    return Path(__file__).resolve().parent.parent / "workspace"


def default_runtime_dir() -> Path:
    """Return the runtime data directory."""
    return Path.home() / ".deepagents" / "wu_tanchang_api"


def _deploy_dir(src: Path, dest: Path, *, symlink: bool = False, ignore: list[str] | None = None) -> None:
    """Deploy a directory from src to dest.

    If symlink is True, tries to create a symlink first.
    If symlink is False or symlinking fails, uses an atomic copytree fallback.
    """
    if symlink:
        if dest.is_symlink():
            try:
                if dest.readlink() == src.resolve():
                    # Already correctly symlinked
                    return
            except OSError:
                pass
            dest.unlink()
        elif dest.exists():
            shutil.rmtree(dest, ignore_errors=True)

        try:
            os.symlink(src.resolve(), dest)
            _logger.info("[WuTanchang] Symlinked directory: %s -> %s", src, dest)
            return
        except (OSError, PermissionError) as exc:
            _logger.warning(
                "[WuTanchang] Symlink failed (%s). Falling back to copytree: %s -> %s",
                exc,
                src,
                dest,
            )

    # Copy / deploy via atomic replace to prevent partial delete or failure window
    temp_dest = dest.with_name(dest.name + ".tmp")
    if temp_dest.exists():
        if temp_dest.is_symlink():
            temp_dest.unlink()
        else:
            shutil.rmtree(temp_dest, ignore_errors=True)

    ignore_callable = shutil.ignore_patterns(*ignore) if ignore else None
    shutil.copytree(src, temp_dest, ignore=ignore_callable)

    if dest.is_symlink():
        dest.unlink()
        os.rename(temp_dest, dest)
    elif dest.exists():
        backup_dest = dest.with_name(dest.name + ".old")
        if backup_dest.exists():
            if backup_dest.is_symlink():
                backup_dest.unlink()
            else:
                shutil.rmtree(backup_dest, ignore_errors=True)
        os.rename(dest, backup_dest)
        os.rename(temp_dest, dest)
        shutil.rmtree(backup_dest, ignore_errors=True)
    else:
        os.rename(temp_dest, dest)
    _logger.info("[WuTanchang] Deployed directory via copy: %s -> %s", src, dest)


def ensure_runtime_workspace(*, workspace_src: Path, runtime_dir: Path) -> Path:
    """Deploy workspace assets (kb, skills) into runtime backend root.

    Symlinks kb/ and skills/ from workspace into runtime_dir so the
    FilesystemBackend can access them at virtual paths /kb/ and /skills/.

    Args:
        workspace_src: Source workspace directory in the repo.
        runtime_dir: Target runtime backend root.

    Returns:
        The runtime_dir path (created if needed).
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # Also copy persona .md and owner.json files from workspace root to runtime
    for pattern in ("*.md", "owner.json"):
        for src_file in workspace_src.glob(pattern):
            if not src_file.name.startswith("kb") and not src_file.name.startswith("memory"):
                dest = runtime_dir / src_file.name
                shutil.copy2(src_file, dest)
                _logger.info("[WuTanchang] Deployed file: %s -> %s", src_file, dest)
    for name in ("kb", "skills"):
        src = workspace_src / name
        if not src.exists():
            continue
        dest = runtime_dir / name
        # Both kb/ and skills/ are read-only! We can safely symlink both to save space.
        _deploy_dir(src, dest, symlink=True)

    # Deploy all workspace directories (e.g. workspace_*, workspace)
    # Writable workspace folders containing persona MDs are copied to preserve isolation,
    # but we ignore kb, skills, and memory folders inside them to avoid redundant big copies.
    for folder in workspace_src.parent.glob("workspace*"):
        if folder.is_dir() and folder.name not in ("kb", "skills"):
            dest = runtime_dir / folder.name
            _deploy_dir(folder, dest, symlink=False, ignore=["kb", "skills", "memory"])

    return runtime_dir


def mask_sensitive_value(value: str | None, show_chars: int = 8) -> str:
    """Mask a sensitive value for logging."""
    if not value:
        return "(not set)"
    if len(value) <= show_chars + 4:
        return "***"
    return f"{value[:show_chars]}...{value[-4:]}"


def get_workspace_agent_id(workspace_path: Path) -> str:
    """Parse the Agent id from MEMORY.md in the workspace.
    If workspace_path is an owner workspace, looks in the corresponding user workspace.
    """
    path = workspace_path
    is_owner = path.name.endswith("_owner")
    if is_owner:
        user_ws_name = path.name[:-6]  # strip '_owner'
        path = path.parent / user_ws_name

    memory_md = path / "MEMORY.md"
    if memory_md.exists():
        try:
            content = memory_md.read_text(encoding="utf-8")
            import re
            match = re.search(r"-\s+\*\*Agent\s+id\*\*:\s*(\S+)", content, re.IGNORECASE)
            if not match:
                match = re.search(r"-\s+Agent\s+id\s*:\s*(\S+)", content, re.IGNORECASE)
            if match:
                agent_id = match.group(1).strip()
                if is_owner:
                    return f"{agent_id}_owner"
                return agent_id
        except Exception:
            pass

    # Fallbacks if parsing fails
    if "1" in workspace_path.name:
        return "yc01_owner" if is_owner else "yc01"
    return "owner" if is_owner else "default"


def get_workspace_owner_name(workspace_path: Path) -> str:
    """Parse the owner name from owner.json in the workspace.
    If workspace_path is an owner workspace, looks in the corresponding user workspace.
    """
    import json
    path = workspace_path
    # Try current workspace first, then fall back to user workspace if it's owner mode
    for target_path in (path, path.parent / path.name.replace("_owner", "")):
        owner_json = target_path / "owner.json"
        if owner_json.exists():
            try:
                data = json.loads(owner_json.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "owner_name" in data:
                    return data["owner_name"]
            except Exception:
                pass

    # Fallbacks
    if "1" in workspace_path.name:
        return "YC老师"
    return "吴探长"


def get_workspace_domain(workspace_path: Path) -> str:
    """Parse the domain/category from IDENTITY.md in the workspace.
    Returns "创业" or "餐饮" (default).
    """
    path = workspace_path
    if path.name.endswith("_owner"):
        user_ws_name = path.name[:-6]
        path = path.parent / user_ws_name

    identity_md = path / "IDENTITY.md"
    if identity_md.exists():
        try:
            content = identity_md.read_text(encoding="utf-8")
            if "创业" in content:
                return "创业"
        except Exception:
            pass
    return "餐饮"
