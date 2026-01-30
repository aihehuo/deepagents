"""Utility functions for agent factory."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

# Use uvicorn's configured logger so output reliably shows up in the terminal.
_logger = logging.getLogger("uvicorn.error")


def copy_example_skills_if_missing(*, dest_skills_dir: Path) -> None:
    """Copy example skills into dest_skills_dir (no overwrite).

    Looks for libs/deepagents-cli/examples/skills relative to repo root
    (resolved from this file: apps/business_cofounder_api/agent_factory -> repo root).
    If not in a monorepo or path missing, skips without error.
    """
    # Path(__file__) = .../apps/business_cofounder_api/agent_factory/utils.py
    # -> parent.parent.parent.parent = repo root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    src = repo_root / "libs" / "deepagents-cli" / "examples" / "skills"
    if not src.exists():
        return

    dest_skills_dir.mkdir(parents=True, exist_ok=True)
    for skill_dir in sorted(src.iterdir()):
        if not skill_dir.is_dir():
            continue
        if not (skill_dir / "SKILL.md").exists():
            continue
        dest = dest_skills_dir / skill_dir.name
        if dest.exists():
            continue
        shutil.copytree(skill_dir, dest)


def copy_default_expertise_if_missing(*, dest_expertise_dir: Path) -> None:
    """Copy all expertise templates from source directory, always overwriting existing files.
    
    This ensures all expertise files in the runtime directory are up-to-date with the
    source files. Always overwrites existing files to ensure updates are reflected.
    Copies from apps/business_cofounder_api/expertise/ to the destination directory.
    """
    dest_expertise_dir.mkdir(parents=True, exist_ok=True)
    
    # Source directory: apps/business_cofounder_api/expertise/
    source_expertise_dir = Path(__file__).parent.parent / "expertise"
    
    if not source_expertise_dir.exists():
        _logger.warning("[Expertise] Source expertise directory not found at %s", source_expertise_dir)
        return
    
    # Copy all .md files from source to destination (always overwrite existing files)
    copied_count = 0
    overwritten_count = 0
    for source_file in source_expertise_dir.glob("*.md"):
        dest_file = dest_expertise_dir / source_file.name
        
        if dest_file.exists():
            overwritten_count += 1
            _logger.info("[Expertise] Overwriting expertise template: %s -> %s", source_file.name, dest_file)
        else:
            copied_count += 1
            _logger.info("[Expertise] Copying new expertise template: %s -> %s", source_file.name, dest_file)
        
        shutil.copy2(source_file, dest_file)
    
    if overwritten_count > 0:
        _logger.info("[Expertise] Overwrote %d existing expertise template(s) in %s", overwritten_count, dest_expertise_dir)
    if copied_count > 0:
        _logger.info("[Expertise] Copied %d new expertise template(s) to %s", copied_count, dest_expertise_dir)
    if overwritten_count == 0 and copied_count == 0:
        _logger.debug("[Expertise] No expertise templates found in source directory")


def mask_sensitive_value(value: str | None, show_chars: int = 8) -> str:
    """Mask a sensitive value for logging (show first N chars and last 4 chars)."""
    if not value:
        return "(not set)"
    if len(value) <= show_chars + 4:
        return "***"  # Too short to mask meaningfully
    return f"{value[:show_chars]}...{value[-4:]}"


def mask_url(url: str | None) -> str:
    """Mask URL for logging (show full URL as it's less sensitive than API keys)."""
    if not url:
        return "(not set)"
    # For URLs, show the full URL since domain/path is not sensitive
    # Only mask query parameters if present
    if "?" in url:
        base_url, query = url.split("?", 1)
        return f"{base_url}?***"
    return url


def get_user_memory_path(base_dir: Path, user_id: str) -> Path:
    """Get the path to user-level memory file.
    
    Args:
        base_dir: Base directory for the API (~/.deepagents/business_cofounder_api)
        user_id: User identifier
        
    Returns:
        Path to user memory file: base_dir/users/{user_id}/agent.md
    """
    user_dir = base_dir / "users" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir / "agent.md"


def get_conversation_memory_path(base_dir: Path, user_id: str, conversation_id: str) -> Path:
    """Get the path to conversation-level memory file.
    
    Args:
        base_dir: Base directory for the API (~/.deepagents/business_cofounder_api)
        user_id: User identifier
        conversation_id: Conversation identifier
        
    Returns:
        Path to conversation memory file: base_dir/users/{user_id}/conversations/{conversation_id}/agent.md
    """
    conversation_dir = base_dir / "users" / user_id / "conversations" / conversation_id
    conversation_dir.mkdir(parents=True, exist_ok=True)
    return conversation_dir / "agent.md"


def ensure_memory_directories_exist(base_dir: Path, user_id: str | None, conversation_id: str | None) -> None:
    """Ensure memory directories exist for the given user and conversation.
    
    Args:
        base_dir: Base directory for the API
        user_id: User identifier (optional)
        conversation_id: Conversation identifier (optional)
    """
    if user_id:
        user_dir = base_dir / "users" / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        if conversation_id:
            conversation_dir = user_dir / "conversations" / conversation_id
            conversation_dir.mkdir(parents=True, exist_ok=True)
