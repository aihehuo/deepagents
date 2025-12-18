"""Test backends for use in integration tests.

This module provides backend wrappers and utilities for testing.
"""

from pathlib import Path
from typing import Any

from deepagents.backends.protocol import BackendProtocol, EditResult, WriteResult


class DirectoryOnlyBackend:
    """Backend wrapper that forces all writes/edits into a specified directory.
    
    This ensures the agent can only write/edit files in the specified directory,
    preventing writes to arbitrary locations (e.g., `/`, `/home/user`, or other system paths).
    
    Reads/ls/grep/etc. are delegated to the underlying backend unchanged.
    """

    def __init__(self, *, backend: BackendProtocol, test_dir: str | Path) -> None:
        """Initialize the directory-only backend.
        
        Args:
            backend: The underlying backend to delegate to.
            test_dir: The directory where writes/edits are allowed.
        """
        self._backend = backend
        self._test_dir = Path(test_dir).expanduser().resolve()
        self._test_dir.mkdir(parents=True, exist_ok=True)

    def _map_write_path(self, file_path: str) -> str:
        """Map any write/edit path to the test directory.
        
        The filesystem middleware normalizes many paths to start with "/".
        We always treat the final component as the desired filename and place it under test_dir/.
        
        Args:
            file_path: The virtual path from the agent (e.g., "/path/to/file.md")
            
        Returns:
            Actual path in test_dir
        """
        # Extract just the filename from the virtual path
        name = Path(file_path).name or "output.txt"
        target = (self._test_dir / name).resolve()
        # Safety: ensure target stays within test_dir
        target.relative_to(self._test_dir)
        return str(target)

    # ---- write/edit (sync + async)
    def write(self, file_path: str, content: str) -> WriteResult:  # type: ignore[override]
        """Write a file, mapping the path to the test directory."""
        return self._backend.write(self._map_write_path(file_path), content)

    async def awrite(self, file_path: str, content: str) -> WriteResult:  # type: ignore[override]
        """Write a file asynchronously, mapping the path to the test directory."""
        return await self._backend.awrite(self._map_write_path(file_path), content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # type: ignore[override]
        """Edit a file, mapping the path to the test directory."""
        return self._backend.edit(self._map_write_path(file_path), old_string, new_string, replace_all=replace_all)

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # type: ignore[override]
        """Edit a file asynchronously, mapping the path to the test directory."""
        return await self._backend.aedit(self._map_write_path(file_path), old_string, new_string, replace_all=replace_all)

    # ---- everything else delegates
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the underlying backend."""
        return getattr(self._backend, name)

