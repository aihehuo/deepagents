from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents.backends.protocol import BackendProtocol, EditResult, WriteResult


class DocsOnlyWriteBackend:
    """Backend wrapper that forces all writes/edits into a docs directory.

    This is used by the Business Co-Founder API to ensure the agent cannot write files
    into arbitrary locations (e.g. `/`, `/home/user`, or host-specific paths).

    Reads/ls/grep/etc. are delegated to the underlying backend unchanged.
    """

    def __init__(self, *, backend: BackendProtocol, docs_dir: str | Path) -> None:
        self._backend = backend
        self._docs_dir = Path(docs_dir).expanduser().resolve()
        self._docs_dir.mkdir(parents=True, exist_ok=True)

    def _map_write_path(self, file_path: str) -> str:
        # The filesystem middleware normalizes many paths to start with "/".
        # We always treat the final component as the desired filename and place it under docs/.
        name = Path(file_path).name or "output.txt"
        target = (self._docs_dir / name).resolve()
        # Safety: ensure target stays within docs_dir
        target.relative_to(self._docs_dir)
        return str(target)

    # ---- write/edit (sync + async)
    def write(self, file_path: str, content: str) -> WriteResult:  # type: ignore[override]
        return self._backend.write(self._map_write_path(file_path), content)

    async def awrite(self, file_path: str, content: str) -> WriteResult:  # type: ignore[override]
        return await self._backend.awrite(self._map_write_path(file_path), content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # type: ignore[override]
        return self._backend.edit(self._map_write_path(file_path), old_string, new_string, replace_all=replace_all)

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # type: ignore[override]
        return await self._backend.aedit(self._map_write_path(file_path), old_string, new_string, replace_all=replace_all)

    # ---- everything else delegates
    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


