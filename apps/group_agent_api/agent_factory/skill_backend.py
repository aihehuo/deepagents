"""Read-only model filesystem, isolated from host-owned profile state."""

from pathlib import Path

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import EditResult, FileUploadResponse, GlobResult, WriteResult
from deepagents.backends.state import StateBackend


class SkillReadOnlyBackend(FilesystemBackend):
    """Expose installed skills only; host business writers use their own store."""

    def _resolve_path(self, key: str) -> Path:
        path = super()._resolve_path(key)
        skills = self.cwd / "skills"
        if not path.is_relative_to(skills):
            msg = "Model filesystem access is limited to installed skills"
            raise PermissionError(msg)
        return path

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Limit even the backend's default glob root to skills.

        Args:
            pattern: Requested file pattern.
            path: Optional virtual search root.

        Returns:
            Matches inside the skills directory, or a permission error.
        """
        result = super().glob(pattern, "/skills" if path in {None, "/"} else path)
        safe = []
        for match in result.matches or []:
            try:
                self._resolve_path(match["path"])
            except (OSError, ValueError, RuntimeError):
                continue
            safe.append(match)
        return GlobResult(matches=safe, error=result.error)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Deny generic writes.

        Args:
            file_path: Requested virtual path.
            content: Requested content.

        Returns:
            A permission error without changing disk state.
        """
        # Retain BackendProtocol parameter names; every payload is rejected.
        del file_path, content
        return WriteResult(error="permission_denied: model filesystem is read-only")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002  # Preserve the BackendProtocol positional signature.
    ) -> EditResult:
        """Deny generic edits.

        Args:
            file_path: Requested virtual path.
            old_string: Requested match.
            new_string: Requested replacement.
            replace_all: Requested replacement policy.

        Returns:
            A permission error without changing disk state.
        """
        # Retain BackendProtocol parameter names; every payload is rejected.
        del file_path, old_string, new_string, replace_all
        return EditResult(error="permission_denied: model filesystem is read-only")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Deny uploads, including tool-output persistence.

        Args:
            files: Requested paths and contents.

        Returns:
            A permission error for each requested upload.
        """
        return [FileUploadResponse(path=path, error="permission_denied") for path, _ in files]


def create_group_agent_backend(runtime: Path) -> CompositeBackend:
    """Keep skills host-read-only while allowing framework artifacts in state."""
    artifacts = StateBackend()
    return CompositeBackend(
        default=SkillReadOnlyBackend(root_dir=str(runtime), virtual_mode=True),
        routes={
            "/conversation_history/": artifacts,
            "/large_tool_results/": artifacts,
        },
    )
