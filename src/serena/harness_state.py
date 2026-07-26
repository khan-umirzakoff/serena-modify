"""External runtime-state storage for the coding harness."""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


def _default_state_root() -> Path:
    """Default user-cache root for harness runtime state."""
    configured_cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_home = Path(configured_cache_home).expanduser() if configured_cache_home else Path.home() / ".cache"
    return cache_home / "serena" / "task-sessions"


def _project_key(project_root: Path) -> str:
    """Stable, readable namespace for one project path."""
    resolved_root = project_root.expanduser().resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved_root.name).strip("-") or "project"
    digest = hashlib.sha256(str(resolved_root).encode("utf-8")).hexdigest()[:16]
    return f"{slug}-{digest}"


@dataclass(frozen=True)
class HarnessStateStore:
    """Project- and workspace-scoped runtime-state location."""

    root: Path

    @classmethod
    def create(cls, root: Path | str | None = None) -> "HarnessStateStore":
        """
        Create a state store rooted outside project working trees.

        :param root: optional explicit state root; defaults to the user cache
        :return: configured state store
        """
        state_root = Path(root).expanduser() if root is not None else _default_state_root()
        return cls(state_root.resolve())

    def path(self, project_root: Path, filename: str, workspace_id: str | None = None) -> Path:
        """
        Return a state file path for one project and workspace.

        :param project_root: active project root
        :param filename: state filename
        :param workspace_id: optional isolated workspace identifier
        :return: external state file path
        """
        if Path(filename).name != filename:
            raise ValueError("state filename must not contain path separators")
        if workspace_id is not None and not workspace_id.isalnum():
            raise ValueError("workspace_id must be alphanumeric")

        session_name = workspace_id or "single"
        return self.root / _project_key(project_root) / session_name / filename
