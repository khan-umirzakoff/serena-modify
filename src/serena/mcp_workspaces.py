"""Workspace lifecycle management for shared MCP servers."""

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serena.agent import SerenaAgent


DEFAULT_WORKSPACE_TTL_SECONDS = 6 * 60 * 60
DEFAULT_MAX_WORKSPACES = 8


class WorkspaceMode(Enum):
    """Operating mode for project routing on a shared MCP server."""

    SINGLE = "single"
    MULTI = "multi"

    @property
    def is_multi(self) -> bool:
        """Return whether explicit multi-workspace routing is enabled."""
        return self is WorkspaceMode.MULTI


class WorkspaceError(Exception):
    """Base error raised by the MCP workspace registry."""


class WorkspaceNotFoundError(WorkspaceError):
    """Raised when a workspace handle is unknown or expired."""


@dataclass
class Workspace:
    """A workspace-bound Serena agent and its lifecycle metadata."""

    workspace_id: str
    project: str
    project_root: str
    agent: "SerenaAgent"
    created_at: str
    last_used_monotonic: float

    def public_info(self, now_monotonic: float) -> dict[str, object]:
        """Return client-safe workspace metadata."""
        active_project = self.agent.get_active_project()
        return {
            "workspace_id": self.workspace_id,
            "project": active_project.project_name if active_project is not None else self.project,
            "project_root": str(active_project.project_root) if active_project is not None else None,
            "created_at": self.created_at,
            "idle_seconds": max(0, int(now_monotonic - self.last_used_monotonic)),
        }


class WorkspaceRegistry:
    """Owns independent Serena agents addressed by explicit workspace handles."""

    def __init__(
        self,
        agent_factory: Callable[[str, str], "SerenaAgent"],
        ttl_seconds: float = DEFAULT_WORKSPACE_TTL_SECONDS,
        max_workspaces: int = DEFAULT_MAX_WORKSPACES,
        allow_shared_worktree: bool = False,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("workspace TTL must be positive")
        if max_workspaces <= 0:
            raise ValueError("maximum workspace count must be positive")
        self._agent_factory = agent_factory
        self._ttl_seconds = ttl_seconds
        self._max_workspaces = max_workspaces
        self._allow_shared_worktree = allow_shared_worktree
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._workspaces: dict[str, Workspace] = {}
        self._shutdown_event = threading.Event()
        cleanup_interval_seconds = min(60.0, ttl_seconds / 2)
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(cleanup_interval_seconds,),
            name="SerenaWorkspaceCleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    @property
    def ttl_seconds(self) -> float:
        """Return the inactivity TTL applied to workspaces."""
        return self._ttl_seconds

    @property
    def max_workspaces(self) -> int:
        """Return the maximum number of simultaneously active workspaces."""
        return self._max_workspaces

    def _pop_expired_locked(self, now: float) -> list[Workspace]:
        expired_ids = [
            workspace_id for workspace_id, workspace in self._workspaces.items() if now - workspace.last_used_monotonic >= self._ttl_seconds
        ]
        return [self._workspaces.pop(workspace_id) for workspace_id in expired_ids]

    @staticmethod
    def _shutdown_workspaces(workspaces: list[Workspace]) -> None:
        for workspace in workspaces:
            workspace.agent.on_shutdown()

    @staticmethod
    def _active_project_root(workspace: Workspace) -> str:
        """Current resolved project root for duplicate-working-tree checks."""
        active_project = workspace.agent.get_active_project()
        return str(Path(active_project.project_root).resolve()) if active_project is not None else workspace.project_root

    def cleanup_expired(self) -> int:
        """Close idle workspaces and return the number removed."""
        with self._lock:
            expired = self._pop_expired_locked(self._monotonic())
        self._shutdown_workspaces(expired)
        return len(expired)

    def _cleanup_loop(self, interval_seconds: float) -> None:
        while not self._shutdown_event.wait(interval_seconds):
            self.cleanup_expired()

    def open(self, project: str) -> dict[str, object]:
        """Create an independent agent for a project and return its workspace metadata."""
        project = project.strip()
        if not project:
            raise WorkspaceError("project must not be empty")
        self.cleanup_expired()
        with self._lock:
            if len(self._workspaces) >= self._max_workspaces:
                raise WorkspaceError(
                    f"Workspace limit reached ({self._max_workspaces}). Close an idle workspace or raise --max-workspaces."
                )

        workspace_id = uuid.uuid4().hex
        agent = self._agent_factory(project, workspace_id)
        active_project = agent.get_active_project()
        if active_project is None:
            agent.on_shutdown()
            raise WorkspaceError(f"Unable to activate project: {project}")
        project_root = str(Path(active_project.project_root).resolve())

        now = self._monotonic()
        workspace = Workspace(
            workspace_id=workspace_id,
            project=project,
            project_root=project_root,
            agent=agent,
            created_at=datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
            last_used_monotonic=now,
        )
        with self._lock:
            duplicate = next(
                (existing for existing in self._workspaces.values() if self._active_project_root(existing) == project_root),
                None,
            )
            if duplicate is not None and not self._allow_shared_worktree:
                agent.on_shutdown()
                raise WorkspaceError(
                    "This Git working tree is already open in another workspace. "
                    "Open a separate Git worktree path for parallel edits, or explicitly start the server with --allow-shared-worktree."
                )
            if len(self._workspaces) >= self._max_workspaces:
                agent.on_shutdown()
                raise WorkspaceError(
                    f"Workspace limit reached ({self._max_workspaces}). Close an idle workspace or raise --max-workspaces."
                )
            self._workspaces[workspace_id] = workspace
        return workspace.public_info(now)

    def get_agent(self, workspace_id: str) -> "SerenaAgent":
        """Return and touch the agent for a workspace handle."""
        self.cleanup_expired()
        now = self._monotonic()
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
            if workspace is None:
                raise WorkspaceNotFoundError(f"Unknown or expired workspace_id {workspace_id!r}. Call open_workspace again.")
            workspace.last_used_monotonic = now
            return workspace.agent

    def assert_project_available(self, workspace_id: str, project_root: Path | str) -> None:
        """
        Reject project switches that would share a working tree across workspaces.

        :param workspace_id: workspace attempting activation
        :param project_root: requested project root
        """
        if self._allow_shared_worktree:
            return

        resolved_project_root = str(Path(project_root).resolve())
        with self._lock:
            duplicate = next(
                (
                    workspace
                    for current_id, workspace in self._workspaces.items()
                    if current_id != workspace_id and self._active_project_root(workspace) == resolved_project_root
                ),
                None,
            )
        if duplicate is not None:
            raise WorkspaceError(
                "This Git working tree is already open in another workspace. Activate a separate Git worktree path for parallel edits."
            )

    def list(self) -> list[dict[str, object]]:
        """Return metadata for all active workspaces without extending their TTL."""
        self.cleanup_expired()
        now = self._monotonic()
        with self._lock:
            workspaces = sorted(self._workspaces.values(), key=lambda workspace: workspace.created_at)
            return [workspace.public_info(now) for workspace in workspaces]

    def close(self, workspace_id: str) -> bool:
        """Close a workspace and release its agent resources."""
        with self._lock:
            workspace = self._workspaces.pop(workspace_id, None)
        if workspace is None:
            raise WorkspaceNotFoundError(f"Unknown or expired workspace_id {workspace_id!r}. It may already be closed.")
        workspace.agent.on_shutdown()
        return True

    def shutdown(self) -> None:
        """Close all workspaces owned by this registry."""
        self._shutdown_event.set()
        with self._lock:
            workspaces = list(self._workspaces.values())
            self._workspaces.clear()
        self._shutdown_workspaces(workspaces)
        if threading.current_thread() is not self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)
