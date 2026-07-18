"""Tests for shared-MCP workspace isolation."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from serena.config.context_mode import SerenaAgentContext
from serena.mcp_workspaces import WorkspaceNotFoundError, WorkspaceRegistry
from serena.tools.tools_base import ToolRegistry
from serena.tools.workflow_tools import _coding_task_context_path, _goal_state_path, _plan_state_path


@dataclass
class FakeProject:
    project_name: str
    project_root: Path


class FakeAgent:
    def __init__(self, project: str, workspace_id: str) -> None:
        self.project = FakeProject(Path(project).name, Path(project))
        self.workspace_id = workspace_id
        self.shutdown_calls = 0

    def get_active_project(self) -> FakeProject:
        return self.project

    def on_shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_workspace_tools_are_registered_and_enabled_for_chatgpt() -> None:
    tool_names = ToolRegistry().get_tool_names()
    context = SerenaAgentContext.load("chatgpt")

    assert {"open_workspace", "list_workspaces", "close_workspace"} <= set(tool_names)
    assert {"open_workspace", "list_workspaces", "close_workspace"} <= set(context.included_optional_tools)


def test_workspace_registry_isolates_agents_and_closes_resources(tmp_path: Path) -> None:
    agents: dict[str, FakeAgent] = {}

    def factory(project: str, workspace_id: str) -> FakeAgent:
        agent = FakeAgent(project, workspace_id)
        agents[workspace_id] = agent
        return agent

    registry = WorkspaceRegistry(factory, ttl_seconds=3600)
    try:
        first = registry.open(str(tmp_path / "first"))
        second = registry.open(str(tmp_path / "second"))
        first_id = str(first["workspace_id"])
        second_id = str(second["workspace_id"])

        assert first_id != second_id
        assert registry.get_agent(first_id) is agents[first_id]
        assert registry.get_agent(second_id) is agents[second_id]
        assert len(registry.list()) == 2

        assert registry.close(first_id) is True
        assert agents[first_id].shutdown_calls == 1
        with pytest.raises(WorkspaceNotFoundError, match="Unknown or expired"):
            registry.get_agent(first_id)
    finally:
        registry.shutdown()

    assert agents[second_id].shutdown_calls == 1


def test_workspace_registry_expires_idle_agents(tmp_path: Path) -> None:
    clock = FakeClock()
    agents: dict[str, FakeAgent] = {}

    def factory(project: str, workspace_id: str) -> FakeAgent:
        agent = FakeAgent(project, workspace_id)
        agents[workspace_id] = agent
        return agent

    registry = WorkspaceRegistry(factory, ttl_seconds=10, monotonic=clock)
    try:
        workspace = registry.open(str(tmp_path))
        workspace_id = str(workspace["workspace_id"])
        clock.value = 11

        assert registry.cleanup_expired() == 1
        assert agents[workspace_id].shutdown_calls == 1
        with pytest.raises(WorkspaceNotFoundError, match="Unknown or expired"):
            registry.get_agent(workspace_id)
    finally:
        registry.shutdown()


def test_workflow_state_paths_are_workspace_scoped(tmp_path: Path) -> None:
    workspace_id = "abc123"

    assert _goal_state_path(tmp_path) == tmp_path / ".serena" / "goal_state.json"
    assert _goal_state_path(tmp_path, workspace_id) == tmp_path / ".serena" / "task-sessions" / workspace_id / "goal_state.json"
    assert _plan_state_path(tmp_path, workspace_id) == tmp_path / ".serena" / "task-sessions" / workspace_id / "plan_state.json"
    assert _coding_task_context_path(tmp_path, workspace_id) == (
        tmp_path / ".serena" / "task-sessions" / workspace_id / "coding_task_context.json"
    )
