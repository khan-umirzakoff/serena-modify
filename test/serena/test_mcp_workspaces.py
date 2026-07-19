"""Tests for shared-MCP workspace isolation."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from serena.mcp import SerenaMCPFactory
from serena.mcp_workspaces import WorkspaceMode, WorkspaceNotFoundError, WorkspaceRegistry
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


def test_workspace_tools_are_exposed_only_in_multi_mode() -> None:
    tool_names = ToolRegistry().get_tool_names()
    single_factory = SerenaMCPFactory(
        transport="stdio",
        context="chatgpt",
        project="/project",
        workspace_mode=WorkspaceMode.SINGLE,
    )
    fixed_factory = SerenaMCPFactory(
        transport="stdio",
        context="chatgpt",
        project="/project",
        workspace_mode=WorkspaceMode.SINGLE,
        fixed_project=True,
    )
    multi_factory = SerenaMCPFactory(transport="stdio", context="chatgpt", workspace_mode=WorkspaceMode.MULTI)

    assert {"open_workspace", "list_workspaces", "close_workspace"} <= set(tool_names)
    assert single_factory.context.single_project is False
    assert fixed_factory.context.single_project is True
    assert not ({"open_workspace", "list_workspaces", "close_workspace"} & set(single_factory.context.included_optional_tools))
    assert multi_factory.context.single_project is False
    assert {"open_workspace", "list_workspaces", "close_workspace"} <= set(multi_factory.context.included_optional_tools)


def test_mcp_schema_matches_workspace_mode() -> None:
    observed: dict[str, tuple[bool, bool, bool, bool]] = {}

    for workspace_mode in WorkspaceMode:
        factory = SerenaMCPFactory(transport="stdio", context="chatgpt", workspace_mode=workspace_mode)
        server = factory.create_mcp_server(
            enable_web_dashboard=False,
            enable_gui_log_window=False,
            open_web_dashboard=False,
        )
        assert factory.agent is not None
        factory._set_mcp_tools(server, openai_tool_compatible=True, structured_output=False)
        tools = server._tool_manager._tools
        observed[workspace_mode.value] = (
            "open_workspace" in tools,
            "workspace_id" in tools["read_file"].parameters["properties"],
            "open_workspace" in factory._get_initial_instructions(),
            "activate_project" in tools,
        )
        if factory._workspace_registry is not None:
            factory._workspace_registry.shutdown()
        factory.agent.on_shutdown()

    assert observed == {
        "single": (False, False, False, True),
        "multi": (True, True, True, True),
    }


def test_fixed_project_hides_project_switching(tmp_path: Path) -> None:
    factory = SerenaMCPFactory(
        transport="stdio",
        context="chatgpt",
        project=str(tmp_path),
        workspace_mode=WorkspaceMode.SINGLE,
        fixed_project=True,
    )
    server = factory.create_mcp_server(
        enable_web_dashboard=False,
        enable_gui_log_window=False,
        open_web_dashboard=False,
    )
    assert factory.agent is not None
    factory._set_mcp_tools(server, openai_tool_compatible=True, structured_output=False)

    assert "activate_project" not in server._tool_manager._tools
    assert "fixed-project mode" in factory._get_initial_instructions()

    factory.agent.on_shutdown()


def test_chatgpt_harness_ux_is_remote_and_unambiguous() -> None:
    factory = SerenaMCPFactory(
        transport="stdio",
        context="chatgpt",
        workspace_mode=WorkspaceMode.SINGLE,
    )
    server = factory.create_mcp_server(
        enable_web_dashboard=False,
        enable_gui_log_window=False,
        open_web_dashboard=False,
    )
    assert factory.agent is not None
    factory._set_mcp_tools(server, openai_tool_compatible=True, structured_output=False)

    tools = server._tool_manager._tools
    system_prompt = factory.agent.create_system_prompt()

    assert "execute_shell_command" not in tools
    assert "activate_project" in tools
    assert "single-workspace mode" in factory._get_initial_instructions()
    assert "startup project is the initial active project, not a permanent lock" in factory._get_initial_instructions()
    assert "remote MCP coding harness" in system_prompt
    assert "output_mode" in system_prompt
    assert 'keys=["ENTER"]' in system_prompt
    assert "desktop app context" not in system_prompt
    assert "separate code editor window" not in system_prompt
    assert "output_mode" in tools["exec_command"].description
    write_properties = tools["write_stdin"].parameters["properties"]
    assert "keys" in write_properties
    assert "submit" not in write_properties
    assert write_properties["keys"]["items"]["$ref"] == "#/$defs/TerminalKey"
    assert "ENTER" in tools["write_stdin"].parameters["$defs"]["TerminalKey"]["enum"]
    assert "semantic terminal keys" in write_properties["keys"]["description"].lower()

    factory.agent.on_shutdown()


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
