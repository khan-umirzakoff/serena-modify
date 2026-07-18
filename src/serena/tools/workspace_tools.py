"""Tools for managing independent agents on a shared MCP server."""

import json

from serena.tools import Tool, ToolMarkerDoesNotRequireActiveProject, ToolMarkerOptional


class OpenWorkspaceTool(Tool, ToolMarkerOptional, ToolMarkerDoesNotRequireActiveProject):
    """Opens an isolated Serena workspace for one chat or task."""

    def apply(self, project: str) -> str:
        """Open a project in a new isolated workspace.

        :param project: absolute project path or registered Serena project name
        :return: JSON metadata containing the workspace_id to pass to subsequent tool calls
        """
        workspace = self.agent.get_workspace_registry_or_raise().open(project)
        return json.dumps(
            {
                "workspace": workspace,
                "usage": "Pass workspace_id to every subsequent Serena tool call for this chat or task.",
            },
            ensure_ascii=False,
            indent=2,
        )


class ListWorkspacesTool(Tool, ToolMarkerOptional, ToolMarkerDoesNotRequireActiveProject):
    """Lists isolated workspaces hosted by this Serena MCP server."""

    def apply(self) -> str:
        """List active isolated workspaces.

        :return: JSON workspace list and inactivity TTL
        """
        registry = self.agent.get_workspace_registry_or_raise()
        return json.dumps(
            {"workspaces": registry.list(), "ttl_seconds": registry.ttl_seconds},
            ensure_ascii=False,
            indent=2,
        )


class CloseWorkspaceTool(Tool, ToolMarkerOptional, ToolMarkerDoesNotRequireActiveProject):
    """Closes an isolated workspace and releases its resources."""

    def apply(self, workspace_id: str) -> str:
        """Close an isolated workspace.

        :param workspace_id: workspace identifier returned by open_workspace
        :return: JSON close result
        """
        closed = self.agent.get_workspace_registry_or_raise().close(workspace_id)
        return json.dumps({"workspace_id": workspace_id, "closed": closed}, ensure_ascii=False, indent=2)
