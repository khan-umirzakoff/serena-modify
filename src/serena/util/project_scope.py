"""Project-scope resolution shared by Codex-compatible harness features."""

import subprocess
from pathlib import Path


def resolve_git_scope_root(project_root: Path, focus_dir: Path) -> Path:
    """
    Return the nearest Git root inside the active Serena project.

    :param project_root: active Serena project root
    :param focus_dir: task focus directory
    :return: nested Git root, or the Serena project root when none applies
    """
    resolved_project_root = project_root.resolve()
    resolved_focus_dir = focus_dir.resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=resolved_focus_dir,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return resolved_project_root

    git_root_text = result.stdout.strip()
    if result.returncode != 0 or not git_root_text:
        return resolved_project_root

    git_root = Path(git_root_text).resolve()
    if git_root == resolved_project_root or resolved_project_root in git_root.parents:
        return git_root
    return resolved_project_root
