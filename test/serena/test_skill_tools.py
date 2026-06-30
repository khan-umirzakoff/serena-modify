from pathlib import Path

from serena.tools.skill_tools import discover_skills, render_skills_summary, skill_roots
from serena.tools.tools_base import ToolRegistry


def _write_skill(path: Path, name: str = "frontend-ui", description: str = "Build polished frontend UI.") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\nUse the design system.\n", encoding="utf-8")


def test_skill_tools_are_registered() -> None:
    names = ToolRegistry().get_tool_names()

    assert "discover_skills" in names
    assert "read_skill" in names


def test_discovers_repo_agents_skills_for_focus_path(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    focus_dir = project_root / "frontend" / "src"
    focus_dir.mkdir(parents=True)
    skill_path = project_root / "frontend" / ".agents" / "skills" / "frontend-ui" / "SKILL.md"
    _write_skill(skill_path)

    outcome = discover_skills(project_root, focus_dir)

    assert not outcome.errors
    assert len(outcome.skills) == 1
    assert outcome.skills[0].name == "frontend-ui"
    assert outcome.skills[0].path_to_skill_md == str(skill_path.resolve())


def test_skill_roots_include_project_and_user_locations(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    focus_dir = project_root / "frontend"
    home_dir = tmp_path / "home"

    roots = skill_roots(project_root, focus_dir, home_dir=home_dir)

    root_paths = [root.path for root in roots]
    assert project_root / ".serena" / "skills" in root_paths
    assert project_root / ".agents" / "skills" in root_paths
    assert focus_dir / ".agents" / "skills" in root_paths
    assert home_dir / ".agents" / "skills" in root_paths


def test_render_skills_summary_is_progressive_disclosure(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    focus_dir = project_root
    skill_path = project_root / ".agents" / "skills" / "backend" / "SKILL.md"
    _write_skill(skill_path, name="backend", description="Implement backend changes safely.")

    outcome = discover_skills(project_root, focus_dir)
    summary = render_skills_summary(outcome.skills)

    assert summary is not None
    assert "backend: Implement backend changes safely" in summary
    assert str(skill_path.resolve()) in summary
    assert "Use the design system" not in summary
    assert "read_skill" in summary
