from pathlib import Path

from serena.tools.skill_tools import discover_skills, missing_tool_dependencies, render_skills_summary, skill_roots
from serena.tools.tools_base import ToolRegistry


def _write_skill(path: Path, name: str = "frontend-ui", description: str = "Build polished frontend UI.") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\nUse the design system.\n", encoding="utf-8")


def _write_openai_yaml(skill_dir: Path) -> None:
    metadata = skill_dir / "agents" / "openai.yaml"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        "short-description: Frontend UI workflow\n"
        "interface:\n"
        "  display_name: Frontend UI\n"
        "  short_description: Current frontend workflow\n"
        "  default_prompt: Build the requested frontend safely.\n"
        "policy:\n"
        "  allow_implicit_invocation: false\n"
        "  allow_scripts: true\n"
        "  allowed_tools:\n"
        "    - read_file\n"
        "dependencies:\n"
        "  tools:\n"
        "    - read_file\n"
        "    - missing_tool\n"
        "    - type: mcp\n"
        "      value: official_docs\n"
        "  mcp_servers:\n"
        "    - SERENA_MCP_local\n"
        "resources:\n"
        "  references:\n"
        "    - references/design.md\n"
        "  scripts:\n"
        "    - scripts/check.py\n",
        encoding="utf-8",
    )


def test_skill_tools_are_registered() -> None:
    names = ToolRegistry().get_tool_names()

    assert "discover_skills" in names
    assert "read_skill" in names


def test_safe_file_tools_are_registered() -> None:
    names = ToolRegistry().get_tool_names()

    assert "create_directory" in names
    assert "create_or_replace_file" in names


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

    admin_dir = tmp_path / "etc" / "codex" / "skills"
    roots = skill_roots(project_root, focus_dir, home_dir=home_dir, admin_skills_dir=admin_dir)

    root_paths = [root.path for root in roots]
    assert project_root / ".serena" / "skills" in root_paths
    assert project_root / ".agents" / "skills" in root_paths
    assert focus_dir / ".agents" / "skills" in root_paths
    assert home_dir / ".agents" / "skills" in root_paths
    assert admin_dir in root_paths


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


def test_reads_openai_metadata_policy_resources_and_dependencies(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    focus_dir = project_root
    skill_dir = project_root / ".agents" / "skills" / "frontend-ui"
    skill_path = skill_dir / "SKILL.md"
    _write_skill(skill_path)
    _write_openai_yaml(skill_dir)

    outcome = discover_skills(project_root, focus_dir)

    assert not outcome.errors
    skill = outcome.skills[0]
    assert skill.short_description == "Current frontend workflow"
    assert skill.interface.display_name == "Frontend UI"
    assert skill.interface.short_description == "Current frontend workflow"
    assert skill.interface.default_prompt == "Build the requested frontend safely."
    assert skill.policy.allow_implicit_invocation is False
    assert skill.policy.allow_scripts is True
    assert skill.policy.allowed_tools == ["read_file"]
    assert skill.dependencies.tools == ["missing_tool", "read_file"]
    assert skill.dependencies.mcp_servers == ["SERENA_MCP_local", "official_docs"]
    assert skill.resources.references == ["references/design.md"]
    assert skill.resources.scripts == ["scripts/check.py"]
    assert missing_tool_dependencies(skill, {"read_file"}) == ["missing_tool"]
    public_metadata = skill.to_public_dict()
    assert public_metadata["interface"] == {
        "display_name": "Frontend UI",
        "default_prompt": "Build the requested frontend safely.",
    }


def test_discovers_existing_resource_directories_without_metadata(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    skill_dir = project_root / ".agents" / "skills" / "backend"
    _write_skill(skill_dir / "SKILL.md", name="backend", description="Backend workflow.")
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "references" / "notes.md").write_text("notes\n", encoding="utf-8")

    outcome = discover_skills(project_root, project_root)

    assert not outcome.errors
    assert outcome.skills[0].resources.references == ["references/notes.md"]


def test_skips_skills_disabled_in_codex_config(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    skill_path = project_root / ".agents" / "skills" / "backend" / "SKILL.md"
    _write_skill(skill_path, name="backend", description="Backend workflow.")
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        f'[[skills.config]]\npath = "{skill_path}"\nenabled = false\n',
        encoding="utf-8",
    )

    outcome = discover_skills(project_root, project_root, codex_home=codex_home)

    assert outcome.skills == []
