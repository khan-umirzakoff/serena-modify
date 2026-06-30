"""
Tools and helpers for Codex-compatible skills.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from serena.tools import Tool
from serena.tools.tools_base import ToolRegistry

SKILLS_FILENAME = "SKILL.md"
SKILL_METADATA_PATH = Path("agents") / "openai.yaml"
AGENTS_DIR_NAME = ".agents"
SKILLS_DIR_NAME = "skills"
MAX_SCAN_DEPTH = 6
MAX_SKILLS_DIRS_PER_ROOT = 2000
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024
DEFAULT_SKILL_SUMMARY_MAX_CHARS = 8000
SCOPE_RANK = {"repo": 0, "user": 1, "system": 2, "admin": 3}


@dataclass(frozen=True)
class SkillPolicy:
    """Invocation policy declared by a skill."""

    allow_implicit_invocation: bool = True
    allow_scripts: bool = False
    allowed_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SkillResources:
    """Relative skill resource paths discovered without reading their contents."""

    references: list[str] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SkillDependencies:
    """Declared tool and MCP dependencies for a skill."""

    tools: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SkillMetadata:
    """Codex-compatible skill metadata discovered from a ``SKILL.md`` file."""

    name: str
    description: str
    path_to_skill_md: str
    scope: str
    short_description: str | None = None
    policy: SkillPolicy = field(default_factory=SkillPolicy)
    resources: SkillResources = field(default_factory=SkillResources)
    dependencies: SkillDependencies = field(default_factory=SkillDependencies)
    metadata_path: str | None = None


@dataclass(frozen=True)
class SkillError:
    """Skill discovery error that does not abort the scan."""

    path: str
    message: str


@dataclass(frozen=True)
class SkillLoadOutcome:
    """Skill discovery result."""

    skills: list[SkillMetadata]
    errors: list[SkillError]
    truncated: bool = False


@dataclass(frozen=True)
class SkillRoot:
    """Root directory to scan for skills."""

    path: Path
    scope: str


def _sanitize_single_line(raw: str) -> str:
    return " ".join(raw.split())


def _extract_frontmatter(contents: str) -> str | None:
    lines = contents.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    frontmatter_lines: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            return "\n".join(frontmatter_lines)
        frontmatter_lines.append(line)
    return None


def _default_skill_name(path: Path) -> str:
    return _sanitize_single_line(path.parent.name) or "skill"


def _validate_required_text(value: str, field_name: str, max_len: int) -> str:
    value = _sanitize_single_line(value)
    if not value:
        raise ValueError(f"missing field `{field_name}`")
    if len(value) > max_len:
        raise ValueError(f"invalid {field_name}: exceeds maximum length of {max_len} characters")
    return value


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_sanitize_single_line(value)] if value.strip() else []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(_sanitize_single_line(item))
            elif isinstance(item, dict):
                name = item.get("name") or item.get("id") or item.get("path")
                if isinstance(name, str) and name.strip():
                    result.append(_sanitize_single_line(name))
        return result
    if isinstance(value, dict):
        return [_sanitize_single_line(str(key)) for key in value.keys() if str(key).strip()]
    return []


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _read_optional_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"invalid YAML mapping: {path}")
    return parsed


def _nested_dict(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    return value if isinstance(value, dict) else {}


def _parse_skill_policy(openai_metadata: dict[str, Any]) -> SkillPolicy:
    policy = _nested_dict(openai_metadata, "policy")
    return SkillPolicy(
        allow_implicit_invocation=_as_bool(policy.get("allow_implicit_invocation"), True),
        allow_scripts=_as_bool(policy.get("allow_scripts"), False),
        allowed_tools=sorted(set(_as_string_list(policy.get("allowed_tools")))),
    )


def _dependency_tools(openai_metadata: dict[str, Any]) -> list[str]:
    dependencies = _nested_dict(openai_metadata, "dependencies")
    tools = _as_string_list(dependencies.get("tools"))
    tools.extend(_as_string_list(openai_metadata.get("tools")))
    return sorted(set(tools))


def _dependency_mcp_servers(openai_metadata: dict[str, Any]) -> list[str]:
    dependencies = _nested_dict(openai_metadata, "dependencies")
    servers = _as_string_list(dependencies.get("mcp_servers"))
    servers.extend(_as_string_list(dependencies.get("mcpServers")))
    servers.extend(_as_string_list(openai_metadata.get("mcp_servers")))
    servers.extend(_as_string_list(openai_metadata.get("mcpServers")))
    return sorted(set(servers))


def _list_existing_relative_files(skill_dir: Path, directory_name: str) -> list[str]:
    root = skill_dir / directory_name
    if not root.is_dir():
        return []
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                files.append(str(path.relative_to(skill_dir)).replace("\\", "/"))
            except ValueError:
                continue
    return files


def _declared_resource_paths(openai_metadata: dict[str, Any], skill_dir: Path) -> SkillResources:
    resources = _nested_dict(openai_metadata, "resources")
    return SkillResources(
        references=_as_string_list(resources.get("references")) or _list_existing_relative_files(skill_dir, "references"),
        scripts=_as_string_list(resources.get("scripts")) or _list_existing_relative_files(skill_dir, "scripts"),
        assets=_as_string_list(resources.get("assets")) or _list_existing_relative_files(skill_dir, "assets"),
        resources=_as_string_list(resources.get("resources")) or _list_existing_relative_files(skill_dir, "resources"),
    )


def _parse_skill_file(path: Path, scope: str) -> SkillMetadata:
    contents = path.read_text(encoding="utf-8")
    frontmatter = _extract_frontmatter(contents)
    if frontmatter is None:
        raise ValueError("missing YAML frontmatter delimited by ---")
    parsed = yaml.safe_load(frontmatter) or {}
    if not isinstance(parsed, dict):
        raise ValueError("invalid YAML frontmatter: expected mapping")

    skill_dir = path.parent
    metadata_path = skill_dir / SKILL_METADATA_PATH
    openai_metadata = _read_optional_yaml(metadata_path)

    name = _validate_required_text(str(parsed.get("name") or _default_skill_name(path)), "name", MAX_NAME_LEN)
    description = _validate_required_text(str(parsed.get("description") or ""), "description", MAX_DESCRIPTION_LEN)
    metadata = parsed.get("metadata") or {}
    short_description = None
    if isinstance(metadata, dict) and metadata.get("short-description"):
        short_description = _validate_required_text(str(metadata["short-description"]), "metadata.short-description", MAX_DESCRIPTION_LEN)
    elif openai_metadata.get("short-description"):
        short_description = _validate_required_text(str(openai_metadata["short-description"]), "short-description", MAX_DESCRIPTION_LEN)

    return SkillMetadata(
        name=name,
        description=description,
        short_description=short_description,
        path_to_skill_md=str(path.resolve()),
        scope=scope,
        policy=_parse_skill_policy(openai_metadata),
        resources=_declared_resource_paths(openai_metadata, skill_dir),
        dependencies=SkillDependencies(
            tools=_dependency_tools(openai_metadata),
            mcp_servers=_dependency_mcp_servers(openai_metadata),
        ),
        metadata_path=str(metadata_path.resolve()) if metadata_path.is_file() else None,
    )


def _dirs_between_project_root_and_focus(project_root: Path, focus_dir: Path) -> list[Path]:
    project_root = project_root.resolve()
    focus_dir = focus_dir.resolve()
    try:
        focus_dir.relative_to(project_root)
    except ValueError:
        return [project_root]

    dirs = [focus_dir, *focus_dir.parents]
    scoped_dirs = [path for path in dirs if path == project_root or project_root in path.parents]
    scoped_dirs.reverse()
    return scoped_dirs


def skill_roots(project_root: Path, focus_dir: Path, home_dir: Path | None = None) -> list[SkillRoot]:
    """Return Codex-compatible skill roots for an active Serena project and focus directory."""
    roots: list[SkillRoot] = [SkillRoot(project_root / ".serena" / SKILLS_DIR_NAME, "repo")]
    for directory in _dirs_between_project_root_and_focus(project_root, focus_dir):
        roots.append(SkillRoot(directory / AGENTS_DIR_NAME / SKILLS_DIR_NAME, "repo"))
    if home_dir is None:
        home_dir = Path.home()
    roots.append(SkillRoot(home_dir / AGENTS_DIR_NAME / SKILLS_DIR_NAME, "user"))

    seen: set[Path] = set()
    deduped: list[SkillRoot] = []
    for root in roots:
        normalized = root.path.expanduser()
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(SkillRoot(normalized, root.scope))
    return deduped


def _iter_skill_files(root: Path) -> tuple[list[Path], bool]:
    try:
        root = root.resolve()
    except OSError:
        return [], False
    if not root.is_dir():
        return [], False

    skill_files: list[Path] = []
    queue: list[tuple[Path, int]] = [(root, 0)]
    visited: set[Path] = {root}
    truncated = False

    while queue:
        directory, depth = queue.pop(0)
        if depth > MAX_SCAN_DEPTH:
            continue
        try:
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
        except OSError:
            continue

        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    resolved_dir = entry.resolve()
                    if len(visited) >= MAX_SKILLS_DIRS_PER_ROOT:
                        truncated = True
                        continue
                    if resolved_dir not in visited:
                        visited.add(resolved_dir)
                        queue.append((resolved_dir, depth + 1))
                elif entry.is_file() and entry.name == SKILLS_FILENAME:
                    skill_files.append(entry)
            except OSError:
                continue
    return skill_files, truncated


def discover_skills(project_root: Path, focus_dir: Path) -> SkillLoadOutcome:
    """Discover Codex-compatible skills using progressive-disclosure metadata only."""
    skills: list[SkillMetadata] = []
    errors: list[SkillError] = []
    truncated = False

    for root in skill_roots(project_root, focus_dir):
        skill_files, root_truncated = _iter_skill_files(root.path)
        truncated = truncated or root_truncated
        for skill_file in skill_files:
            try:
                skills.append(_parse_skill_file(skill_file, root.scope))
            except (OSError, ValueError, yaml.YAMLError) as error:
                errors.append(SkillError(path=str(skill_file), message=str(error)))

    seen_paths: set[str] = set()
    deduped: list[SkillMetadata] = []
    for skill in skills:
        if skill.path_to_skill_md not in seen_paths:
            seen_paths.add(skill.path_to_skill_md)
            deduped.append(skill)

    deduped.sort(key=lambda skill: (SCOPE_RANK.get(skill.scope, 99), skill.name, skill.path_to_skill_md))
    return SkillLoadOutcome(skills=deduped, errors=errors, truncated=truncated)


def missing_tool_dependencies(skill: SkillMetadata, available_tools: set[str]) -> list[str]:
    """Return declared Serena tool dependencies that are not active."""
    return sorted(tool for tool in skill.dependencies.tools if tool not in available_tools)


def skill_dependency_reports(skills: list[SkillMetadata], available_tools: set[str]) -> dict[str, dict[str, Any]]:
    """Return compact dependency status for each skill."""
    reports: dict[str, dict[str, Any]] = {}
    for skill in skills:
        reports[skill.name] = {
            "required_tools": skill.dependencies.tools,
            "missing_tools": missing_tool_dependencies(skill, available_tools),
            "required_mcp_servers": skill.dependencies.mcp_servers,
            "mcp_server_status": "declared_only",
        }
    return reports


def render_skills_summary(skills: list[SkillMetadata], max_chars: int = DEFAULT_SKILL_SUMMARY_MAX_CHARS) -> str | None:
    """Render a Codex-style skills summary without loading full skill instructions."""
    if not skills:
        return None

    lines = [
        "## Skills",
        "A skill is a task-specific local instruction package stored in a SKILL.md file.",
        "Only skill name, description, policy, resources, dependencies, and file path are listed here.",
        "Do not follow a skill until you first call read_skill for its name or SKILL.md path.",
        "",
        "### Available skills",
    ]
    for skill in skills:
        path = skill.path_to_skill_md.replace("\\", "/")
        description = skill.short_description or skill.description
        policy_bits = [f"implicit={skill.policy.allow_implicit_invocation}", f"scripts={skill.policy.allow_scripts}"]
        dependency_bits = []
        if skill.dependencies.tools:
            dependency_bits.append("tools=" + ",".join(skill.dependencies.tools))
        if skill.dependencies.mcp_servers:
            dependency_bits.append("mcp=" + ",".join(skill.dependencies.mcp_servers))
        resource_count = sum(len(items) for items in asdict(skill.resources).values())
        if resource_count:
            dependency_bits.append(f"resources={resource_count}")
        suffix = "; ".join([*policy_bits, *dependency_bits])
        lines.append(f"- {skill.name}: {description} ({suffix}; file: {path})")

    lines.extend(
        [
            "",
            "### Skill workflow",
            "- Match the task to the skill name and description.",
            "- If a skill is relevant, call read_skill before following its instructions.",
            "- Treat scripts/assets/references as local resources; inspect before running or relying on them.",
            "- If required tools are missing, report the missing dependency instead of guessing.",
        ]
    )
    rendered = "\n".join(lines)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max(0, max_chars - 80)].rstrip() + "\n...\n[skills summary truncated]"


def _resolve_focus_dir(project_root: Path, relative_path: str) -> Path:
    candidate = (project_root / relative_path).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as error:
        raise ValueError(f"relative_path escapes project root: {relative_path}") from error
    return candidate.parent if candidate.is_file() else candidate


def _json_response(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _truncate_text(text: str, max_answer_chars: int) -> str:
    if max_answer_chars < 0 or len(text) <= max_answer_chars:
        return text
    return text[: max(0, max_answer_chars - 80)].rstrip() + "\n...\n[truncated]"


def _active_tool_names() -> set[str]:
    return set(ToolRegistry().get_tool_names())


class DiscoverSkillsTool(Tool):
    """Discovers Codex-compatible skills without loading full SKILL.md instructions."""

    def apply(self, relative_path: str = ".", max_skills: int = 50) -> str:
        """
        Return available Codex-compatible skills as name, description, scope, and SKILL.md path.

        :param relative_path: project-relative file or directory used to choose scoped repo skills
        :param max_skills: maximum number of skills to return
        :return: JSON skill discovery result
        """
        project_root = Path(self.get_project_root()).resolve()
        focus_dir = _resolve_focus_dir(project_root, relative_path)
        outcome = discover_skills(project_root, focus_dir)
        skills = outcome.skills[: max(0, max_skills)]
        available_tools = _active_tool_names()
        response = {
            "project_root": str(project_root),
            "focus_dir": str(focus_dir),
            "skills": [asdict(skill) for skill in skills],
            "dependency_report": skill_dependency_reports(skills, available_tools),
            "omitted_skill_count": max(0, len(outcome.skills) - len(skills)),
            "errors": [asdict(error) for error in outcome.errors],
            "truncated": outcome.truncated,
            "progressive_disclosure": "Full SKILL.md contents are not included; call read_skill before using a skill.",
        }
        return _json_response(response)


class ReadSkillTool(Tool):
    """Reads the full SKILL.md instructions for one discovered Codex-compatible skill."""

    def apply(self, skill: str, relative_path: str = ".", max_answer_chars: int = 50000) -> str:
        """
        Return full SKILL.md instructions for a skill name or SKILL.md path.

        :param skill: exact skill name or project/absolute path to a SKILL.md file
        :param relative_path: project-relative file or directory used to choose scoped repo skills when resolving by name
        :param max_answer_chars: maximum response length; ``-1`` disables truncation
        :return: JSON skill instructions with metadata and contents
        """
        project_root = Path(self.get_project_root()).resolve()
        focus_dir = _resolve_focus_dir(project_root, relative_path)
        outcome = discover_skills(project_root, focus_dir)
        selected: SkillMetadata | None = None
        skill_path = Path(skill).expanduser()
        if skill_path.is_absolute() or skill.endswith(SKILLS_FILENAME) or "/" in skill or "\\" in skill:
            candidate = skill_path if skill_path.is_absolute() else (project_root / skill_path)
            try:
                resolved_candidate = candidate.resolve()
            except OSError:
                resolved_candidate = candidate
            for item in outcome.skills:
                if Path(item.path_to_skill_md).resolve() == resolved_candidate:
                    selected = item
                    break
        else:
            matches = [item for item in outcome.skills if item.name == skill]
            if len(matches) > 1:
                return _json_response(
                    {"error": f"multiple skills named {skill!r}; use a SKILL.md path", "matches": [asdict(item) for item in matches]}
                )
            selected = matches[0] if matches else None
        if selected is None:
            return _json_response({"error": f"skill not found: {skill}", "available_skills": [asdict(item) for item in outcome.skills]})
        contents = Path(selected.path_to_skill_md).read_text(encoding="utf-8")
        available_tools = _active_tool_names()
        response = {
            "skill": asdict(selected),
            "dependency_report": skill_dependency_reports([selected], available_tools).get(selected.name, {}),
            "contents": contents,
            "usage": "Follow these instructions only for the current task when this skill is relevant.",
        }
        return _truncate_text(_json_response(response), max_answer_chars)
