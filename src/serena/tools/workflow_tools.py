"""
Tools supporting the general workflow of the agent
"""

import json
import os
import platform
import re
import shlex
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from serena.harness_state import HarnessStateStore
from serena.tools import Tool, ToolMarkerCanEdit, ToolMarkerDoesNotRequireActiveProject, ToolMarkerOptional, WriteMemoryTool
from serena.tools.skill_tools import discover_skills, render_skills_summary, skill_dependency_reports
from serena.tools.task_catalog import VALIDATION_KINDS, TaskCatalog, ValidationHints, discover_task_catalog, infer_validation_hints
from serena.util.project_scope import resolve_git_scope_root


@dataclass(frozen=True)
class ProjectInstructionDocument:
    """Project-level instruction document discovered for a coding task."""

    path: str
    relative_path: str
    contents: str
    truncated: bool


@dataclass(frozen=True)
class CommandSnapshot:
    """Result of a bounded command used in a coding-task snapshot."""

    command: str
    exit_code: int | None
    output: str


@dataclass(frozen=True)
class ProjectInstructionSettings:
    """Codex-compatible project instruction discovery settings."""

    codex_home: Path
    fallback_filenames: tuple[str, ...]
    max_bytes: int


@dataclass(frozen=True)
class CodingTaskSnapshot:
    """Codex-style project snapshot prepared before code editing."""

    project_name: str
    project_root: str
    focus_path: str
    git_root: str
    context: str
    modes: list[str]
    active_tools: list[str]
    instruction_documents: list[ProjectInstructionDocument]
    edit_policy: dict[str, Any]
    validation_hints: ValidationHints
    task_catalog_summary: dict[str, Any]
    task_catalog: dict[str, Any]
    active_goal: dict[str, Any] | None
    active_plan: dict[str, Any] | None
    available_skills: list[dict[str, Any]]
    skills_summary: str | None
    skill_dependency_report: dict[str, Any]
    git_status: CommandSnapshot
    git_diff_stat: CommandSnapshot
    git_diff_cached_stat: CommandSnapshot


@dataclass(frozen=True)
class ProjectSwitchGuidance:
    """Exact project activation and retry arguments for an out-of-project focus path."""

    project: str
    relative_path: str


GOAL_STATE_FILENAME = "goal_state.json"
PLAN_STATE_FILENAME = "plan_state.json"
CODING_TASK_CONTEXT_FILENAME = "coding_task_context.json"
MAX_GOAL_OBJECTIVE_CHARS = 4000
MODEL_SETTABLE_GOAL_STATUSES = {"complete", "blocked"}
PLAN_STATUSES = {"pending", "in_progress", "completed"}
DEFAULT_REVIEW_DIFF_MAX_CHARS = 60_000
DEFAULT_PROJECT_DOC_MAX_BYTES = 32 * 1024


def _serena_state_path(
    project_root: Path,
    filename: str,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> Path:
    """External runtime-state path for one project and workspace."""
    return (state_store or HarnessStateStore.create()).path(project_root, filename, workspace_id)


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _goal_state_path(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> Path:
    return _serena_state_path(project_root, GOAL_STATE_FILENAME, workspace_id, state_store)


def _plan_state_path(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> Path:
    return _serena_state_path(project_root, PLAN_STATE_FILENAME, workspace_id, state_store)


def _coding_task_context_path(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> Path:
    return _serena_state_path(project_root, CODING_TASK_CONTEXT_FILENAME, workspace_id, state_store)


def _save_coding_task_context(
    project_root: Path,
    state: dict[str, Any],
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> None:
    path = _coding_task_context_path(project_root, workspace_id, state_store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_coding_task_context(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> dict[str, Any] | None:
    path = _coding_task_context_path(project_root, workspace_id, state_store)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def _validate_goal_objective(objective: str) -> None:
    if not objective:
        raise ValueError("goal objective must not be empty")
    if len(objective) > MAX_GOAL_OBJECTIVE_CHARS:
        raise ValueError(f"goal objective must be at most {MAX_GOAL_OBJECTIVE_CHARS} characters")


def _load_goal_state(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> dict[str, Any] | None:
    path = _goal_state_path(project_root, workspace_id, state_store)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def _save_goal_state(
    project_root: Path,
    state: dict[str, Any],
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> None:
    path = _goal_state_path(project_root, workspace_id, state_store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_goal_state(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> dict[str, Any]:
    state = _load_goal_state(project_root, workspace_id, state_store)
    if state is None:
        raise ValueError("No Serena goal exists for this project. Call create_goal first.")
    return state


def _goal_time_used_seconds(state: dict[str, Any]) -> int:
    created_at = state.get("created_at")
    if not isinstance(created_at, str):
        return 0
    created = _parse_time(created_at)
    if created is None:
        return 0
    if state.get("status") in {"complete", "blocked", "budget_limited"}:
        end_value = state.get("updated_at")
    else:
        end_value = _utc_now()
    end = _parse_time(end_value) if isinstance(end_value, str) else None
    if end is None:
        return 0
    return max(0, int((end - created).total_seconds()))


def _goal_remaining_tokens(state: dict[str, Any]) -> int | None:
    token_budget = state.get("token_budget")
    if not isinstance(token_budget, int):
        return None
    tokens_used = state.get("tokens_used", 0)
    if not isinstance(tokens_used, int):
        tokens_used = 0
    return max(0, token_budget - tokens_used)


def _goal_guidance(goal: dict[str, Any] | None) -> list[str] | None:
    """Return compact model-facing guidance for an active goal."""
    if goal is None or goal.get("status") != "active":
        return None

    return [
        "Continue toward the full objective and use current worktree or external evidence as authoritative.",
        "Keep update_plan current for meaningful multi-step work; a plan update does not replace doing the work.",
        "Mark complete only after every requirement is verified. Mark blocked only after the same blocker repeats for three goal turns and no meaningful progress remains.",
    ]


def _goal_public_state(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> dict[str, Any] | None:
    state = _load_goal_state(project_root, workspace_id, state_store)
    if state is None:
        return None
    state = dict(state)
    state["state_path"] = str(_goal_state_path(project_root, workspace_id, state_store))
    state["time_used_seconds"] = _goal_time_used_seconds(state)
    state["remaining_tokens"] = _goal_remaining_tokens(state)
    return state


def _compact_goal_public_state(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> dict[str, Any] | None:
    """
    Return the active goal fields useful in a coding task snapshot.

    :param project_root: active project root
    :return: compact goal state, or None if no goal is active
    """
    state = _goal_public_state(project_root, workspace_id, state_store)
    if state is None:
        return None
    return {
        "objective": state.get("objective"),
        "status": state.get("status"),
        "remaining_tokens": state.get("remaining_tokens"),
        "time_used_seconds": state.get("time_used_seconds"),
    }


def _goal_response(goal: dict[str, Any] | None, include_completion_report: bool = False) -> dict[str, Any]:
    report = None
    if include_completion_report and goal is not None and goal.get("token_budget") is not None:
        report = (
            "Goal marked complete. Serena cannot measure ChatGPT Web/App token usage directly; "
            f"stored tokens_used={goal.get('tokens_used', 0)} and token_budget={goal.get('token_budget')}."
        )
    return {
        "goal": goal,
        "remaining_tokens": _goal_remaining_tokens(goal) if goal is not None else None,
        "guidance": _goal_guidance(goal),
        "completion_budget_report": report,
    }


def _append_goal_note(state: dict[str, Any], note: str) -> None:
    progress = state.setdefault("progress", [])
    if not isinstance(progress, list):
        progress = []
        state["progress"] = progress
    progress.append({"at": _utc_now(), "note": note})


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    tail = max_chars - half
    omitted = len(text) - max_chars
    return text[:half] + f"\n...[omitted {omitted} chars from middle]...\n" + text[-tail:], True


def _save_plan_state(
    project_root: Path,
    state: dict[str, Any],
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> None:
    path = _plan_state_path(project_root, workspace_id, state_store)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_plan_state(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> dict[str, Any] | None:
    path = _plan_state_path(project_root, workspace_id, state_store)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state if isinstance(state, dict) else None


def _plan_public_state(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> dict[str, Any] | None:
    state = _load_plan_state(project_root, workspace_id, state_store)
    if state is None:
        return None
    state = dict(state)
    state["state_path"] = str(_plan_state_path(project_root, workspace_id, state_store))
    return state


def _compact_plan_public_state(
    project_root: Path,
    workspace_id: str | None = None,
    state_store: HarnessStateStore | None = None,
) -> dict[str, Any] | None:
    """
    Return active plan fields useful in a coding task snapshot.

    :param project_root: active project root
    :return: compact plan state, or None if no plan is active
    """
    state = _plan_public_state(project_root, workspace_id, state_store)
    if state is None:
        return None
    return {
        "updated_at": state.get("updated_at"),
        "explanation": state.get("explanation"),
        "plan": state.get("plan"),
    }


def _validate_plan(plan: list[dict[str, Any]]) -> None:
    if not isinstance(plan, list):
        raise ValueError("plan must be a list of plan items")

    allowed_item_keys = {"step", "status"}
    in_progress_count = 0
    all_completed = True

    for index, item in enumerate(plan):
        if not isinstance(item, dict):
            raise ValueError(f"plan item {index} must be an object")

        extra_keys = set(item) - allowed_item_keys
        if extra_keys:
            raise ValueError(f"plan item {index} has unsupported fields: {sorted(extra_keys)}")

        step = item.get("step")
        status = item.get("status")
        if not isinstance(step, str) or not step.strip():
            raise ValueError(f"plan item {index} must include a non-empty step")
        if status not in PLAN_STATUSES:
            raise ValueError(f"plan item {index} status must be one of {sorted(PLAN_STATUSES)}")
        if status == "in_progress":
            in_progress_count += 1
        if status != "completed":
            all_completed = False

    if in_progress_count > 1:
        raise ValueError("at most one plan item can be in_progress")
    if plan and not all_completed and in_progress_count != 1:
        raise ValueError("exactly one plan item must be in_progress until all items are completed")


def _format_plan_markdown(plan: list[dict[str, Any]]) -> str:
    markers = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    return "\n".join(f"- {markers.get(item['status'], '[ ]')} {item['step']}" for item in plan)


def _resolve_git_root(project_root: Path, focus_dir: Path) -> Path:
    """Nearest Git repository root for a scoped coding task."""
    return resolve_git_scope_root(project_root, focus_dir)


def _run_git_text(project_root: Path, command: list[str], max_chars: int = 12000) -> dict[str, Any]:
    snapshot = _run_git_snapshot(project_root, command)
    output, truncated = _truncate_text(snapshot.output, max_chars)
    return {
        "command": snapshot.command,
        "exit_code": snapshot.exit_code,
        "output": output,
        "truncated": truncated,
    }


def _review_findings_schema() -> dict[str, Any]:
    return {
        "findings": [
            {
                "title": "[P1] Short actionable issue title",
                "body": "One concise paragraph explaining why this is a bug and when it matters.",
                "confidence_score": 0.0,
                "priority": 1,
                "code_location": {
                    "absolute_file_path": "/absolute/path/to/file",
                    "line_range": {"start": 1, "end": 1},
                },
            }
        ],
        "overall_correctness": "patch is correct | patch is incorrect",
        "overall_explanation": "1-3 sentence explanation.",
        "overall_confidence_score": 0.0,
    }


def _review_rubric() -> str:
    return """Review stance: prioritize discrete, actionable bugs introduced by the change.
Flag issues that affect correctness, security, performance, reliability, or meaningful maintainability.
Do not flag trivial style, speculative risks, pre-existing issues, or intentional behavior changes unless evidence shows a bug.
Every finding must cite the smallest useful changed line range, preferably no more than 5-10 lines.
Use priorities P0-P3. Prefer no findings when there is no issue the author would clearly fix.
Return the exact JSON review schema; do not include markdown fences or extra prose."""


def _resolve_review_target(
    project_root: Path, target: str, base_branch: str | None, commit_sha: str | None, instructions: str | None
) -> dict[str, Any]:
    normalized = target.strip().lower().replace("-", "_")
    if normalized in {"uncommitted", "uncommitted_changes", ""}:
        return {
            "target": "uncommitted_changes",
            "user_facing_hint": "current changes",
            "prompt": "Review the current code changes, including staged, unstaged, and untracked files. Provide prioritized, actionable findings.",
            "diff_commands": [
                ["git", "diff", "--cached"],
                ["git", "diff"],
            ],
            "extra_commands": [
                ["git", "status", "--short"],
                ["git", "ls-files", "--others", "--exclude-standard"],
            ],
        }
    if normalized == "base_branch":
        if not base_branch or not base_branch.strip():
            raise ValueError("base_branch is required when target='base_branch'")
        branch = base_branch.strip()
        merge_base = _run_git_snapshot(project_root, ["merge-base", "HEAD", branch]).output.strip()
        if merge_base:
            prompt = (
                f"Review the code changes against base branch '{branch}'. The merge base is {merge_base}. "
                "Provide prioritized, actionable findings."
            )
            diff_commands = [["git", "diff", merge_base]]
        else:
            prompt = (
                f"Review the code changes against base branch '{branch}'. Determine the merge base if needed, "
                "inspect the merge diff, and provide prioritized, actionable findings."
            )
            diff_commands = [["git", "diff", branch]]
        return {
            "target": "base_branch",
            "user_facing_hint": f"changes against '{branch}'",
            "prompt": prompt,
            "diff_commands": diff_commands,
            "extra_commands": [["git", "status", "--short"]],
        }
    if normalized == "commit":
        if not commit_sha or not commit_sha.strip():
            raise ValueError("commit_sha is required when target='commit'")
        sha = commit_sha.strip()
        title = _run_git_snapshot(project_root, ["show", "-s", "--format=%s", sha]).output.strip()
        prompt = f"Review the code changes introduced by commit {sha}"
        if title:
            prompt += f' ("{title}")'
        prompt += ". Provide prioritized, actionable findings."
        return {
            "target": "commit",
            "user_facing_hint": f"commit {sha[:7]}" + (f": {title}" if title else ""),
            "prompt": prompt,
            "diff_commands": [["git", "show", "--format=medium", "--patch", sha]],
            "extra_commands": [["git", "status", "--short"]],
        }
    if normalized == "custom":
        if not instructions or not instructions.strip():
            raise ValueError("instructions is required when target='custom'")
        return {
            "target": "custom",
            "user_facing_hint": instructions.strip(),
            "prompt": instructions.strip(),
            "diff_commands": [["git", "diff", "--cached"], ["git", "diff"]],
            "extra_commands": [["git", "status", "--short"]],
        }
    raise ValueError("target must be one of: uncommitted, base_branch, commit, custom")


def _relative_path(path: Path, root: Path) -> str:
    """
    Return a POSIX-like project-relative path.

    :param path: absolute path
    :param root: absolute project root
    :return: relative path or absolute path if ``path`` is outside ``root``
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _resolve_focus_dir(project_root: Path, relative_path: str) -> Path:
    """
    Resolve the directory that scopes hierarchical project instructions.

    :param project_root: active Serena project root
    :param relative_path: project-relative file or directory path
    :return: absolute directory inside the project
    """
    focus_path = (project_root / relative_path).resolve()
    try:
        focus_path.relative_to(project_root)
    except ValueError as e:
        raise ValueError(f"Focus path must stay inside the active project: {relative_path}") from e

    if focus_path.is_file():
        return focus_path.parent
    if not focus_path.exists() and focus_path.suffix:
        return focus_path.parent
    return focus_path


def _project_switch_guidance(project_root: Path, relative_path: str) -> ProjectSwitchGuidance:
    """Resolve a likely project root and task-relative retry path for an external focus path."""
    requested_path = (project_root / relative_path).resolve()
    candidate = requested_path
    if requested_path.is_file() or (not requested_path.exists() and requested_path.suffix):
        candidate = requested_path.parent

    switch_root = candidate
    if candidate.is_dir():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=candidate,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0 and result.stdout.strip():
            git_root = Path(result.stdout.strip()).resolve()
            if requested_path == git_root or git_root in requested_path.parents:
                switch_root = git_root

    try:
        retry_path = requested_path.relative_to(switch_root).as_posix() or "."
    except ValueError:
        retry_path = "."
    return ProjectSwitchGuidance(project=str(switch_root), relative_path=retry_path)


def _resolve_coding_task_focus_dir(
    project_root: Path,
    relative_path: str,
    *,
    activate_project_available: bool,
    workspace_id: str | None,
) -> Path:
    """Resolve coding-task focus and provide an actionable project-switch error when possible."""
    try:
        return _resolve_focus_dir(project_root, relative_path)
    except ValueError as error:
        if not activate_project_available:
            raise
        guidance = _project_switch_guidance(project_root, relative_path)
        workspace_argument = f', workspace_id="{workspace_id}"' if workspace_id is not None else ""
        raise ValueError(
            f"Focus path is outside the active project: {(project_root / relative_path).resolve()}.\n"
            f'Call activate_project(project="{guidance.project}"{workspace_argument}), then '
            f'prepare_coding_task(relative_path="{guidance.relative_path}"{workspace_argument}).'
        ) from error


def _instruction_search_dirs(project_root: Path, focus_dir: Path) -> list[Path]:
    """
    Return directories from the project root down to the focus directory.

    :param project_root: active Serena project root
    :param focus_dir: absolute focus directory
    :return: ordered directories following Codex's root-to-CWD rule
    """
    relative_parts = focus_dir.relative_to(project_root).parts
    search_dirs = [project_root]
    cursor = project_root
    for part in relative_parts:
        cursor = cursor / part
        search_dirs.append(cursor)
    return search_dirs


def _project_instruction_settings(codex_home: Path | None = None) -> ProjectInstructionSettings:
    """Load Codex-compatible global project instruction settings."""
    if codex_home is None:
        configured_home = os.environ.get("CODEX_HOME")
        codex_home = Path(configured_home).expanduser() if configured_home else Path.home() / ".codex"

    config = _read_toml_file(codex_home / "config.toml")
    raw_fallbacks = config.get("project_doc_fallback_filenames")
    fallback_filenames: list[str] = []
    if isinstance(raw_fallbacks, list):
        for value in raw_fallbacks:
            if not isinstance(value, str):
                continue
            filename = value.strip()
            if filename and Path(filename).name == filename and filename not in fallback_filenames:
                fallback_filenames.append(filename)

    raw_max_bytes = config.get("project_doc_max_bytes")
    max_bytes = raw_max_bytes if isinstance(raw_max_bytes, int) and raw_max_bytes > 0 else DEFAULT_PROJECT_DOC_MAX_BYTES
    return ProjectInstructionSettings(
        codex_home=codex_home,
        fallback_filenames=tuple(fallback_filenames),
        max_bytes=max_bytes,
    )


def _load_first_instruction_document(
    directory: Path,
    filenames: tuple[str, ...],
    project_root: Path | None,
    remaining_bytes: int,
) -> tuple[ProjectInstructionDocument | None, int]:
    """Load the first non-empty instruction document in precedence order."""
    for filename in filenames:
        candidate = directory / filename
        if not candidate.is_file():
            continue

        data = candidate.read_bytes()
        if not data.strip():
            continue

        truncated = len(data) > remaining_bytes
        if truncated:
            data = data[:remaining_bytes]
        text = data.decode("utf-8", errors="replace")
        relative_path = _relative_path(candidate, project_root) if project_root is not None else str(candidate)
        return (
            ProjectInstructionDocument(
                path=str(candidate),
                relative_path=relative_path,
                contents=text,
                truncated=truncated,
            ),
            len(data),
        )
    return None, 0


def _load_instruction_documents(
    project_root: Path,
    focus_dir: Path,
    max_total_bytes: int,
    settings: ProjectInstructionSettings | None = None,
) -> list[ProjectInstructionDocument]:
    """
    Load Codex-style AGENTS documents for the focused task scope.

    :param project_root: active Serena project root
    :param focus_dir: absolute focus directory
    :param max_total_bytes: maximum bytes to include across all instruction documents
    :return: ordered instruction documents
    """
    if max_total_bytes <= 0:
        return []

    fallback_filenames = settings.fallback_filenames if settings is not None else ()
    candidate_filenames = ("AGENTS.override.md", "AGENTS.md", *fallback_filenames)
    documents: list[ProjectInstructionDocument] = []
    remaining_bytes = max_total_bytes

    if settings is not None:
        global_document, consumed_bytes = _load_first_instruction_document(
            settings.codex_home,
            ("AGENTS.override.md", "AGENTS.md"),
            project_root=None,
            remaining_bytes=remaining_bytes,
        )
        if global_document is not None:
            documents.append(global_document)
            remaining_bytes -= consumed_bytes

    for directory in _instruction_search_dirs(project_root, focus_dir):
        if remaining_bytes <= 0:
            break

        document, consumed_bytes = _load_first_instruction_document(
            directory,
            candidate_filenames,
            project_root=project_root,
            remaining_bytes=remaining_bytes,
        )
        if document is not None:
            documents.append(document)
            remaining_bytes -= consumed_bytes

    return documents


def _run_git_snapshot(project_root: Path, args: list[str]) -> CommandSnapshot:
    """
    Run a bounded git command for the active project.

    :param project_root: active Serena project root
    :param args: git arguments without the ``git`` executable
    :return: command snapshot
    """
    normalized_args = args[1:] if args and args[0] == "git" else args
    command = ["git", *normalized_args]
    try:
        result = subprocess.run(command, cwd=project_root, text=True, capture_output=True, timeout=10, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return CommandSnapshot(command=" ".join(command), exit_code=None, output=str(e))

    output = result.stdout.strip()
    stderr = result.stderr.strip()
    if stderr:
        output = f"{output}\n{stderr}".strip()
    if len(output) > 12000:
        output = output[:12000] + "\n...[truncated]"

    return CommandSnapshot(command=" ".join(command), exit_code=result.returncode, output=output)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_toml_file(path: Path) -> dict[str, Any]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _detect_node_package_manager(project_root: Path, package_files: list[str], managers: list[str]) -> str:
    if (project_root / "pnpm-lock.yaml").is_file():
        _append_unique(package_files, "pnpm-lock.yaml")
        _append_unique(managers, "pnpm")
        return "pnpm"
    if (project_root / "yarn.lock").is_file():
        _append_unique(package_files, "yarn.lock")
        _append_unique(managers, "yarn")
        return "yarn"
    if (project_root / "package-lock.json").is_file():
        _append_unique(package_files, "package-lock.json")
        _append_unique(managers, "npm")
        return "npm"
    _append_unique(managers, "npm")
    return "npm"


def _node_run_command(package_manager: str, script: str) -> str:
    if package_manager == "yarn":
        return f"yarn {script}"
    if package_manager == "pnpm":
        return f"pnpm {script}"
    return f"npm run {script}" if script not in {"test"} else "npm test"


def _infer_validation_hints(project_root: Path) -> ValidationHints:
    """
    Infer likely validation commands from the universal task catalog.

    :param project_root: active Serena project root
    :return: likely validation commands and source files
    """
    return infer_validation_hints(project_root)


def _compact_task_dict(task: Any) -> dict[str, Any]:
    """
    Return the minimal task fields an agent needs to choose or run a task.

    :param task: discovered project task
    :return: compact task dictionary
    """
    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "command": task.command,
        "workdir": task.workdir,
        "long_running": task.long_running,
    }


def _task_catalog_agent_view(
    full_catalog: TaskCatalog,
    returned_catalog: TaskCatalog,
    *,
    include_details: bool = False,
) -> dict[str, Any]:
    """
    Return a compact task catalog view for model-facing responses.

    :param full_catalog: complete discovered task catalog
    :param returned_catalog: filtered catalog whose tasks may be returned
    :param include_details: whether full package file and validation detail should be included
    :return: model-facing task catalog view
    """
    compact_summary = full_catalog.summary()
    task_count_value = compact_summary.get("task_count", 0)
    task_count = task_count_value if isinstance(task_count_value, int) else 0
    omitted_task_count = max(0, task_count - len(returned_catalog.tasks))

    view: dict[str, Any] = {
        "summary": {
            **compact_summary,
            "returned_task_count": len(returned_catalog.tasks),
            "omitted_task_count": omitted_task_count,
        },
        "top_tasks": [_compact_task_dict(task) for task in returned_catalog.tasks],
        "details_available": bool(omitted_task_count or full_catalog.package_files),
    }

    if include_details:
        view["catalog"] = asdict(returned_catalog)

    return view


def _normalize_validation_id(validation_id: str) -> str:
    """
    Normalize validation shortcuts to task catalog kinds.

    :param validation_id: user-facing validation shortcut or task id
    :return: normalized shortcut
    """
    normalized = validation_id.strip().lower().replace("_", "-")
    aliases = {
        "type-check": "typecheck",
        "type": "typecheck",
        "types": "typecheck",
        "tests": "test",
        "unit": "test",
        "unit-test": "test",
        "unit-tests": "test",
        "fmt": "format",
        "format-check": "format",
        "check-format": "format",
        "checks": "check",
        "verify-all": "verify",
    }
    return aliases.get(normalized, normalized)


def _select_validation_task(catalog: TaskCatalog, validation_id: str) -> Any | None:
    """
    Select the best matching validation task from a catalog.

    :param catalog: filtered task catalog
    :param validation_id: validation kind, alias, command name, or exact task id
    :return: matching task or None
    """
    normalized = _normalize_validation_id(validation_id)
    if not normalized:
        return None

    for task in catalog.tasks:
        if task.task_id == validation_id or task.task_id == normalized:
            return task

    for task in catalog.tasks:
        if task.kind == normalized:
            return task

    for task in catalog.tasks:
        task_name = task.task_id.rsplit(":", 1)[-1].lower().replace("_", "-")
        if task.kind in VALIDATION_KINDS and task_name == normalized:
            return task

    for task in catalog.tasks:
        if task.kind in VALIDATION_KINDS and normalized in task.task_id.lower():
            return task

    return None


def _validation_file_args(project_root: Path, relative_path: str, files: list[str] | None) -> list[str]:
    """Validate project-relative focused validation file arguments."""
    if not files:
        return []
    scope_root = (project_root / relative_path).resolve()
    if scope_root.is_file():
        scope_root = scope_root.parent
    if scope_root != project_root and project_root not in scope_root.parents:
        raise ValueError(f"Validation scope escapes project root: {relative_path}")

    result: list[str] = []
    for file_path in files:
        raw_path = Path(file_path)
        if raw_path.is_absolute():
            raise ValueError(f"Focused validation files must be project-relative: {file_path}")
        resolved = (project_root / raw_path).resolve()
        if resolved != project_root and project_root not in resolved.parents:
            raise ValueError(f"Focused validation file escapes project root: {file_path}")
        if scope_root != project_root and resolved != scope_root and scope_root not in resolved.parents:
            raise ValueError(f"Focused validation file is outside relative_path scope: {file_path}")
        result.append(resolved.relative_to(project_root).as_posix())
    return result


def _replace_dot_arg(parts: list[str], file_args: list[str]) -> list[str]:
    """Replace the final dot argument with focused file args when present."""
    if parts and parts[-1] == ".":
        return [*parts[:-1], *file_args]
    return [*parts, *file_args]


def _focused_validation_command(task: Any, file_args: list[str]) -> str | None:
    """Return a focused validation command for safe, known runners."""
    if not file_args:
        return None

    parts = shlex.split(task.command)
    if not parts:
        return None

    joined = " ".join(parts)
    if "ruff check" in joined:
        return " ".join(shlex.quote(part) for part in _replace_dot_arg(parts, file_args))
    if "ruff format" in joined:
        return " ".join(shlex.quote(part) for part in _replace_dot_arg(parts, file_args))
    if "pytest" in parts:
        return " ".join(shlex.quote(part) for part in _replace_dot_arg(parts, file_args))

    return None


def _select_focused_validation_task(catalog: TaskCatalog, validation_id: str, file_args: list[str]) -> tuple[Any | None, str | None]:
    """Select a validation task, preferring focus-capable commands when files are provided."""
    selected = _select_validation_task(catalog, validation_id)
    if not file_args:
        return selected, None

    normalized = _normalize_validation_id(validation_id)
    candidates = [task for task in catalog.tasks if task.kind == normalized or task.task_id == validation_id]
    if selected is not None and selected not in candidates:
        candidates.append(selected)

    for candidate in candidates:
        command = _focused_validation_command(candidate, file_args)
        if command is not None:
            return candidate, command

    return selected, None


def _task_service_status(task: Any, output: str) -> dict[str, Any]:
    """Return service readiness metadata for long-running tasks."""
    if not getattr(task, "ready_pattern", None):
        return {}
    return {
        "ready_pattern": task.ready_pattern,
        "ready": task.ready_pattern in output,
    }


def _parse_validation_diagnostics(output: str, max_items: int = 20) -> list[dict[str, Any]]:
    """Parse common validation output into compact diagnostics."""
    diagnostics: list[dict[str, Any]] = []
    for line in output.splitlines():
        flutter_machine = re.match(
            r"^(ERROR|WARNING|INFO)\|[^|]*\|[^|]*\|([^|]+)\|(\d+)\|(\d+)\|\d+\|(.+)$",
            line,
        )
        flutter_human = re.match(
            r"^\s*(error|warning|info)\s+[•-]\s+(.+?)\s+[•-]\s+(.+?):(\d+):(\d+)\s+[•-]\s+(\S+)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        dart_human = re.match(
            r"^\s*(error|warning|info)\s+-\s+(.+?):(\d+):(\d+)\s+-\s+(.+?)(?:\s+-\s+(\S+))?\s*$",
            line,
            flags=re.IGNORECASE,
        )

        if flutter_machine is not None:
            severity, path, line_number, column, message = flutter_machine.groups()
            diagnostics.append(
                {
                    "tool": "dart",
                    "severity": severity.lower(),
                    "path": path,
                    "line": int(line_number),
                    "column": int(column),
                    "message": message.strip(),
                }
            )
        elif flutter_human is not None:
            severity, message, path, line_number, column, code = flutter_human.groups()
            diagnostics.append(
                {
                    "tool": "flutter",
                    "severity": severity.lower(),
                    "path": path,
                    "line": int(line_number),
                    "column": int(column),
                    "code": code,
                    "message": message.strip(),
                }
            )
        elif dart_human is not None:
            severity, path, line_number, column, message, code = dart_human.groups()
            diagnostics.append(
                {
                    "tool": "dart",
                    "severity": severity.lower(),
                    "path": path,
                    "line": int(line_number),
                    "column": int(column),
                    "code": code,
                    "message": message.strip(),
                }
            )
        elif line.startswith("FAILED "):
            parts = line.split(" ", 2)
            diagnostics.append({"tool": "pytest", "path": parts[1] if len(parts) > 1 else "", "message": line})
        else:
            parts = line.split(":", 3)
            if len(parts) == 4 and parts[1].isdigit() and parts[2].isdigit():
                diagnostics.append(
                    {
                        "path": parts[0],
                        "line": int(parts[1]),
                        "column": int(parts[2]),
                        "message": parts[3].strip(),
                    }
                )
        if len(diagnostics) >= max_items:
            break
    return diagnostics


def _edit_policy_contract() -> dict[str, Any]:
    """
    Return the compact edit-tool policy agents should follow.

    :return: ordered edit policy contract
    """
    return {
        "default_order": [
            "inspect before editing unfamiliar code",
            "symbol tools for symbol-aware changes",
            "replace_content for exact small text edits with allow_multiple_occurrences=false",
            "apply_patch for atomic multi-file or structured textual patches",
        ],
        "apply_patch": {
            "role": "atomic multi-file or structured textual edit tool, not the default small-edit tool",
            "dry_run": "use dry_run=true first for complex or risky patches",
            "failure_contract": "on failure no patch changes are written and changes=[]",
        },
        "replace_content": {
            "role": "exact small text replacement",
            "safety": "keep allow_multiple_occurrences=false unless every match was inspected",
            "precheck": "use search_for_pattern before broad regex or multi-occurrence edits",
        },
        "semantic_tools": ["replace_symbol_body", "insert_before_symbol", "insert_after_symbol", "rename_symbol", "safe_delete_symbol"],
        "guardrails": [
            "preserve unrelated user changes",
            "do not overwrite whole files when a smaller edit is enough",
            "reject ambiguous edits and inspect before retrying",
        ],
    }


def _compact_validation_hints(catalog: TaskCatalog, max_commands: int = 12) -> ValidationHints:
    """
    Return validation hints without manifest path noise.

    :param catalog: filtered task catalog used for model-facing validation hints
    :param max_commands: maximum likely commands to include
    :return: compact validation hints
    """
    commands = [task.command for task in catalog.tasks if task.kind in VALIDATION_KINDS]
    return ValidationHints(
        package_files=[],
        detected_package_managers=catalog.detected_package_managers,
        likely_commands=commands[:max_commands],
    )


def _active_serena_state(agent: Any) -> tuple[str, list[str], list[str]]:
    context = getattr(agent.get_context(), "name", str(agent.get_context()))
    active_modes = agent.get_active_modes()
    mode_names = active_modes.get_mode_names() if hasattr(active_modes, "get_mode_names") else []
    return context, list(mode_names), sorted(agent.get_active_tool_names())


APPLY_PATCH_BEGIN = "*** Begin Patch"
APPLY_PATCH_END = "*** End Patch"
APPLY_PATCH_ADD = "*** Add File: "
APPLY_PATCH_DELETE = "*** Delete File: "
APPLY_PATCH_UPDATE = "*** Update File: "
APPLY_PATCH_MOVE = "*** Move to: "


@dataclass(frozen=True)
class ApplyPatchChangeSummary:
    """Summary of a file operation produced by ``apply_patch``."""

    operation: str
    path: str
    move_path: str | None = None
    line_count: int | None = None


@dataclass(frozen=True)
class ApplyPatchResult:
    """Result returned after parsing and optionally applying a Codex-style patch."""

    success: bool
    dry_run: bool
    atomic: bool
    summary: str
    changes: list[ApplyPatchChangeSummary]
    errors: list[str]
    would_change: list[str]


@dataclass(frozen=True)
class _ParsedPatchOperation:
    """Parsed patch operation before filesystem application."""

    operation: str
    path: str
    move_path: str | None
    lines: list[str]


def _strip_apply_patch_heredoc(patch: str) -> str:
    """Normalized patch body with optional heredoc wrapper removed."""
    lines = patch.strip().splitlines()
    if len(lines) >= 4 and lines[0] in {"<<EOF", "<<'EOF'", '<<"EOF"'} and lines[-1].endswith("EOF"):
        return "\n".join(lines[1:-1]).strip()
    return patch.strip()


def _parse_apply_patch_operations(patch: str) -> list[_ParsedPatchOperation]:
    """Parsed Codex-style patch operations.

    The accepted envelope follows Codex's Add/Delete/Update/Move patch grammar.
    The parser is intentionally small and strict enough to return actionable
    errors before any filesystem changes are attempted.
    """
    patch = _strip_apply_patch_heredoc(patch)
    lines = patch.splitlines()

    if not lines or lines[0].strip() != APPLY_PATCH_BEGIN:
        raise ValueError("The first line of the patch must be '*** Begin Patch'")
    if lines[-1].strip() != APPLY_PATCH_END:
        raise ValueError("The last line of the patch must be '*** End Patch'")

    operations: list[_ParsedPatchOperation] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith(APPLY_PATCH_ADD):
            file_path = line.removeprefix(APPLY_PATCH_ADD).strip()
            index += 1
            payload: list[str] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                if not lines[index].startswith("+"):
                    raise ValueError(f"Add File lines must start with '+': {file_path}")
                payload.append(lines[index][1:])
                index += 1
            operations.append(_ParsedPatchOperation("add", file_path, None, payload))
            continue

        if line.startswith(APPLY_PATCH_DELETE):
            file_path = line.removeprefix(APPLY_PATCH_DELETE).strip()
            operations.append(_ParsedPatchOperation("delete", file_path, None, []))
            index += 1
            continue

        if line.startswith(APPLY_PATCH_UPDATE):
            file_path = line.removeprefix(APPLY_PATCH_UPDATE).strip()
            index += 1
            move_path: str | None = None
            if index < len(lines) - 1 and lines[index].startswith(APPLY_PATCH_MOVE):
                move_path = lines[index].removeprefix(APPLY_PATCH_MOVE).strip()
                index += 1

            payload = []
            while (
                index < len(lines) - 1
                and not lines[index].startswith("*** Add File: ")
                and not lines[index].startswith("*** Delete File: ")
                and not lines[index].startswith("*** Update File: ")
            ):
                payload.append(lines[index])
                index += 1
            operations.append(_ParsedPatchOperation("update", file_path, move_path, payload))
            continue

        raise ValueError(f"Unsupported patch header on line {index + 1}: {line}")

    if not operations:
        raise ValueError("No files were modified.")
    return operations


def _resolve_patch_path(project_root: Path, relative_workdir: str, patch_path: str) -> Path:
    """Resolved project-local path from a patch file reference."""
    raw_path = Path(patch_path)
    if raw_path.is_absolute():
        raise ValueError(f"Patch paths must be relative, got absolute path: {patch_path}")

    workdir = (project_root / relative_workdir).resolve()
    candidate = (workdir / raw_path).resolve()
    if candidate != project_root and project_root not in candidate.parents:
        raise ValueError(f"Patch path escapes the active project: {patch_path}")
    return candidate


def _relative_patch_path(project_root: Path, path: Path) -> str:
    """Project-relative path spelling for JSON responses."""
    return path.relative_to(project_root).as_posix()


def _split_patch_update_hunks(lines: list[str]) -> list[list[str]]:
    """Split an Update File payload into hunk bodies."""
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("@@"):
            if current is not None:
                hunks.append(current)
            current = []
            continue
        if line == "*** End of File":
            continue
        if current is None:
            raise ValueError("Update File hunks must start with '@@'")
        if not line or line[0] not in {" ", "-", "+"}:
            raise ValueError(f"Invalid hunk line: {line}")
        current.append(line)
    if current is not None:
        hunks.append(current)
    if not hunks:
        raise ValueError("Update File operation must contain at least one hunk.")
    return hunks


def _find_unique_subsequence(haystack: list[str], needle: list[str]) -> int:
    """Unique location of a hunk's old lines inside file content."""
    if not needle:
        return len(haystack)

    matches = []
    limit = len(haystack) - len(needle) + 1
    for index in range(max(0, limit)):
        if haystack[index : index + len(needle)] == needle:
            matches.append(index)
            if len(matches) > 1:
                break

    if not matches:
        raise ValueError("Patch hunk did not match file content.")
    if len(matches) > 1:
        raise ValueError("Patch hunk matched multiple locations; add more context.")
    return matches[0]


def _apply_patch_update_text(original: str, payload_lines: list[str]) -> str:
    """Updated file text after applying parsed Update File hunks."""
    content = original.splitlines()
    trailing_newline = original.endswith("\n")

    for hunk in _split_patch_update_hunks(payload_lines):
        old_lines = [line[1:] for line in hunk if line.startswith((" ", "-"))]
        new_lines = [line[1:] for line in hunk if line.startswith((" ", "+"))]
        start = _find_unique_subsequence(content, old_lines)
        content[start : start + len(old_lines)] = new_lines

    updated = "\n".join(content)
    if trailing_newline or original == "":
        updated += "\n"
    return updated


def _apply_parsed_patch_operation(
    project_root: Path,
    relative_workdir: str,
    operation: _ParsedPatchOperation,
    dry_run: bool,
) -> ApplyPatchChangeSummary:
    """Apply one parsed patch operation to the active project filesystem."""
    target = _resolve_patch_path(project_root, relative_workdir, operation.path)
    relative_target = _relative_patch_path(project_root, target)

    if operation.operation == "add":
        if target.exists():
            raise ValueError(f"Cannot add file that already exists: {relative_target}")
        content = "\n".join(operation.lines)
        if operation.lines:
            content += "\n"
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return ApplyPatchChangeSummary("add", relative_target, line_count=len(operation.lines))

    if operation.operation == "delete":
        if not target.exists():
            raise ValueError(f"Cannot delete missing file: {relative_target}")
        content = target.read_text(encoding="utf-8")
        if not dry_run:
            target.unlink()
        return ApplyPatchChangeSummary("delete", relative_target, line_count=len(content.splitlines()))

    if operation.operation == "update":
        if not target.exists():
            raise ValueError(f"Cannot update missing file: {relative_target}")
        original = target.read_text(encoding="utf-8")
        updated = _apply_patch_update_text(original, operation.lines)
        move_target = None
        if operation.move_path is not None:
            move_target = _resolve_patch_path(project_root, relative_workdir, operation.move_path)
            if move_target.exists() and move_target != target:
                raise ValueError(f"Cannot move to existing file: {_relative_patch_path(project_root, move_target)}")

        if not dry_run:
            output_target = move_target or target
            output_target.parent.mkdir(parents=True, exist_ok=True)
            output_target.write_text(updated, encoding="utf-8")
            if move_target is not None and move_target != target:
                target.unlink()

        return ApplyPatchChangeSummary(
            "update",
            relative_target,
            move_path=_relative_patch_path(project_root, move_target) if move_target is not None else None,
            line_count=len(updated.splitlines()),
        )

    raise ValueError(f"Unsupported patch operation: {operation.operation}")


def _apply_codex_style_patch(project_root: Path, relative_workdir: str, patch: str, dry_run: bool) -> ApplyPatchResult:
    """Apply a Codex-style patch as an all-or-nothing transaction."""
    changes: list[ApplyPatchChangeSummary] = []
    staged_content: dict[Path, str | None] = {}

    def exists_in_stage(path: Path) -> bool:
        if path in staged_content:
            return staged_content[path] is not None
        return path.exists()

    def read_from_stage(path: Path) -> str:
        if path in staged_content:
            content = staged_content[path]
            if content is None:
                raise ValueError(f"Cannot read deleted file: {_relative_patch_path(project_root, path)}")
            return content
        return path.read_text(encoding="utf-8")

    def stage_operation(operation: _ParsedPatchOperation) -> ApplyPatchChangeSummary:
        target = _resolve_patch_path(project_root, relative_workdir, operation.path)
        relative_target = _relative_patch_path(project_root, target)

        if operation.operation == "add":
            if exists_in_stage(target):
                raise ValueError(f"Cannot add file that already exists: {relative_target}")
            content = "\n".join(operation.lines)
            if operation.lines:
                content += "\n"
            staged_content[target] = content
            return ApplyPatchChangeSummary("add", relative_target, line_count=len(operation.lines))

        if operation.operation == "delete":
            if not exists_in_stage(target):
                raise ValueError(f"Cannot delete missing file: {relative_target}")
            content = read_from_stage(target)
            staged_content[target] = None
            return ApplyPatchChangeSummary("delete", relative_target, line_count=len(content.splitlines()))

        if operation.operation == "update":
            if not exists_in_stage(target):
                raise ValueError(f"Cannot update missing file: {relative_target}")
            original = read_from_stage(target)
            updated = _apply_patch_update_text(original, operation.lines)
            move_target = None
            if operation.move_path is not None:
                move_target = _resolve_patch_path(project_root, relative_workdir, operation.move_path)
                if move_target != target and exists_in_stage(move_target):
                    raise ValueError(f"Cannot move to existing file: {_relative_patch_path(project_root, move_target)}")

            if move_target is not None and move_target != target:
                staged_content[target] = None
                staged_content[move_target] = updated
            else:
                staged_content[target] = updated

            return ApplyPatchChangeSummary(
                "update",
                relative_target,
                move_path=_relative_patch_path(project_root, move_target) if move_target is not None else None,
                line_count=len(updated.splitlines()),
            )

        raise ValueError(f"Unsupported patch operation: {operation.operation}")

    try:
        operations = _parse_apply_patch_operations(patch)
        for operation in operations:
            changes.append(stage_operation(operation))
    except Exception as error:
        return ApplyPatchResult(
            success=False,
            dry_run=dry_run,
            atomic=True,
            summary=f"patch failed before writing: {error}",
            changes=[],
            errors=[str(error)],
            would_change=sorted({_relative_patch_path(project_root, path) for path in staged_content}),
        )

    would_change = sorted({_relative_patch_path(project_root, path) for path in staged_content})

    if not dry_run:
        snapshots: dict[Path, bytes | None] = {}
        for path in staged_content:
            snapshots[path] = path.read_bytes() if path.exists() else None
        try:
            for path, content in staged_content.items():
                if content is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
            for path, content in staged_content.items():
                if content is None and path.exists():
                    path.unlink()
        except Exception as error:
            for path, original_content in snapshots.items():
                if original_content is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(original_content)
            return ApplyPatchResult(
                success=False,
                dry_run=dry_run,
                atomic=True,
                summary=f"patch failed while writing and was rolled back: {error}",
                changes=[],
                errors=[str(error)],
                would_change=would_change,
            )

    return ApplyPatchResult(
        success=True,
        dry_run=dry_run,
        atomic=True,
        summary=f"patch {'validated' if dry_run else 'applied'} atomically: {len(changes)} file operation(s)",
        changes=changes,
        errors=[],
        would_change=would_change,
    )


class ApplyPatchTool(Tool, ToolMarkerCanEdit):
    """
    Applies Codex-style file patches inside the active project.
    """

    def apply(self, patch: str, relative_workdir: str = ".", dry_run: bool = False, max_answer_chars: int = -1) -> str:
        """
        Apply or validate a Codex-style Add/Delete/Update/Move patch.

        :param patch: patch body enclosed by ``*** Begin Patch`` and ``*** End Patch``
        :param relative_workdir: project-relative directory used to resolve patch paths
        :param dry_run: validate and summarize the patch without writing files
        :param max_answer_chars: maximum JSON response length; ``-1`` uses Serena's default response limit
        :return: JSON patch result with success, changed files, and errors
        """
        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()

        workdir = _resolve_focus_dir(project_root, relative_workdir)
        result = _apply_codex_style_patch(project_root, _relative_path(workdir, project_root), patch, dry_run)
        response = json.dumps(asdict(result), ensure_ascii=False, indent=2)

        if max_answer_chars >= 0:
            output, _ = _truncate_text(response, max_answer_chars)
            return output
        return response


class PrepareCodingTaskTool(Tool):
    """
    Prepares a Codex-style coding task snapshot before editing code.
    """

    def apply(self, relative_path: str = ".", max_instruction_bytes: int | None = None) -> str:
        """
        Prepare a coding task snapshot with scoped project instructions, git state, active Serena context, and likely validation commands.

        :param relative_path: project-relative file or directory path that scopes AGENTS.md discovery
        :param max_instruction_bytes: maximum bytes across global and project AGENTS documents; defaults to Codex config or 32 KiB
        :return: JSON snapshot for planning the coding task
        """
        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        workspace_id = self.agent.get_workspace_id()
        state_store = self.agent.get_harness_state_store()

        focus_dir = _resolve_coding_task_focus_dir(
            project_root,
            relative_path,
            activate_project_available=self.agent.tool_is_exposed("activate_project"),
            workspace_id=workspace_id,
        )
        focus_path = _relative_path(focus_dir, project_root)
        git_root = _resolve_git_root(project_root, focus_dir)
        instruction_settings = _project_instruction_settings()
        instruction_budget = instruction_settings.max_bytes if max_instruction_bytes is None else max_instruction_bytes

        instruction_documents = _load_instruction_documents(
            project_root=git_root,
            focus_dir=focus_dir,
            max_total_bytes=instruction_budget,
            settings=instruction_settings,
        )

        context, modes, active_tools = _active_serena_state(self.agent)
        full_task_catalog = discover_task_catalog(project_root, relative_path=relative_path)
        task_catalog = full_task_catalog.filtered(include_internal=False, max_tasks=15)
        skills_outcome = discover_skills(git_root, focus_dir)
        skills_summary = render_skills_summary(skills_outcome.skills)
        skill_dependency_report = skill_dependency_reports(skills_outcome.skills, set(self.agent.get_active_tool_names()))

        now = _utc_now()
        _save_coding_task_context(
            project_root,
            {
                "project_root": str(project_root),
                "active_project_name": active_project.project_name,
                "relative_path": relative_path,
                "focus_path": focus_path,
                "git_root": str(git_root),
                "created_at": now,
                "updated_at": now,
            },
            workspace_id,
            state_store,
        )

        snapshot = CodingTaskSnapshot(
            project_name=active_project.project_name,
            project_root=str(project_root),
            focus_path=focus_path,
            git_root=str(git_root),
            context=context,
            modes=modes,
            active_tools=active_tools,
            instruction_documents=instruction_documents,
            edit_policy=_edit_policy_contract(),
            validation_hints=_compact_validation_hints(task_catalog),
            task_catalog_summary=full_task_catalog.summary(),
            task_catalog=_task_catalog_agent_view(full_task_catalog, task_catalog, include_details=False),
            active_goal=_compact_goal_public_state(project_root, workspace_id, state_store),
            active_plan=_compact_plan_public_state(project_root, workspace_id, state_store),
            available_skills=[skill.to_public_dict() for skill in skills_outcome.skills],
            skills_summary=skills_summary,
            skill_dependency_report=skill_dependency_report,
            git_status=_run_git_snapshot(git_root, ["status", "--short"]),
            git_diff_stat=_run_git_snapshot(git_root, ["diff", "--stat"]),
            git_diff_cached_stat=_run_git_snapshot(git_root, ["diff", "--cached", "--stat"]),
        )

        return json.dumps(asdict(snapshot), ensure_ascii=False, indent=2)


class GetCodingHarnessInstructionsTool(Tool):
    """
    Provides a coding workflow contract.
    """

    def apply(self) -> str:
        """
        Return a JSON workflow contract.

        :return: JSON workflow contract
        """
        contract = {
            "objective": "Make ChatGPT use Serena like a disciplined Codex-style coding harness while Serena remains the MCP server.",
            "task_start": {
                "first_step": "Call prepare_coding_task with the most relevant file or directory path.",
                "goal": "For explicitly requested multi-turn goals, use create_goal/get_goal. Do not infer goals for ordinary small tasks.",
                "instructions": "Treat returned AGENTS.md and AGENTS.override.md contents as scoped project instructions.",
                "dirty_tree": "Preserve existing user changes and inspect git status/diff before editing touched areas.",
            },
            "tool_use": {
                "plan": (
                    "Use update_plan as a lightweight checklist for meaningful multi-step work. "
                    "It must not switch Serena modes or disable tools; after calling it, briefly reflect the current plan in chat."
                ),
                "code_understanding": "Inspect first with Serena search/symbol tools before editing unfamiliar code.",
                "edits": (
                    "Keep changes scoped. Use semantic tools for symbol-aware edits, replace_content for exact small text edits, "
                    "and apply_patch only for atomic multi-file or structured textual patches. "
                    "Reject ambiguous edits and do not overwrite unknown user changes."
                ),
                "edit_policy": _edit_policy_contract(),
                "terminal": (
                    "Use exec_command for commands; set tty=true for interactive stdin/REPL/prompt workflows. "
                    "Use write_stdin for stdin, terminal_status for non-consuming status checks, "
                    "and send_terminal_signal for SIGINT/SIGTERM/SIGKILL instead of raw control-character hacks. "
                    "Keep write_stdin chars literal and use semantic keys such as ENTER for terminal actions."
                ),
                "flow": "Inspect → plan when useful → edit minimally → validate focused → inspect failures → fix task-related issues → finalize.",
                "output": "Keep command output bounded; use log_path for full logs.",
                "sessions": "Use list_terminal_sessions, terminal_status, send_terminal_signal, and stop_terminal_session to account for or control background processes.",
            },
            "validation": {
                "source": "Use run_validation for common checks like lint, test, typecheck, format, build, or verify; use get_validation_commands/discover_project_tasks only when choosing is ambiguous.",
                "behavior": "Run validation when practical. If blocked by missing tools or environment setup, report the blocker precisely.",
            },
            "skills": {
                "discovery": "prepare_coding_task and discover_skills expose only skill metadata and source path, not full instructions.",
                "progressive_disclosure": "Before following a skill, call read_skill with the skill name or source path.",
                "policy": "Respect skill policy, resources, and dependency_report before use.",
            },
            "review": {
                "start": "Use prepare_review_task for review requests, then inspect the returned diff/context and produce prioritized findings.",
                "finish": "Use finalize_review_task to wrap review output when the user initiated a review workflow.",
            },
            "finalization": {
                "goal_status": "Use update_goal only to mark an existing goal complete or genuinely blocked.",
                "completion_audit": "Mark complete only when current evidence proves the full objective is achieved.",
                "blocked_audit": "Mark blocked only after the same blocker repeats for at least three consecutive goal turns and no meaningful progress is possible.",
                "before_final": "Call finalize_coding_task after edits and validation attempts.",
                "final_answer": ["what changed", "changed files", "validation results", "running sessions/services", "remaining risks"],
            },
            "terminal_tools": [
                "exec_command",
                "write_stdin",
                "terminal_status",
                "send_terminal_signal",
                "list_terminal_sessions",
                "stop_terminal_session",
            ],
            "workflow_tools": [
                "update_plan",
                "get_goal",
                "create_goal",
                "update_goal",
                "record_goal_progress",
                "apply_patch",
                "prepare_coding_task",
                "prepare_review_task",
                "finalize_review_task",
                "get_validation_commands",
                "run_validation",
                "discover_skills",
                "read_skill",
                "finalize_coding_task",
            ],
            "final_response_contract": ["what changed", "changed files", "validation", "running services", "risks"],
        }
        return json.dumps(contract, ensure_ascii=False, indent=2)


class FinalizeCodingTaskTool(Tool):
    """
    Produces a final coding task snapshot.
    """

    def apply(
        self,
        verification_results: str | None = None,
        remaining_risks: str | None = None,
        relative_path: str | None = None,
    ) -> str:
        """
        Return final git state for completing a coding task.

        :param verification_results: optional concise summary of validation commands and their results
        :param remaining_risks: optional concise risk summary for the final response
        :param relative_path: project-relative file or directory used to choose the Git root; defaults to the latest prepare_coding_task focus
        :return: JSON final coding task snapshot
        """
        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        workspace_id = self.agent.get_workspace_id()
        state_store = self.agent.get_harness_state_store()
        context = _load_coding_task_context(project_root, workspace_id, state_store) if relative_path is None else None
        resolved_relative_path = relative_path or str((context or {}).get("focus_path") or ".")
        focus_dir = _resolve_focus_dir(project_root, resolved_relative_path)
        focus_path = _relative_path(focus_dir, project_root)
        git_root = _resolve_git_root(project_root, focus_dir)
        snapshot = {
            "project_name": active_project.project_name,
            "project_root": str(project_root),
            "focus_path": focus_path,
            "git_root": str(git_root),
            "active_goal": _goal_public_state(project_root, workspace_id, state_store),
            "active_plan": _plan_public_state(project_root, workspace_id, state_store),
            "git_status": asdict(_run_git_snapshot(git_root, ["status", "--short"])),
            "git_diff_stat": asdict(_run_git_snapshot(git_root, ["diff", "--stat"])),
            "git_diff_cached_stat": asdict(_run_git_snapshot(git_root, ["diff", "--cached", "--stat"])),
            "git_diff_names": asdict(_run_git_snapshot(git_root, ["diff", "--name-only"])),
            "git_diff_cached_names": asdict(_run_git_snapshot(git_root, ["diff", "--cached", "--name-only"])),
            "validation_results": verification_results,
            "remaining_risks": remaining_risks,
            "final_response_contract": ["result", "changed_files", "validation", "risks"],
        }
        return json.dumps(snapshot, ensure_ascii=False, indent=2)


class GetValidationCommandsTool(Tool):
    """
    Returns inferred project validation commands.
    """

    def apply(self, include_internal: bool = False, max_tasks: int = 30, relative_path: str = ".") -> str:
        """
        Return likely validation commands for the active project.

        :param include_internal: whether internal helper tasks should be included
        :param max_tasks: maximum number of validation tasks to consider
        :param relative_path: project-relative file or directory used to scope task discovery
        :return: JSON validation command hints
        """
        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        catalog = discover_task_catalog(project_root, relative_path=relative_path).filtered(
            include_internal=include_internal,
            max_tasks=max_tasks,
        )
        return json.dumps(asdict(catalog.validation_hints), ensure_ascii=False, indent=2)


class DiscoverProjectTasksTool(Tool):
    """
    Discovers runnable project tasks without requiring semantic indexing.
    """

    def apply(
        self,
        max_depth: int = 5,
        include_internal: bool = False,
        include_fixtures: bool = False,
        include_examples: bool = False,
        include_ignored: bool = False,
        max_tasks: int = 30,
        include_details: bool = False,
        relative_path: str = ".",
    ) -> str:
        """
        Return the universal project task catalog.

        :param max_depth: maximum directory depth for manifest discovery
        :param include_internal: whether internal helper tasks should be included
        :param include_fixtures: whether fixture and test-resource manifests should be included
        :param include_examples: whether example, sample, and demo manifests should be included
        :param include_ignored: whether generated/dependency/cache directories should be scanned
        :param max_tasks: maximum number of tasks to include in the task list; set to 0 for none
        :param include_details: whether full package files and validation hints should be returned
        :param relative_path: project-relative file or directory used to scope task discovery
        :return: JSON project task catalog
        """
        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        catalog = discover_task_catalog(
            project_root,
            max_depth=max_depth,
            include_fixtures=include_fixtures,
            include_examples=include_examples,
            include_ignored=include_ignored,
            relative_path=relative_path,
        )
        visible_catalog = catalog.filtered(include_internal=include_internal, max_tasks=max_tasks)
        response = _task_catalog_agent_view(
            catalog,
            visible_catalog,
            include_details=include_details,
        )
        response["filtered"] = {
            "include_internal": include_internal,
            "include_fixtures": include_fixtures,
            "include_examples": include_examples,
            "include_ignored": include_ignored,
            "max_tasks": max_tasks,
            "include_details": include_details,
            "relative_path": relative_path,
        }
        return json.dumps(response, ensure_ascii=False, indent=2)


class RunTaskTool(Tool, ToolMarkerCanEdit):
    """
    Runs a discovered project task by task_id instead of raw shell text.
    """

    def apply(
        self,
        task_id: str,
        relative_path: str = ".",
        yield_time_ms: int = 10000,
        max_output_tokens: int | None = None,
        tty: bool | None = None,
    ) -> str:
        """
        Run a task from the discovered task catalog.

        :param task_id: task identifier returned by discover_project_tasks
        :param relative_path: project-relative file or directory used to scope task lookup
        :param yield_time_ms: wait before yielding output
        :param max_output_tokens: approximate output budget
        :param tty: override whether to run through PTY; defaults to task metadata
        :return: JSON terminal response
        """
        from serena.tools.cmd_tools import _json_response

        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        catalog = discover_task_catalog(project_root, relative_path=relative_path)
        task = next((candidate for candidate in catalog.tasks if candidate.task_id == task_id), None)
        if task is None:
            catalog = discover_task_catalog(
                project_root,
                include_fixtures=True,
                include_examples=True,
                relative_path=relative_path,
            )
            task = next((candidate for candidate in catalog.tasks if candidate.task_id == task_id), None)
        if task is None:
            visible_catalog = catalog.filtered(include_internal=False, max_tasks=30)
            return json.dumps(
                {
                    "ok": False,
                    "error": f"Unknown task_id: {task_id}",
                    "relative_path": relative_path,
                    "available_task_ids": [candidate.task_id for candidate in visible_catalog.tasks],
                },
                ensure_ascii=False,
                indent=2,
            )
        if task.depends_on:
            if task.depends_order not in (None, "", "sequence"):
                return json.dumps(
                    {
                        "ok": False,
                        "error": "Only sequence compound task execution is supported.",
                        "task_id": task.task_id,
                        "depends_order": task.depends_order,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            tasks_by_id = {candidate.task_id: candidate for candidate in catalog.tasks}
            results = []
            for dependency_id in task.depends_on:
                dependency = tasks_by_id.get(dependency_id)
                if dependency is None:
                    return json.dumps(
                        {"ok": False, "error": f"Unknown dependency task_id: {dependency_id}", "task_id": task.task_id},
                        ensure_ascii=False,
                        indent=2,
                    )
                if dependency.depends_on:
                    return json.dumps(
                        {"ok": False, "error": "Nested compound tasks are not supported.", "task_id": dependency.task_id},
                        ensure_ascii=False,
                        indent=2,
                    )
                dependency_workdir = (project_root / dependency.workdir).resolve()
                if project_root not in [dependency_workdir, *dependency_workdir.parents]:
                    return json.dumps(
                        {"ok": False, "error": f"Task workdir escapes project root: {dependency.workdir}"},
                        ensure_ascii=False,
                        indent=2,
                    )
                dependency_response = self.agent.get_terminal_process_manager().exec_command(
                    command=dependency.command,
                    cwd=dependency_workdir,
                    yield_time_ms=yield_time_ms,
                    max_output_tokens=max_output_tokens,
                    tty=dependency.interactive if tty is None else tty,
                )
                payload = json.loads(_json_response(dependency_response))
                payload["selected_task"] = _compact_task_dict(dependency)
                payload["service"] = _task_service_status(dependency, str(payload.get("output", "")))
                results.append(payload)
                if payload.get("running") or payload.get("exit_code") not in (0, None):
                    return json.dumps(
                        {"ok": False, "task_id": task.task_id, "depends_order": "sequence", "results": results},
                        ensure_ascii=False,
                        indent=2,
                    )
            return json.dumps(
                {"ok": True, "task_id": task.task_id, "depends_order": "sequence", "results": results},
                ensure_ascii=False,
                indent=2,
            )

        workdir = (project_root / task.workdir).resolve()
        if project_root not in [workdir, *workdir.parents]:
            return json.dumps({"ok": False, "error": f"Task workdir escapes project root: {task.workdir}"}, ensure_ascii=False, indent=2)

        response = self.agent.get_terminal_process_manager().exec_command(
            command=task.command,
            cwd=workdir,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            tty=task.interactive if tty is None else tty,
        )
        payload = json.loads(_json_response(response))
        payload["selected_task"] = _compact_task_dict(task)
        payload["service"] = _task_service_status(task, str(payload.get("output", "")))
        return json.dumps(payload, ensure_ascii=False, indent=2)


class RunValidationTool(Tool, ToolMarkerCanEdit):
    """
    Runs a likely validation task by kind or shortcut.
    """

    def apply(
        self,
        validation_id: str,
        relative_path: str = ".",
        files: list[str] | None = None,
        include_internal: bool = False,
        yield_time_ms: int = 10000,
        max_output_tokens: int | None = None,
        tty: bool | None = None,
    ) -> str:
        """
        Run the best matching validation task without requiring a raw command or task catalog lookup.

        :param validation_id: validation kind, alias, command name, or exact task id, such as lint, test, typecheck, format, build, verify
        :param relative_path: project-relative file or directory used to scope validation lookup
        :param files: optional project-relative files or test paths for focused validation
        :param include_internal: whether internal helper tasks may be selected
        :param yield_time_ms: wait before yielding output
        :param max_output_tokens: approximate output budget
        :param tty: override whether to run through PTY; defaults to task metadata
        :return: JSON terminal response with selected validation task metadata
        """
        from serena.tools.cmd_tools import _json_response

        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        try:
            file_args = _validation_file_args(project_root, relative_path, files)
        except Exception as error:
            return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2)

        catalog = discover_task_catalog(project_root, relative_path=relative_path).filtered(
            include_internal=include_internal,
            max_tasks=10000,
        )
        task, focused_command = _select_focused_validation_task(catalog, validation_id, file_args)
        if task is None:
            visible_catalog = catalog.filtered(include_internal=include_internal, max_tasks=20)
            return json.dumps(
                {
                    "ok": False,
                    "error": f"No validation task matched: {validation_id}",
                    "relative_path": relative_path,
                    "available_validations": [
                        _compact_task_dict(candidate) for candidate in visible_catalog.tasks if candidate.kind in VALIDATION_KINDS
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        if file_args and focused_command is None:
            return json.dumps({"ok": False, "error": "Focused validation unavailable"}, ensure_ascii=False, indent=2)
        if task.depends_on:
            return json.dumps(
                {
                    "ok": False,
                    "error": "Compound validation tasks are not implemented yet; run dependency task_ids directly.",
                    "validation_id": validation_id,
                    "task_id": task.task_id,
                    "depends_on": list(task.depends_on),
                    "depends_order": task.depends_order,
                },
                ensure_ascii=False,
                indent=2,
            )

        workdir = (project_root / task.workdir).resolve()
        if project_root not in [workdir, *workdir.parents]:
            return json.dumps({"ok": False, "error": f"Task workdir escapes project root: {task.workdir}"}, ensure_ascii=False, indent=2)

        response = self.agent.get_terminal_process_manager().exec_command(
            command=focused_command or task.command,
            cwd=workdir,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            tty=task.interactive if tty is None else tty,
        )
        payload = json.loads(_json_response(response))
        payload["validation_id"] = validation_id
        payload["relative_path"] = relative_path
        payload["focused_files"] = file_args
        payload["focused_command"] = focused_command
        payload["diagnostics"] = _parse_validation_diagnostics(str(payload.get("output", "")))
        payload["selected_task"] = _compact_task_dict(task)
        return json.dumps(payload, ensure_ascii=False, indent=2)


class UpdatePlanTool(Tool):
    """
    Codex-style lightweight checklist tool for ongoing coding work.
    """

    def apply(self, plan: list[dict[str, str]], explanation: str | None = None) -> str:
        """
        Update the current task plan without switching Serena modes or disabling tools.

        :param plan: list of plan items, each with `step` and status `pending`, `in_progress`, or `completed`
        :param explanation: optional concise reason for this plan update
        :return: JSON plan state plus markdown that should be reflected to the user in chat
        """
        normalized_plan = [{"step": item.get("step", "").strip(), "status": item.get("status", "")} for item in plan]
        _validate_plan(normalized_plan)

        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        workspace_id = self.agent.get_workspace_id()
        state_store = self.agent.get_harness_state_store()
        state = {
            "updated_at": _utc_now(),
            "explanation": explanation.strip() if isinstance(explanation, str) and explanation.strip() else None,
            "plan": normalized_plan,
            "state_path": str(_plan_state_path(project_root, workspace_id, state_store)),
        }
        _save_plan_state(project_root, state, workspace_id, state_store)
        markdown = _format_plan_markdown(normalized_plan)
        return json.dumps(
            {
                "message": "Plan updated",
                "plan": normalized_plan,
                "explanation": state["explanation"],
                "markdown": markdown,
                "user_visible_instruction": "Briefly reflect this plan/status in chat, then continue the work.",
            },
            ensure_ascii=False,
            indent=2,
        )


class PrepareReviewTaskTool(Tool):
    """
    Prepares a Codex-style review task context for the main ChatGPT/Serena agent.
    """

    def apply(
        self,
        target: str = "uncommitted",
        base_branch: str | None = None,
        commit_sha: str | None = None,
        instructions: str | None = None,
        max_diff_chars: int = DEFAULT_REVIEW_DIFF_MAX_CHARS,
    ) -> str:
        """
        Prepare review prompt, git context, and bounded diff for a review task.

        :param target: one of `uncommitted`, `base_branch`, `commit`, or `custom`
        :param base_branch: base branch for target `base_branch`
        :param commit_sha: commit SHA for target `commit`
        :param instructions: custom review instructions for target `custom`
        :param max_diff_chars: maximum characters to include across diff outputs
        :return: JSON review task context
        """
        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        resolved = _resolve_review_target(project_root, target, base_branch, commit_sha, instructions)
        per_diff_budget = max(1000, max_diff_chars // max(1, len(resolved["diff_commands"])))
        diffs = [_run_git_text(project_root, command, per_diff_budget) for command in resolved["diff_commands"]]
        extra_context = [_run_git_text(project_root, command, 12000) for command in resolved["extra_commands"]]
        payload = {
            "target": resolved["target"],
            "user_facing_hint": resolved["user_facing_hint"],
            "review_prompt": resolved["prompt"],
            "review_rubric": _review_rubric(),
            "findings_schema": _review_findings_schema(),
            "project_root": str(project_root),
            "git_context": extra_context,
            "diffs": diffs,
            "output_contract": (
                "Findings first, ordered by severity. If no actionable issues are found, say so clearly. "
                "Use finalize_review_task when wrapping a user-initiated review result."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


class FinalizeReviewTaskTool(Tool):
    """
    Wraps review output using Codex-style review completion context.
    """

    def apply(self, review_results: str | None = None, interrupted: bool = False) -> str:
        """
        Finalize a review task.

        :param review_results: JSON or text review output
        :param interrupted: true if the review was interrupted
        :return: JSON containing Codex-style review wrapper text
        """
        if interrupted:
            wrapper = (
                "<user_action>\n"
                "  <context>User initiated a review task, but it was interrupted. If asked, tell them to re-initiate the review and wait for it to complete.</context>\n"
                "  <action>review</action>\n"
                "  <results>\n"
                "  None.\n"
                "  </results>\n"
                "</user_action>\n"
            )
        else:
            results = (review_results or "").strip() or "No review output was provided."
            wrapper = (
                "<user_action>\n"
                "  <context>User initiated a review task. Here's the full review output. User may select one or more comments to resolve.</context>\n"
                "  <action>review</action>\n"
                "  <results>\n"
                f"  {results}\n"
                "  </results>\n"
                "</user_action>\n"
            )
        return json.dumps({"review_wrapper": wrapper}, ensure_ascii=False, indent=2)


class GetGoalTool(Tool):
    """
    Gets the current project goal, including status, budget fields, and remaining token budget.
    """

    def apply(self) -> str:
        """
        Get the current goal for this project.

        :return: JSON goal response
        """
        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        workspace_id = self.agent.get_workspace_id()
        state_store = self.agent.get_harness_state_store()
        return json.dumps(_goal_response(_goal_public_state(project_root, workspace_id, state_store)), ensure_ascii=False, indent=2)


class CreateGoalTool(Tool):
    """
    Creates a goal only when explicitly requested by the user or harness instructions.
    """

    def apply(self, objective: str, token_budget: int | None = None) -> str:
        """
        Create a project goal. Do not infer goals from ordinary tasks.

        :param objective: concrete objective to pursue
        :param token_budget: optional positive token budget, only when explicitly requested
        :return: JSON goal response
        """
        objective = objective.strip()
        _validate_goal_objective(objective)
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be positive when provided")

        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        workspace_id = self.agent.get_workspace_id()
        state_store = self.agent.get_harness_state_store()
        existing = _load_goal_state(project_root, workspace_id, state_store)
        if existing is not None and existing.get("status") != "complete":
            raise ValueError("cannot create a new goal because this project has an unfinished goal; complete the existing goal first")

        now = _utc_now()
        state: dict[str, Any] = {
            "objective": objective,
            "status": "active",
            "token_budget": token_budget,
            "tokens_used": 0,
            "time_used_seconds": 0,
            "created_at": now,
            "updated_at": now,
            "progress": [],
            "validation_results": None,
            "remaining_risks": None,
            "project_root": str(project_root),
        }
        _save_goal_state(project_root, state, workspace_id, state_store)
        return json.dumps(_goal_response(_goal_public_state(project_root, workspace_id, state_store)), ensure_ascii=False, indent=2)


class UpdateGoalTool(Tool):
    """
    Updates the existing goal only to mark it complete or genuinely blocked.
    """

    def apply(self, status: str) -> str:
        """
        Update the existing goal status.

        :param status: required status, either complete or blocked
        :return: JSON goal response
        """
        normalized_status = status.strip().lower()
        if normalized_status not in MODEL_SETTABLE_GOAL_STATUSES:
            raise ValueError(
                "update_goal can only mark the existing goal complete or blocked; pause, resume, budget-limited, "
                "and usage-limited status changes are controlled outside this tool"
            )

        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        workspace_id = self.agent.get_workspace_id()
        state_store = self.agent.get_harness_state_store()
        state = _require_goal_state(project_root, workspace_id, state_store)
        if state.get("status") == "complete":
            raise ValueError("cannot update goal because it is already complete")
        state["time_used_seconds"] = _goal_time_used_seconds(state)
        state["status"] = normalized_status
        state["updated_at"] = _utc_now()
        _save_goal_state(project_root, state, workspace_id, state_store)
        return json.dumps(
            _goal_response(
                _goal_public_state(project_root, workspace_id, state_store),
                include_completion_report=normalized_status == "complete",
            ),
            ensure_ascii=False,
            indent=2,
        )


class RecordGoalProgressTool(Tool):
    """
    Records Serena-native progress notes without changing goal status.
    """

    def apply(self, note: str, validation_results: str | None = None, remaining_risks: str | None = None) -> str:
        """
        Append a progress note to the active goal.

        Use `update_goal` only for Codex-style complete/blocked status changes.

        :param note: concise progress note
        :param validation_results: optional validation summary
        :param remaining_risks: optional risk/blocker summary
        :return: JSON goal response
        """
        note = note.strip()
        if not note:
            raise ValueError("progress note must not be empty")
        active_project = self.agent.get_active_project_or_raise()
        project_root = Path(active_project.project_root).resolve()
        workspace_id = self.agent.get_workspace_id()
        state_store = self.agent.get_harness_state_store()
        state = _require_goal_state(project_root, workspace_id, state_store)
        _append_goal_note(state, note)
        if validation_results is not None:
            state["validation_results"] = validation_results
        if remaining_risks is not None:
            state["remaining_risks"] = remaining_risks
        state["time_used_seconds"] = _goal_time_used_seconds(state)
        state["updated_at"] = _utc_now()
        _save_goal_state(project_root, state, workspace_id, state_store)
        return json.dumps(_goal_response(_goal_public_state(project_root, workspace_id, state_store)), ensure_ascii=False, indent=2)


class OnboardingTool(Tool):
    """
    Performs onboarding (identifying the project structure and essential tasks, e.g. for testing or building).
    """

    def apply(self) -> str:
        """
        Call this tool if onboarding was not performed yet.
        You will call this tool at most once per conversation.

        :return: instructions on how to create the onboarding information
        """
        write_memory_tool_available = self.agent.tool_is_exposed(WriteMemoryTool.get_name_from_cls())
        if not write_memory_tool_available:
            return "Memory writing tool not activated, skipping onboarding."
        system = platform.system()
        # seed the project-local memory-maintenance memory (or detect a global override) so
        # the prompt can point the agent at the conventions before it writes anything
        memory_maintenance_name = self.memory_manager.ensure_memory_maintenance_memory()
        return self.prompt_factory.create_onboarding_prompt(system=system, memory_maintenance_name=memory_maintenance_name)


class InitialInstructionsTool(Tool, ToolMarkerDoesNotRequireActiveProject):
    """
    Provides instructions Serena usage (i.e. the 'Serena Instructions Manual')
    for clients that do not read the initial instructions when the MCP server is connected.
    """

    # noinspection PyIncorrectDocstring
    # (session_id is injected via apply_ex)
    def apply(self, session_id: str) -> str:
        """
        Provides the 'Serena Instructions Manual', which contains essential information on how to use the Serena toolbox.
        IMPORTANT: If you have not yet read the manual, call this tool immediately after you are given your task by the user,
        as it will critically inform you!
        """
        return self.agent.create_system_prompt(session_id=session_id)


class SerenaInfoTool(Tool, ToolMarkerOptional, ToolMarkerDoesNotRequireActiveProject):
    """
    Provides information about an advanced topic on demand, facilitating context-efficiency.
    """

    def apply(self, topic: str) -> str:
        """
        Retrieves Serena-specific information
        :param topic: the topic, which you must have been given explicitly
        """
        match topic:
            case "jet_brains_debug_repl":
                return self.agent.prompt_factory.create_info_jet_brains_debug_repl()
            case _:
                raise ValueError("Invalid topic: " + topic)
