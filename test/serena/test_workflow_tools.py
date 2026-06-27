from pathlib import Path

from serena.tools.task_catalog import discover_task_catalog
from serena.tools.tools_base import ToolRegistry
from serena.tools.workflow_tools import (
    MAX_GOAL_OBJECTIVE_CHARS,
    _apply_codex_style_patch,
    _compact_validation_hints,
    _edit_policy_contract,
    _focused_validation_command,
    _format_plan_markdown,
    _goal_public_state,
    _goal_response,
    _goal_state_path,
    _infer_validation_hints,
    _load_instruction_documents,
    _normalize_validation_id,
    _parse_validation_diagnostics,
    _resolve_focus_dir,
    _resolve_review_target,
    _run_git_snapshot,
    _save_goal_state,
    _select_focused_validation_task,
    _select_validation_task,
    _task_catalog_agent_view,
    _task_service_status,
    _validate_goal_objective,
    _validate_plan,
    _validation_file_args,
)


def test_coding_workflow_tools_are_registered() -> None:
    names = ToolRegistry().get_tool_names()

    assert "apply_patch" in names
    assert "prepare_coding_task" in names
    assert "get_coding_harness_instructions" in names
    assert "get_validation_commands" in names
    assert "discover_project_tasks" in names
    assert "run_task" in names
    assert "run_validation" in names
    assert "finalize_coding_task" in names
    assert "update_plan" in names
    assert "prepare_review_task" in names
    assert "finalize_review_task" in names
    assert "get_goal" in names
    assert "create_goal" in names
    assert "update_goal" in names
    assert "record_goal_progress" in names


def test_load_instruction_documents_prefers_override_and_preserves_scope_order(tmp_path: Path) -> None:
    project_root = tmp_path
    focus_dir = project_root / "src" / "feature"
    focus_dir.mkdir(parents=True)
    (project_root / "AGENTS.md").write_text("root instructions", encoding="utf-8")
    (project_root / "src" / "AGENTS.md").write_text("ignored default", encoding="utf-8")
    (project_root / "src" / "AGENTS.override.md").write_text("src override", encoding="utf-8")
    (focus_dir / "AGENTS.md").write_text("feature instructions", encoding="utf-8")

    documents = _load_instruction_documents(project_root, focus_dir, max_total_bytes=65536)

    assert [document.relative_path for document in documents] == [
        "AGENTS.md",
        "src/AGENTS.override.md",
        "src/feature/AGENTS.md",
    ]
    assert [document.contents for document in documents] == ["root instructions", "src override", "feature instructions"]


def test_load_instruction_documents_truncates_to_byte_budget(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "AGENTS.md").write_text("123456789", encoding="utf-8")

    documents = _load_instruction_documents(project_root, project_root, max_total_bytes=4)

    assert len(documents) == 1
    assert documents[0].contents == "1234"
    assert documents[0].truncated is True


def test_resolve_focus_dir_rejects_paths_outside_project(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"

    try:
        _resolve_focus_dir(tmp_path, f"../{outside.name}")
    except ValueError as e:
        assert "inside the active project" in str(e)
    else:
        raise AssertionError("Expected focus path outside the project to be rejected")


def test_infer_validation_hints_detects_python_and_node_commands(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """
[tool.poe.tasks]
lint = "ruff check src test"
test = "pytest test"
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "pnpm-lock.yaml").write_text("", encoding="utf-8")
    (tmp_path / "package.json").write_text(
        '{"scripts": {"lint": "eslint .", "test": "vitest", "build": "vite build"}}',
        encoding="utf-8",
    )

    hints = _infer_validation_hints(tmp_path)

    assert hints.package_files == ["package.json", "pyproject.toml"]
    assert hints.detected_package_managers == ["pnpm", "uv", "poe"]
    assert hints.likely_commands == ["pnpm lint", "uv run poe lint", "pnpm test", "uv run poe test", "pnpm build"]


def test_discover_task_catalog_supports_broad_manifests(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'demo'\nversion = '0.1.0'\n", encoding="utf-8")
    (tmp_path / "go.mod").write_text("module example.com/demo\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("test:\n\ttrue\nbuild:\n\ttrue\n", encoding="utf-8")
    (tmp_path / "justfile").write_text("lint:\n    true\n", encoding="utf-8")
    (tmp_path / "Taskfile.yml").write_text("tasks:\n  verify:\n    cmds:\n      - true\n", encoding="utf-8")
    (tmp_path / ".serena").mkdir()
    (tmp_path / ".serena" / "tasks.json").write_text(
        '{"tasks": [{"task_id": "custom:verify", "kind": "verify", "command": "echo ok"}]}',
        encoding="utf-8",
    )

    catalog = discover_task_catalog(tmp_path)
    task_ids = {task.task_id for task in catalog.tasks}

    assert "custom:verify" in task_ids
    assert "root:cargo:check:check" in task_ids
    assert "root:go:test:test" in task_ids
    assert "root:just:lint:lint" in task_ids
    assert "root:task:verify:verify" in task_ids
    assert "cargo" in catalog.detected_package_managers
    assert "go" in catalog.detected_package_managers
    assert "make" in catalog.detected_package_managers


def test_task_catalog_filtered_view_hides_internal_tasks_and_prefers_root_tasks(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.poe.tasks]\n_hidden = 'echo hidden'\nlint = 'echo lint'\n", encoding="utf-8")
    nested = tmp_path / "test" / "resources" / "repos" / "rust"
    nested.mkdir(parents=True)
    (nested / "Cargo.toml").write_text("[package]\nname = 'fixture'\nversion = '0.1.0'\n", encoding="utf-8")

    catalog = discover_task_catalog(tmp_path)
    filtered = catalog.filtered(include_internal=False, max_tasks=2)

    assert all(task.visibility == "public" for task in filtered.tasks)
    assert [task.workdir for task in filtered.tasks] == [".", "."]
    assert all("test/resources" not in package_file for package_file in catalog.package_files)
    assert "echo hidden" not in filtered.validation_hints.likely_commands

    catalog_with_fixtures = discover_task_catalog(tmp_path, include_fixtures=True)
    assert any("test/resources" in package_file for package_file in catalog_with_fixtures.package_files)


def test_discover_task_catalog_can_be_scoped_to_relative_path(tmp_path: Path) -> None:
    frontend = tmp_path / "apps" / "frontend"
    backend = tmp_path / "apps" / "backend"
    frontend.mkdir(parents=True)
    backend.mkdir(parents=True)
    (frontend / "package.json").write_text('{"scripts": {"lint": "eslint ."}}', encoding="utf-8")
    (backend / "package.json").write_text('{"scripts": {"test": "vitest"}}', encoding="utf-8")

    catalog = discover_task_catalog(tmp_path, relative_path="apps/frontend")

    assert [task.workdir for task in catalog.tasks] == ["apps/frontend"]
    assert [task.task_id for task in catalog.tasks] == ["apps:frontend:npm:lint:lint"]
    assert catalog.package_files == ["apps/frontend/package.json"]


def test_task_override_v2_metadata_is_preserved(tmp_path: Path) -> None:
    (tmp_path / ".serena").mkdir()
    (tmp_path / ".serena" / "tasks.json").write_text(
        """
{
  "version": 2,
  "tasks": [
    {
      "id": "frontend:dev",
      "kind": "dev",
      "command": "npm run dev",
      "workdir": "apps/frontend",
      "long_running": true,
      "ready_pattern": "ready",
      "problem_matcher": "vite"
    },
    {
      "id": "verify",
      "kind": "verify",
      "depends_on": ["frontend:lint", "backend:test"],
      "depends_order": "sequence"
    }
  ]
}
""",
        encoding="utf-8",
    )
    (tmp_path / "apps" / "frontend").mkdir(parents=True)

    catalog = discover_task_catalog(tmp_path)
    tasks = {task.task_id: task for task in catalog.tasks}

    assert tasks["frontend:dev"].ready_pattern == "ready"
    assert tasks["frontend:dev"].problem_matcher == "vite"
    assert tasks["frontend:dev"].long_running is True
    assert tasks["verify"].runner == "compound"
    assert tasks["verify"].depends_on == ("frontend:lint", "backend:test")
    assert tasks["verify"].depends_order == "sequence"


def test_task_catalog_agent_view_is_compact_by_default(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{"scripts": {"lint": "eslint .", "test": "vitest", "build": "vite build"}}',
        encoding="utf-8",
    )
    nested = tmp_path / "apps" / "web"
    nested.mkdir(parents=True)
    (nested / "package.json").write_text(
        '{"scripts": {"lint": "eslint app", "build": "vite build"}}',
        encoding="utf-8",
    )

    catalog = discover_task_catalog(tmp_path)
    visible = catalog.filtered(max_tasks=2)

    view = _task_catalog_agent_view(catalog, visible)
    hints = _compact_validation_hints(visible)

    assert set(view) == {"summary", "top_tasks", "details_available"}
    assert "catalog" not in view
    assert "package_files" not in view
    assert len(view["top_tasks"]) == 2
    assert set(view["top_tasks"][0]) == {"task_id", "kind", "command", "workdir", "long_running"}
    assert view["summary"]["package_files_count"] == 2
    assert hints.package_files == []
    assert hints.likely_commands

    detailed_view = _task_catalog_agent_view(catalog, visible, include_details=True)
    assert "catalog" in detailed_view
    assert detailed_view["catalog"]["package_files"] == ["package.json", "apps/web/package.json"]


def test_goal_objective_validation_matches_codex_limits() -> None:
    _validate_goal_objective("ship the thing")

    try:
        _validate_goal_objective("")
    except ValueError as e:
        assert "must not be empty" in str(e)
    else:
        raise AssertionError("Expected empty goal objective to be rejected")

    try:
        _validate_goal_objective("x" * (MAX_GOAL_OBJECTIVE_CHARS + 1))
    except ValueError as e:
        assert "at most" in str(e)
    else:
        raise AssertionError("Expected overlong goal objective to be rejected")


def test_goal_public_state_and_response_include_remaining_budget(tmp_path: Path) -> None:
    _save_goal_state(
        tmp_path,
        {
            "objective": "finish harness",
            "status": "active",
            "token_budget": 100,
            "tokens_used": 25,
            "time_used_seconds": 0,
            "created_at": "2026-06-24T00:00:00Z",
            "updated_at": "2026-06-24T00:00:00Z",
            "progress": [],
        },
    )

    public = _goal_public_state(tmp_path)
    assert public is not None
    assert public["state_path"] == str(_goal_state_path(tmp_path))
    assert public["remaining_tokens"] == 75

    response = _goal_response(public, include_completion_report=True)
    assert response["remaining_tokens"] == 75
    assert "tokens_used=25" in response["completion_budget_report"]
    assert response["runtime_prompts"]["continuation"].startswith('<codex_internal_context source="serena_goal_continuation">')
    continuation = response["runtime_prompts"]["continuation"]
    assert "<objective>\nfinish harness\n</objective>" in continuation
    assert "Continuation behavior:" in continuation
    assert "Completion audit:" in continuation
    assert "Blocked audit:" in continuation
    assert "Do not call update_goal unless the goal is complete" in continuation
    assert "Tokens remaining: 75" in response["runtime_prompts"]["objective_updated"]
    assert "Time spent pursuing goal:" in response["runtime_prompts"]["budget_limit"]


def test_update_plan_validation_matches_codex_shape() -> None:
    plan = [
        {"step": "Inspect Codex plan tool", "status": "completed"},
        {"step": "Implement Serena plan tool", "status": "in_progress"},
        {"step": "Run tests", "status": "pending"},
    ]

    _validate_plan(plan)

    assert _format_plan_markdown(plan) == ("- [x] Inspect Codex plan tool\n- [~] Implement Serena plan tool\n- [ ] Run tests")

    try:
        _validate_plan(
            [
                {"step": "one", "status": "in_progress"},
                {"step": "two", "status": "in_progress"},
            ]
        )
    except ValueError as e:
        assert "at most one" in str(e)
    else:
        raise AssertionError("Expected multiple in_progress plan items to be rejected")

    try:
        _validate_plan(
            [
                {"step": "one", "status": "completed"},
                {"step": "two", "status": "pending"},
            ]
        )
    except ValueError as e:
        assert "exactly one" in str(e)
    else:
        raise AssertionError("Expected non-complete plans without in_progress to be rejected")

    try:
        _validate_plan([{"step": "one", "status": "in_progress", "extra": "unsupported"}])
    except ValueError as e:
        assert "unsupported fields" in str(e)
    else:
        raise AssertionError("Expected unsupported plan fields to be rejected")

    _validate_plan(
        [
            {"step": "one", "status": "completed"},
            {"step": "two", "status": "completed"},
        ]
    )


def test_resolve_review_target_uncommitted_and_commit(tmp_path: Path) -> None:
    uncommitted = _resolve_review_target(tmp_path, "uncommitted", None, None, None)

    assert uncommitted["target"] == "uncommitted_changes"
    assert ["git", "diff", "--cached"] in uncommitted["diff_commands"]
    assert "staged, unstaged, and untracked" in uncommitted["prompt"]

    commit = _resolve_review_target(tmp_path, "commit", None, "abc1234", None)

    assert commit["target"] == "commit"
    assert commit["user_facing_hint"].startswith("commit abc1234")
    assert commit["diff_commands"] == [["git", "show", "--format=medium", "--patch", "abc1234"]]


def test_planning_mode_no_longer_disables_editing_tools() -> None:
    planning_mode = Path(__file__).parents[2] / "src" / "serena" / "resources" / "config" / "modes" / "planning.yml"
    contents = planning_mode.read_text(encoding="utf-8")

    assert "excluded_tools: []" in contents
    assert "read-only planning mode" in contents


def test_edit_policy_prefers_semantic_then_exact_then_patch() -> None:
    policy = _edit_policy_contract()

    assert policy["default_order"] == [
        "inspect before editing unfamiliar code",
        "symbol tools for symbol-aware changes",
        "replace_content for exact small text edits with allow_multiple_occurrences=false",
        "apply_patch for atomic multi-file or structured textual patches",
    ]
    assert "not the default small-edit tool" in policy["apply_patch"]["role"]
    assert policy["apply_patch"]["failure_contract"] == "on failure no patch changes are written and changes=[]"
    assert "allow_multiple_occurrences=false" in policy["replace_content"]["safety"]


def test_apply_patch_add_update_move_and_delete(tmp_path: Path) -> None:
    add_result = _apply_codex_style_patch(
        tmp_path,
        ".",
        """*** Begin Patch
*** Add File: notes.txt
+hello
+world
*** End Patch""",
        dry_run=False,
    )

    assert add_result.success is True
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert add_result.changes[0].operation == "add"

    update_result = _apply_codex_style_patch(
        tmp_path,
        ".",
        """*** Begin Patch
*** Update File: notes.txt
*** Move to: docs/renamed.txt
@@
 hello
-world
+serena
*** End Patch""",
        dry_run=False,
    )

    assert update_result.success is True
    assert not (tmp_path / "notes.txt").exists()
    assert (tmp_path / "docs" / "renamed.txt").read_text(encoding="utf-8") == "hello\nserena\n"
    assert update_result.changes[0].move_path == "docs/renamed.txt"

    delete_result = _apply_codex_style_patch(
        tmp_path,
        ".",
        """*** Begin Patch
*** Delete File: docs/renamed.txt
*** End Patch""",
        dry_run=False,
    )

    assert delete_result.success is True
    assert not (tmp_path / "docs" / "renamed.txt").exists()


def test_apply_patch_dry_run_and_safety(tmp_path: Path) -> None:
    dry_result = _apply_codex_style_patch(
        tmp_path,
        ".",
        """*** Begin Patch
*** Add File: dry.txt
+not written
*** End Patch""",
        dry_run=True,
    )

    assert dry_result.success is True
    assert not (tmp_path / "dry.txt").exists()

    unsafe_result = _apply_codex_style_patch(
        tmp_path,
        ".",
        """*** Begin Patch
*** Add File: ../outside.txt
+bad
*** End Patch""",
        dry_run=False,
    )

    assert unsafe_result.success is False
    assert "escapes the active project" in unsafe_result.errors[0]


def test_apply_patch_rejects_ambiguous_update_hunks(tmp_path: Path) -> None:
    target = tmp_path / "dupes.txt"
    target.write_text("same\nsame\n", encoding="utf-8")

    result = _apply_codex_style_patch(
        tmp_path,
        ".",
        """*** Begin Patch
*** Update File: dupes.txt
@@
-same
+changed
*** End Patch""",
        dry_run=False,
    )

    assert result.success is False
    assert "matched multiple locations" in result.errors[0]
    assert target.read_text(encoding="utf-8") == "same\nsame\n"


def test_apply_patch_multifile_success_is_atomic(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")

    result = _apply_codex_style_patch(
        tmp_path,
        ".",
        """*** Begin Patch
*** Update File: a.txt
@@
-one
+two
*** Add File: b.txt
+created
*** End Patch""",
        dry_run=False,
    )

    assert result.success is True
    assert result.atomic is True
    assert result.would_change == ["a.txt", "b.txt"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "two\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "created\n"


def test_apply_patch_failure_does_not_partially_write(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    first.write_text("old\n", encoding="utf-8")

    result = _apply_codex_style_patch(
        tmp_path,
        ".",
        """*** Begin Patch
*** Update File: first.txt
@@
-old
+new
*** Update File: missing.txt
@@
-missing
+changed
*** End Patch""",
        dry_run=False,
    )

    assert result.success is False
    assert result.atomic is True
    assert result.changes == []
    assert first.read_text(encoding="utf-8") == "old\n"
    assert not (tmp_path / "missing.txt").exists()


def test_apply_patch_dry_run_multifile_does_not_write(tmp_path: Path) -> None:
    existing = tmp_path / "existing.txt"
    remove_me = tmp_path / "remove-me.txt"
    existing.write_text("before\n", encoding="utf-8")
    remove_me.write_text("delete\n", encoding="utf-8")

    result = _apply_codex_style_patch(
        tmp_path,
        ".",
        """*** Begin Patch
*** Update File: existing.txt
@@
-before
+after
*** Add File: created.txt
+created
*** Delete File: remove-me.txt
*** End Patch""",
        dry_run=True,
    )

    assert result.success is True
    assert result.atomic is True
    assert existing.read_text(encoding="utf-8") == "before\n"
    assert remove_me.read_text(encoding="utf-8") == "delete\n"
    assert not (tmp_path / "created.txt").exists()


def test_run_git_snapshot_tolerates_leading_git(tmp_path: Path) -> None:
    subprocess_result = _run_git_snapshot(tmp_path, ["git", "status", "--short"])

    assert subprocess_result.command == "git status --short"
    assert "git git" not in subprocess_result.command


def test_validation_diagnostics_parser_extracts_ruff_and_pytest() -> None:
    diagnostics = _parse_validation_diagnostics(
        "src/demo.py:10:5: F401 unused import\nFAILED test/test_demo.py::test_demo - AssertionError"
    )

    assert diagnostics[0] == {"path": "src/demo.py", "line": 10, "column": 5, "message": "F401 unused import"}
    assert diagnostics[1]["tool"] == "pytest"
    assert diagnostics[1]["path"] == "test/test_demo.py::test_demo"


def test_task_service_status_uses_ready_pattern() -> None:
    class DemoTask:
        ready_pattern = "ready"

    assert _task_service_status(DemoTask(), "server ready") == {"ready_pattern": "ready", "ready": True}
    assert _task_service_status(DemoTask(), "starting") == {"ready_pattern": "ready", "ready": False}


def test_task_service_status_is_empty_without_pattern() -> None:
    class DemoTask:
        ready_pattern = None

    assert _task_service_status(DemoTask(), "ready") == {}


def test_focused_validation_command_replaces_dot_with_files(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n", encoding="utf-8")

    catalog = discover_task_catalog(tmp_path).filtered(include_internal=False, max_tasks=10000)
    task = _select_validation_task(catalog, "lint")

    assert task is not None
    assert _focused_validation_command(task, ["src/example.py"]) == "python -m ruff check src/example.py"


def test_select_focused_validation_prefers_focusable_task(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.poe.tasks]\nlint = 'ruff check .'\n[tool.ruff]\n",
        encoding="utf-8",
    )

    catalog = discover_task_catalog(tmp_path).filtered(include_internal=False, max_tasks=10000)
    task, command = _select_focused_validation_task(catalog, "lint", ["src/example.py"])

    assert task is not None
    assert command == "python -m ruff check src/example.py"


def test_validation_file_args_rejects_files_outside_scope(tmp_path: Path) -> None:
    scope = tmp_path / "apps" / "frontend"
    outside = tmp_path / "apps" / "backend"
    scope.mkdir(parents=True)
    outside.mkdir(parents=True)
    (scope / "page.py").write_text("", encoding="utf-8")
    (outside / "api.py").write_text("", encoding="utf-8")

    assert _validation_file_args(tmp_path, "apps/frontend", ["apps/frontend/page.py"]) == ["apps/frontend/page.py"]

    try:
        _validation_file_args(tmp_path, "apps/frontend", ["apps/backend/api.py"])
    except ValueError as error:
        assert "outside relative_path scope" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_select_validation_task_accepts_shortcuts(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.poe.tasks]\nlint = 'ruff check .'\ntest = 'pytest'\ntype-check = 'mypy .'\n",
        encoding="utf-8",
    )

    catalog = discover_task_catalog(tmp_path).filtered(include_internal=False, max_tasks=10000)

    assert _normalize_validation_id("type-check") == "typecheck"
    assert _select_validation_task(catalog, "lint").task_id == "root:poe:lint:lint"
    assert _select_validation_task(catalog, "tests").task_id == "root:poe:test:test"
    assert _select_validation_task(catalog, "type").kind == "typecheck"
    assert _select_validation_task(catalog, "root:poe:lint:lint").kind == "lint"
    assert _select_validation_task(catalog, "missing") is None
