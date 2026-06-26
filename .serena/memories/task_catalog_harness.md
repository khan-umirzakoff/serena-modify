# Task Catalog Harness

Serena MCP should reduce ChatGPT tool-call filter issues by making common project tasks structured instead of raw shell. Raw `exec_command` stays available, but normal coding agents should prefer task discovery and `task_id` execution.

## Current v1 design

- `src/serena/tools/task_catalog.py` owns the task discovery engine.
- `discover_project_tasks` exposes a compact catalog view as a tool: summary plus bounded `top_tasks` by default; full package files, validation hint detail, and full filtered catalog require `include_details=true`.
- `run_task(task_id)` executes a discovered task through the existing Codex-style terminal manager.
- `run_validation(validation_id)` is the preferred shortcut for common validation checks when task choice is obvious.
- `get_validation_commands` remains a backward-compatible wrapper around catalog validation hints.
- `prepare_coding_task` includes `task_catalog_summary`, compact `edit_policy`, and a small public top-task list, so the agent can choose safe tasks early without dumping every command.
- `discover_project_tasks`, `get_validation_commands`, `run_task`, and `run_validation` support `relative_path` for scoped multi-repo/package workflows.
- `.serena/tasks.json` v2 metadata is preserved for custom tasks: `id`, `kind`, `command`, `workdir`, `interactive`, `long_running`, `ready_pattern`, `problem_matcher`, `depends_on`, and `depends_order`.
- Compound task metadata is catalogued, but compound execution is intentionally not automatic yet; agents should run dependency task IDs directly until a dedicated runner is implemented.
- Terminal control is structured: use `exec_command(tty=true)` for interactive commands, `write_stdin` for prompt input, `terminal_status` for non-consuming session status/output inspection, and `send_terminal_signal` for SIGINT/SIGTERM/SIGKILL.
- Terminal `workdir` is not the same as Serena active project. `exec_command`/`execute_shell_command` responses warn when a command runs in a nested Git repo or outside the active project root while semantic tools still point elsewhere.

## Edit policy contract

- Inspect unfamiliar code before editing.
- Use semantic tools for symbol-aware edits: `replace_symbol_body`, `insert_before_symbol`, `insert_after_symbol`, `rename_symbol`, `safe_delete_symbol`.
- Use `replace_content` for exact small text edits and keep `allow_multiple_occurrences=false` unless every match was inspected.
- Use `apply_patch` for atomic multi-file or structured textual patches, not as the default small-edit tool.
- For complex or risky patches, validate with `dry_run=true` first.
- If `apply_patch` fails, no patch changes are written and `changes=[]`.

## Supported v1 sources

- `.serena/tasks.json` overrides
- `package.json`
- `pyproject.toml`
- `Cargo.toml`
- `go.mod`
- Makefiles
- justfiles
- Taskfile.yml / Taskfile.yaml
- `composer.json`
- `pom.xml`
- Gradle files
- `Gemfile`
- `CMakeLists.txt`
- `*.csproj`

## Design principles

- Do not require semantic indexing for task discovery.
- Use bounded filesystem manifest scanning.
- Skip heavy/generated directories.
- Include runner in `task_id` to avoid collisions.
- Keep project override support for unknown frameworks.
- Hide internal helper tasks by default with `include_internal=false`.
- Exclude fixture/test-resource manifests by default unless `include_fixtures=true` is explicit.
- Exclude example/sample/demo manifests by default unless `include_examples=true` is explicit.
- Exclude generated/dependency/cache directories by default unless `include_ignored=true` is explicit.
- Sort root/workspace tasks ahead of nested manifests when broader discovery is enabled.
- Prefer summary-first output, structured JSON, and bounded terminal logs.

## Future versions

- V2: safe runner introspection (`just --list`, `task --list`, `npm run`, `cargo metadata`) when bounded.
- V3: validation failure classifier with file/line/message extraction.
- V4: learn successful user-provided commands into `.serena/tasks.json` suggestions.
