# Task Catalog Harness

Serena's ChatGPT/Web MCP workflow should not depend on raw shell commands for common validation and project tasks. Raw shell remains available through `exec_command`, but normal agent workflows should prefer a structured task catalog.

## Goals

- Discover project tasks without requiring semantic indexing.
- Support many ecosystems through manifest scanning, not a narrow framework list.
- Let agents run `task_id` values instead of sending risky raw command strings.
- Keep output bounded through the existing Codex-style terminal manager.
- Allow project-specific overrides for unknown or custom stacks.

## V1 detection model

Task Catalog v1 scans the filesystem with bounded depth and ignores heavy generated directories such as `.git`, `.venv`, `node_modules`, `target`, `dist`, and build caches.

Supported manifests in v1:

- `package.json`
- `pyproject.toml`
- `Cargo.toml`
- `go.mod`
- `Makefile`, `makefile`, `GNUmakefile`
- `justfile`, `Justfile`, `.justfile`
- `Taskfile.yml`, `Taskfile.yaml`
- `composer.json`
- `pom.xml`
- `build.gradle`, `build.gradle.kts`
- `Gemfile`
- `CMakeLists.txt`
- `*.csproj`

The engine also reads `.serena/tasks.json` for explicit project overrides.

## Task shape

Each task has:

- `task_id`
- `kind`
- `runner`
- `command`
- `workdir`
- `source`
- `confidence`
- `interactive`
- `long_running`
- `visibility`
- `priority`

Task IDs include runner information to avoid collisions, for example:

- `root:pnpm:lint:lint`
- `root:poe:lint:lint`
- `root:cargo:check:check`

## Tool flow

- `discover_project_tasks(include_internal=false, include_fixtures=false, include_examples=false, include_ignored=false, max_tasks=30, include_details=false, relative_path=".")` returns only summary plus bounded `top_tasks` by default; full package files and validation hint detail are opt-in.
- `discover_project_tasks(include_internal=true)` is reserved for debugging or advanced agent use when helper tasks are needed.
- `discover_project_tasks(include_fixtures=true, include_examples=true)` is reserved for explicit inspection of test fixtures, samples, demos, and example projects.
- `discover_project_tasks(include_ignored=true)` is a last-resort diagnostic option for generated, dependency, cache, or normally ignored directories.
- `run_task(task_id, relative_path=".")` executes a known task through the existing terminal process manager after scoped lookup.
- `run_validation(validation_id, relative_path=".", files=[...])` selects and runs validation; `files` narrows safe known runners such as `ruff check`, `ruff format`, and `pytest`.
- `get_validation_commands(relative_path=".")` stays as a backward-compatible public-only wrapper around the catalog's validation hints.
- `prepare_coding_task` includes `task_catalog_summary`, compact `edit_policy`, and only a small public top-task list by default.
- Edit policy is explicit: inspect unfamiliar code first, use semantic tools for symbol-aware changes, use `replace_content` for exact small edits with `allow_multiple_occurrences=false`, and reserve `apply_patch` for atomic multi-file or structured textual patches.
- Scoped task lookup is supported through `relative_path`, so multi-repo workspaces can target `frontend`, `backend`, or a nested package without scanning/running unrelated tasks.
- `.serena/tasks.json` v2 metadata is preserved for custom tasks: `id`, `kind`, `command`, `workdir`, `interactive`, `long_running`, `ready_pattern`, `problem_matcher`, `depends_on`, and `depends_order`.
- Compound `depends_on` tasks execute automatically when `depends_order` is `sequence`; parallel and nested compound tasks still fail closed.
- Long-running task responses include service readiness metadata when a task defines `ready_pattern`.
- Validation responses include compact parsed diagnostics for common `ruff` and `pytest` output.
- Terminal control is structured: use `exec_command(tty=true)` for interactive commands, `write_stdin` for prompt input, `terminal_status` for non-consuming session status/output inspection, and `send_terminal_signal` for SIGINT/SIGTERM/SIGKILL.
- Terminal `workdir` and Serena active project are separate contexts. If `exec_command` runs inside a nested Git repository or outside the active project root, the terminal response includes a warning telling the agent to call `activate_project(...)` when semantic tools should follow that repo.

## Noise control

The catalog is intentionally summary-first and root-first:

- internal helper tasks whose names start with `_` are hidden by default;
- fixture and test-resource manifests are excluded by default;
- example, sample, and demo manifests are excluded by default;
- generated, dependency, cache, and vendor directories are excluded by default;
- root/workspace tasks sort ahead of nested manifests when explicit broader discovery is enabled;
- public validation tasks such as `typecheck`, `lint`, `test`, `format`, and `build` sort before generic tasks;
- package file lists, validation hint detail, and full filtered catalogs remain available only with `include_details=true`.

## Future versions

V2 should add runner introspection for installed tools where safe and bounded, such as `just --list`, `task --list`, `npm run`, and `cargo metadata`.

V3 should add failure classification that maps common compiler/test output into file, line, and message diagnostics.

V4 should support learning from successful user-provided commands by writing `.serena/tasks.json` suggestions instead of hardcoding framework-specific behavior.
