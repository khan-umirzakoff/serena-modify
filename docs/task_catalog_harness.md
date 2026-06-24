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

- `discover_project_tasks(include_internal=false, include_fixtures=false, include_examples=false, include_ignored=false, max_tasks=30, include_details=false)` returns only summary plus bounded `top_tasks` by default; full package files and validation hint detail are opt-in.
- `discover_project_tasks(include_internal=true)` is reserved for debugging or advanced agent use when helper tasks are needed.
- `discover_project_tasks(include_fixtures=true, include_examples=true)` is reserved for explicit inspection of test fixtures, samples, demos, and example projects.
- `discover_project_tasks(include_ignored=true)` is a last-resort diagnostic option for generated, dependency, cache, or normally ignored directories.
- `run_task(task_id)` executes a known task through the existing terminal process manager.
- `get_validation_commands` stays as a backward-compatible public-only wrapper around the catalog's validation hints.
- `prepare_coding_task` includes `task_catalog_summary` and only a small public top-task list by default.
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

V3 should add failure classification and a structured validation runner that maps common compiler/test output into file, line, and message diagnostics.

V4 should support learning from successful user-provided commands by writing `.serena/tasks.json` suggestions instead of hardcoding framework-specific behavior.
