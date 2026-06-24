"""
Project task discovery helpers for coding harness workflows.
"""

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALWAYS_SKIPPED_DIRS = {
    ".git",
}

SKIPPED_DIRS = {
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".serena",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

FIXTURE_DIR_NAMES = {
    "__fixtures__",
    "fixture",
    "fixtures",
    "testdata",
    "test_data",
}

FIXTURE_PATH_PREFIXES = {
    ("test", "resources"),
    ("tests", "resources"),
    ("test", "fixtures"),
    ("tests", "fixtures"),
}

EXAMPLE_DIR_NAMES = {
    "demo",
    "demos",
    "example",
    "examples",
    "sample",
    "samples",
}

MANIFEST_NAMES = {
    "package.json",
    "Cargo.toml",
    "pyproject.toml",
    "Makefile",
    "makefile",
    "GNUmakefile",
    "justfile",
    "Justfile",
    ".justfile",
    "Taskfile.yml",
    "Taskfile.yaml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
    "Gemfile",
    "CMakeLists.txt",
}

VALIDATION_KINDS = {"build", "check", "format", "lint", "test", "typecheck", "verify"}


@dataclass(frozen=True)
class ValidationHints:
    """Likely validation commands inferred from discovered tasks."""

    package_files: list[str]
    detected_package_managers: list[str]
    likely_commands: list[str]


@dataclass(frozen=True)
class ProjectTask:
    """Runnable project task discovered from manifests or project overrides."""

    task_id: str
    kind: str
    runner: str
    command: str
    workdir: str
    source: str
    confidence: str
    interactive: bool = False
    long_running: bool = False
    visibility: str = "public"
    priority: int = 100


@dataclass(frozen=True)
class TaskCatalog:
    """Project task catalog inferred without requiring semantic indexing."""

    package_files: list[str]
    detected_package_managers: list[str]
    tasks: list[ProjectTask]
    validation_hints: ValidationHints

    def filtered(self, include_internal: bool = False, max_tasks: int | None = None) -> "TaskCatalog":
        """
        Return a catalog view suitable for user-facing agent decisions.

        :param include_internal: whether internal helper tasks should be included
        :param max_tasks: maximum number of tasks to include in the returned view
        :return: filtered task catalog
        """
        tasks = self.tasks if include_internal else [task for task in self.tasks if task.visibility != "internal"]
        if max_tasks is not None:
            tasks = tasks[: max(0, max_tasks)]

        hints = ValidationHints(
            package_files=self.package_files,
            detected_package_managers=self.detected_package_managers,
            likely_commands=[task.command for task in tasks if task.kind in VALIDATION_KINDS],
        )
        return TaskCatalog(
            package_files=self.package_files,
            detected_package_managers=self.detected_package_managers,
            tasks=tasks,
            validation_hints=hints,
        )

    def summary(self) -> dict[str, object]:
        """
        Return a compact task catalog summary.

        :return: summary grouped by kind and visibility
        """
        by_kind: dict[str, int] = {}
        by_visibility: dict[str, int] = {}
        for task in self.tasks:
            by_kind[task.kind] = by_kind.get(task.kind, 0) + 1
            by_visibility[task.visibility] = by_visibility.get(task.visibility, 0) + 1

        return {
            "package_files_count": len(self.package_files),
            "detected_package_managers": self.detected_package_managers,
            "task_count": len(self.tasks),
            "by_kind": dict(sorted(by_kind.items())),
            "by_visibility": dict(sorted(by_visibility.items())),
        }


def discover_task_catalog(
    project_root: Path,
    max_depth: int = 5,
    include_fixtures: bool = False,
    include_examples: bool = False,
    include_ignored: bool = False,
) -> TaskCatalog:
    """
    Discover runnable project tasks from manifests and overrides.

    :param project_root: project root directory
    :param max_depth: maximum directory depth for manifest discovery
    :param include_fixtures: whether fixture and test-resource manifests should be scanned
    :param include_examples: whether example, sample, and demo manifests should be scanned
    :param include_ignored: whether generated/dependency/cache directories should be scanned
    :return: discovered task catalog
    """
    project_root = project_root.resolve()
    package_files: list[str] = []
    managers: list[str] = []
    tasks: list[ProjectTask] = []

    for override_path in _task_override_files(project_root):
        _append_unique(package_files, _relative_path(override_path, project_root))
        tasks.extend(_tasks_from_override(project_root, override_path, managers))

    manifests = _iter_manifest_files(
        project_root,
        max_depth=max_depth,
        include_fixtures=include_fixtures,
        include_examples=include_examples,
        include_ignored=include_ignored,
    )
    for manifest in manifests:
        _append_unique(package_files, _relative_path(manifest, project_root))
        tasks.extend(_tasks_from_manifest(project_root, manifest, managers))

    tasks = _dedupe_tasks(tasks)
    hints = ValidationHints(
        package_files=package_files,
        detected_package_managers=managers,
        likely_commands=[task.command for task in tasks if task.kind in VALIDATION_KINDS],
    )
    return TaskCatalog(package_files=package_files, detected_package_managers=managers, tasks=tasks, validation_hints=hints)


def infer_validation_hints(project_root: Path, include_internal: bool = False, max_tasks: int = 30) -> ValidationHints:
    """
    Infer validation hints from the universal task catalog.

    :param project_root: project root directory
    :param include_internal: whether internal helper tasks should be included
    :param max_tasks: maximum number of tasks to consider
    :return: likely validation commands and sources
    """
    return discover_task_catalog(project_root).filtered(include_internal=include_internal, max_tasks=max_tasks).validation_hints


def _task_override_files(project_root: Path) -> list[Path]:
    """Project override files in priority order."""
    return [path for path in [project_root / ".serena" / "tasks.json"] if path.is_file()]


def _iter_manifest_files(
    project_root: Path,
    max_depth: int,
    include_fixtures: bool,
    include_examples: bool,
    include_ignored: bool,
) -> list[Path]:
    """Manifest files discovered by bounded directory traversal."""
    manifests: list[Path] = []
    frontier = [project_root]

    while frontier:
        directory = frontier.pop(0)
        depth = len(directory.relative_to(project_root).parts)
        if depth > max_depth:
            continue
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            continue

        for child in children:
            relative_parts = child.relative_to(project_root).parts
            if child.is_dir():
                if not _should_scan_directory(relative_parts, include_fixtures, include_examples, include_ignored):
                    continue
                frontier.append(child)
                continue
            if (child.name in MANIFEST_NAMES or child.suffix == ".csproj") and _should_scan_manifest(
                relative_parts,
                include_fixtures,
                include_examples,
            ):
                manifests.append(child)

    return manifests


def _should_scan_directory(
    relative_parts: tuple[str, ...],
    include_fixtures: bool,
    include_examples: bool,
    include_ignored: bool,
) -> bool:
    """
    Return whether a directory should be traversed during task discovery.

    :param relative_parts: project-relative directory parts
    :param include_fixtures: whether fixture/test-resource directories are allowed
    :param include_examples: whether example/sample/demo directories are allowed
    :param include_ignored: whether generated/dependency/cache directories are allowed
    :return: whether the directory should be scanned
    """
    name = relative_parts[-1]

    if name in ALWAYS_SKIPPED_DIRS:
        return False
    if not include_ignored and (name in SKIPPED_DIRS or name.startswith(".")):
        return False
    if not include_fixtures and _is_fixture_path(relative_parts):
        return False
    if not include_examples and _is_example_path(relative_parts):
        return False

    return True


def _should_scan_manifest(relative_parts: tuple[str, ...], include_fixtures: bool, include_examples: bool) -> bool:
    """
    Return whether a manifest should be included in the discovered catalog.

    :param relative_parts: project-relative manifest path parts
    :param include_fixtures: whether fixture/test-resource manifests are allowed
    :param include_examples: whether example/sample/demo manifests are allowed
    :return: whether the manifest should be scanned
    """
    parent_parts = relative_parts[:-1]

    if not include_fixtures and _is_fixture_path(parent_parts):
        return False
    if not include_examples and _is_example_path(parent_parts):
        return False

    return True


def _is_fixture_path(relative_parts: tuple[str, ...]) -> bool:
    """
    Return whether a path points into fixture or test-resource content.

    :param relative_parts: project-relative path parts
    :return: whether the path is fixture-like
    """
    parts = tuple(part.lower() for part in relative_parts)
    if any(part in FIXTURE_DIR_NAMES for part in parts):
        return True
    return any(parts[: len(prefix)] == prefix for prefix in FIXTURE_PATH_PREFIXES)


def _is_example_path(relative_parts: tuple[str, ...]) -> bool:
    """
    Return whether a path points into example, sample, or demo content.

    :param relative_parts: project-relative path parts
    :return: whether the path is example-like
    """
    return any(part.lower() in EXAMPLE_DIR_NAMES for part in relative_parts)


def _tasks_from_manifest(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks inferred from a single manifest file."""
    if manifest.name == "package.json":
        return _tasks_from_package_json(project_root, manifest, managers)
    if manifest.name == "pyproject.toml":
        return _tasks_from_pyproject(project_root, manifest, managers)
    if manifest.name == "Cargo.toml":
        return _tasks_from_cargo(project_root, manifest, managers)
    if manifest.name == "go.mod":
        return _tasks_from_go(project_root, manifest, managers)
    if manifest.name in {"Makefile", "makefile", "GNUmakefile"}:
        return _tasks_from_makefile(project_root, manifest, managers)
    if manifest.name in {"justfile", "Justfile", ".justfile"}:
        return _tasks_from_justfile(project_root, manifest, managers)
    if manifest.name in {"Taskfile.yml", "Taskfile.yaml"}:
        return _tasks_from_taskfile(project_root, manifest, managers)
    if manifest.name == "composer.json":
        return _tasks_from_composer(project_root, manifest, managers)
    if manifest.name in {"pom.xml", "build.gradle", "build.gradle.kts", "Gemfile", "CMakeLists.txt"} or manifest.suffix == ".csproj":
        return _tasks_from_generic_manifest(project_root, manifest, managers)
    return []


def _tasks_from_package_json(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks inferred from package.json scripts."""
    scripts = _read_json_file(manifest).get("scripts", {})
    if not isinstance(scripts, dict):
        return []

    package_manager = _detect_node_package_manager(manifest.parent, managers)
    tasks: list[ProjectTask] = []
    for script in sorted(scripts):
        kind = _classify_task_name(script)
        tasks.append(
            _task(
                project_root,
                manifest.parent,
                script,
                kind,
                package_manager,
                _node_run_command(package_manager, script),
                f"{_relative_path(manifest, project_root)}:scripts.{script}",
                "high",
                long_running=kind == "dev",
            )
        )
    return tasks


def _tasks_from_pyproject(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks inferred from Python project metadata."""
    data = _read_toml_file(manifest)
    runner = "uv" if (manifest.parent / "uv.lock").is_file() else "python"
    _append_unique(managers, runner)
    tasks: list[ProjectTask] = []

    poe_tasks = data.get("tool", {}).get("poe", {}).get("tasks", {})
    if isinstance(poe_tasks, dict):
        _append_unique(managers, "poe")
        prefix = "uv run poe" if runner == "uv" else "poe"
        for name in sorted(poe_tasks):
            kind = _classify_task_name(name)
            tasks.append(
                _task(
                    project_root,
                    manifest.parent,
                    name,
                    kind,
                    "poe",
                    f"{prefix} {name}",
                    f"{_relative_path(manifest, project_root)}:tool.poe.tasks.{name}",
                    "high",
                    long_running=kind == "dev",
                )
            )

    prefix = "uv run" if runner == "uv" else "python -m"
    if any((manifest.parent / name).exists() for name in ("tests", "test", "pytest.ini")):
        tasks.append(
            _task(
                project_root,
                manifest.parent,
                "test",
                "test",
                runner,
                f"{prefix} pytest",
                f"{_relative_path(manifest, project_root)}:pytest-fallback",
                "medium",
            )
        )
    if isinstance(data.get("tool", {}).get("ruff"), dict):
        tasks.append(
            _task(
                project_root,
                manifest.parent,
                "lint",
                "lint",
                runner,
                f"{prefix} ruff check .",
                f"{_relative_path(manifest, project_root)}:tool.ruff",
                "medium",
            )
        )
    if isinstance(data.get("tool", {}).get("mypy"), dict):
        tasks.append(
            _task(
                project_root,
                manifest.parent,
                "typecheck",
                "typecheck",
                runner,
                f"{prefix} mypy .",
                f"{_relative_path(manifest, project_root)}:tool.mypy",
                "medium",
            )
        )
    return tasks


def _tasks_from_cargo(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks inferred from Cargo projects."""
    _append_unique(managers, "cargo")
    return [
        _task(
            project_root,
            manifest.parent,
            "check",
            "check",
            "cargo",
            "cargo check",
            f"{_relative_path(manifest, project_root)}:builtin.check",
            "high",
        ),
        _task(
            project_root,
            manifest.parent,
            "test",
            "test",
            "cargo",
            "cargo test",
            f"{_relative_path(manifest, project_root)}:builtin.test",
            "high",
        ),
        _task(
            project_root,
            manifest.parent,
            "clippy",
            "lint",
            "cargo",
            "cargo clippy --all-targets --all-features",
            f"{_relative_path(manifest, project_root)}:builtin.clippy",
            "medium",
        ),
        _task(
            project_root,
            manifest.parent,
            "fmt-check",
            "format",
            "cargo",
            "cargo fmt --check",
            f"{_relative_path(manifest, project_root)}:builtin.fmt",
            "medium",
        ),
    ]


def _tasks_from_go(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks inferred from Go modules."""
    _append_unique(managers, "go")
    return [
        _task(
            project_root,
            manifest.parent,
            "test",
            "test",
            "go",
            "go test ./...",
            f"{_relative_path(manifest, project_root)}:builtin.test",
            "high",
        ),
        _task(
            project_root,
            manifest.parent,
            "vet",
            "lint",
            "go",
            "go vet ./...",
            f"{_relative_path(manifest, project_root)}:builtin.vet",
            "medium",
        ),
    ]


def _tasks_from_makefile(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks inferred from Makefile targets."""
    _append_unique(managers, "make")
    return [
        _task(
            project_root,
            manifest.parent,
            target,
            _classify_task_name(target),
            "make",
            f"make {target}",
            f"{_relative_path(manifest, project_root)}:{target}",
            "medium",
            long_running=_classify_task_name(target) == "dev",
        )
        for target in _parse_make_targets(manifest)
    ]


def _tasks_from_justfile(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks inferred from justfile recipes."""
    _append_unique(managers, "just")
    return [
        _task(
            project_root,
            manifest.parent,
            recipe,
            _classify_task_name(recipe),
            "just",
            f"just {recipe}",
            f"{_relative_path(manifest, project_root)}:{recipe}",
            "medium",
            long_running=_classify_task_name(recipe) == "dev",
        )
        for recipe in _parse_just_recipes(manifest)
    ]


def _tasks_from_taskfile(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks inferred from Taskfile task names."""
    _append_unique(managers, "task")
    return [
        _task(
            project_root,
            manifest.parent,
            name,
            _classify_task_name(name),
            "task",
            f"task {name}",
            f"{_relative_path(manifest, project_root)}:tasks.{name}",
            "medium",
            long_running=_classify_task_name(name) == "dev",
        )
        for name in _parse_taskfile_names(manifest)
    ]


def _tasks_from_composer(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks inferred from composer.json scripts."""
    _append_unique(managers, "composer")
    scripts = _read_json_file(manifest).get("scripts", {})
    if not isinstance(scripts, dict):
        return []
    return [
        _task(
            project_root,
            manifest.parent,
            name,
            _classify_task_name(name),
            "composer",
            f"composer run {name}",
            f"{_relative_path(manifest, project_root)}:scripts.{name}",
            "high",
        )
        for name in sorted(scripts)
    ]


def _tasks_from_generic_manifest(project_root: Path, manifest: Path, managers: list[str]) -> list[ProjectTask]:
    """Conservative built-in tasks for common non-script manifests."""
    relative = _relative_path(manifest, project_root)
    if manifest.name == "pom.xml":
        _append_unique(managers, "maven")
        return [_task(project_root, manifest.parent, "test", "test", "maven", "mvn test", f"{relative}:builtin.test", "medium")]
    if manifest.name in {"build.gradle", "build.gradle.kts"}:
        _append_unique(managers, "gradle")
        executable = "./gradlew" if (manifest.parent / "gradlew").is_file() else "gradle"
        return [_task(project_root, manifest.parent, "test", "test", "gradle", f"{executable} test", f"{relative}:builtin.test", "medium")]
    if manifest.name == "Gemfile":
        _append_unique(managers, "bundler")
        command = "bundle exec rake test" if (manifest.parent / "Rakefile").is_file() else "bundle exec ruby -c"
        return [_task(project_root, manifest.parent, "test", "test", "bundler", command, f"{relative}:builtin.test", "low")]
    if manifest.name == "CMakeLists.txt":
        _append_unique(managers, "cmake")
        return [_task(project_root, manifest.parent, "build", "build", "cmake", "cmake --build build", f"{relative}:builtin.build", "low")]
    if manifest.suffix == ".csproj":
        _append_unique(managers, "dotnet")
        return [_task(project_root, manifest.parent, "test", "test", "dotnet", "dotnet test", f"{relative}:builtin.test", "medium")]
    return []


def _tasks_from_override(project_root: Path, override_path: Path, managers: list[str]) -> list[ProjectTask]:
    """Tasks loaded from .serena/tasks.json override."""
    data = _read_json_file(override_path)
    raw_tasks = data.get("tasks", []) if isinstance(data, dict) else []
    if not isinstance(raw_tasks, list):
        return []

    tasks: list[ProjectTask] = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            continue
        task_id = str(raw_task.get("task_id") or raw_task.get("id") or "").strip()
        command = str(raw_task.get("command") or "").strip()
        if not task_id or not command:
            continue
        runner = str(raw_task.get("runner") or "custom")
        _append_unique(managers, runner)
        tasks.append(
            ProjectTask(
                task_id=task_id,
                kind=str(raw_task.get("kind") or _classify_task_name(task_id)),
                runner=runner,
                command=command,
                workdir=str(raw_task.get("workdir") or "."),
                source=_relative_path(override_path, project_root),
                confidence=str(raw_task.get("confidence") or "override"),
                interactive=bool(raw_task.get("interactive", False)),
                long_running=bool(raw_task.get("long_running", False)),
            )
        )
    return tasks


def _task(
    project_root: Path,
    workdir: Path,
    name: str,
    kind: str,
    runner: str,
    command: str,
    source: str,
    confidence: str,
    *,
    interactive: bool = False,
    long_running: bool = False,
) -> ProjectTask:
    """ProjectTask constructed with a stable project-relative task id."""
    workdir_text = _relative_path(workdir, project_root)
    prefix = "root" if workdir_text == "." else workdir_text.replace("/", ":")
    runner_key = runner.replace("/", ":")
    return ProjectTask(
        task_id=f"{prefix}:{runner_key}:{kind}:{name}" if not name.startswith(f"{prefix}:") else name,
        kind=kind,
        runner=runner,
        command=command,
        workdir=workdir_text,
        source=source,
        confidence=confidence,
        interactive=interactive,
        long_running=long_running,
        visibility=_task_visibility(name),
        priority=_task_priority(kind, name),
    )


def _task_visibility(name: str) -> str:
    """Visibility level for catalog presentation."""
    if name.startswith("_"):
        return "internal"
    return "public"


def _task_priority(kind: str, name: str) -> int:
    """Sort priority for task catalog presentation."""
    base_priority = {
        "check": 10,
        "typecheck": 20,
        "lint": 30,
        "test": 40,
        "format": 50,
        "build": 60,
        "dev": 70,
        "run": 80,
        "migrate": 90,
        "task": 100,
    }.get(kind, 100)
    if name.startswith("_"):
        return base_priority + 1000
    return base_priority


def _classify_task_name(name: str) -> str:
    """Semantic task kind inferred from a script, recipe, or target name."""
    normalized = name.lower().replace("_", "-")
    if any(token in normalized for token in ("typecheck", "type-check", "tsc")):
        return "typecheck"
    if any(token in normalized for token in ("lint", "clippy", "vet")):
        return "lint"
    if any(token in normalized for token in ("fmt", "format")):
        return "format"
    if normalized == "check":
        return "check"
    if "test" in normalized or normalized == "spec":
        return "test"
    if any(token in normalized for token in ("build", "compile", "package")):
        return "build"
    if any(token in normalized for token in ("dev", "serve", "watch", "start", "run")):
        return "dev"
    if any(token in normalized for token in ("verify", "validate", "ci")):
        return "verify"
    if any(token in normalized for token in ("migrate", "migration")):
        return "migrate"
    return "task"


def _detect_node_package_manager(package_root: Path, managers: list[str]) -> str:
    """Node package manager inferred from lock files near package.json."""
    if (package_root / "pnpm-lock.yaml").is_file():
        _append_unique(managers, "pnpm")
        return "pnpm"
    if (package_root / "bun.lockb").is_file() or (package_root / "bun.lock").is_file():
        _append_unique(managers, "bun")
        return "bun"
    if (package_root / "yarn.lock").is_file():
        _append_unique(managers, "yarn")
        return "yarn"
    _append_unique(managers, "npm")
    return "npm"


def _node_run_command(package_manager: str, script: str) -> str:
    """Run command for package.json scripts."""
    if package_manager == "yarn":
        return f"yarn {script}"
    if package_manager == "pnpm":
        return f"pnpm {script}"
    if package_manager == "bun":
        return f"bun run {script}"
    return "npm test" if script == "test" else f"npm run {script}"


def _parse_make_targets(path: Path) -> list[str]:
    """User-facing Make targets parsed conservatively."""
    targets: list[str] = []
    for line in _read_text_lines(path):
        if line.startswith(("\t", ".")):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*):(?:\s|$)", line)
        if match:
            _append_unique(targets, match.group(1))
    return targets[:100]


def _parse_just_recipes(path: Path) -> list[str]:
    """Just recipes parsed without running external commands."""
    recipes: list[str] = []
    for line in _read_text_lines(path):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or line.startswith((" ", "\t")):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)(?:\s|:|$)", stripped)
        if match and not stripped.startswith(("set ", "export ", "alias ")):
            _append_unique(recipes, match.group(1))
    return recipes[:100]


def _parse_taskfile_names(path: Path) -> list[str]:
    """Taskfile task names parsed from the top-level tasks section."""
    names: list[str] = []
    in_tasks = False
    for line in _read_text_lines(path):
        if re.match(r"^tasks:\s*$", line):
            in_tasks = True
            continue
        if in_tasks and line and not line.startswith(" "):
            break
        if in_tasks:
            match = re.match(r"^\s{2}([A-Za-z0-9_.-]+):\s*$", line)
            if match:
                _append_unique(names, match.group(1))
    return names[:100]


def _dedupe_tasks(tasks: list[ProjectTask]) -> list[ProjectTask]:
    """Tasks deduplicated by task_id with first task winning."""
    seen: set[str] = set()
    deduped: list[ProjectTask] = []
    for task in tasks:
        if task.task_id in seen:
            continue
        seen.add(task.task_id)
        deduped.append(task)
    return sorted(deduped, key=lambda task: (_workdir_depth(task.workdir), task.priority, task.workdir, task.task_id))


def _workdir_depth(workdir: str) -> int:
    """Directory depth used to keep root workspace tasks ahead of nested manifests."""
    if workdir in {"", "."}:
        return 0
    return len(Path(workdir).parts)


def _read_json_file(path: Path) -> dict[str, Any]:
    """JSON object read from a file, returning an empty object on parse errors."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_toml_file(path: Path) -> dict[str, Any]:
    """TOML object read from a file, returning an empty object on parse errors."""
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _read_text_lines(path: Path) -> list[str]:
    """Text file lines with invalid or missing files treated as empty."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _append_unique(items: list[str], item: str) -> None:
    """Append item when not already present."""
    if item not in items:
        items.append(item)


def _relative_path(path: Path, project_root: Path) -> str:
    """Project-relative path text."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix() or "."
    except ValueError:
        return path.as_posix()
