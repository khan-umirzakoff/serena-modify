"""
Tools supporting the execution of (external) commands
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import cast

from serena.tools import Tool, ToolMarkerCanEdit
from solidlsp.util.subprocess_util import subprocess_kwargs, terminate_process_tree_with_kill_fallback

DEFAULT_YIELD_TIME_MS = 10_000
DEFAULT_STDIN_YIELD_TIME_MS = 250
DEFAULT_EMPTY_POLL_YIELD_TIME_MS = 5_000
MIN_YIELD_TIME_MS = 250
MAX_YIELD_TIME_MS = 30_000
MAX_EMPTY_POLL_YIELD_TIME_MS = 300_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
MAX_LIVE_TERMINAL_SESSIONS = 64
PROTECTED_RECENT_TERMINAL_SESSIONS = 8
EXPECTED_CODING_CONTEXT_MAX_AGE_SECONDS = 12 * 60 * 60
INTERRUPT = "\u0003"
TERMINAL_SIGNAL_NAMES: dict[str, signal.Signals] = {
    "INT": signal.SIGINT,
    "SIGINT": signal.SIGINT,
    "TERM": signal.SIGTERM,
    "SIGTERM": signal.SIGTERM,
    "KILL": signal.SIGKILL,
    "SIGKILL": signal.SIGKILL,
}
if hasattr(signal, "SIGHUP"):
    TERMINAL_SIGNAL_NAMES.update({"HUP": signal.SIGHUP, "SIGHUP": signal.SIGHUP})
if hasattr(signal, "SIGQUIT"):
    TERMINAL_SIGNAL_NAMES.update({"QUIT": signal.SIGQUIT, "SIGQUIT": signal.SIGQUIT})
UNIFIED_EXEC_ENV = {
    "NO_COLOR": "1",
    "TERM": "dumb",
    "LANG": "C.UTF-8",
    "LC_CTYPE": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "COLORTERM": "",
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GH_PAGER": "cat",
    "SERENA_CI": "1",
}


def _bounded(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _max_output_bytes(max_output_tokens: int | None) -> int:
    tokens = max_output_tokens if max_output_tokens is not None and max_output_tokens > 0 else DEFAULT_MAX_OUTPUT_TOKENS
    return _bounded(tokens * 4, 1024, 1024 * 1024)


def _approx_token_count(text: str) -> int:
    return max(len(text) // 4, len(text.split()))


def _chunk_id() -> str:
    return secrets.token_hex(3)


def _resolve_terminal_signal(signal_name: str) -> tuple[str, signal.Signals]:
    normalized = signal_name.strip().upper()
    if not normalized:
        raise ValueError("terminal signal must not be empty")

    signum = TERMINAL_SIGNAL_NAMES.get(normalized)
    if signum is None:
        allowed = ", ".join(sorted(TERMINAL_SIGNAL_NAMES))
        raise ValueError(f"Unsupported terminal signal: {signal_name}. Allowed: {allowed}")

    canonical = f"SIG{normalized}" if not normalized.startswith("SIG") else normalized
    return canonical, signum


class HeadTailBuffer:
    """
    Bounded byte buffer that keeps a stable prefix and suffix while dropping the middle.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max(0, max_bytes)
        self._head_budget = self._max_bytes // 2
        self._tail_budget = self._max_bytes - self._head_budget
        self._head = bytearray()
        self._tail = bytearray()
        self._omitted_bytes = 0

    @property
    def omitted_bytes(self) -> int:
        return self._omitted_bytes

    def push(self, chunk: bytes) -> None:
        if not chunk:
            return
        if self._max_bytes == 0:
            self._omitted_bytes += len(chunk)
            return

        remaining = chunk
        if len(self._head) < self._head_budget:
            take = min(self._head_budget - len(self._head), len(remaining))
            self._head.extend(remaining[:take])
            remaining = remaining[take:]

        if not remaining:
            return

        if self._tail_budget == 0:
            self._omitted_bytes += len(remaining)
            return

        self._tail.extend(remaining)
        if len(self._tail) > self._tail_budget:
            excess = len(self._tail) - self._tail_budget
            del self._tail[:excess]
            self._omitted_bytes += excess

    def snapshot(self) -> bytes:
        if self._omitted_bytes:
            marker = f"\n...[omitted {self._omitted_bytes} bytes from middle]...\n".encode()
            return bytes(self._head) + marker + bytes(self._tail)
        return bytes(self._head) + bytes(self._tail)

    def drain(self) -> tuple[bytes, int]:
        data = self.snapshot()
        omitted = self._omitted_bytes
        self._head.clear()
        self._tail.clear()
        self._omitted_bytes = 0
        return data, omitted


@dataclass(frozen=True)
class TerminalResponse:
    chunk_id: str
    command: str
    cwd: str
    session_id: int | None
    pid: int
    running: bool
    exit_code: int | None
    duration_ms: int
    wall_time_seconds: float
    output: str
    original_token_count: int
    omitted_bytes: int
    log_path: str
    transport: str
    timed_out: bool = False
    pty_requested_but_pipe_used: bool = False
    warnings: list[str] = field(default_factory=list)


class TerminalSession:
    """
    Terminal session with Codex-like bounded recent output and a full temp log.
    """

    def __init__(
        self,
        session_id: int,
        command: str,
        cwd: Path,
        process: subprocess.Popen[bytes],
        output_fd: int,
        write_fd: int | None,
        log_path: Path,
        transport: str,
        max_output_bytes: int,
    ) -> None:
        self.session_id = session_id
        self.command = command
        self.cwd = cwd
        self.process = process
        self.output_fd = output_fd
        self.write_fd = write_fd
        self.log_path = log_path
        self.transport = transport
        self.started_at = time.monotonic()
        self.last_used = self.started_at
        self._pending_output = HeadTailBuffer(max_output_bytes)
        self._all_output = HeadTailBuffer(max_output_bytes)
        self._condition = threading.Condition()
        self._reader_done = False
        self._log_file = log_path.open("ab", buffering=0)
        self._reader_thread = threading.Thread(target=self._read_output, name=f"serena-terminal-{session_id}", daemon=True)
        self._reader_thread.start()

    def _read_output(self) -> None:
        try:
            while True:
                try:
                    chunk = os.read(self.output_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                with self._condition:
                    self._pending_output.push(chunk)
                    self._all_output.push(chunk)
                    self._log_file.write(chunk)
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._reader_done = True
                self._condition.notify_all()
            try:
                self._log_file.close()
            except OSError:
                pass
            try:
                os.close(self.output_fd)
            except OSError:
                pass

    def _wait_for_yield(self, yield_time_ms: int) -> bool:
        deadline = time.monotonic() + (yield_time_ms / 1000)
        while True:
            if self.process.poll() is not None:
                self._reader_thread.join(timeout=0.2)
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            with self._condition:
                self._condition.wait(timeout=min(remaining, 0.05))

    def write(self, chars: str) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"Terminal session {self.session_id} has already exited")
        self.last_used = time.monotonic()
        if self.transport != "pty":
            if chars == INTERRUPT:
                self.interrupt()
                return
            raise RuntimeError("stdin is closed for non-PTY sessions; start exec_command with tty=true for interactive input")
        if self.write_fd is None:
            raise RuntimeError(f"Terminal session {self.session_id} stdin is closed")
        os.write(self.write_fd, chars.encode())

    def send_signal(self, signal_name: str) -> str:
        canonical, signum = _resolve_terminal_signal(signal_name)
        self.last_used = time.monotonic()
        if self.process.poll() is not None:
            return canonical
        if os.name == "posix":
            try:
                os.killpg(self.process.pid, signum)
                return canonical
            except OSError:
                pass
        self.process.send_signal(signum)
        return canonical

    def interrupt(self) -> None:
        self.send_signal("SIGINT")

    def collect_response(self, yield_time_ms: int, drain: bool = True) -> TerminalResponse:
        self.last_used = time.monotonic()
        timed_out = self._wait_for_yield(yield_time_ms)
        running = self.process.poll() is None
        if drain:
            output_bytes, omitted = self._pending_output.drain()
        else:
            output_bytes = self._all_output.snapshot()
            omitted = self._all_output.omitted_bytes
        output = output_bytes.decode("utf-8", errors="replace")
        duration_ms = int((time.monotonic() - self.started_at) * 1000)
        return TerminalResponse(
            chunk_id=_chunk_id(),
            command=self.command,
            cwd=str(self.cwd),
            session_id=self.session_id if running else None,
            pid=self.process.pid,
            running=running,
            exit_code=self.process.returncode,
            duration_ms=duration_ms,
            wall_time_seconds=duration_ms / 1000,
            output=output,
            original_token_count=_approx_token_count(output),
            omitted_bytes=omitted,
            log_path=str(self.log_path),
            transport=self.transport,
            timed_out=timed_out and running,
            pty_requested_but_pipe_used=False,
        )

    def terminate(self) -> None:
        if self.process.poll() is None:
            terminate_process_tree_with_kill_fallback(self.process, terminate_timeout=2.0, process_name="Terminal session")
        if self.write_fd is not None:
            try:
                os.close(self.write_fd)
            except OSError:
                pass

    def info(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "pid": self.process.pid,
            "command": self.command,
            "cwd": str(self.cwd),
            "transport": self.transport,
            "running": self.process.poll() is None,
            "exit_code": self.process.returncode,
            "duration_ms": int((time.monotonic() - self.started_at) * 1000),
            "last_used_age_ms": int((time.monotonic() - self.last_used) * 1000),
            "log_path": str(self.log_path),
        }

    def status(self, include_output: bool = False, max_output_tokens: int | None = None) -> dict[str, object]:
        self.last_used = time.monotonic()
        self.process.poll()
        status = self.info()
        status["reader_done"] = self._reader_done
        if include_output:
            output_bytes = self._all_output.snapshot()
            omitted = self._all_output.omitted_bytes
            if max_output_tokens is not None:
                buffer = HeadTailBuffer(_max_output_bytes(max_output_tokens))
                buffer.push(output_bytes)
                output_bytes, extra_omitted = buffer.drain()
                omitted += extra_omitted
            output = output_bytes.decode("utf-8", errors="replace")
            status.update(
                {
                    "output": output,
                    "original_token_count": _approx_token_count(output),
                    "omitted_bytes": omitted,
                }
            )
        return status


class TerminalProcessManager:
    """
    Serena-native process/session manager inspired by Codex unified exec.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_session_id = 1
        self._sessions: dict[int, TerminalSession] = {}

    def _allocate_session_id(self) -> int:
        with self._lock:
            session_id = self._next_session_id
            self._next_session_id += 1
            return session_id

    def _register(self, session: TerminalSession) -> None:
        with self._lock:
            self._cleanup_exited_locked()
            if len(self._sessions) >= MAX_LIVE_TERMINAL_SESSIONS:
                pruned = self._prune_session_locked()
                if pruned is None:
                    raise RuntimeError(f"Too many live terminal sessions; maximum is {MAX_LIVE_TERMINAL_SESSIONS}")
            self._sessions[session.session_id] = session

    def _cleanup_exited_locked(self) -> None:
        exited = [session for session in self._sessions.values() if session.process.poll() is not None]
        protected = {
            session.session_id
            for session in sorted(exited, key=lambda item: item.last_used, reverse=True)[:PROTECTED_RECENT_TERMINAL_SESSIONS]
        }
        for session in exited:
            if session.session_id not in protected:
                del self._sessions[session.session_id]

    def _prune_session_locked(self) -> TerminalSession | None:
        if not self._sessions:
            return None
        by_recency = sorted(self._sessions.values(), key=lambda item: item.last_used, reverse=True)
        protected = {session.session_id for session in by_recency[:PROTECTED_RECENT_TERMINAL_SESSIONS]}
        lru = sorted(self._sessions.values(), key=lambda item: item.last_used)
        victim = next((session for session in lru if session.session_id not in protected and session.process.poll() is not None), None)
        if victim is None:
            victim = next((session for session in lru if session.session_id not in protected), None)
        if victim is None:
            return None
        self._sessions.pop(victim.session_id, None)
        victim.terminate()
        return victim

    def _get(self, session_id: int) -> TerminalSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"No running terminal session with id {session_id}")
        return session

    def _unregister_if_exited(self, session: TerminalSession) -> None:
        if session.process.poll() is None:
            return
        with self._lock:
            self._cleanup_exited_locked()

    def exec_command(
        self,
        command: str,
        cwd: Path,
        yield_time_ms: int = DEFAULT_YIELD_TIME_MS,
        max_output_tokens: int | None = None,
        shell: str | None = None,
        login: bool = False,
        tty: bool = False,
    ) -> TerminalResponse:
        session_id = self._allocate_session_id()
        log_path = Path(tempfile.gettempdir()) / f"serena-exec-{session_id}-{int(time.time() * 1000)}.log"
        env = os.environ.copy()
        env.update(UNIFIED_EXEC_ENV)

        shell_path = shell or os.environ.get("SHELL")
        if shell_path:
            argv: str | list[str] = [shell_path, "-lc" if login else "-c", command]
            use_shell = False
        else:
            argv = command
            use_shell = True

        transport = "pty" if tty and os.name == "posix" else "pipe"
        pty_requested_but_pipe_used = tty and transport != "pty"
        master_fd: int | None = None
        slave_fd: int | None = None
        if transport == "pty":
            import pty

            master_fd, slave_fd = pty.openpty()
            process = subprocess.Popen(
                argv,
                cwd=str(cwd),
                shell=use_shell,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
                close_fds=True,
                start_new_session=True,
                **subprocess_kwargs(),
            )
            os.close(slave_fd)
            output_fd = master_fd
            write_fd = master_fd
        else:
            process = subprocess.Popen(
                argv,
                cwd=str(cwd),
                shell=use_shell,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=os.name == "posix",
                **subprocess_kwargs(),
            )
            assert process.stdout is not None
            output_fd = os.dup(process.stdout.fileno())
            write_fd = None
        process = cast(subprocess.Popen[bytes], process)
        session = TerminalSession(
            session_id=session_id,
            command=command,
            cwd=cwd,
            process=process,
            output_fd=output_fd,
            write_fd=write_fd,
            log_path=log_path,
            transport=transport,
            max_output_bytes=_max_output_bytes(max_output_tokens),
        )
        try:
            self._register(session)
        except Exception:
            session.terminate()
            raise
        response = session.collect_response(_bounded(yield_time_ms, MIN_YIELD_TIME_MS, MAX_YIELD_TIME_MS))
        if pty_requested_but_pipe_used:
            response = replace(response, pty_requested_but_pipe_used=True)
        self._unregister_if_exited(session)
        return response

    def write_stdin(
        self,
        session_id: int,
        chars: str = "",
        yield_time_ms: int | None = None,
        max_output_tokens: int | None = None,
    ) -> TerminalResponse:
        session = self._get(session_id)
        if max_output_tokens is not None:
            # Apply the requested budget to newly returned output without changing the session's full retained log.
            session._pending_output = HeadTailBuffer(_max_output_bytes(max_output_tokens))
        if chars:
            session.write(chars)
            effective_yield = DEFAULT_STDIN_YIELD_TIME_MS if yield_time_ms is None else yield_time_ms
            effective_yield = _bounded(effective_yield, MIN_YIELD_TIME_MS, MAX_YIELD_TIME_MS)
        else:
            effective_yield = DEFAULT_EMPTY_POLL_YIELD_TIME_MS if yield_time_ms is None else yield_time_ms
            effective_yield = _bounded(effective_yield, MIN_YIELD_TIME_MS, MAX_EMPTY_POLL_YIELD_TIME_MS)
        response = session.collect_response(effective_yield)
        self._unregister_if_exited(session)
        return response

    def list_sessions(self) -> list[dict[str, object]]:
        with self._lock:
            self._cleanup_exited_locked()
            sessions = sorted(self._sessions.values(), key=lambda item: item.session_id)
            return [session.info() for session in sessions if session.process.poll() is None]

    def session_status(
        self,
        session_id: int,
        include_output: bool = False,
        max_output_tokens: int | None = None,
    ) -> dict[str, object]:
        session = self._get(session_id)
        status = session.status(include_output=include_output, max_output_tokens=max_output_tokens)
        self._unregister_if_exited(session)
        return status

    def send_signal(
        self,
        session_id: int,
        signal_name: str = "SIGINT",
        yield_time_ms: int = DEFAULT_STDIN_YIELD_TIME_MS,
        max_output_tokens: int | None = None,
    ) -> tuple[str, TerminalResponse]:
        session = self._get(session_id)
        if max_output_tokens is not None:
            session._pending_output = HeadTailBuffer(_max_output_bytes(max_output_tokens))
        canonical = session.send_signal(signal_name)
        response = session.collect_response(_bounded(yield_time_ms, MIN_YIELD_TIME_MS, MAX_YIELD_TIME_MS))
        self._unregister_if_exited(session)
        return canonical, response

    def stop_session(self, session_id: int) -> bool:
        session = self._get(session_id)
        session.terminate()
        with self._lock:
            self._sessions.pop(session_id, None)
        return True


TERMINAL_PROCESS_MANAGER = TerminalProcessManager()


def _json_response(response: TerminalResponse) -> str:
    payload = asdict(response)
    # Keep Codex's trained output field names first while retaining Serena diagnostics.
    ordered = {
        "chunk_id": payload["chunk_id"],
        "wall_time_seconds": payload["wall_time_seconds"],
        "exit_code": payload["exit_code"],
        "session_id": payload["session_id"],
        "original_token_count": payload["original_token_count"],
        "output": payload["output"],
        "warnings": payload["warnings"],
        "command": payload["command"],
        "cwd": payload["cwd"],
        "pid": payload["pid"],
        "running": payload["running"],
        "duration_ms": payload["duration_ms"],
        "omitted_bytes": payload["omitted_bytes"],
        "log_path": payload["log_path"],
        "transport": payload["transport"],
        "timed_out": payload["timed_out"],
        "pty_requested_but_pipe_used": payload["pty_requested_but_pipe_used"],
    }
    return json.dumps(ordered, ensure_ascii=False, indent=2)


def _resolve_workdir(project_root: str, workdir: str | None) -> Path:
    if workdir is None or workdir == "":
        resolved = Path(project_root).resolve()
    else:
        candidate = Path(workdir)
        resolved = candidate.resolve() if candidate.is_absolute() else (Path(project_root) / candidate).resolve()

    if not resolved.is_dir():
        raise FileNotFoundError(f"Working directory does not exist: {resolved}")
    return resolved


def _load_expected_coding_git_root(project_root: Path) -> Path | None:
    """
    Return a fresh coding-task Git root recorded by ``prepare_coding_task``.
    """
    context_path = project_root / ".serena" / "coding_task_context.json"
    if not context_path.is_file():
        return None
    try:
        if time.time() - context_path.stat().st_mtime > EXPECTED_CODING_CONTEXT_MAX_AGE_SECONDS:
            return None
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(context, dict):
        return None

    context_project_root = context.get("project_root")
    if not isinstance(context_project_root, str) or Path(context_project_root).resolve() != project_root.resolve():
        return None

    git_root = context.get("git_root")
    if not isinstance(git_root, str):
        return None
    resolved_git_root = Path(git_root).resolve()
    try:
        resolved_git_root.relative_to(project_root.resolve())
    except ValueError:
        return None
    return resolved_git_root


def _find_enclosing_git_root(path: Path) -> Path | None:
    """
    Return the nearest enclosing Git repository root.
    """
    for current in [path, *path.parents]:
        if (current / ".git").exists():
            return current
    return None


def _terminal_context_warnings(project_root: Path, workdir: Path) -> list[str]:
    """
    Return warnings for terminal context that differs from Serena's active project.
    """
    warnings: list[str] = []
    project_root = project_root.resolve()
    workdir = workdir.resolve()

    try:
        workdir.relative_to(project_root)
    except ValueError:
        warnings.append(
            "Terminal workdir is outside the active Serena project root. "
            f"Terminal cwd: {workdir}. Active project root: {project_root}. "
            "Semantic tools and task discovery still use the active project; call activate_project(...) if this is the intended project."
        )
        return warnings

    workdir_git_root = _find_enclosing_git_root(workdir)
    if workdir_git_root is not None and workdir_git_root.resolve() != project_root:
        resolved_workdir_git_root = workdir_git_root.resolve()
        expected_git_root = _load_expected_coding_git_root(project_root)
        if expected_git_root == resolved_workdir_git_root:
            warnings.append(
                "Terminal workdir matches the latest prepare_coding_task nested Git root. "
                f"Nested Git root: {resolved_workdir_git_root}. Active Serena workspace root: {project_root}. "
                "Terminal commands are scoped to the expected nested repo."
            )
        else:
            warnings.append(
                "Terminal workdir is inside a nested Git repository while the active Serena project is different. "
                f"Nested Git root: {resolved_workdir_git_root}. Active project root: {project_root}. "
                "Terminal commands will run in the nested repo, but semantic tools and task discovery still use the active project; "
                "call activate_project(...) for that repo if intended."
            )

    return warnings


def _with_context_warnings(response: TerminalResponse, project_root: Path, workdir: Path) -> TerminalResponse:
    """
    Return a terminal response annotated with project/workdir context warnings.
    """
    warnings = [*response.warnings, *_terminal_context_warnings(project_root, workdir)]
    if not warnings:
        return response
    return replace(response, warnings=warnings)


class ExecuteShellCommandTool(Tool, ToolMarkerCanEdit):
    """
    Compatibility wrapper around Serena's Codex-style terminal engine.
    """

    def apply(
        self,
        command: str,
        cwd: str | None = None,
        capture_stderr: bool = True,
        max_answer_chars: int = -1,
        timeout_seconds: int = 120,
        background: bool = False,
    ) -> str:
        """
        Execute a shell command and return its output. If there is a memory about suggested commands, read that first.
        Prefer `exec_command` for new terminal work and `write_stdin` to continue an ongoing command.

        :param command: the shell command to execute
        :param cwd: the working directory to execute the command in. If None, the project root will be used.
        :param capture_stderr: kept for compatibility; stderr is merged into the bounded terminal output.
        :param max_answer_chars: maximum JSON response length; -1 uses the configured default.
        :param timeout_seconds: initial wait before returning a session for longer-running commands.
        :param background: if true, return quickly with a session ID for the running command.
        :return: a JSON object containing command metadata, bounded output, exit code, and session/log information
        """
        del capture_stderr
        project_root = Path(self.get_project_root()).resolve()
        workdir = _resolve_workdir(str(project_root), cwd)
        yield_time_ms = MIN_YIELD_TIME_MS if background else _bounded(timeout_seconds * 1000, MIN_YIELD_TIME_MS, MAX_YIELD_TIME_MS)
        response = TERMINAL_PROCESS_MANAGER.exec_command(command=command, cwd=workdir, yield_time_ms=yield_time_ms)
        response = _with_context_warnings(response, project_root, workdir)
        return self._limit_length(_json_response(response), max_answer_chars)


class ExecCommandTool(Tool, ToolMarkerCanEdit):
    """
    Runs a command locally, returning bounded output or a session ID for ongoing interaction.
    """

    def apply(
        self,
        cmd: str,
        workdir: str | None = None,
        yield_time_ms: int = DEFAULT_YIELD_TIME_MS,
        max_output_tokens: int | None = None,
        shell: str | None = None,
        login: bool = False,
        tty: bool = False,
    ) -> str:
        """
        Run a shell command with Codex-style bounded output and session handling.

        :param cmd: shell command to execute
        :param workdir: working directory for the command. Relative paths resolve inside the active project root.
        :param yield_time_ms: wait before yielding output. Defaults to 10000 ms and is capped to 250-30000 ms.
        :param max_output_tokens: approximate output budget. Defaults to 10000 tokens.
        :param shell: shell binary to use. Defaults to the user's SHELL when available.
        :param login: run the shell with login semantics when supported by the selected shell.
        :param tty: request a PTY for interactive commands that need stdin, prompts, REPLs, or console behavior.
        :return: JSON terminal response with output, exit code or session ID, and temp log path
        """
        project_root = Path(self.get_project_root()).resolve()
        workdir_path = _resolve_workdir(str(project_root), workdir)
        response = TERMINAL_PROCESS_MANAGER.exec_command(
            command=cmd,
            cwd=workdir_path,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            shell=shell,
            login=login,
            tty=tty,
        )
        response = _with_context_warnings(response, project_root, workdir_path)
        return _json_response(response)


class WriteStdinTool(Tool, ToolMarkerCanEdit):
    """
    Writes to or polls an existing `exec_command` terminal session.
    """

    def apply(
        self,
        terminal_session_id: int,
        chars: str = "",
        yield_time_ms: int | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """
        Write characters to an existing terminal session and return recent bounded output.

        :param terminal_session_id: terminal session identifier returned as `session_id` by `exec_command`
        :param chars: bytes/characters to write to stdin. Empty string polls without writing.
        :param yield_time_ms: wait before yielding output. Non-empty writes default to 250 ms; empty polls default to 5000 ms.
        :param max_output_tokens: approximate output budget for this response. Defaults to 10000 tokens.
        :return: JSON terminal response with recent output, exit status, and continuing session ID if still running
        """
        response = TERMINAL_PROCESS_MANAGER.write_stdin(
            session_id=terminal_session_id,
            chars=chars,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
        )
        return _json_response(response)


class ListTerminalSessionsTool(Tool):
    """
    Lists running `exec_command` terminal sessions.
    """

    def apply(self) -> str:
        """
        Return currently running terminal sessions.

        :return: JSON list of running terminal sessions
        """
        return json.dumps({"sessions": TERMINAL_PROCESS_MANAGER.list_sessions()}, ensure_ascii=False, indent=2)


class TerminalStatusTool(Tool):
    """
    Returns current status for an `exec_command` terminal session without consuming pending output.
    """

    def apply(self, terminal_session_id: int, include_output: bool = False, max_output_tokens: int | None = None) -> str:
        """
        Return terminal session status.

        :param terminal_session_id: terminal session identifier returned as `session_id` by `exec_command`
        :param include_output: include retained bounded output without draining pending output
        :param max_output_tokens: approximate output budget when include_output is true
        :return: JSON terminal status
        """
        return json.dumps(
            TERMINAL_PROCESS_MANAGER.session_status(
                terminal_session_id,
                include_output=include_output,
                max_output_tokens=max_output_tokens,
            ),
            ensure_ascii=False,
            indent=2,
        )


class SendTerminalSignalTool(Tool, ToolMarkerCanEdit):
    """
    Sends a signal such as SIGINT, SIGTERM, or SIGKILL to an `exec_command` terminal session.
    """

    def apply(
        self,
        terminal_session_id: int,
        signal_name: str = "SIGINT",
        yield_time_ms: int = DEFAULT_STDIN_YIELD_TIME_MS,
        max_output_tokens: int | None = None,
    ) -> str:
        """
        Signal a running terminal session and return recent bounded output.

        :param terminal_session_id: terminal session identifier returned as `session_id` by `exec_command`
        :param signal_name: supported signal name such as SIGINT, SIGTERM, SIGKILL, SIGHUP, or SIGQUIT
        :param yield_time_ms: wait before yielding output after the signal
        :param max_output_tokens: approximate output budget for this response
        :return: JSON terminal response with signal metadata
        """
        sent_signal, response = TERMINAL_PROCESS_MANAGER.send_signal(
            terminal_session_id,
            signal_name=signal_name,
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
        )
        payload = json.loads(_json_response(response))
        return json.dumps({"signal": sent_signal, **payload}, ensure_ascii=False, indent=2)


class StopTerminalSessionTool(Tool, ToolMarkerCanEdit):
    """
    Stops a running `exec_command` terminal session.
    """

    def apply(self, terminal_session_id: int) -> str:
        """
        Terminate a running terminal session and remove it from the session registry.

        :param terminal_session_id: terminal session identifier returned as `session_id` by `exec_command`
        :return: JSON stop result
        """
        stopped = TERMINAL_PROCESS_MANAGER.stop_session(terminal_session_id)
        return json.dumps({"session_id": terminal_session_id, "stopped": stopped}, ensure_ascii=False, indent=2)
