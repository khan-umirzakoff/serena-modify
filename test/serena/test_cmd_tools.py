import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from serena.tools.cmd_tools import (
    SendTerminalSignalTool,
    StopTerminalSessionTool,
    TerminalProcessManager,
    TerminalStatusTool,
    WriteStdinTool,
    _json_response,
    _terminal_context_warnings,
)
from serena.tools.tools_base import Tool, ToolRegistry


def _python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def test_exec_command_tool_and_write_stdin_tool_are_registered() -> None:
    names = ToolRegistry().get_tool_names()

    assert "exec_command" in names
    assert "write_stdin" in names
    assert "list_terminal_sessions" in names
    assert "terminal_status" in names
    assert "send_terminal_signal" in names
    assert "stop_terminal_session" in names


@pytest.mark.parametrize(
    "tool_class",
    (WriteStdinTool, TerminalStatusTool, SendTerminalSignalTool, StopTerminalSessionTool),
)
def test_terminal_continuation_tools_use_codex_session_id_with_legacy_alias(tool_class: type[Tool]) -> None:
    metadata = tool_class.get_apply_fn_metadata_from_cls()

    assert "terminal_session_id" in metadata.arg_model.model_json_schema()["properties"]
    assert tool_class.get_param_aliases() == {"session_id": "terminal_session_id"}
    assert tool_class.get_public_param_aliases() == {"session_id": "terminal_session_id"}


def test_terminal_context_warns_for_nested_repo(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    nested_repo = project_root / "app"
    nested_repo.mkdir(parents=True)
    (nested_repo / ("." + "git")).mkdir()

    warnings = _terminal_context_warnings(project_root, nested_repo)

    assert warnings
    assert "nested Git repository" in warnings[0]


def test_terminal_context_marks_expected_nested_repo(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    nested_repo = project_root / "app"
    nested_repo.mkdir(parents=True)
    (nested_repo / ".git").mkdir()
    context_dir = project_root / ".serena"
    context_dir.mkdir()
    context_file = context_dir / "coding_task_context.json"
    context_file.write_text(
        '{"project_root": "' + str(project_root) + '", "git_root": "' + str(nested_repo) + '"}',
        encoding="utf-8",
    )

    warnings = _terminal_context_warnings(project_root, nested_repo)

    assert warnings
    assert "matches the latest prepare_coding_task nested Git root" in warnings[0]
    assert "active Serena project is different" not in warnings[0]


def test_terminal_context_ignores_mismatched_expected_repo_context(tmp_path: Path) -> None:
    project_root = tmp_path / "workspace"
    nested_repo = project_root / "app"
    nested_repo.mkdir(parents=True)
    (nested_repo / ".git").mkdir()
    context_dir = project_root / ".serena"
    context_dir.mkdir()
    context_file = context_dir / "coding_task_context.json"
    other_project = tmp_path / "other"
    context_file.write_text(
        '{"project_root": "' + str(other_project) + '", "git_root": "' + str(nested_repo) + '"}',
        encoding="utf-8",
    )

    warnings = _terminal_context_warnings(project_root, nested_repo)

    assert warnings
    assert "active Serena project is different" in warnings[0]


def test_short_command_exits_and_returns_output(tmp_path: Path) -> None:
    manager = TerminalProcessManager()

    response = manager.exec_command(_python_command("print('hello')"), cwd=tmp_path, yield_time_ms=1000)

    assert response.running is False
    assert response.session_id is None
    assert response.exit_code == 0
    assert response.output.strip() == "hello"
    assert Path(response.log_path).read_text(encoding="utf-8").strip() == "hello"


def test_long_running_command_returns_session_and_polling_finishes(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command("import time; print('ready', flush=True); time.sleep(0.5); print('done', flush=True)")

    first = manager.exec_command(command, cwd=tmp_path, yield_time_ms=250)

    assert first.running is True
    assert first.session_id is not None
    assert "ready" in first.output

    second = manager.write_stdin(first.session_id, yield_time_ms=2000)

    assert second.running is False
    assert second.session_id is None
    assert second.exit_code == 0
    assert "done" in second.output


def test_write_stdin_sends_input_to_process(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command(
        "import sys; print('prompt', flush=True); line = sys.stdin.readline(); print('echo:' + line.strip(), flush=True)"
    )

    first = manager.exec_command(command, cwd=tmp_path, yield_time_ms=250, tty=True)
    assert first.session_id is not None
    assert "prompt" in first.output
    assert first.transport == "pty"

    second = manager.write_stdin(first.session_id, chars="hello\n", yield_time_ms=1000)

    assert second.running is False
    assert second.exit_code == 0
    assert "echo:hello" in second.output


def test_write_stdin_submits_text_with_enter(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command("import sys; line = sys.stdin.readline(); print('submitted:' + line.strip(), flush=True)")

    first = manager.exec_command(command, cwd=tmp_path, yield_time_ms=250, tty=True)
    assert first.session_id is not None

    second = manager.write_stdin(first.session_id, chars="atomic task", submit=True, yield_time_ms=1000)

    assert second.running is False
    assert second.exit_code == 0
    assert "submitted:atomic task" in second.output
    assert second.output_mode == "screen"


def test_pty_response_renders_ansi_instead_of_returning_raw_sequences(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command("import sys; sys.stdout.write('old\\x1b[3Dnew'); sys.stdout.flush()")

    response = manager.exec_command(command, cwd=tmp_path, yield_time_ms=1000, tty=True)

    assert response.exit_code == 0
    assert response.output == "new"
    assert response.output_mode == "screen"
    assert "\x1b[" not in response.output
    assert Path(response.log_path).read_bytes() == b"old\x1b[3Dnew"
    assert not {
        "rendered_screen",
        "cursor_row",
        "cursor_column",
        "terminal_rows",
        "terminal_columns",
        "synchronized_output",
        "bracketed_paste",
    } & set(asdict(response))

    public_payload = json.loads(_json_response(response))
    assert list(public_payload) == [
        "output",
        "output_mode",
        "session_id",
        "running",
        "exit_code",
        "wall_time_seconds",
        "log_path",
    ]


def test_pty_rendered_snapshot_obeys_output_budget(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command("print('x' * 5000, end='', flush=True)")

    response = manager.exec_command(
        command,
        cwd=tmp_path,
        yield_time_ms=1000,
        max_output_tokens=256,
        tty=True,
        rows=10,
        columns=1000,
    )

    assert response.output_mode == "screen"
    assert len(response.output.encode()) < 1200
    assert response.omitted_bytes > 0
    assert b"x" * 5000 in Path(response.log_path).read_bytes()


def test_pty_initial_size_and_resize_are_reported_to_process(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command(
        "import os, sys; "
        "size = os.get_terminal_size(); print(f'initial:{size.columns}x{size.lines}', flush=True); "
        "sys.stdin.readline(); "
        "size = os.get_terminal_size(); print(f'resized:{size.columns}x{size.lines}', flush=True)"
    )

    first = manager.exec_command(command, cwd=tmp_path, yield_time_ms=250, tty=True, rows=30, columns=100)
    assert first.session_id is not None
    assert "initial:100x30" in first.output

    second = manager.write_stdin(first.session_id, submit=True, rows=40, columns=120, yield_time_ms=1000)

    assert second.exit_code == 0
    assert "resized:120x40" in second.output
    assert second.output_mode == "screen"


def test_pipe_session_rejects_regular_stdin(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command("import time; time.sleep(1)")

    first = manager.exec_command(command, cwd=tmp_path, yield_time_ms=250)

    assert first.session_id is not None
    assert first.transport == "pipe"
    with pytest.raises(RuntimeError, match="stdin is closed"):
        manager.write_stdin(first.session_id, chars="hello\n", yield_time_ms=250)
    manager.stop_session(first.session_id)


def test_write_stdin_interrupts_pipe_session(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command(
        "import signal, time; "
        "signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt)); "
        "print('ready', flush=True); "
        "time.sleep(10)"
    )

    first = manager.exec_command(command, cwd=tmp_path, yield_time_ms=250)

    assert first.session_id is not None
    assert "ready" in first.output
    second = manager.write_stdin(first.session_id, chars="\u0003", yield_time_ms=1000)

    assert second.running is False
    assert second.exit_code != 0


def test_send_signal_interrupts_pipe_session(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command(
        "import signal, time; "
        "signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt)); "
        "print('ready', flush=True); "
        "time.sleep(10)"
    )

    first = manager.exec_command(command, cwd=tmp_path, yield_time_ms=250)

    assert first.session_id is not None
    assert "ready" in first.output
    sent_signal, second = manager.send_signal(first.session_id, signal_name="int", yield_time_ms=1000)

    assert sent_signal == "SIGINT"
    assert second.running is False
    assert second.exit_code != 0


def test_huge_output_is_bounded(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command("print('START' + ('x' * 20000) + 'END')")

    response = manager.exec_command(command, cwd=tmp_path, yield_time_ms=1000, max_output_tokens=100)

    assert response.running is False
    assert response.omitted_bytes > 0
    assert "START" in response.output
    assert "END" in response.output
    assert "omitted" in response.output
    assert len(response.output) < 1500


def test_session_can_be_polled_after_clean_exit(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command("import time; time.sleep(0.5); print('done', flush=True)")

    first = manager.exec_command(command, cwd=tmp_path, yield_time_ms=250)
    session_id = first.session_id
    assert session_id is not None

    final = manager.write_stdin(session_id, yield_time_ms=1000)
    assert final.running is False
    assert final.session_id is None
    assert "done" in final.output

    repeated = manager.write_stdin(session_id, yield_time_ms=250)
    assert repeated.running is False
    assert repeated.session_id is None


def test_list_and_stop_terminal_sessions(tmp_path: Path) -> None:
    manager = TerminalProcessManager()
    command = _python_command("import time; print('ready', flush=True); time.sleep(10)")

    first = manager.exec_command(command, cwd=tmp_path, yield_time_ms=250, tty=True)

    assert first.session_id is not None
    sessions = manager.list_sessions()
    assert [session["session_id"] for session in sessions] == [first.session_id]
    assert sessions[0]["transport"] == "pty"
    assert sessions[0]["output_mode"] == "screen"

    status = manager.session_status(first.session_id, include_output=True)
    assert status["session_id"] == first.session_id
    assert status["running"] is True
    assert status["transport"] == "pty"
    assert "ready" in str(status["output"])

    assert manager.stop_session(first.session_id) is True
    assert manager.list_sessions() == []
