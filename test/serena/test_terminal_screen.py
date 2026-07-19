import pytest

from serena.tools.terminal_screen import TerminalDimensions, TerminalKey, TerminalScreen


def test_terminal_screen_renders_cursor_movement_and_restore() -> None:
    screen = TerminalScreen(TerminalDimensions(rows=4, columns=20))

    screen.feed(b"first\x1b7\x1b[2;1Hsecond\x1b8!")

    snapshot = screen.snapshot()
    assert snapshot.text == "first!\nsecond"
    assert snapshot.cursor_row == 1
    assert snapshot.cursor_column == 7


def test_terminal_screen_hides_incomplete_synchronized_frame() -> None:
    screen = TerminalScreen(TerminalDimensions(rows=4, columns=20))
    screen.feed(b"old frame")

    screen.feed(b"\x1b[?2026")
    screen.feed(b"h\rnew frame")

    incomplete = screen.snapshot()
    assert incomplete.text == "old frame"
    assert incomplete.synchronized_output is True

    screen.feed(b"\x1b[?2026l")

    complete = screen.snapshot()
    assert complete.text == "new frame"
    assert complete.synchronized_output is False


def test_terminal_screen_prepares_bracketed_paste_and_enter_together() -> None:
    screen = TerminalScreen(TerminalDimensions())
    screen.feed(b"\x1b[?2004h")

    payload = screen.prepare_input("task text", keys=[TerminalKey.ENTER])

    assert payload == b"\x1b[200~task text\x1b[201~\r"


@pytest.mark.parametrize(
    ("key", "expected"),
    (
        (TerminalKey.ENTER, b"\r"),
        (TerminalKey.TAB, b"\t"),
        (TerminalKey.ESCAPE, b"\x1b"),
        (TerminalKey.BACKSPACE, b"\x7f"),
        (TerminalKey.DELETE, b"\x1b[3~"),
        (TerminalKey.INSERT, b"\x1b[2~"),
        (TerminalKey.UP, b"\x1b[A"),
        (TerminalKey.DOWN, b"\x1b[B"),
        (TerminalKey.LEFT, b"\x1b[D"),
        (TerminalKey.RIGHT, b"\x1b[C"),
        (TerminalKey.HOME, b"\x1b[H"),
        (TerminalKey.END, b"\x1b[F"),
        (TerminalKey.PAGE_UP, b"\x1b[5~"),
        (TerminalKey.PAGE_DOWN, b"\x1b[6~"),
        (TerminalKey.CTRL_D, b"\x04"),
    ),
)
def test_terminal_screen_encodes_semantic_keys(key: TerminalKey, expected: bytes) -> None:
    screen = TerminalScreen(TerminalDimensions())

    assert screen.prepare_input("", keys=[key]) == expected


def test_terminal_screen_keeps_newlines_literal_without_enter_key() -> None:
    screen = TerminalScreen(TerminalDimensions())

    assert screen.prepare_input("first\nsecond\n", keys=[]) == b"first\nsecond\n"


def test_terminal_screen_resize_updates_snapshot_dimensions() -> None:
    screen = TerminalScreen(TerminalDimensions(rows=24, columns=80))

    screen.resize(TerminalDimensions(rows=40, columns=120))

    snapshot = screen.snapshot()
    assert snapshot.dimensions == TerminalDimensions(rows=40, columns=120)
