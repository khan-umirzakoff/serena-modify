from serena.tools.terminal_screen import TerminalDimensions, TerminalScreen


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

    payload = screen.prepare_input("task text", submit=True)

    assert payload == b"\x1b[200~task text\x1b[201~\r"


def test_terminal_screen_resize_updates_snapshot_dimensions() -> None:
    screen = TerminalScreen(TerminalDimensions(rows=24, columns=80))

    screen.resize(TerminalDimensions(rows=40, columns=120))

    snapshot = screen.snapshot()
    assert snapshot.dimensions == TerminalDimensions(rows=40, columns=120)
