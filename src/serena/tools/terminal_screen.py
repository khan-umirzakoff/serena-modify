"""
Headless terminal screen rendering for PTY-backed command sessions.
"""

from __future__ import annotations

import codecs
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import pyte

DEFAULT_TERMINAL_COLUMNS = 80
DEFAULT_TERMINAL_ROWS = 24
MIN_TERMINAL_DIMENSION = 1
MAX_TERMINAL_DIMENSION = 1_000
SYNCHRONIZED_OUTPUT_ENABLE = "\x1b[?2026h"
SYNCHRONIZED_OUTPUT_DISABLE = "\x1b[?2026l"
BRACKETED_PASTE_ENABLE = "\x1b[?2004h"
BRACKETED_PASTE_DISABLE = "\x1b[?2004l"
TRACKED_PRIVATE_MODES = (
    SYNCHRONIZED_OUTPUT_ENABLE,
    SYNCHRONIZED_OUTPUT_DISABLE,
    BRACKETED_PASTE_ENABLE,
    BRACKETED_PASTE_DISABLE,
)


class TerminalKey(StrEnum):
    """Semantic terminal key supported by interactive input."""

    ENTER = "ENTER"
    TAB = "TAB"
    ESCAPE = "ESCAPE"
    BACKSPACE = "BACKSPACE"
    DELETE = "DELETE"
    INSERT = "INSERT"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    HOME = "HOME"
    END = "END"
    PAGE_UP = "PAGE_UP"
    PAGE_DOWN = "PAGE_DOWN"
    CTRL_D = "CTRL_D"


TERMINAL_KEY_SEQUENCES: dict[TerminalKey, bytes] = {
    TerminalKey.ENTER: b"\r",
    TerminalKey.TAB: b"\t",
    TerminalKey.ESCAPE: b"\x1b",
    TerminalKey.BACKSPACE: b"\x7f",
    TerminalKey.DELETE: b"\x1b[3~",
    TerminalKey.INSERT: b"\x1b[2~",
    TerminalKey.UP: b"\x1b[A",
    TerminalKey.DOWN: b"\x1b[B",
    TerminalKey.LEFT: b"\x1b[D",
    TerminalKey.RIGHT: b"\x1b[C",
    TerminalKey.HOME: b"\x1b[H",
    TerminalKey.END: b"\x1b[F",
    TerminalKey.PAGE_UP: b"\x1b[5~",
    TerminalKey.PAGE_DOWN: b"\x1b[6~",
    TerminalKey.CTRL_D: b"\x04",
}


@dataclass(frozen=True)
class TerminalDimensions:
    """Validated terminal dimensions."""

    rows: int = DEFAULT_TERMINAL_ROWS
    columns: int = DEFAULT_TERMINAL_COLUMNS

    def __post_init__(self) -> None:
        for name, value in (("rows", self.rows), ("columns", self.columns)):
            if not MIN_TERMINAL_DIMENSION <= value <= MAX_TERMINAL_DIMENSION:
                raise ValueError(f"terminal {name} must be between {MIN_TERMINAL_DIMENSION} and {MAX_TERMINAL_DIMENSION}, got {value}")


@dataclass(frozen=True)
class TerminalScreenSnapshot:
    """Stable rendered state of a terminal screen."""

    text: str
    cursor_row: int
    cursor_column: int
    dimensions: TerminalDimensions
    synchronized_output: bool
    bracketed_paste: bool


class TerminalScreen:
    """
    Thread-safe VT-compatible terminal screen backed by :mod:`pyte`.

    Synchronized-output markers are handled outside ``pyte`` so callers never
    receive a partially rendered frame between DECSET/DECRST 2026.
    """

    def __init__(self, dimensions: TerminalDimensions) -> None:
        self._dimensions = dimensions
        self._screen = pyte.Screen(columns=dimensions.columns, lines=dimensions.rows)
        self._stream = pyte.Stream(self._screen)
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending_mode_prefix = ""
        self._synchronized_output = False
        self._bracketed_paste = False
        self._lock = threading.RLock()
        self._stable_snapshot = self._capture()

    @property
    def dimensions(self) -> TerminalDimensions:
        """Current terminal dimensions."""
        with self._lock:
            return self._dimensions

    def feed(self, chunk: bytes) -> None:
        """Feed PTY output into the virtual terminal."""
        if not chunk:
            return

        with self._lock:
            decoded = self._decoder.decode(chunk)
            self._feed_text(self._pending_mode_prefix + decoded)

    def finish(self) -> None:
        """Flush incomplete UTF-8 and private-mode fragments."""
        with self._lock:
            decoded = self._decoder.decode(b"", final=True)
            pending = self._pending_mode_prefix + decoded
            self._pending_mode_prefix = ""
            if pending:
                self._stream.feed(pending)
            if not self._synchronized_output:
                self._stable_snapshot = self._capture()

    def resize(self, dimensions: TerminalDimensions) -> None:
        """Resize the virtual terminal screen."""
        with self._lock:
            self._screen.resize(lines=dimensions.rows, columns=dimensions.columns)
            self._dimensions = dimensions
            if not self._synchronized_output:
                self._stable_snapshot = self._capture()

    def snapshot(self) -> TerminalScreenSnapshot:
        """Return the latest complete rendered frame."""
        with self._lock:
            if self._synchronized_output:
                return TerminalScreenSnapshot(
                    text=self._stable_snapshot.text,
                    cursor_row=self._stable_snapshot.cursor_row,
                    cursor_column=self._stable_snapshot.cursor_column,
                    dimensions=self._stable_snapshot.dimensions,
                    synchronized_output=True,
                    bracketed_paste=self._bracketed_paste,
                )
            return self._stable_snapshot

    def prepare_input(self, text: str, keys: Sequence[TerminalKey]) -> bytes:
        """Encode literal text followed by semantic terminal keys as one payload."""
        with self._lock:
            text_payload = text.encode()
            if text and keys and self._bracketed_paste:
                text_payload = b"\x1b[200~" + text_payload + b"\x1b[201~"
            key_payload = b"".join(TERMINAL_KEY_SEQUENCES[key] for key in keys)
            return text_payload + key_payload

    def _feed_text(self, text: str) -> None:
        """Feed text while preserving incomplete tracked mode markers."""
        self._pending_mode_prefix = ""
        remaining = text

        while remaining:
            marker_index, marker = self._find_next_mode_marker(remaining)
            if marker is None:
                prefix_length = self._tracked_prefix_length(remaining)
                stable_text = remaining[:-prefix_length] if prefix_length else remaining
                self._pending_mode_prefix = remaining[-prefix_length:] if prefix_length else ""
                self._feed_stable_text(stable_text)
                return

            self._feed_stable_text(remaining[:marker_index])
            self._apply_mode_marker(marker)
            remaining = remaining[marker_index + len(marker) :]

    def _feed_stable_text(self, text: str) -> None:
        """Feed ordinary terminal output and commit it outside synchronization."""
        if not text:
            return

        self._stream.feed(text)
        if not self._synchronized_output:
            self._stable_snapshot = self._capture()

    def _apply_mode_marker(self, marker: str) -> None:
        """Apply a tracked private-mode marker."""
        if marker == SYNCHRONIZED_OUTPUT_ENABLE:
            if not self._synchronized_output:
                self._stable_snapshot = self._capture()
            self._synchronized_output = True
        elif marker == SYNCHRONIZED_OUTPUT_DISABLE:
            self._synchronized_output = False
        elif marker == BRACKETED_PASTE_ENABLE:
            self._bracketed_paste = True
        elif marker == BRACKETED_PASTE_DISABLE:
            self._bracketed_paste = False

        self._stream.feed(marker)
        if not self._synchronized_output:
            self._stable_snapshot = self._capture()

    def _capture(self) -> TerminalScreenSnapshot:
        """Capture the current virtual screen as readable plain text."""
        lines = [line.rstrip() for line in self._screen.display]
        while lines and not lines[-1]:
            lines.pop()

        return TerminalScreenSnapshot(
            text="\n".join(lines),
            cursor_row=self._screen.cursor.y + 1,
            cursor_column=self._screen.cursor.x + 1,
            dimensions=self._dimensions,
            synchronized_output=self._synchronized_output,
            bracketed_paste=self._bracketed_paste,
        )

    @staticmethod
    def _find_next_mode_marker(text: str) -> tuple[int, str | None]:
        """Find the next tracked private-mode marker."""
        matches = ((text.find(marker), marker) for marker in TRACKED_PRIVATE_MODES)
        found = [(index, marker) for index, marker in matches if index >= 0]
        if not found:
            return -1, None
        return min(found, key=lambda item: item[0])

    @staticmethod
    def _tracked_prefix_length(text: str) -> int:
        """Measure the trailing fragment that may start a tracked mode marker."""
        maximum = min(len(text), max(len(marker) for marker in TRACKED_PRIVATE_MODES) - 1)
        for length in range(maximum, 0, -1):
            suffix = text[-length:]
            if any(marker.startswith(suffix) for marker in TRACKED_PRIVATE_MODES):
                return length
        return 0
