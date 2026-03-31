from __future__ import annotations

import queue
import sys
import threading


class InputController:
    def __init__(self):
        self.buffer = ""
        self._completed: list[str] = []
        self._fallback_queue: queue.Queue[str] | None = None
        self._fallback_thread: threading.Thread | None = None
        try:
            import msvcrt  # type: ignore

            self._msvcrt = msvcrt
            self._use_msvcrt = True
        except ImportError:
            self._msvcrt = None
            self._use_msvcrt = False
            self._start_fallback_reader()

    def _start_fallback_reader(self) -> None:
        self._fallback_queue = queue.Queue()

        def _reader() -> None:
            while True:
                line = sys.stdin.readline()
                if line == "":
                    self._fallback_queue.put("/quit")
                    return
                self._fallback_queue.put(line.rstrip("\n"))

        self._fallback_thread = threading.Thread(target=_reader, daemon=True, name="stdin-reader")
        self._fallback_thread.start()

    def poll_completed_lines(self) -> list[str]:
        if self._use_msvcrt:
            self._poll_windows_keyboard()
        else:
            self._poll_fallback_queue()
        lines = list(self._completed)
        self._completed.clear()
        return lines

    def _poll_fallback_queue(self) -> None:
        if self._fallback_queue is None:
            return
        while True:
            try:
                self._completed.append(self._fallback_queue.get_nowait())
            except queue.Empty:
                return

    def _poll_windows_keyboard(self) -> None:
        while self._msvcrt.kbhit():
            char = self._msvcrt.getwch()
            if char in ("\x00", "\xe0"):
                if self._msvcrt.kbhit():
                    self._msvcrt.getwch()
                continue
            if char == "\x03":
                self._completed.append("/quit")
                self.buffer = ""
                continue
            if char in ("\r", "\n"):
                line = self.buffer.strip()
                self.buffer = ""
                if line:
                    self._completed.append(line)
                continue
            if char == "\x08":
                self.buffer = self.buffer[:-1]
                continue
            if char.isprintable():
                self.buffer += char

