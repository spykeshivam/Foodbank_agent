"""Centralised logging — FIFO file capped at MAX_LINES."""
import logging
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "foodbank_agent.log")
MAX_LINES = 3000
_TRIM_EVERY = 50  # re-check line count every N emits


class _LineCapHandler(logging.FileHandler):
    def __init__(self, filename: str, max_lines: int = MAX_LINES):
        super().__init__(filename, encoding="utf-8")
        self.max_lines = max_lines
        self._emit_count = 0
        try:
            with open(filename, "r", encoding="utf-8") as f:
                self._line_count = sum(1 for _ in f)
        except FileNotFoundError:
            self._line_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self._line_count += 1
        self._emit_count += 1
        if self._emit_count >= _TRIM_EVERY and self._line_count > self.max_lines:
            self._trim()
            self._emit_count = 0

    def _trim(self) -> None:
        try:
            with open(self.baseFilename, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > self.max_lines:
                with open(self.baseFilename, "w", encoding="utf-8") as f:
                    f.writelines(lines[-self.max_lines :])
                self._line_count = self.max_lines
        except Exception:
            pass


def setup_logging() -> None:
    """Call once at startup — idempotent."""
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = _LineCapHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Suppress chatty third-party loggers
    for name in ("httpx", "httpcore", "google", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
