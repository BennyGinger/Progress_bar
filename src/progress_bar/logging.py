from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class LogBuffer:
    """In-memory ring buffer of rendered log lines (for on-screen display)."""
    max_lines: int = 30
    _lines: Deque[str] = None  # type: ignore

    def __post_init__(self) -> None:
        self._lines = deque(maxlen=self.max_lines)

    def clear(self) -> None:
        self._lines.clear()

    def append(self, line: str) -> None:
        self._lines.append(line)

    def lines(self) -> list[str]:
        return list(self._lines)


class BufferHandler(logging.Handler):
    """
    Logging handler that formats records and stores them into a LogBuffer.

    This does NOT print to console and does NOT touch other handlers.
    It’s safe to add alongside your existing FileHandler(s).
    """
    def __init__(self, buffer: LogBuffer, level: int = logging.INFO, on_emit: Callable | None = None,) -> None:
        super().__init__(level=level)
        self.buffer = buffer
        self.on_emit = on_emit
        # Default formatter; fits can override this after creation if desired
        self.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.buffer.append(msg)
            if self.on_emit:
                self.on_emit()
        except Exception:
            self.handleError(record)


def attach_buffer_handler(
    logger: logging.Logger,
    buffer: LogBuffer,
    *,
    level: int = logging.INFO,
    on_emit: Callable | None = None,
) -> BufferHandler:
    """
    Attach a BufferHandler to an existing logger without changing its other handlers.
    Returns the handler so fits can remove it later.
    """
    h = BufferHandler(buffer, level=level, on_emit=on_emit)
    logger.addHandler(h)
    return h


def detach_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    """Remove a handler safely."""
    try:
        logger.removeHandler(handler)
    finally:
        handler.close()

@dataclass
class HandlerSwap:
    removed: list[tuple[logging.Logger, logging.Handler]]


def _iter_all_loggers() -> Iterable[logging.Logger]:
    yield logging.getLogger()  # root
    mgr = logging.root.manager
    for obj in mgr.loggerDict.values():
        if isinstance(obj, logging.Logger):
            yield obj


def detach_console_stream_handlers() -> HandlerSwap:
    """
    Detach console StreamHandlers (but keep FileHandlers) so logs don't corrupt Live output.
    File handlers stay, so log files remain untouched.
    """
    removed: list[tuple[logging.Logger, logging.Handler]] = []
    for lg in _iter_all_loggers():
        for h in list(lg.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                lg.removeHandler(h)
                removed.append((lg, h))
    return HandlerSwap(removed=removed)


def restore_handlers(swap: HandlerSwap) -> None:
    for lg, h in swap.removed:
        lg.addHandler(h)