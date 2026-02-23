from __future__ import annotations

import logging
from dataclasses import dataclass

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskID, ProgressColumn, Task
from rich.text import Text

from progress_bar.logging import LogBuffer, attach_buffer_handler, detach_console_stream_handlers, detach_handler, HandlerSwap, restore_handlers

def _format_hhmmss(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class IterPerSecColumn(ProgressColumn):
    """Renders iterations per second with safe fallback."""
    def render(self, task: Task) -> Text:
        speed = task.speed
        if speed is None:
            return Text("--.- it/s")
        return Text(f"{speed:.2f} it/s")


class ElapsedEtaColumn(ProgressColumn):
    """Renders elapsed/eta as 0:00:02/0:10:02 with safe fallbacks."""
    def render(self, task: Task) -> Text:
        elapsed = _format_hhmmss(task.elapsed)
        remaining = _format_hhmmss(task.time_remaining)
        return Text(f"{elapsed}/{remaining}")

@dataclass
class ProgressManager:
    console: Console
    progress: Progress
    log_buffer: LogBuffer
    show_logs: bool = True

    _live: Live | None = None
    _swap: HandlerSwap | None = None
    _buf_handler: logging.Handler | None = None
    _capture_logger: logging.Logger | None = None

    @classmethod
    def create(cls, *, max_log_lines: int = 30, show_logs: bool = True) -> ProgressManager:
        console = Console(force_terminal=True, force_interactive=True)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}", justify="right"),
            IterPerSecColumn(),          # <-- safe it/s
            ElapsedEtaColumn(),          # <-- safe elapsed/ETA
            console=console,
            transient=False,
            auto_refresh=False,
        )
        buf = LogBuffer(max_lines=max_log_lines)
        return cls(console=console, progress=progress, log_buffer=buf, show_logs=show_logs)

    # ---- progress API ----
    def add_task(self, description: str, total: int) -> TaskID:
        tid = self.progress.add_task(description, total=total)
        self.refresh()
        return tid

    def advance(self, task_id: TaskID, step: int = 1) -> None:
        self.progress.advance(task_id, step)
        self.refresh()

    # ---- log UI ----
    def _render_log_panel(self) -> Panel:
        text = Text("\n".join(self.log_buffer.lines()))
        return Panel(text, title="Logs", border_style="dim", height=10)

    def refresh(self) -> None:
        if self._live is None:
            return
        self.progress.refresh()
        if self.show_logs:
            self._live.update(Group(self.progress, self._render_log_panel()), refresh=True)
        else:
            self._live.update(self.progress, refresh=True)

    def set_experiment(self, name: str) -> None:
        self.log_buffer.clear()
        self.log_buffer.append(f"--- {name} ---")
        self.refresh()

    # ---- logging capture ----
    def capture_logs(self, logger: logging.Logger | None = None, *, level: int = logging.INFO, mute_console: bool = True) -> None:
        if self._buf_handler is not None:
            return

        if mute_console:
            self._swap = detach_console_stream_handlers()

        self._capture_logger = logger or logging.getLogger()  # root by default
        self._buf_handler = attach_buffer_handler(
            self._capture_logger,
            self.log_buffer,
            level=level,
            on_emit=self.refresh,
        )

    def stop_capture_logs(self) -> None:
        if self._capture_logger is not None and self._buf_handler is not None:
            detach_handler(self._capture_logger, self._buf_handler)

        self._buf_handler = None
        self._capture_logger = None

        if self._swap is not None:
            restore_handlers(self._swap)
            self._swap = None

    # ---- context manager ----
    def __enter__(self) -> ProgressManager:
        if self.show_logs:
            layout = Group(self.progress, self._render_log_panel())
        else:
            layout = self.progress
        
        self._live = Live(
            layout,
            console=self.console,
            refresh_per_second=10,
            redirect_stdout=True,
            redirect_stderr=True,
        )
        self._live.__enter__()

        if self.show_logs:
            # capture + mute console (clean UI)
            self.capture_logs(logging.getLogger(), level=logging.INFO, mute_console=True)
        else:
            # batch mode: don't capture, but still mute console so logs don't trash the pbar
            self._swap = detach_console_stream_handlers()

        self.refresh()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_capture_logs()
        
        # if we only muted console (batch mode), restore here
        if self._swap is not None:
            restore_handlers(self._swap)
            self._swap = None
        
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None
