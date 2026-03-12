from __future__ import annotations

from rich.progress import TaskID

from progress_bar.progress import LogMode, ProgressManager


class ProgressBar:
    """
    Context-managed progress bar with explicit log control.
    
    Use this with a `with` statement to display a progress bar for loops or batch operations.
    Call `advance()` to move the progress bar forward.
    
    Example:
        with pbar(total=100, desc="Processing", logs="buffered") as pb:
            for item in items:
                # do work
                pb.advance()
    """
    
    def __init__(self, *, total: int | None = None, desc: str = "Working", logs: LogMode = "off", log_lines: int = 6, transient: bool = False,):
        """
        Initialize a progress bar.
        
        Args:
            total: Total number of items to process (None = indeterminate)
            desc: Description text to show next to the progress bar
            logs: Log display mode:
                - "off": suppress console logs while bar is active
                - "buffered": capture logs into a fixed rolling panel below the bar
                - "live": let logs flow normally (may interfere with bar display)
            log_lines: Number of log lines to show in buffered mode
            transient: If True, remove the progress bar when done
        """
        self._total: int | None = total
        self._desc: str = desc
        self._logs: LogMode = logs
        self._log_lines: int = log_lines
        self._transient: bool = transient
        
        self._manager: ProgressManager | None = None
        self._task_id: TaskID | None = None
    
    def __enter__(self) -> ProgressBar:
        """Start the progress bar display."""
        self._manager = ProgressManager.create(max_log_lines=self._log_lines,
                                               log_mode=self._logs,
                                               transient=self._transient,)
        self._manager = self._manager.__enter__()

        self._task_id = self._manager.add_task(description=self._desc, total=self._total,)

        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the progress bar display and restore logging."""
        if self._manager is not None:
            self._manager.__exit__(exc_type, exc_val, exc_tb)
            self._manager = None
    
    def advance(self, n: int = 1) -> None:
        """
        Advance the progress bar by n steps.
        
        Args:
            n: Number of steps to advance (default: 1)
        """
        if self._manager is None or self._task_id is None:
            raise RuntimeError("ProgressBar is not active. Use within a 'with' block.")

        self._manager.advance(self._task_id, step=n)
    
    def update(self, *, total: int | None = None, completed: int | None = None, description: str | None = None) -> None:
        """
        Update progress bar properties.
        
        Args:
            total: Update the total number of items
            completed: Set the completed count directly
            description: Update the description text
        """
        if self._manager is None or self._task_id is None:
            raise RuntimeError("ProgressBar is not active. Use within a 'with' block.")
        
        if total is not None:
            self._total = total
            self._manager.progress.update(self._task_id, total=total)
        
        if completed is not None:
            self._manager.progress.update(self._task_id, completed=completed)
        
        if description is not None:
            self._desc = description
            self._manager.progress.update(self._task_id, description=description)
        
        self._manager.refresh()
    
    def close(self) -> None:
        """
        Manually close the progress bar.
        
        Normally you should use the context manager (`with` statement) instead.
        """
        if self._manager is not None:
            self._manager.__exit__(None, None, None)
            self._manager = None
    
    def refresh(self) -> None:
        """Manually refresh the display (usually not needed)."""
        if self._manager is not None:
            self._manager.refresh()


def pbar(*, total: int | None = None, desc: str = "Working", logs: LogMode = "off", log_lines: int = 6, transient: bool = False,) -> ProgressBar:
    """
    Create a context-managed progress bar with explicit log control.
    
    This is the main entry point for the progress_bar package. Use it to wrap
    loops or batch operations with a visual progress indicator.
    
    Args:
        total: Total number of items to process (None = indeterminate)
        desc: Description text to show next to the progress bar
        logs: Log display mode:
            - "off": suppress console logs while bar is active (default)
            - "buffered": capture logs into a fixed rolling panel below the bar
            - "live": let logs flow normally (may interfere with bar display)
        log_lines: Number of log lines to show in buffered mode (default: 6)
        transient: If True, remove the progress bar when done (default: False)
    
    Returns:
        A ProgressBar context manager
    
    Examples:
        Basic usage::
        
            with pbar(total=len(states), desc="Convert") as pb:
                for state in states:
                    process(state)
                    pb.advance()
        
        With buffered logs::
        
            with pbar(total=len(states), desc="Segment", logs="buffered", log_lines=6) as pb:
                for state in states:
                    process(state)  # logs are captured and displayed below bar
                    pb.advance()
        
        Quiet mode::
        
            with pbar(total=seeded_tasks, desc="Pipeline", logs="off") as pb:
                while not done:
                    # work
                    pb.advance()
    """
    return ProgressBar(total=total, desc=desc, logs=logs, log_lines=log_lines, transient=transient)