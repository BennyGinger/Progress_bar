from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest
from rich.progress import Task

from progress_bar.progress import (
    EstimatedRemainingColumn,
    ElapsedColumn,
    TimePerIterColumn,
    _format_hhmmss,
    ProgressManager,
)
from progress_bar.logging import LogBuffer


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "--:--:--"),
        (0, "0:00:00"),
        (-10, "0:00:00"),
        (45, "0:00:45"),
        (125, "0:02:05"),
        (3661, "1:01:01"),
        (36000, "10:00:00"),
    ],
)
def test_format_hhmmss(seconds: float | None, expected: str) -> None:
    """Test time formatting with various inputs."""
    assert _format_hhmmss(seconds) == expected


@pytest.mark.parametrize(
    ("iteration_time", "expected"),
    [
        (None, "--:--:-- /iter"),
        (0.0, "0:00:00 /iter"),
        (2.0, "0:00:02 /iter"),
        (65.0, "0:01:05 /iter"),
    ],
)
def test_time_per_iter_column(
        iteration_time: float | None, expected: str) -> None:
    """Test TimePerIterColumn renders the latest iteration duration."""
    column = TimePerIterColumn()
    task = Mock(spec=Task)
    task.fields = {"last_iteration_time": iteration_time}

    result = column.render(task)
    assert result.plain == expected


@pytest.mark.parametrize(
    ("deadline", "expected"),
    [
        (None, "est. --:--:--"),
        (3761, "est. 1:01:01"),
        (220, "est. 0:02:00"),
        (90, "est. +0:00:10"),
    ],
)
def test_estimated_remaining_column(deadline: float | None, expected: str) -> None:
    """Test EstimatedRemainingColumn renders estimate correctly."""
    column = EstimatedRemainingColumn()
    task = Mock(spec=Task)
    task.fields = {"eta_deadline": deadline}
    task.get_time.return_value = 100

    result = column.render(task)
    assert result.plain == expected


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (None, "--:--:--"),
        (0, "0:00:00"),
        (65, "0:01:05"),
        (3661, "1:01:01"),
    ],
)
def test_elapsed_column(elapsed: float | None, expected: str) -> None:
    """Test ElapsedColumn renders elapsed time correctly."""
    column = ElapsedColumn()
    task = Mock(spec=Task)
    task.elapsed = elapsed

    result = column.render(task)
    assert result.plain == expected


def test_log_buffer_initially_empty() -> None:
    """LogBuffer should start empty."""
    buf = LogBuffer(max_lines=5)
    assert buf.lines() == []


def test_log_buffer_append_single_line() -> None:
    """Should append a single line."""
    buf = LogBuffer(max_lines=5)
    buf.append("First line")
    assert buf.lines() == ["First line"]


def test_log_buffer_append_multiple_lines() -> None:
    """Should append multiple lines in order."""
    buf = LogBuffer(max_lines=5)
    buf.append("Line 1")
    buf.append("Line 2")
    buf.append("Line 3")
    assert buf.lines() == ["Line 1", "Line 2", "Line 3"]


def test_log_buffer_respects_max_lines() -> None:
    """Should evict oldest lines when max_lines is exceeded."""
    buf = LogBuffer(max_lines=3)
    buf.append("1")
    buf.append("2")
    buf.append("3")
    buf.append("4")  # Should evict "1"
    assert buf.lines() == ["2", "3", "4"]


def test_log_buffer_clear_removes_all_lines() -> None:
    """Clear should remove all lines."""
    buf = LogBuffer(max_lines=5)
    buf.append("Line 1")
    buf.append("Line 2")
    buf.clear()
    assert buf.lines() == []


def test_set_experiment_adds_header() -> None:
    """set_experiment should clear buffer and add experiment header."""
    buf = LogBuffer(max_lines=5)
    buf.append("Old log")
    
    # Use a ProgressManager to test set_experiment
    pm = ProgressManager.create(max_log_lines=5, show_logs=False)
    pm.log_buffer = buf
    pm.set_experiment("Experiment_42")
    
    lines = buf.lines()
    assert len(lines) == 1
    assert lines[0] == "--- Experiment_42 ---"


def test_progress_manager_capture_logs_attaches_handler() -> None:
    """capture_logs should attach handler and stop_capture_logs should remove it."""
    pm = ProgressManager.create(max_log_lines=5, show_logs=False)
    test_logger = logging.getLogger("test_capture")
    initial_handler_count = len(test_logger.handlers)

    pm.capture_logs(test_logger, level=logging.INFO, mute_console=False)
    assert len(test_logger.handlers) == initial_handler_count + 1
    assert pm._buf_handler is not None

    pm.stop_capture_logs()
    assert len(test_logger.handlers) == initial_handler_count
    assert pm._buf_handler is None


def test_captured_logs_appear_in_buffer() -> None:
    """Captured logs should be stored in the buffer."""
    pm = ProgressManager.create(max_log_lines=5, show_logs=False)
    test_logger = logging.getLogger("test_buffer_capture")
    test_logger.setLevel(logging.DEBUG)

    pm.capture_logs(test_logger, level=logging.INFO, mute_console=False)
    test_logger.info("Test message")
    pm.stop_capture_logs()

    lines = pm.log_buffer.lines()
    assert len(lines) == 1
    assert "Test message" in lines[0]


def test_stop_capture_logs_is_idempotent() -> None:
    """stop_capture_logs should be safe to call multiple times."""
    pm = ProgressManager.create(max_log_lines=5, show_logs=False)
    test_logger = logging.getLogger("test_idempotent")

    pm.capture_logs(test_logger, level=logging.INFO, mute_console=False)
    pm.stop_capture_logs()
    pm.stop_capture_logs()  # Should not raise


def test_capture_logs_is_idempotent() -> None:
    """capture_logs should not add duplicate handlers."""
    pm = ProgressManager.create(max_log_lines=5, show_logs=False)
    test_logger = logging.getLogger("test_double_capture")
    initial_count = len(test_logger.handlers)

    pm.capture_logs(test_logger, level=logging.INFO, mute_console=False)
    pm.capture_logs(test_logger, level=logging.INFO, mute_console=False)
    
    # Should not add duplicate handlers
    assert len(test_logger.handlers) == initial_count + 1
    
    pm.stop_capture_logs()


@pytest.mark.parametrize(
    ("show_logs", "expected"),
    [
        (True, True),
        (False, False),
    ],
)
def test_progress_manager_show_logs_flag(show_logs: bool, expected: bool) -> None:
    """show_logs flag should be correctly set."""
    pm = ProgressManager.create(show_logs=show_logs)
    assert pm.show_logs is expected


def test_default_show_logs_is_true() -> None:
    """show_logs should default to True."""
    pm = ProgressManager.create()
    assert pm.show_logs is True


def test_context_manager_sets_up_live() -> None:
    """Context manager should set up and tear down Live display."""
    pm = ProgressManager.create(max_log_lines=5, show_logs=False)
    assert pm._live is None

    with pm:
        assert pm._live is not None

    assert pm._live is None


def test_context_manager_captures_logs_when_show_logs_true() -> None:
    """When show_logs=True, logs from captured loggers appear in buffer."""
    pm = ProgressManager.create(max_log_lines=5, show_logs=True)
    test_logger = logging.getLogger("test_context_capture")
    test_logger.setLevel(logging.DEBUG)

    with pm:
        pm.capture_logs(test_logger, level=logging.DEBUG, mute_console=False)
        test_logger.info("Test log in context")

    # After exit, logs should be in buffer
    lines = pm.log_buffer.lines()
    assert any("Test log in context" in line for line in lines)


def test_exit_cleans_up_handlers() -> None:
    """Context manager exit should clean up all handlers and state."""
    pm = ProgressManager.create(max_log_lines=5, show_logs=True)
    
    with pm:
        pass  # Just enter and exit

    assert pm._buf_handler is None
    assert pm._swap is None
    assert pm._live is None
