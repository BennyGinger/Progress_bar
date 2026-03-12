"""
Tests for the new explicit progress bar API.

These tests verify the context-managed ProgressBar and pbar() function.
"""
from __future__ import annotations

import pytest

from progress_bar import pbar, ProgressBar


def test_pbar_returns_progress_bar_instance() -> None:
    """pbar() should return a ProgressBar instance."""
    pb = pbar(total=10, desc="Test")
    assert isinstance(pb, ProgressBar)


def test_progress_bar_raises_on_advance_before_enter() -> None:
    """Calling advance() before entering context should raise."""
    pb = pbar(total=10, desc="Test")
    
    with pytest.raises(RuntimeError, match="not active"):
        pb.advance()


def test_progress_bar_raises_on_update_before_enter() -> None:
    """Calling update() before entering context should raise."""
    pb = pbar(total=10, desc="Test")
    
    with pytest.raises(RuntimeError, match="not active"):
        pb.update(total=20)


def test_pbar_creates_task_with_correct_total(patch_progress_manager_create) -> None:
    """The progress bar should create a task with the specified total."""
    with pbar(total=50, desc="Test Task", logs="off"):
        pass
    
    dummy_pm = patch_progress_manager_create
    assert dummy_pm.add_task_calls == [("Test Task", 50)]


def test_pbar_advance_increments_progress(patch_progress_manager_create) -> None:
    """advance() should increment progress by the specified amount."""
    with pbar(total=10, desc="Test", logs="off") as pb:
        pb.advance()
        pb.advance()
        pb.advance(3)
    
    dummy_pm = patch_progress_manager_create
    assert len(dummy_pm.advance_calls) == 3
    assert dummy_pm.advance_calls[0][1] == 1  # default step
    assert dummy_pm.advance_calls[1][1] == 1
    assert dummy_pm.advance_calls[2][1] == 3


def test_pbar_respects_logs_off_mode(patch_progress_manager_create) -> None:
    """logs='off' should pass log_mode to ProgressManager."""
    with pbar(total=10, desc="Test", logs="off"):
        pass
    
    dummy_pm = patch_progress_manager_create
    assert dummy_pm.created_kwargs is not None
    assert dummy_pm.created_kwargs.get("log_mode") == "off"


def test_pbar_respects_logs_buffered_mode(patch_progress_manager_create) -> None:
    """logs='buffered' should pass log_mode to ProgressManager."""
    with pbar(total=10, desc="Test", logs="buffered"):
        pass
    
    dummy_pm = patch_progress_manager_create
    assert dummy_pm.created_kwargs is not None
    assert dummy_pm.created_kwargs.get("log_mode") == "buffered"


def test_pbar_respects_log_lines_param(patch_progress_manager_create) -> None:
    """log_lines parameter should be passed to ProgressManager."""
    with pbar(total=10, desc="Test", logs="buffered", log_lines=12):
        pass
    
    dummy_pm = patch_progress_manager_create
    assert dummy_pm.created_kwargs is not None
    assert dummy_pm.created_kwargs.get("max_log_lines") == 12


def test_pbar_respects_transient_param(patch_progress_manager_create) -> None:
    """transient parameter should be passed to ProgressManager."""
    with pbar(total=10, desc="Test", logs="off", transient=True):
        pass
    
    dummy_pm = patch_progress_manager_create
    assert dummy_pm.created_kwargs is not None
    assert dummy_pm.created_kwargs.get("transient") is True


def test_pbar_default_log_lines(patch_progress_manager_create) -> None:
    """Default log_lines should be 6."""
    with pbar(total=10, desc="Test", logs="buffered"):
        pass
    
    dummy_pm = patch_progress_manager_create
    assert dummy_pm.created_kwargs is not None
    assert dummy_pm.created_kwargs.get("max_log_lines") == 6


def test_pbar_cleans_up_on_exception(patch_progress_manager_create) -> None:
    """Progress bar should clean up properly even if an exception occurs."""
    dummy_pm = patch_progress_manager_create
    
    with pytest.raises(ValueError):
        with pbar(total=10, desc="Test", logs="off") as pb:
            pb.advance()
            raise ValueError("Test error")
    
    # __exit__ should have been called with exception info
    assert dummy_pm.exit_called


def test_pbar_supports_basic_loop(patch_progress_manager_create) -> None:
    """Basic loop pattern should work correctly."""
    items = [1, 2, 3, 4, 5]
    
    processed = []
    with pbar(total=len(items), desc="Process", logs="off") as pb:
        for item in items:
            processed.append(item * 2)
            pb.advance()
    
    assert processed == [2, 4, 6, 8, 10]
    
    dummy_pm = patch_progress_manager_create
    assert len(dummy_pm.advance_calls) == 5


def test_pbar_update_changes_description(patch_progress_manager_create) -> None:
    """update() should be able to change the description."""
    with pbar(total=10, desc="Phase 1", logs="off") as pb:
        pb.advance()
        pb.update(description="Phase 2")
        pb.advance()
    
    dummy_pm = patch_progress_manager_create
    # Verify the progress object had update called on it
    assert len(dummy_pm.progress.update_calls) >= 1


def test_pbar_update_changes_total(patch_progress_manager_create) -> None:
    """update() should be able to change the total."""
    with pbar(total=10, desc="Test", logs="off") as pb:
        pb.advance(5)
        pb.update(total=20)
        pb.advance(5)
    
    dummy_pm = patch_progress_manager_create
    assert len(dummy_pm.progress.update_calls) >= 1


def test_pbar_none_total_uses_fallback(patch_progress_manager_create) -> None:
    """None total should be passed through for indeterminate progress."""
    with pbar(total=None, desc="Loading", logs="off"):
        pass
    
    dummy_pm = patch_progress_manager_create
    assert dummy_pm.add_task_calls == [("Loading", None)]


def test_pbar_respects_logs_live_mode(patch_progress_manager_create) -> None:
    """logs='live' should pass log_mode to ProgressManager."""
    dummy_pm = patch_progress_manager_create

    with pbar(total=10, desc="Test", logs="live"):
        pass

    assert dummy_pm.created_kwargs is not None
    assert dummy_pm.created_kwargs.get("log_mode") == "live"

