from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class DummyTask:
    """Simple stand-in for a Rich TaskID."""
    task_id: int = 1


class DummyProgress:
    """Mock Rich Progress object."""
    def __init__(self) -> None:
        self.update_calls: list[tuple[int, dict[str, Any]]] = []
    
    def update(self, task_id: int, **kwargs: Any) -> None:
        self.update_calls.append((task_id, kwargs))


class DummyProgressManager:
    """
    A dummy ProgressManager that mimics the subset of the API your decorator uses:
    - context manager
    - add_task(desc, total) -> task_id
    - advance(task_id)
    """
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.exit_called = False

        self.created_kwargs: dict[str, Any] | None = None
        self.add_task_calls: list[tuple[str, int]] = []
        self.advance_calls: list[tuple[int, int]] = []  # (task_id, step)
        
        # Add mock progress object
        self.progress = DummyProgress()
        
        # Add mock swap for handler testing
        self._swap: Any = None

    def __enter__(self) -> "DummyProgressManager":
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True
        self.exit_called = True

    def add_task(self, description: str, total: int) -> int:
        self.add_task_calls.append((description, total))
        return 1

    def advance(self, task_id: int, step: int = 1) -> None:
        self.advance_calls.append((task_id, step))
    
    def refresh(self) -> None:
        """Mock refresh method."""
        pass


@pytest.fixture
def dummy_pm() -> DummyProgressManager:
    return DummyProgressManager()


@pytest.fixture
def patch_progress_manager_create(monkeypatch: pytest.MonkeyPatch, dummy_pm: DummyProgressManager):
    """
    Patch progress_bar.progress.ProgressManager.create to return DummyProgressManager,
    while capturing the kwargs it was called with (e.g. show_logs=...).
    """
    # Import inside fixture to ensure tests use the installed package/module under test.
    import progress_bar.progress as progress_mod

    def _fake_create(*args: Any, **kwargs: Any) -> DummyProgressManager:
        dummy_pm.created_kwargs = dict(kwargs)
        return dummy_pm

    monkeypatch.setattr(progress_mod.ProgressManager, "create", staticmethod(_fake_create))
    return dummy_pm
