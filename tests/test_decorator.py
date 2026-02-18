from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pytest

from progress_bar.decorator import pbar


@dataclass
class Settings:
    execution: str = "serial"


def test_pbar_raises_if_missing_exp_states() -> None:
    @pbar(desc="X")
    def f(settings) -> Iterable[Sequence[int]]:
        yield [1]

    with pytest.raises(TypeError):
        # only 1 positional arg -> should raise
        f(Settings())  # type: ignore[misc]


def test_pbar_uses_len_of_exp_states_for_total(patch_progress_manager_create) -> None:
    @pbar(desc="Convert")
    def step(settings, exp_states) -> Iterable[Sequence[int]]:
        yield [1]
        yield [2]

    result = step(Settings(execution="serial"), [10, 20, 30])

    # flatten correctness
    assert result == [1, 2]

    # add_task called once with total == len(exp_states)
    dummy_pm = patch_progress_manager_create
    assert dummy_pm.add_task_calls == [("Convert", 3)]

    # advanced once per chunk
    assert len(dummy_pm.advance_calls) == 2


def test_pbar_sets_show_logs_true_for_serial(patch_progress_manager_create) -> None:
    @pbar(desc="Test")
    def step(settings, exp_states) -> Iterable[Sequence[int]]:
        yield [1]

    _ = step(Settings(execution="serial"), [1, 2])

    dummy_pm = patch_progress_manager_create
    assert dummy_pm.created_kwargs is not None
    assert dummy_pm.created_kwargs.get("show_logs") is True


def test_pbar_sets_show_logs_false_for_thread(patch_progress_manager_create) -> None:
    @pbar(desc="Test")
    def step(settings, exp_states) -> Iterable[Sequence[int]]:
        yield [1]

    _ = step(Settings(execution="thread"), [1, 2])

    dummy_pm = patch_progress_manager_create
    assert dummy_pm.created_kwargs is not None
    assert dummy_pm.created_kwargs.get("show_logs") is False


def test_pbar_default_execution_is_serial_when_missing_attr(patch_progress_manager_create) -> None:
    class NoExecution:
        pass

    @pbar(desc="Test")
    def step(settings, exp_states) -> Iterable[Sequence[int]]:
        yield [1]

    _ = step(NoExecution(), [1, 2])  # type: ignore[arg-type]

    dummy_pm = patch_progress_manager_create
    assert dummy_pm.created_kwargs is not None
    assert dummy_pm.created_kwargs.get("show_logs") is True


def test_pbar_flattens_sequences_in_order(patch_progress_manager_create) -> None:
    @pbar(desc="Flatten")
    def step(settings, exp_states) -> Iterable[Sequence[int]]:
        yield [1, 2]
        yield [3]
        yield []
        yield [4, 5]

    result = step(Settings(execution="serial"), [0, 0, 0, 0])
    assert result == [1, 2, 3, 4, 5]

    dummy_pm = patch_progress_manager_create
    # one tick per yielded chunk, even empty chunks
    assert len(dummy_pm.advance_calls) == 4
