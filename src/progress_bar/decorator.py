from __future__ import annotations

from collections.abc import Iterable, Sequence, Sized
from functools import wraps
from typing import Callable, ParamSpec, TypeVar, cast

from progress_bar.progress import ProgressManager

P = ParamSpec("P")
T = TypeVar("T")


def pbar(*, desc: str) -> Callable[[Callable[P, Iterable[Sequence[T]]]], Callable[P, list[T]]]:
    """
    Decorate a function that returns an iterable of chunks (Sequence[T]).
    
    Policy:
        - The wrapped function must return an Iterable[Sequence[T]] (e.g., the iterator returned by `execute(...)` which yields lists of ExperimentState).
        - The arguments to the wrapped function must include a sized container as the second positional argument (args[1]), which is used to set the total for the progress bar. Like run_step(settings: StepSettings, exp_states: list[ExperimentState], ...) -> Iterable[ExperimentState]
    
    Progress ticks once per chunk (i.e., once per input experiment), and we flatten chunks into list[T].

    Assumption for FITS runners: args[1] is exp_states, so total = len(exp_states).
    """
    def decorator(func: Callable[P, Iterable[Sequence[T]]]) -> Callable[P, list[T]]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> list[T]:
            if len(args) < 2:
                raise TypeError(
                    f"{func.__name__} must be called with (settings, exp_states, ...). "
                    f"Got {len(args)} positional args."
                )

            settings = args[0]
            exp_states = cast(Sized, args[1])

            try:
                total = len(exp_states) 
            except Exception as e:
                raise TypeError(
                    f"{func.__name__}: expected args[1] (exp_states, list[ExperimentStates]) to support len(...)."
                ) from e

            # Serial => show log box; parallel => only pbar
            exec_mode = getattr(settings, "execution", "serial")
            show_logs = (exec_mode == "serial")

            iterable = func(*args, **kwargs)

            out: list[T] = []
            with ProgressManager.create(show_logs=show_logs) as pm:
                tid = pm.add_task(desc, total=total)
                for chunk in iterable:
                    out.extend(chunk)
                    pm.advance(tid)
            return out

        return wrapper
    return decorator