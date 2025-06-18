from __future__ import annotations
import logging
from typing import Iterable, Iterator, TypeVar, Union, Callable, cast
from pathlib import Path
from os import PathLike
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from IPython.core.getipython import get_ipython


T = TypeVar("T")
PathType = Union[str, Path, PathLike[str]]

logger = logging.getLogger(__name__)

def is_run_from_ipython()-> bool:
    """Check if the pipeline is run in a notebook or not"""
    return get_ipython() is not None

def get_corresponding_tqdm() -> Callable[..., Iterator[object]]:
    """
    Returns either tqdm.notebook.tqdm or the standard tqdm.tqdm,
    both of which accept the same signature and yield Iterator[object].
    """
    if is_run_from_ipython():
        from tqdm.notebook import tqdm as notebook_tqdm
        return notebook_tqdm  # type: ignore[return-value]
    from tqdm import tqdm as standard_tqdm
    return standard_tqdm  # type: ignore[return-value]

def setup_progress_monitor(iterable: Iterable[T],*,
                desc: str | None = None,
                colour: str | None = None,
                total: int | None = None,) -> Iterator[T]:
    """
    Progress bar function from the tqdm library. Either the notebook or the terminal version is used. See tqdm documentation for more information.
    
    Main Args:
        iterable: iterable object
        desc: str, description of the progress bar
        colour: str, colour of the progress bar
        total: int, total number of iterations, used in multiprocessing
    """
    tqdm_obj = get_corresponding_tqdm()
    return cast(Iterator[T], tqdm_obj(iterable,
                    desc=desc,
                    colour=colour,
                    total=total))

def pbar_desc(desc: str)-> str:
    """Get the description of the progress bar"""
    if is_run_from_ipython():
        return desc
    return f"\033[94m{desc}\033[0m"

def run_parallel(
    func: Callable[[T], T],
    iterable: Iterable[T],
    *,
    executor: str = 'thread',       # 'thread' or 'process'
    max_workers: int | None = None,
    desc: str | None = None,
    colour: str | None = None,
) -> list[T]:
    """
    Execute a function over an iterable in parallel, showing a progress bar.

    Parameters
    ----------
    func : Callable[[T], Any]
        The function to execute on each item.
    iterable : Iterable[T]
        The sequence of inputs to process.
    executor : {'thread', 'process'}
        Whether to use ThreadPoolExecutor or ProcessPoolExecutor.
    max_workers : int | None
        Number of worker threads/processes. Defaults to os.cpu_count() if None.
    desc : str | None
        Description for the progress bar.
    colour : str | None
        Colour for the progress bar.

    Returns
    -------
    List of results in the same order as `iterable`.
    """
    exec_type = executor.lower()
    if exec_type not in ('thread', 'process'):
        raise ValueError("executor must be 'thread' or 'process'")

    # Determine executor class
    ExecPool = ThreadPoolExecutor if exec_type == 'thread' else ProcessPoolExecutor

    # Attempt to infer total length
    total = None
    try:
        total = len(iterable)  # type: ignore
    except Exception:
        pass

    results: list[T] = []
    with ExecPool(max_workers=max_workers) as pool:
        # Submit all tasks
        futures = [pool.submit(func, item) for item in iterable]
        # Iterate as tasks complete, with progress bar
        for future in setup_progress_monitor(
            as_completed(futures), desc=desc, colour=colour, total=total
        ):
            item = future_to_item[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Item {item} failed: {e}")
    return results