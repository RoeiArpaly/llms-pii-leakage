"""Parallel execution utilities for I/O-bound detector calls."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable

from pandas import Series


def parallel_apply(func: Callable, series: Series, max_workers: int = 8, **kwargs) -> list:
    func_with_kwargs = partial(func, **kwargs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func_with_kwargs, series))
    return results
