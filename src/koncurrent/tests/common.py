import threading
from collections.abc import Iterator

import pytest

from koncurrent import EXECUTOR_ENV_VAR, WORKERS_ENV_VAR


def results[R](it: Iterator[R | Exception]) -> list[R]:
    """
    The results of map_results(), in input order.
    Raises the first error rather than returning it, so a failing test shows the original exception and its traceback.
    """
    values: list[R] = []
    for res in it:
        if isinstance(res, Exception):
            raise res
        values.append(res)
    return values


def thread_name(_: object) -> str:
    return threading.current_thread().name


def use_thread_pool(monkeypatch: pytest.MonkeyPatch, *, workers: int) -> None:
    """
    Makes the shared pool a thread pool, regardless of which kind koncurrent would pick for this Python build.
    """
    monkeypatch.setenv(EXECUTOR_ENV_VAR, 'threads')
    monkeypatch.setenv(WORKERS_ENV_VAR, str(workers))
