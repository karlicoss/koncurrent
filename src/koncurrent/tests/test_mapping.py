import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import override

import pytest

from koncurrent import SynchronousExecutor, default_buffersize, default_executor, map_results

from .common import results, use_thread_pool


def _make_executor(kind: str) -> Executor | None:
    match kind:
        case 'default':
            return None
        case 'synchronous':
            return SynchronousExecutor()
        case 'threads':
            return ThreadPoolExecutor(max_workers=2)
        case 'processes':
            return ProcessPoolExecutor(max_workers=2)
        case _:
            raise AssertionError(kind)


@pytest.mark.parametrize('kind', ['synchronous', 'threads', 'processes'])
def test_errors_dont_stop_later_items(kind: str) -> None:
    executor = _make_executor(kind)
    try:
        outcomes = [res for _, res in map_results(int, ['1', 'x', '3'], executor=executor)]
    finally:
        if executor is not None:
            executor.shutdown()
    assert outcomes[0] == 1
    assert outcomes[2] == 3
    error = outcomes[1]
    assert isinstance(error, ValueError), error
    if kind == 'processes':
        # the worker's traceback is attached when the error crosses the process boundary
        assert error.__cause__ is not None


def test_results_in_input_order() -> None:
    """
    Results come out in input order, even when later items finish first.
    Executors don't promise any completion order, so map_results always waits for the oldest pending item.
    """
    finished: list[int] = []
    others_done = threading.Event()

    def work(x: int) -> int:
        if x == 0:
            # timeout so the test fails rather than hangs if the other items never run
            assert others_done.wait(timeout=10)
        finished.append(x)
        if len(finished) == 3:
            others_done.set()
        return x

    with ThreadPoolExecutor(max_workers=4) as executor:
        res = results(map_results(work, range(4), executor=executor))
    assert finished[-1] == 0  # the first item really did finish last
    assert res == [0, 1, 2, 3]


@pytest.mark.parametrize('kind', ['default', 'synchronous'])
def test_serial_doesnt_run_ahead(kind: str) -> None:
    """
    When running serially, an item only runs once the consumer asks for its result, so nothing is computed ahead.
    That keeps memory as low as a plain loop, e.g. only one parsed file at a time.
    Native Executor.map(buffersize=1) would already have run the next item:
      https://github.com/python/cpython/issues/131466
    """
    calls: list[int] = []

    def work(x: int) -> int:
        calls.append(x)
        return x

    it = map_results(work, range(100), executor=_make_executor(kind))
    assert calls == []  # nothing runs until iteration starts
    for i in range(3):
        assert next(it) == (i, i)
        assert calls == list(range(i + 1))
    it.close()


class _CountingExecutor(ThreadPoolExecutor):
    submitted = 0

    @override
    def submit[**P, R](self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        self.submitted += 1
        return super().submit(fn, *args, **kwargs)


@pytest.mark.parametrize(('buffersize', 'expected'), [pytest.param(None, 4, id='default'), (3, 3)])
def test_read_ahead_is_bounded(buffersize: int | None, expected: int) -> None:
    """
    Never more than buffersize items are submitted ahead of the consumer, which defaults to twice the number of workers.
    Native Executor.map(buffersize=...) would have submitted one more: https://github.com/python/cpython/issues/131466
    """
    with _CountingExecutor(max_workers=2) as executor:
        it = map_results(str, range(100), executor=executor, buffersize=buffersize)
        assert next(it) == (0, '0')
        assert executor.submitted == expected
        it.close()


def test_default_buffersize(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Twice the number of workers, or 1 for a SynchronousExecutor.
    Also works for koncurrent's shared pool.
    """
    assert default_buffersize(SynchronousExecutor()) == 1
    assert default_buffersize(SynchronousExecutor(max_workers=4)) == 1  # still no workers
    with ThreadPoolExecutor(max_workers=3) as executor:
        assert default_buffersize(executor) == 6

    use_thread_pool(monkeypatch, workers=2)
    assert default_buffersize(default_executor()) == 4


class _Delegating(Executor):
    """
    An executor that doesn't say how many workers it has, e.g. a wrapper around some other executor.
    """

    def __init__(self, inner: Executor) -> None:
        self.inner = inner

    @override
    def submit[**P, R](self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        return self.inner.submit(fn, *args, **kwargs)


def test_unknown_executor_needs_explicit_buffersize() -> None:
    """
    For an executor that doesn't say how many workers it has, the default buffersize fails rather than guessing.
    Passing buffersize explicitly works.
    """
    executor = _Delegating(SynchronousExecutor())
    with pytest.raises(AssertionError, match='pass buffersize explicitly'):
        default_buffersize(executor)
    with pytest.raises(AssertionError, match='pass buffersize explicitly'):
        results(map_results(str, range(3), executor=executor))
    assert results(map_results(str, range(3), executor=executor, buffersize=2)) == ['0', '1', '2']


def test_closing_early_cancels_queued_work() -> None:
    started: list[int] = []
    release = threading.Event()

    def work(x: int) -> int:
        started.append(x)
        if x > 0:
            release.wait()
        return x

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            it = map_results(work, range(100), executor=executor, buffersize=4)
            assert next(it) == (0, 0)
            it.close()
        finally:
            # otherwise a failing assert above would hang on leaving the with block, waiting for the blocked item
            release.set()
    # item 1 might have started before closing, the rest were still queued and got cancelled
    assert started in ([0], [0, 1]), started
