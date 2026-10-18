"""
map_results(), the ordered parallel map that libraries use, and default_buffersize() for sizing its read-ahead.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Executor, Future
from contextlib import AbstractContextManager
from typing import override

from .shared import default_executor


def default_buffersize(executor: Executor) -> int:
    """
    How many items to submit to the executor ahead of the consumer, when the caller doesn't say otherwise.
    map_results() uses it, and it suits other ordered maps too, e.g. Executor.map(fn, items, buffersize=...) on 3.14+.

    Twice the executor's number of workers: enough for each worker to have an item running and about one more queued,
      so the workers rarely wait for the consumer.
    A smaller buffer leaves workers idle,
      while a larger one holds more finished results in memory, and wastes more work if the consumer stops early.
    A SynchronousExecutor has no workers, so it's 1: it runs each item as soon as it's submitted,
      so running ahead would only use more memory.
    Fails for executors that don't say how many workers they have, rather than guessing.
    Pass buffersize explicitly for those.
    """
    # executors have no public way to tell this,
    #   but stdlib ones, SynchronousExecutor and koncurrent's shared pool have _max_workers
    workers: int | None = getattr(executor, '_max_workers', None)
    assert workers is not None, f"Don't know how many workers {executor} has, pass buffersize explicitly"
    return max(1, 2 * workers)


def map_results[T, R](
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    executor: Executor | None = None,
    buffersize: int | None = None,
) -> MapResults[R | Exception]:
    """
    Yields fn(item) for each item, in input order.
    If fn raises an Exception, yields the exception instead and carries on with later items.
    Other exceptions (e.g. KeyboardInterrupt) and errors from iterating items propagate, which ends the map.

    executor: where to run fn.
      None means default_executor(), looked up when iteration starts rather than when map_results() is called,
        so the shared pool isn't created unless something actually iterates.
    buffersize: how many items may be submitted ahead of the consumer, which bounds memory if the consumer is slow.
      None means default_buffersize(executor).

    Closing the iterator early cancels the work that hasn't started yet.
    That also happens once it's garbage collected, e.g. right after breaking out of `for ... in map_results(...)`,
      unless something else still refers to it.
    The iterator is also a context manager that closes it on exit, like os.scandir(),
      e.g. `with map_results(...) as results:` in a generator that yields from it.
    That cancels the queued work even if something still refers to the map, e.g. the traceback of an exception kept around.
    On a process pool, one more item than the running ones may still run, since it's already queued for the workers.

    To know which item a result or an error belongs to, zip them, e.g. `zip(paths, map_results(parse, paths), strict=True)`.
    For a one-shot iterator, `inputs, to_map = itertools.tee(items)` works,
      and only keeps about buffersize items around, since that's how far the map reads ahead.

    >>> list(map_results(int, ['1', 'x', '3']))
    [1, ValueError("invalid literal for int() with base 10: 'x'"), 3]
    """
    return MapResults(_map_results(fn, items, executor=executor, buffersize=buffersize))


class MapResults[T](Iterator[T], AbstractContextManager['MapResults[T]', None]):
    """
    What map_results() returns: an iterator that's also a context manager closing it on exit, like os.scandir().
    __iter__() and __enter__() come from the base classes, and both return the MapResults itself.
    """

    def __init__(self, results: Generator[T, None, None]) -> None:
        self._results = results

    @override
    def __next__(self) -> T:
        return next(self._results)

    def close(self) -> None:
        """
        Cancel the work that hasn't started yet.
        Does nothing once the map is finished.
        """
        self._results.close()

    @override
    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _map_results[T, R](
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    executor: Executor | None = None,
    buffersize: int | None = None,
) -> Generator[R | Exception, None, None]:
    """
    The generator behind map_results(), see there.
    """
    if executor is None:
        executor = default_executor()

    if buffersize is None:
        buffersize = default_buffersize(executor)
    assert buffersize > 0, buffersize

    # NOTE: this is roughly Executor.map(fn, items, buffersize=...), which doesn't fit here because:
    # - it stops at the first error, whereas here errors are returned and later items carry on
    # - it submits one more item than buffersize: https://github.com/python/cpython/issues/131466
    # - buffersize only exists since 3.14
    pending: deque[Future[R]] = deque()

    def pop_oldest() -> R | Exception:
        future = pending.popleft()
        try:
            return future.result()
        except Exception as e:
            return e

    try:
        for item in items:
            pending.append(executor.submit(fn, item))
            if len(pending) >= buffersize:
                yield pop_oldest()
        while len(pending) > 0:
            yield pop_oldest()
    finally:
        # only non-empty when leaving early: the iterator got closed,
        #   or something raised (e.g. iterating items, or Ctrl-C), so drop the work nobody will look at
        for future in pending:
            future.cancel()
