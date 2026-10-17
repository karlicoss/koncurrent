"""
The stdlib doesn't have an executor like this, but it has been discussed:
- https://github.com/python/cpython/issues/80576 (bpo-36395): a request for a fake executor for tests,
    which would only run the callable once result() is called.
  The reporter's hand-rolled fake futures broke concurrent.futures.wait().
  SynchronousExecutor runs callables during submit() and returns standard futures instead,
    so wait() and as_completed() work with it.
- https://mail.python.org/archives/list/python-ideas@python.org/thread/LMTQ2AI6A7UXEFVHRGHKWD33H24FGM6G/:
    a python-ideas proposal to add a SerialExecutor to concurrent.futures.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor, Future
from typing import Any, override


class SynchronousExecutor(Executor):
    """
    Runs each callable right away in the thread calling submit(), and returns an already completed Future.

    Useful when an executor is required, but everything should run serially,
      e.g. to opt out of the shared pool for one object, or for debugging and profiling.
    Unlike pool executors, it lets BaseExceptions like KeyboardInterrupt propagate from submit() right away,
      since they happen in the caller's thread.

    It takes the keyword arguments that all stdlib pools share (max_workers, initializer and initargs),
      so it can stand in for them, e.g. in koncurrent.configure().
    But it has no workers, so they don't do anything: max_workers is ignored, and initializer is never called.
    Calling it in the caller's thread instead could break the caller,
      e.g. a common initializer for process pools makes the workers ignore Ctrl-C.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        initializer: Callable[..., object] | None = None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        # there are no workers, so nothing to size or set up
        del max_workers, initializer, initargs
        self._max_workers = 0  # same attribute as stdlib pools, e.g. for default_buffersize()
        self._shutdown = False

    @override
    def submit[**P, R](self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        if self._shutdown:
            raise RuntimeError('cannot schedule new futures after shutdown')
        future: Future[R] = Future()
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            future.set_exception(e)
        else:
            future.set_result(result)
        return future

    @override
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        # everything submitted has already completed, so there is nothing to wait for or cancel
        self._shutdown = True
