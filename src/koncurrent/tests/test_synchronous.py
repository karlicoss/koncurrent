import threading
import traceback
from concurrent.futures import as_completed, wait

import pytest

from koncurrent import SynchronousExecutor


def thread_name() -> str:
    return threading.current_thread().name


def test_runs_in_calling_thread() -> None:
    """
    The callable runs in the thread that called submit(), not in some worker thread.
    """
    future = SynchronousExecutor().submit(thread_name)
    assert future.done()
    assert future.result() == threading.current_thread().name


def test_runs_during_submit() -> None:
    """
    The callable runs during submit() itself, rather than later (e.g. when someone asks for the result).
    So it behaves like a plain function call: side effects happen in order with the caller's own code,
      and work submitted just for its side effects runs even if nobody calls result().
    """
    calls: list[int] = []
    executor = SynchronousExecutor()
    first = executor.submit(calls.append, 1)
    calls.append(2)
    second = executor.submit(calls.append, 3)
    # nobody asked for any results, but both calls already happened, in order with the caller's own code
    assert calls == [1, 2, 3]
    assert first.done()
    assert second.done()


def test_futures_are_done_when_submit_returns() -> None:
    """
    Code written for pool executors often waits via concurrent.futures.wait() or as_completed() rather than result().
    With a SynchronousExecutor, nothing else would ever complete the futures,
      so they have to be done by the time submit() returns, otherwise such code would block forever.
    """
    executor = SynchronousExecutor()
    futures = [executor.submit(int, s) for s in ['1', '2']]
    _, not_done = wait(futures, timeout=0)
    assert not_done == set()
    assert sorted(f.result() for f in as_completed(futures, timeout=0)) == [1, 2]


def test_exception_goes_into_future() -> None:
    """
    An exception raised by the callable doesn't escape from submit(), it's stored in the returned future instead.
    Same as with pool executors, so callers handle errors the same way whichever executor they get.
    """
    future = SynchronousExecutor().submit(int, 'x')
    assert future.done()
    assert isinstance(future.exception(), ValueError)


def test_exception_keeps_traceback() -> None:
    """
    The stored exception keeps its traceback, so it still points at where it was raised.
    result() reraises that same exception, adding its own frames on top of the original ones.
    """

    def fail() -> None:
        raise ValueError('boom')

    future = SynchronousExecutor().submit(fail)
    assert future.done()
    error = future.exception()
    assert error is not None
    assert traceback.extract_tb(error.__traceback__)[-1].name == 'fail'

    with pytest.raises(ValueError) as excinfo:
        future.result()
    assert excinfo.value is error
    frames = [frame.name for frame in traceback.extract_tb(error.__traceback__)]
    assert frames[0] == 'test_exception_keeps_traceback'
    assert frames[-1] == 'fail'


def test_keyboard_interrupt_propagates_right_away() -> None:
    """
    Unlike ordinary exceptions, BaseExceptions like KeyboardInterrupt propagate from submit() right away.
    The callable runs in the caller's thread, so this is the caller's own Ctrl-C:
      storing it in the future would delay it until someone calls result(), or swallow it if nobody does.
    """

    def interrupt() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        SynchronousExecutor().submit(interrupt)


def test_no_submit_after_shutdown() -> None:
    """
    Like other executors, it refuses new work with RuntimeError after shutdown(), e.g. after leaving a `with executor:` block.
    """
    executor = SynchronousExecutor()
    executor.shutdown()
    with pytest.raises(RuntimeError):
        executor.submit(int, '1')
