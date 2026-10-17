"""
The process-wide pool behind default_executor().
The app sets it up with configure(),
  and the KONCURRENT_EXECUTOR and KONCURRENT_WORKERS environment variables override that.
Either way, the pool is only created when something first needs it.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import threading
import warnings
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextvars import ContextVar
from functools import partial
from typing import Any, Literal, assert_never, override

from .synchronous import SynchronousExecutor

EXECUTOR_ENV_VAR = 'KONCURRENT_EXECUTOR'
WORKERS_ENV_VAR = 'KONCURRENT_WORKERS'

_FREE_THREADED_BUILD = sysconfig.get_config_var('Py_GIL_DISABLED') == 1


# set in the shared pool's own workers by _init_worker()
# a ContextVar rather than a thread-local, so it also reaches threads that a task hands work to,
#   if they run in a copy of the task's context: e.g. via asyncio.to_thread(),
#   or any thread on free-threaded 3.14+ builds, since those start with a copy of the starting thread's context
_in_worker: ContextVar[bool] = ContextVar('koncurrent_in_worker', default=False)


def _init_worker(initializer: Callable[..., object] | None, /, *initargs: Any) -> None:
    """
    The shared pool's initializer: marks the worker, then calls configure()'s initializer (if any) with the initargs.
    It's a module-level function, so that process pools can pickle it.
    """
    # stdlib pools run all of a worker's tasks in the context their initializer ran in, so they all see the mark
    _in_worker.set(True)
    if initializer is not None:
        initializer(*initargs)


class _SharedSynchronousExecutor(SynchronousExecutor):
    """
    What default_executor() hands out when there's no pool: a single instance, shared by the whole process.

    It has to outlive the default_executor() call,
      since Executor.map(..., buffersize=...) only keeps a weak reference to its executor,
      and silently stops submitting once that's garbage collected, e.g. when a helper returns the map's iterator.
    """

    @override
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        # shared by the whole process, so borrowers don't get to shut it down (e.g. via a with block)
        pass


_synchronous = _SharedSynchronousExecutor()


class _SharedExecutor(Executor):
    """
    What default_executor() hands out when there is a pool:
      a wrapper that keeps borrowers from shutting it down or deadlocking it.

    - shutdown() does nothing, since the pool is shared by the whole process.
      Otherwise e.g. a library using the executor in a with block would break the pool for everyone else.
    - Submitting from one of the pool's own workers runs the callable right away in that worker,
        since waiting for the pool from inside it could deadlock once all the workers are busy waiting.
      default_executor() already returns a SynchronousExecutor inside the workers,
        so this is for code that got hold of the pool outside of them, e.g. a task closing over it.
    """

    def __init__(self, inner: Executor) -> None:
        self._inner = inner
        # same attribute as stdlib executors, so default_buffersize() works like for any other pool
        self._max_workers: int | None = getattr(inner, '_max_workers', None)

    @override
    def submit[**P, R](self, fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> Future[R]:
        if _in_worker.get():
            return _synchronous.submit(fn, *args, **kwargs)
        if isinstance(self._inner, ThreadPoolExecutor):
            # the GIL only matters for workers running in this process
            _warn_if_gil_reenabled()
        return self._inner.submit(fn, *args, **kwargs)

    @override
    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        pass  # see the class docstring


type _Config = tuple[Callable[..., Executor], dict[str, Any]]  # the pool's factory and its keyword arguments
type _PoolKind = Literal['threads', 'processes']
type _ExecutorKind = Literal['synchronous'] | _PoolKind  # what KONCURRENT_EXECUTOR can be


def _env_executor() -> _ExecutorKind | None:
    value = os.environ.get(EXECUTOR_ENV_VAR)
    match value:
        case None | 'synchronous' | 'threads' | 'processes':
            return value
        case _:
            raise AssertionError(f'{EXECUTOR_ENV_VAR} must be synchronous, threads or processes, got {value!r}')


def _env_workers() -> int | None:
    value = os.environ.get(WORKERS_ENV_VAR)
    if value is None:
        return None
    workers = int(value)
    assert workers >= 0, f'{WORKERS_ENV_VAR} must be non-negative, got {value!r}'
    return workers


def _cpu_count() -> int:
    """
    How many CPUs there are, or 1 if the OS can't tell.
    On 3.13+ that's only the CPUs this process may use (its CPU affinity, e.g. set by taskset),
      while on 3.12 it's all of them.
    ProcessPoolExecutor's default size is the same number (capped at 61 on Windows).
    """
    # 'or 1' rather than an explicit None check, which mypy would flag as always false on Linux,
    #   where typeshed says process_cpu_count() never returns None
    if sys.version_info >= (3, 13):
        return os.process_cpu_count() or 1
    return os.cpu_count() or 1


def _kind_config(kind: _PoolKind) -> _Config:
    """
    How koncurrent creates its own pool of this kind,
      when the environment variables choose the kind rather than configure().
    """
    match kind:
        case 'threads':
            # sized for CPU-bound work (e.g. parsing on free-threaded builds), like ProcessPoolExecutor,
            #   rather than by ThreadPoolExecutor's own default, min(32, cpus + 4), which is meant for I/O
            return ThreadPoolExecutor, {'max_workers': _cpu_count(), 'thread_name_prefix': 'koncurrent'}
        case 'processes':
            return ProcessPoolExecutor, {}
        case _:
            assert_never(kind)


def _pool_config(config: _Config | None, *, env_executor: _PoolKind | None, env_workers: int | None) -> _Config | None:
    """
    How to create the shared pool, from what was passed to configure() and the environment variables,
      or None if nothing set up parallelism.
    default_executor() has already handled the serial cases (KONCURRENT_EXECUTOR=synchronous, KONCURRENT_WORKERS=0).
    """
    if env_executor is not None:
        # replaces configure() completely: its factory could be anything,
        #   so its arguments might not suit this kind of executor
        factory, kwargs = _kind_config(env_executor)
    elif config is not None:
        factory, kwargs = config
    elif env_workers is not None:
        # nothing says which kind to use, so pick the one that can run Python code in parallel on this build
        # by build rather than by whether the GIL is enabled right now: if an extension module turned it back on,
        #   switching to processes would suddenly need picklable functions, whereas threads just get a warning
        factory, kwargs = _kind_config('threads' if _FREE_THREADED_BUILD else 'processes')
    else:
        return None
    if env_workers is not None:
        kwargs = {**kwargs, 'max_workers': env_workers}
    return factory, kwargs


class _LazyPool:
    """
    The process-wide pool: how to create it, and the pool itself once it's created.

    The lock makes configuring and creating the pool atomic, so that:
    - threads using the pool for the first time at the same moment don't each create one (all but one would leak)
    - a configure() call racing with the pool's creation either takes effect, or fails because it's too late,
        rather than being silently ignored
    """

    def __init__(self, config: _Config | None = None) -> None:
        self.config = config  # set by configure()
        self.executor: _SharedExecutor | None = None
        self._lock = threading.Lock()

    def configure(self, config: _Config) -> None:
        with self._lock:
            assert self.executor is None, 'configure() is too late, the shared pool is already running'
            self.config = config

    def get(self, *, env_executor: _PoolKind | None, env_workers: int | None) -> _SharedExecutor | None:
        """
        The pool, created by the first call, or None if nothing set up parallelism.
        """
        with self._lock:
            if self.executor is None:
                config = _pool_config(self.config, env_executor=env_executor, env_workers=env_workers)
                if config is None:
                    return None
                factory, kwargs = config
                initializer = partial(_init_worker, kwargs.get('initializer'))
                inner = factory(**{**kwargs, 'initializer': initializer})
                self.executor = _SharedExecutor(inner)
            return self.executor


_lazy_pool = _LazyPool()


if hasattr(os, 'register_at_fork'):  # only exists where os.fork() does, e.g. not on Windows, WASI, Android or iOS

    def _after_fork_in_child() -> None:
        global _lazy_pool
        # the child inherits the pool object, but none of its threads, so tasks submitted to it could wait forever
        # the lock might be stuck too, if some other thread held it during the fork
        # so start afresh, with the same settings
        _lazy_pool = _LazyPool(_lazy_pool.config)

    os.register_at_fork(after_in_child=_after_fork_in_child)


def configure(factory: Callable[..., Executor], /, **kwargs: Any) -> None:
    """
    Sets up the shared pool, e.g. configure(ThreadPoolExecutor, max_workers=8).
    It's meant for the app to call, e.g. early in its startup, rather than for libraries.
    The pool is only created as factory(**kwargs) when something first needs it.
    Until then, a later call replaces an earlier one, and after that, configure() fails with an AssertionError.

    koncurrent passes its own initializer to mark the pool's workers (calling yours from it, if you pass one),
      so the factory has to accept initializer and initargs, like all stdlib executors and SynchronousExecutor do.
    KONCURRENT_EXECUTOR and KONCURRENT_WORKERS take precedence, see default_executor().
    """
    _lazy_pool.configure((factory, kwargs))


def default_executor() -> Executor:
    """
    The executor used when none is passed explicitly: the shared pool, or a SynchronousExecutor if there's none.

    The pool is created by the first call that needs it, and fixed from then on.
    Its shutdown() does nothing, since it's shared by the whole process.
    What you get, from the highest precedence:
    - KONCURRENT_EXECUTOR=synchronous or KONCURRENT_WORKERS=0: a SynchronousExecutor, regardless of anything else.
    - KONCURRENT_EXECUTOR=threads or processes: that kind of pool, ignoring configure(),
        with KONCURRENT_WORKERS workers if that's set, and one per CPU otherwise.
    - configure(): its executor, with max_workers replaced by KONCURRENT_WORKERS if that's set,
        so a configured SynchronousExecutor stays serial.
    - Only KONCURRENT_WORKERS: threads on free-threaded builds (even if something turned the GIL back on),
        and processes otherwise.
    - Nothing: a SynchronousExecutor.
    Called from one of the pool's own workers, it's always a SynchronousExecutor,
      so nested maps run inline rather than deadlocking.
    """
    env_executor = _env_executor()
    env_workers = _env_workers()
    if env_executor == 'synchronous' or env_workers == 0 or _in_worker.get():
        return _synchronous
    executor = _lazy_pool.get(env_executor=env_executor, env_workers=env_workers)
    return _synchronous if executor is None else executor


def _gil_reenabled() -> bool:
    """
    True if this is a free-threaded build, but the GIL is enabled anyway.
    Usually that's because an extension module without free-threading support got imported, or because of PYTHON_GIL=1.
    """
    if sys.version_info < (3, 13):
        return False  # free-threaded builds only exist since 3.13
    return _FREE_THREADED_BUILD and sys._is_gil_enabled()


_gil_warned = False


def _warn_if_gil_reenabled() -> None:
    """
    Warns once if the shared thread pool can't run Python code in parallel,
      because the GIL is enabled on a free-threaded build.
    It's checked on every submit to a thread pool, so it also catches the GIL getting enabled while the pool is in use,
      e.g. by a task importing an extension module without free-threading support.
    """
    global _gil_warned
    if _gil_warned or not _gil_reenabled():
        return
    _gil_warned = True
    warnings.warn(
        'koncurrent: this is a free-threaded Python build, but the GIL is enabled, '
        "so the shared pool's threads can't run Python code in parallel. "
        'Usually that comes from importing an extension module without free-threading support: '
        'look for an earlier RuntimeWarning naming it.',
        RuntimeWarning,
        stacklevel=3,  # point at whatever submitted to the pool
    )
