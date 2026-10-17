import contextvars
import os
import select
import signal
import sys
import threading
import warnings
from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from koncurrent import (
    EXECUTOR_ENV_VAR,
    WORKERS_ENV_VAR,
    SynchronousExecutor,
    configure,
    default_buffersize,
    default_executor,
    map_results,
    shared,
)

from .common import results, thread_name, use_thread_pool


def test_serial_by_default() -> None:
    assert results(map_results(thread_name, range(3))) == [threading.current_thread().name] * 3
    assert shared._lazy_pool.executor is None


def test_shared_pool_is_created_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    use_thread_pool(monkeypatch, workers=2)
    it = map_results(thread_name, range(4))
    assert shared._lazy_pool.executor is None  # calling map_results() alone doesn't start anything
    names = results(it)
    assert shared._lazy_pool.executor is not None
    assert all(name.startswith('koncurrent') for name in names), names


def test_explicit_executor_avoids_shared_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WORKERS_ENV_VAR, '2')
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='mine') as executor:
        names = results(map_results(thread_name, range(4), executor=executor))
    assert all(name.startswith('mine') for name in names), names
    assert shared._lazy_pool.executor is None


def _nested_map(_: int) -> tuple[str, list[str]]:
    """
    A task that maps again: returns the name of the thread it runs in, and the ones the nested map ran in.
    """
    return threading.current_thread().name, results(map_results(thread_name, range(3)))


def test_nested_map_runs_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    # with a single worker, waiting on the pool from inside it would deadlock
    use_thread_pool(monkeypatch, workers=1)
    for name, inner in results(map_results(_nested_map, range(2))):
        assert name.startswith('koncurrent'), name
        assert inner == [name] * 3


def test_captured_shared_executor_runs_inline_in_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The shared pool, obtained outside of it (e.g. kept by a library),
      runs tasks inline when they're submitted from one of its own workers.
    """
    # a spare worker, so that a regression shows up as the task running on the other worker, rather than as a deadlock
    use_thread_pool(monkeypatch, workers=2)
    executor = default_executor()

    def use_captured(_: int) -> tuple[str, str]:
        return threading.current_thread().name, executor.submit(thread_name, None).result()

    [(name, inner)] = results(map_results(use_captured, [0]))
    assert inner == name


@pytest.mark.parametrize('start', ['copied context', 'plain thread'])
def test_nested_map_in_helper_thread_runs_inline(monkeypatch: pytest.MonkeyPatch, start: str) -> None:
    """
    A task can hand a map to another thread and wait for it,
      as long as that thread runs in a copy of the task's context.
    The map then runs inline in that thread,
      rather than waiting for the pool, whose only worker is busy waiting for the thread.
    Plain threads only start with a copy of the context with sys.flags.thread_inherit_context,
      which is the default on free-threaded builds (3.14+).
    """
    if start == 'plain thread' and getattr(sys.flags, 'thread_inherit_context', 0) == 0:
        pytest.skip('plain threads start with an empty context here')
    use_thread_pool(monkeypatch, workers=1)

    def via_helper_thread(_: int) -> tuple[str, list[str]]:
        nested: list[str] = []

        def run() -> None:
            nested.extend(results(map_results(thread_name, range(2))))

        if start == 'copied context':
            # what asyncio.to_thread() does
            thread = threading.Thread(target=contextvars.copy_context().run, args=(run,))
        else:
            thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive(), 'the helper thread is stuck waiting for the pool'
        return thread.name, nested

    [(name, nested)] = results(map_results(via_helper_thread, [0]))
    assert nested == [name] * 2


@pytest.mark.parametrize('setup', ['serial', 'threads'])
def test_borrowers_cant_shut_down_default_executor(monkeypatch: pytest.MonkeyPatch, setup: str) -> None:
    """
    default_executor() hands the same executor to everyone, whether it's the shared pool or the serial default,
      so shutting it down mustn't stop it from working for everyone else.
    """
    if setup == 'threads':
        use_thread_pool(monkeypatch, workers=2)
    executor = default_executor()
    # e.g. a library closing whatever executor it was given
    with executor:
        assert executor.submit(int, '1').result() == 1
    assert default_executor() is executor
    assert executor.submit(int, '2').result() == 2


@pytest.mark.parametrize('setup', ['nothing set', 'no workers', 'synchronous'])
def test_serial_default_survives_buffered_map(monkeypatch: pytest.MonkeyPatch, setup: str) -> None:
    """
    Executor.map(..., buffersize=...) only keeps a weak reference to its executor,
      and stops submitting once that's gone.
    So when running serially, the executor from default_executor() has to stay alive even if the caller drops it,
      e.g. when a helper returns the map's iterator.
    """
    if sys.version_info < (3, 14):
        pytest.skip('Executor.map only has buffersize since Python 3.14')
    match setup:
        case 'no workers':
            monkeypatch.setenv(WORKERS_ENV_VAR, '0')
        case 'synchronous':
            monkeypatch.setenv(EXECUTOR_ENV_VAR, 'synchronous')

    def parse_all(items: list[str]) -> Iterator[int]:
        executor = default_executor()
        return executor.map(int, items, buffersize=default_buffersize(executor))

    assert list(parse_all([str(i) for i in range(10)])) == list(range(10))


@pytest.mark.parametrize('use', ['map_results', 'directly'])
def test_warns_once_if_gil_got_reenabled(monkeypatch: pytest.MonkeyPatch, use: str) -> None:
    """
    Using the shared pool warns once if the GIL got enabled in a free-threaded build,
      whether it's used through map_results() or directly, e.g. with Executor.map.
    """
    use_thread_pool(monkeypatch, workers=2)
    monkeypatch.setattr(shared, '_gil_reenabled', lambda: True)
    monkeypatch.setattr(shared, '_gil_warned', False)

    def run(x: str) -> None:
        if use == 'map_results':
            assert results(map_results(int, [x])) == [int(x)]
        else:
            assert list(default_executor().map(int, [x])) == [int(x)]

    with pytest.warns(RuntimeWarning, match='GIL'):
        run('1')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        run('2')


def test_forked_child_gets_fresh_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform != 'linux':
        pytest.skip("forks with the pool's threads running, which isn't possible on Windows or safe on macOS")
    use_thread_pool(monkeypatch, workers=2)
    assert results(map_results(int, ['1'])) == [1]  # starts the pool in this process

    read_fd, write_fd = os.pipe()
    with warnings.catch_warnings():
        # forking while the pool's threads are alive is exactly the situation being tested
        warnings.simplefilter('ignore', DeprecationWarning)
        pid = os.fork()
    if pid == 0:
        # reusing the parent's pool would hang here, since its threads didn't survive the fork
        try:
            os.write(write_fd, repr(results(map_results(thread_name, range(2)))).encode())
        finally:
            os._exit(0)

    os.close(write_fd)
    ready, _, _ = select.select([read_fd], [], [], 10)
    if len(ready) == 0:
        os.kill(pid, signal.SIGKILL)
    output = os.read(read_fd, 1000).decode() if len(ready) > 0 else ''
    os.close(read_fd)
    os.waitpid(pid, 0)
    assert 'koncurrent' in output, 'child process hung or failed'


def test_configure() -> None:
    """
    configure() sets up the shared pool without any environment variables, and it's still only created on first use.
    """
    configure(ThreadPoolExecutor, max_workers=3, thread_name_prefix='configured')
    assert shared._lazy_pool.executor is None
    names = results(map_results(thread_name, range(4)))
    assert all(name.startswith('configured') for name in names), names


def test_configure_after_first_use_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    use_thread_pool(monkeypatch, workers=2)
    assert results(map_results(int, ['1'])) == [1]  # starts the pool
    with pytest.raises(AssertionError, match='already running'):
        configure(ThreadPoolExecutor, max_workers=4)


def test_configure_keeps_own_initializer() -> None:
    """
    An initializer passed to configure() still runs in each worker, along with koncurrent's own one marking the workers.
    """
    initialized: list[str] = []
    configure(ThreadPoolExecutor, max_workers=1, initializer=initialized.append, initargs=('done',))
    for name, inner in results(map_results(_nested_map, range(2))):
        assert inner == [name] * 3  # the worker is still marked, so the nested map ran inline
    assert initialized == ['done']


@pytest.mark.parametrize('env_workers', [None, '4'])
def test_configure_synchronous(monkeypatch: pytest.MonkeyPatch, env_workers: str | None) -> None:
    """
    configure(SynchronousExecutor) runs everything serially.
    KONCURRENT_WORKERS doesn't change that, since it only replaces max_workers, which SynchronousExecutor ignores.
    """
    configure(SynchronousExecutor)
    if env_workers is not None:
        monkeypatch.setenv(WORKERS_ENV_VAR, env_workers)
    assert results(map_results(thread_name, range(2))) == [threading.current_thread().name] * 2
    assert default_buffersize(default_executor()) == 1
    # its initializer never ran, so this thread isn't mistaken for one of the pool's workers
    assert not shared._in_worker.get()


# module-level, so that process pools can pickle them
def _where(_: object) -> tuple[int, str]:
    return os.getpid(), threading.current_thread().name


def _nested_in_process(_: int) -> tuple[tuple[int, str], list[tuple[int, str]]]:
    return _where(None), results(map_results(_where, range(2)))


def test_configure_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The shared pool can be a process pool, e.g. for CPU-bound pure Python code on builds with the GIL.
    Maps nested inside a worker process still run inline there, rather than starting another pool in the worker.
    """
    # inherited by the worker processes, so without the marker they'd each start a pool of their own
    monkeypatch.setenv(WORKERS_ENV_VAR, '2')
    configure(ProcessPoolExecutor)
    for worker, inner in results(map_results(_nested_in_process, range(2))):
        assert worker[0] != os.getpid()
        assert inner == [worker, worker]


# module-level, so that process pools can pickle them
_worker_state: dict[str, str] = {}


def _remember(value: str) -> None:
    _worker_state['initialized'] = value


def _recall(_: int) -> str | None:
    return _worker_state.get('initialized')


def test_configure_process_pool_keeps_own_initializer() -> None:
    """
    An initializer passed to configure() runs in process pool workers too, with its initargs.
    koncurrent's own initializer, which calls it, gets pickled along with it under spawn and forkserver.
    """
    configure(ProcessPoolExecutor, max_workers=1, initializer=_remember, initargs=('done',))
    assert results(map_results(_recall, range(2))) == ['done', 'done']
    assert _worker_state == {}  # it only ran in the worker


class _Configured(ThreadPoolExecutor):
    """
    Stands in for whatever executor the app passes to configure().
    """


_KINDS: dict[type[Executor], str] = {
    _Configured: 'configured',
    ThreadPoolExecutor: 'threads',
    ProcessPoolExecutor: 'processes',
}


def _default_workers(kind: str) -> int:
    """
    How many workers koncurrent's own pool of this kind has when nothing says how many.
    """
    if kind == 'threads':
        return shared._cpu_count()
    # whatever ProcessPoolExecutor picks itself
    with ProcessPoolExecutor() as executor:
        return default_buffersize(executor) // 2


# fmt: off
_PRECEDENCE = [
    # configure()  KONCURRENT_EXECUTOR  KONCURRENT_WORKERS  kind          workers
    (False,        None,                None,               'serial',     None),
    (False,        None,                '0',                'serial',     None),
    (False,        None,                '2',                'auto',       2),
    (False,        'synchronous',       None,               'serial',     None),
    (False,        'synchronous',       '0',                'serial',     None),
    (False,        'synchronous',       '2',                'serial',     None),
    (False,        'threads',           None,               'threads',    None),
    (False,        'threads',           '0',                'serial',     None),
    (False,        'threads',           '2',                'threads',    2),
    (False,        'processes',         None,               'processes',  None),
    (False,        'processes',         '0',                'serial',     None),
    (False,        'processes',         '2',                'processes',  2),
    (True,         None,                None,               'configured', 3),
    (True,         None,                '0',                'serial',     None),
    (True,         None,                '2',                'configured', 2),
    (True,         'synchronous',       None,               'serial',     None),
    (True,         'synchronous',       '0',                'serial',     None),
    (True,         'synchronous',       '2',                'serial',     None),
    (True,         'threads',           None,               'threads',    None),
    (True,         'threads',           '0',                'serial',     None),
    (True,         'threads',           '2',                'threads',    2),
    (True,         'processes',         None,               'processes',  None),
    (True,         'processes',         '0',                'serial',     None),
    (True,         'processes',         '2',                'processes',  2),
]
# fmt: on


@pytest.mark.parametrize(('configured', 'env_executor', 'env_workers', 'kind', 'workers'), _PRECEDENCE)
def test_precedence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    configured: bool,
    env_executor: str | None,
    env_workers: str | None,
    kind: str,
    workers: int | None,
) -> None:
    """
    What default_executor() gives for every combination of configure() and the environment variables.
    configure() is called as configure(_Configured, max_workers=3).
    kind 'auto' means threads on free-threaded builds, and processes otherwise.
    workers=None means the pool's default size, or no pool at all for 'serial'.
    """
    if configured:
        configure(_Configured, max_workers=3)
    if env_executor is not None:
        monkeypatch.setenv(EXECUTOR_ENV_VAR, env_executor)
    if env_workers is not None:
        monkeypatch.setenv(WORKERS_ENV_VAR, env_workers)

    executor = default_executor()
    if isinstance(executor, SynchronousExecutor):
        actual: tuple[str, int | None] = ('serial', None)
    else:
        assert isinstance(executor, shared._SharedExecutor)
        actual = (_KINDS[type(executor._inner)], executor._max_workers)

    if kind == 'auto':
        kind = 'threads' if shared._FREE_THREADED_BUILD else 'processes'
    if kind in {'threads', 'processes'} and workers is None:
        workers = _default_workers(kind)
    assert actual == (kind, workers)


@pytest.mark.parametrize(
    ('free_threaded', 'gil', 'pool_type'),
    [
        (False, True, ProcessPoolExecutor),
        (True, False, ThreadPoolExecutor),
        # e.g. an extension module without free-threading support turned the GIL back on
        (True, True, ThreadPoolExecutor),
    ],
)
def test_kind_follows_build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free_threaded: bool,
    gil: bool,
    pool_type: type[Executor],
) -> None:
    """
    With only KONCURRENT_WORKERS set, the pool uses whatever can run Python code in parallel on this build:
      threads on free-threaded builds, and processes otherwise.
    It goes by the build rather than by whether the GIL is enabled right now,
      so that something turning the GIL back on doesn't suddenly require picklable functions.
    """
    monkeypatch.setattr(shared, '_FREE_THREADED_BUILD', free_threaded)
    monkeypatch.setattr(shared, '_gil_reenabled', lambda: free_threaded and gil)
    monkeypatch.setenv(WORKERS_ENV_VAR, '2')
    executor = default_executor()
    assert isinstance(executor, shared._SharedExecutor)
    assert isinstance(executor._inner, pool_type)


def test_thread_pool_has_one_worker_per_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    koncurrent's own thread pool is meant for CPU-bound work, so it has one worker per CPU,
      rather than ThreadPoolExecutor's default, min(32, cpus + 4), which is meant for I/O.
    """
    # process_cpu_count() only exists since 3.13, and unlike cpu_count() it respects CPU affinity
    monkeypatch.setattr(os, 'process_cpu_count' if sys.version_info >= (3, 13) else 'cpu_count', lambda: 3)
    monkeypatch.setenv(EXECUTOR_ENV_VAR, 'threads')
    executor = default_executor()
    assert isinstance(executor, shared._SharedExecutor)
    assert isinstance(executor._inner, ThreadPoolExecutor)
    assert executor._max_workers == 3


def test_invalid_executor_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTOR_ENV_VAR, 'fibers')
    with pytest.raises(AssertionError, match='must be synchronous, threads or processes'):
        default_executor()
