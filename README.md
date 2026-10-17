# koncurrent

Parallel map for libraries, through a process-wide executor that the app controls.

Libraries often have loops that could run in parallel, e.g. parsing hundreds of export files.
But a library shouldn't start threads on its own: if every library created its own pool, they would multiply,
  and only the app knows how much CPU it can spare.
With koncurrent, a library writes such a loop as `koncurrent.map_results(fn, items)`,
  and the app decides whether and how much to parallelise it.
Without any setup, it runs serially in the calling thread, like a plain loop.

Needs Python 3.12+.
Work in progress, the API may change.

## Library side

```python
import json
from collections.abc import Iterator
from concurrent.futures import Executor
from pathlib import Path

import koncurrent


def _parse(path: Path) -> list[dict]:
    return json.loads(path.read_text())['items']


class DAL:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.executor: Executor | None = None  # None: koncurrent's default, looked up when iteration starts
        self.buffersize: int | None = None  # None: twice the executor's number of workers, or 1 when running serially

    def items(self) -> Iterator[dict | Exception]:
        results = koncurrent.map_results(_parse, self.paths, executor=self.executor, buffersize=self.buffersize)
        for path, res in results:
            if isinstance(res, Exception):
                res.add_note(f'^ while processing {path}')
                yield res
            else:
                yield from res
```

`map_results` yields `(item, result)` pairs in input order.
If the function raises an `Exception` for some item, the exception takes the place of the result,
  and later items are still processed.
Other exceptions (e.g. `KeyboardInterrupt`) and errors from iterating the items propagate, and end the map.
At most `buffersize` items are submitted ahead of the consumer, so a slow consumer doesn't pile up results in memory.
Closing the iterator early, e.g. by breaking out of `for ... in map_results(...)`,
  cancels the work that hasn't started yet.

The library doesn't choose where `fn` runs, so it has to work in all of these:

- serially, in the calling thread
- in several threads at once, which really run in parallel on free-threaded builds,
    so it shouldn't change shared state without a lock
- in worker processes, e.g. with only `KONCURRENT_WORKERS` set on a build with the GIL.
  Then `fn`, the items, the results and any exceptions have to be picklable:
    `fn` should be a module-level function (or a `functools.partial` of one),
    and exceptions need `args` that are enough to rebuild them.

Tasks on the shared thread pool run in their worker thread's `contextvars` context rather than the caller's,
  like with any `ThreadPoolExecutor`.
On free-threaded builds that includes the warnings filters,
  so `warnings.catch_warnings()` or `pytest.warns()` around a map miss warnings from its tasks.

A map nested inside a task on the shared pool (e.g. a DAL used inside another parallel loop) runs inline,
  rather than deadlocking while waiting for the pool.
So do maps in threads that the task starts, if they inherit its context,
  e.g. via `asyncio.to_thread()`, or any thread on free-threaded builds (3.14+).
A plain thread on a build with the GIL starts with an empty context though,
  unless `PYTHON_THREAD_INHERIT_CONTEXT=1` is set (3.14+), so a task waiting for one that maps can still deadlock.

Code that needs an executor rather than a map can get the shared pool with `koncurrent.default_executor()`.
Its `shutdown()` does nothing, since the pool is shared by the whole process.
`koncurrent.default_buffersize(executor)` gives the same default as `map_results`,
  e.g. for `Executor.map(fn, items, buffersize=...)` on 3.14+.

## App side

Threads only run Python code in parallel on free-threaded builds (e.g. `python3.14t`).
On builds with the GIL, they still overlap I/O and work that releases the GIL (e.g. decompression),
  but CPU-bound parsing needs processes to get faster.

The app sets up the pool shared by the whole process, e.g. early in its startup:

```python
from concurrent.futures import ThreadPoolExecutor

import koncurrent

koncurrent.configure(ThreadPoolExecutor, max_workers=8)
```

It's only created when something first needs it, as `ThreadPoolExecutor(max_workers=8)`.
Any executor works, as long as it accepts `initializer` and `initargs` like the stdlib ones,
  since koncurrent passes its own initializer (calling yours, if you pass one) to recognise the pool's workers.
If it doesn't report its number of workers the way stdlib ones do (`_max_workers`),
  libraries have to pass `buffersize` themselves.
`koncurrent.SynchronousExecutor` takes those arguments too, so `configure(SynchronousExecutor)` keeps everything serial,
  unless `KONCURRENT_EXECUTOR` says otherwise.
A `ProcessPoolExecutor` needs the app's main script to have an `if __name__ == '__main__':` guard,
  unless its workers are forked, which is the default only on Linux before 3.14.

Environment variables take precedence over `configure()`, e.g. to change things for a single run.
They only affect the shared pool, not executors passed explicitly.

- `KONCURRENT_EXECUTOR=synchronous` or `KONCURRENT_WORKERS=0`: everything runs serially in the calling thread,
    regardless of the rest.
- `KONCURRENT_EXECUTOR=threads` or `processes`: replaces `configure()` completely, including its arguments.
  The pool has one worker per CPU, unless `KONCURRENT_WORKERS` is set too.
  That's fewer threads than `ThreadPoolExecutor`'s own default, which is meant for I/O,
    since koncurrent is meant for CPU-bound work.
- `KONCURRENT_WORKERS=4`: replaces `max_workers` of the configured executor, without changing its kind,
    so a configured `SynchronousExecutor` stays serial.
  Without `configure()`, the pool uses threads on free-threaded builds, and processes otherwise.
  That goes by the build, so it still uses threads if something turned the GIL back on.

If nothing sets up parallelism, everything runs serially.

Objects that take an executor, like `DAL.executor` above, can bypass the shared pool,
  e.g. by setting it to a `ThreadPoolExecutor`, or to `koncurrent.SynchronousExecutor()` to run that object serially.

Importing an extension module that doesn't declare free-threading support turns the GIL back on for the whole process.
On a free-threaded build, if the GIL is enabled when something runs on the shared thread pool, koncurrent warns once.

## Prior art

Other languages give libraries a shared pool by default, or make one unnecessary:

- Java's [`ForkJoinPool.commonPool()`](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/concurrent/ForkJoinPool.html#commonPool()):
    one process-wide pool, sized by a system property, which parallel streams use without anyone passing it around.
- .NET's [managed thread pool](https://learn.microsoft.com/en-us/dotnet/standard/threading/the-managed-thread-pool):
    one per process, which the Task Parallel Library (`Task.Run`, `Parallel.For`) runs on.
- Rust's [Rayon](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html):
    a global pool shared by every crate calling `par_iter()`, set up on first use and sized by `RAYON_NUM_THREADS`,
    much like koncurrent's shared pool.
  A thread waiting inside Rayon runs other queued tasks meanwhile (work stealing),
    so nested parallel loops don't deadlock, whereas koncurrent runs nested maps inline.
- Go has no pools to share: goroutines are cheap,
    and the runtime schedules them all onto [`GOMAXPROCS`](https://pkg.go.dev/runtime#GOMAXPROCS) OS threads.

Python's standard library has nothing like this.
The closest things in Python:

- asyncio's default executor is the only shared pool in the standard library:
    `loop.run_in_executor(None, ...)` and `asyncio.to_thread()` run on a thread pool
    that the event loop creates on first use.
  It doesn't fit koncurrent's use case:
  - it's only reachable from async code, and there's one per event loop rather than one per process
  - `loop.set_default_executor()` only accepts a `ThreadPoolExecutor`
  - its default size, `min(32, cpus + 4)`, is meant for I/O
  - asyncio's own DNS lookups run on it, so CPU-heavy work there would hold up networking,
      as [Yury Selivanov pointed out](https://bugs.python.org/msg355846) when `asyncio.to_thread()` was being designed
- joblib's [`parallel_config()`](https://joblib.readthedocs.io/en/stable/generated/joblib.parallel_config.html)
    sets how library code using joblib's `Parallel` runs, without passing anything through the library's API.
  But it's a context manager around the calls rather than a process-wide setting, and it only applies to joblib.
- dask lets the app set a global pool for its threaded scheduler, via
    [`dask.config.set(pool=ThreadPoolExecutor(4))`](https://docs.dask.org/en/stable/scheduler-overview.html).
  That's close in shape to koncurrent, but it only applies to dask's own scheduler.
