from collections.abc import Iterator

import pytest

from koncurrent import EXECUTOR_ENV_VAR, WORKERS_ENV_VAR, shared


@pytest.fixture(autouse=True)
def fresh_pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # the shared pool and the shared serial executor are process-wide state, so each test starts with fresh ones
    monkeypatch.delenv(EXECUTOR_ENV_VAR, raising=False)
    monkeypatch.delenv(WORKERS_ENV_VAR, raising=False)
    monkeypatch.setattr(shared, '_lazy_pool', shared._LazyPool())
    monkeypatch.setattr(shared, '_synchronous', shared._SharedSynchronousExecutor())
    yield
    executor = shared._lazy_pool.executor
    if executor is not None:
        executor._inner.shutdown()
