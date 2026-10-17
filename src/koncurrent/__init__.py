"""
Parallel map for libraries, through a process-wide executor that the app controls.

Libraries call map_results() for loops that can run in parallel, e.g. parsing many export files.
Without any setup, these run serially in the calling thread, like a plain loop.
The app decides whether and how much to parallelise them,
  via configure() or the KONCURRENT_EXECUTOR and KONCURRENT_WORKERS environment variables,
  or by passing an explicit executor.
"""

from .mapping import default_buffersize, map_results
from .shared import EXECUTOR_ENV_VAR, WORKERS_ENV_VAR, configure, default_executor
from .synchronous import SynchronousExecutor

__all__ = [
    'EXECUTOR_ENV_VAR',
    'WORKERS_ENV_VAR',
    'SynchronousExecutor',
    'configure',
    'default_buffersize',
    'default_executor',
    'map_results',
]
