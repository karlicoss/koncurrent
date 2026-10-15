"""
Parallel map for libraries, through a process-wide executor that the app controls.
"""

from .synchronous import SynchronousExecutor

__all__ = [
    'SynchronousExecutor',
]
