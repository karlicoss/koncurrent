# koncurrent

Parallel map for libraries, through a process-wide executor that the app controls.

Work in progress, so far it only provides `koncurrent.SynchronousExecutor`:
  an executor that runs each callable right away in the calling thread, and returns an already completed future.
