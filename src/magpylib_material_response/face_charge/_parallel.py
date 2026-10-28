"""The assembly's and the fields' chunks on every core.

NumPy releases the interpreter lock in the arrays' arithmetic, and each chunk
writes its own block, so threads share the work without copies; a chunk is
sized so that all of them together take the memory one took alone.
"""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import magpylib as magpy
import numpy as np

CHUNK = 2_000_000  # kernel evaluations per call, about 0.5 GB, shared by the threads
# below this many kernel evaluations the threads cost more than they save: 100
# small cubes coupled pair by pair ran 2.5 times slower on 14 threads
MIN_WORK = 200_000

_pool, _pool_threads = None, 0  # made on first use and kept while the count holds
_worker = threading.local()


def threads():
    """The threads to run on: magpylib's ``defaults.threads`` where magpylib
    has it, else counted as it counts them: the CPUs this process may use, up
    to 8, and one in WebAssembly builds such as Pyodide, which cannot start
    threads."""
    count = getattr(magpy.defaults, "threads", None)
    if count is None:
        if sys.platform in ("emscripten", "wasi"):
            return 1
        if hasattr(os, "process_cpu_count"):  # Python 3.13+
            count = os.process_cpu_count()
        elif hasattr(os, "sched_getaffinity"):  # Linux
            count = len(os.sched_getaffinity(0))
        else:
            count = os.cpu_count()
        count = min(8, count or 1)
    return max(1, int(count))


def _mark(fn):
    # a new thread starts from NumPy's default error settings: take the caller's
    errstate, errcall = np.geterr(), np.geterrcall()

    def call(item):
        _worker.inside = True
        with np.errstate(call=errcall, **errstate):
            return fn(item)

    return call


def run(fn, items, work=None):
    """``fn(item)`` for each item, on :func:`threads` threads; the first error
    raised is raised here. On one thread when the ``work`` (kernel
    evaluations) is small, and inside a worker already: the pool does not wait
    on itself."""
    global _pool, _pool_threads  # noqa: PLW0603  # pylint: disable=global-statement
    items = list(items)
    count = threads()
    serial = (
        count == 1
        or len(items) < 2
        or getattr(_worker, "inside", False)
        or (work is not None and work < MIN_WORK)
    )
    if serial:
        for item in items:
            fn(item)
        return
    if _pool_threads != count:
        if _pool is not None:
            _pool.shutdown(wait=False)
        _pool = ThreadPoolExecutor(count, thread_name_prefix="face-charge")
        _pool_threads = count
    for _ in _pool.map(_mark(fn), items):
        pass
