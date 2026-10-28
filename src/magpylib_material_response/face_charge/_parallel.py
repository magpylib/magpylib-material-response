"""The assembly's and the fields' chunks on every core.

NumPy releases the interpreter lock in the arrays' arithmetic, and each chunk
writes its own block, so threads share the work without copies; a chunk is
sized so that all of them together take the memory one took alone.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

THREADS = os.cpu_count() or 1
CHUNK = 2_000_000  # kernel evaluations per call, about 0.5 GB, shared by the threads
# below this many kernel evaluations the threads cost more than they save: 100
# small cubes coupled pair by pair ran 2.5 times slower on 14 threads
MIN_WORK = 200_000

_pool = None  # made on first use and kept: a new pool per call cost small models
_worker = threading.local()


def _mark(fn):
    def call(item):
        _worker.inside = True
        return fn(item)

    return call


def run(fn, items, work=None):
    """``fn(item)`` for each item, on ``THREADS`` threads; the first error
    raised is raised here. On one thread when the ``work`` (kernel
    evaluations) is small, and inside a worker already: the pool does not wait
    on itself."""
    global _pool  # noqa: PLW0603  # pylint: disable=global-statement
    items = list(items)
    serial = (
        THREADS == 1
        or len(items) < 2
        or getattr(_worker, "inside", False)
        or (work is not None and work < MIN_WORK)
    )
    if serial:
        for item in items:
            fn(item)
        return
    if _pool is None:
        _pool = ThreadPoolExecutor(THREADS, thread_name_prefix="face-charge")
    for _ in _pool.map(_mark(fn), items):
        pass
