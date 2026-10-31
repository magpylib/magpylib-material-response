"""The assembly's and the fields' chunks on every core.

NumPy releases the interpreter lock in the arrays' arithmetic, and each chunk
writes its own block, so threads share the work without copies; a chunk is
sized so that all of them together take the memory one took alone.
"""

from __future__ import annotations

import os
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


def _forget_pool():
    # a forked child inherits the pool but none of its threads: work handed to
    # it would wait forever (a multiprocessing.Pool on Linux, after a solve)
    global _pool, _pool_threads  # pylint: disable=global-statement
    _pool, _pool_threads = None, 0


if hasattr(os, "register_at_fork"):  # not on Windows
    os.register_at_fork(after_in_child=_forget_pool)


def threads():
    """The threads to run on: magpylib's ``defaults.threads``."""
    return max(1, int(magpy.defaults.threads))


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
