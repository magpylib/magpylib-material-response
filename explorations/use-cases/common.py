"""Helpers shared by the use-case scripts: timing, the split of a face-charge
solve into its source-field part and the rest, peak memory, result files."""

from __future__ import annotations

import contextlib
import json
import os
import resource
import sys
import tempfile
import time
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np
import scipy.linalg

warnings.filterwarnings("ignore")
mm = 1e-3
HERE = Path(__file__).parent
RESULTS = HERE / "results"


def peak_rss_gb():
    """This process's peak resident memory in GB (macOS reports bytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if sys.platform == "darwin" else 1e6)


class SourceFieldTimer:
    """Time spent in magpylib.getB while active: inside apply_face_charge that is
    the field of the fixed sources at the panel points, the load of the solve,
    the one part that changes when only sources move."""

    def __init__(self):
        self.seconds = 0.0
        self._getB = magpy.getB

    def __enter__(self):
        def timed(*args, **kwargs):
            t = time.perf_counter()
            try:
                return self._getB(*args, **kwargs)
            finally:
                self.seconds += time.perf_counter() - t

        magpy.getB = timed
        return self

    def __exit__(self, *exc):
        magpy.getB = self._getB


def solve_timed(solver, collection, **kwargs):
    """Solve and return (solved, total s, source-field s)."""
    with SourceFieldTimer() as src:
        t = time.perf_counter()
        solved = solver(collection, **kwargs)
        total = time.perf_counter() - t
    return solved, total, src.seconds


def backsub_seconds(n, rhs=1, repeat=5):
    """Median time of one LU back-substitution for n unknowns, `rhs` columns:
    what a kept factorization costs per configuration."""
    rng = np.random.default_rng(0)
    lu = scipy.linalg.lu_factor(rng.standard_normal((n, n)) + n * np.eye(n))
    b = rng.standard_normal((n, rhs))
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        scipy.linalg.lu_solve(lu, b)
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def write_result(name, data):
    """results/<name>.json, written atomically so a stopped run leaves no half file."""
    RESULTS.mkdir(exist_ok=True)
    data = {**data, "peak_rss_gb": round(peak_rss_gb(), 2)}
    fd, tmp = tempfile.mkstemp(dir=RESULTS, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1, default=float)
    os.replace(tmp, RESULTS / f"{name}.json")
    print(f"wrote results/{name}.json")


def done(name):
    return (RESULTS / f"{name}.json").exists()
