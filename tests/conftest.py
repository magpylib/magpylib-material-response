from __future__ import annotations

import os
import time

import magpylib as magpy
import numpy as np
import pytest


def _getH_tensor(cells):
    """The tensor ``T[k, i, j, m]`` of ``demag_tensor`` straight from
    ``magpy.getH``: the field at each cell's centre of each cell polarized
    along k. A point-matched reference that shares no code with the solver."""
    cells = [c.copy() for c in cells]
    pos = np.array([c.centroid for c in cells])
    n = len(cells)
    T = []
    for unit in np.eye(3):
        for c in cells:
            c.polarization = c.orientation.inv().apply(unit)
        T.append(np.reshape(magpy.getH(cells, pos), (n, n, 3)))
    return np.array(T)


@pytest.fixture
def getH_tensor():
    return _getH_tensor


#: seconds a test may take unless marked slow, on the CI job of Ubuntu with
#: Python 3.14 (MR_TEST_BUDGET set to it there); a slower test proves its
#: behaviour on a smaller model, or is marked slow, which a pull request runs
#: in a job of its own
BUDGET = float(os.environ.get("MR_TEST_BUDGET") or "inf")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    start = time.perf_counter()
    result = yield
    seconds = time.perf_counter() - start
    if seconds > BUDGET and item.get_closest_marker("slow") is None:
        pytest.fail(
            f"took {seconds:.1f} s, over the {BUDGET:.0f} s a test may take "
            "unless marked slow",
            pytrace=False,
        )
    return result
