from __future__ import annotations

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
