"""The case and tools of the refinement study.

A soft plate (10 x 10 x 4 mm, chi 1000) with a magnet (4 x 4 x 3 mm, 1.2 T,
chi 0) 1 mm above it. B is read in the gap 0.25 mm above the plate, above the
rim, and far away. The plate's mesh is a tensor product: one list of edges per
axis. It can carry extra lines, each graded toward like a touching body's
footprint, and extra edges, plain bisections.
"""

import os
import sys
import time

import magpylib as magpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from bem import Panels  # noqa: E402
from multibody import Body, field_B, multi_layer_edges, solve_system  # noqa: E402
from p1 import P1  # noqa: E402

DIM = (0.01, 0.01, 0.004)
CHI = 1000.0
MAGNET = magpy.magnet.Cuboid(dimension=(0.004, 0.004, 0.003), position=(0, 0, 0.0045), polarization=(0, 0, 1.2))
FOOTPRINT = [[-0.002, 0.002], [-0.002, 0.002], []]  # the magnet's outline over the plate, per axis
POINTS = np.array([[0, 0, 2.25], [1, 1, 2.25], [2, 0, 2.25], [3, 0, 2.25], [4.5, 0, 2.25], [0, 0, 8.0], [8, 0, 0]]) * 1e-3
NAMES = ["gap centre", "gap (1, 1)", "gap, magnet edge", "gap, outside", "above rim", "far above", "far side"]
NEAR = slice(0, 5)  # within 0.25 mm of the plate


class PlateBody(Body):
    """The plate, with `lines` graded toward (like a footprint) and `extra`
    edges added as they are, per axis."""

    def __init__(self, m, lines=None, extra=None, g=2.5):
        super().__init__(DIM, chi=CHI, m=m, g=g, lines=[list(x) for x in (lines or [[], [], []])])
        self.extra = [np.asarray(e, float) for e in (extra or [[], [], []])]

    def edges(self):
        m, g, kappa = self.mesh_args
        c = np.min(self.half)
        out = []
        for k, h in enumerate(self.half):
            e = np.concatenate([multi_layer_edges(h, self.lines[k], kappa * c, m, g), self.extra[k]])
            out.append(np.unique(np.round(e / h, 12)) * h)
        return out

    def build(self):
        self.pan = Panels(self.half, 0, axis_edges=self.edges())
        self.s = P1(self.pan, nq=self.nq)
        self.G = self.s.assemble()
        return self

    def unknowns(self):
        """Unknowns before building: 3 per panel."""
        return 3 * Panels(self.half, 0, axis_edges=self.edges()).N


def solve(body):
    """Build and solve: (coefficients, B at POINTS, seconds)."""
    t = time.perf_counter()
    body.build()
    c = solve_system([body], lambda p: magpy.getH(MAGNET, p))[0]
    B = field_B([body], [c], POINTS) + MAGNET.getB(POINTS)
    return c, B, time.perf_counter() - t


def corners(body, c):
    """Each panel's corners (N, 4, 3) and its charge there (N, 4)."""
    p = body.pan
    c = np.asarray(c).reshape(p.N, 3)
    X = np.zeros((p.N, 4, 3))
    idx = np.arange(p.N)
    for r, (pp, qq) in enumerate(((p.p1, p.q1), (p.p2, p.q1), (p.p2, p.q2), (p.p1, p.q2))):
        X[idx, r, p.i] = pp
        X[idx, r, p.j] = qq
        X[idx, r, p.k] = p.w0
    s, t = np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1])
    return X, c[:, :1] + c[:, 1:2] * s + c[:, 2:3] * t


def jumps(body, c):
    """Per panel, the largest jump of the charge at its corners against the
    panels of the same face meeting there, as a share of the peak charge."""
    X, q = corners(body, c)
    key = np.concatenate([np.round(X.reshape(-1, 3) / body.half.max(), 9), np.repeat(body.pan.face, 4)[:, None]], axis=1)
    inv = np.unique(key, axis=0, return_inverse=True)[1].ravel()
    hi = np.full(inv.max() + 1, -np.inf)
    lo = np.full(inv.max() + 1, np.inf)
    np.maximum.at(hi, inv, q.ravel())
    np.minimum.at(lo, inv, q.ravel())
    return (hi - lo)[inv].reshape(-1, 4).max(1) / np.abs(q).max()


def indicator(body, c):
    """Per panel, its jump weighted by its size: jump x sqrt(area) / sqrt(total
    area). The plain jump stays large in the first panels at a soft body's
    edges, where the charge is singular and the graded panels are tiny."""
    return jumps(body, c) * np.sqrt(body.pan.area / body.pan.area.sum())


def errors(B, ref):
    """|B - ref| / |ref| per point."""
    return np.linalg.norm(B - ref, axis=1) / np.linalg.norm(ref, axis=1)
