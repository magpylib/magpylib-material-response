"""Test cases: a cuboid with susceptibility chi (and maybe remanence) near a magnet."""
import numpy as np, magpylib as magpy
from bem import Panels, kprime_gal, solve, face_average
MU0 = 4e-7*np.pi

SHAPES = {"cube": np.array([5e-3, 5e-3, 5e-3]),       # half sides (m)
          "plate": np.array([10e-3, 5e-3, 1.25e-3])}
DIST = np.array([0.1, 0.25, 0.5, 1, 2, 5]) * 1e-3

def observers(half):
    """Points at distance d off the top-face centre, an edge midpoint and a corner."""
    pts, tags = [], []
    for d in DIST:
        pts.append([0.3*half[0], 0.2*half[1], half[2] + d]); tags.append(("face", d))
        pts.append([half[0] + d/np.sqrt(2), 0.1*half[1], half[2] + d/np.sqrt(2)]); tags.append(("edge", d))
        pts.append(half + d/np.sqrt(3)); tags.append(("corner", d))
    return np.array(pts), tags

def source(half):
    """A 5 mm NdFeB cube below the body, 2 mm gap, offset sideways, tilted."""
    m = magpy.magnet.Cuboid(dimension=(5e-3,)*3, polarization=(0.2, 0, 1.2),
                            position=(0.4*half[0], -0.3*half[1], -half[2] - 2e-3 - 2.5e-3))
    m.rotate_from_angax(20, "y")
    return m

def bem_case(shape, chi, n, J0=None, uniform=None, K=None, pan=None):
    """Solve and return (pan, sigma, K). J0: remanent polarization (T) of the body.
    uniform: applied mu0 H (T) instead of the magnet."""
    half = SHAPES[shape]
    if pan is None:
        pan = Panels(half, n, "power", 2.0)
    if K is None:
        K = kprime_gal(pan, 3)
    if uniform is not None:
        f = pan.normal @ (np.asarray(uniform) / MU0)
    else:
        src = source(half)
        f = face_average(pan, lambda p: magpy.getH(src, p))
    m0n = None if J0 is None else pan.normal @ (np.asarray(J0) / MU0)
    sigma = solve(pan, chi, f, m0n=m0n, K=K)
    return pan, sigma, K

def body_B(pan, sigma, obs):
    """B (T) of the body alone at outside observers."""
    return MU0 * pan.field(obs, sigma)
