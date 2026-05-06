"""Unit tests for the Newell analytical demag tensor."""

from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.demag import apply_demag, demag_tensor
from magpylib_material_response.meshing import mesh_Cuboid
from magpylib_material_response.newell import (
    demag_block,
    demag_tensor_newell,
    newell_f,
    newell_g,
    self_demag_factors,
)


def test_self_demag_cube_one_third():
    factors = self_demag_factors([1.0, 1.0, 1.0])
    np.testing.assert_allclose(factors, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)


@pytest.mark.parametrize(
    "dim",
    [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 0.5),
        (1.0, 1.0, 0.1),  # oblate
        (0.1, 0.1, 5.0),  # prolate
        (3.7, 1.2, 0.8),
    ],
)
def test_self_demag_trace_is_one(dim):
    """Brown's identity: Nxx + Nyy + Nzz = 1 for any rectangular prism."""
    factors = self_demag_factors(dim)
    np.testing.assert_allclose(factors.sum(), 1.0, atol=1e-9)


def test_self_block_offdiagonal_zero():
    """Off-diagonal demag elements vanish at zero displacement (mirror symmetry)."""
    N = demag_block(np.zeros(3), [1.5, 0.7, 1.1])
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert abs(N[i, j]) < 1e-12, f"N[{i},{j}] = {N[i, j]} not zero"


def test_block_symmetric():
    rng = np.random.default_rng(0)
    disp = rng.normal(size=(5, 3))
    N = demag_block(disp, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(N, np.swapaxes(N, -1, -2), atol=1e-12)


def test_newell_f_origin_zero():
    assert newell_f(0.0, 0.0, 0.0) == 0.0
    assert newell_g(0.0, 0.0, 0.0) == 0.0


def test_demag_tensor_matches_magpylib_far_field():
    """Newell volume-averaged tensor agrees with magpylib point-matched tensor
    for well-separated cells. Volume averaging vs point sampling differs at
    O((d/r)^2), so the agreement is not exact for adjacent cells; we compare
    on a Frobenius-norm-normalised scale and require < 5 % relative deviation
    on the well-separated-pair sub-block."""
    a = 1.0
    pos = np.array(
        [(i * a, j * a, k * a) for i in range(3) for j in range(3) for k in range(3)],
        dtype=float,
    )

    T_newell = demag_tensor_newell(pos, [a, a, a], magpy.mu_0)

    rotation = R.from_euler("z", 1e-9)
    cells_rotated = []
    for p in pos:
        c = magpy.magnet.Cuboid(dimension=(a, a, a), position=p)
        c.orientation = rotation
        cells_rotated.append(c)
    T_magpy = demag_tensor(cells_rotated)

    # Far-field mask: drop self and first-neighbour pairs.
    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    far = dists > 1.5 * a

    # Per-pair Frobenius norms; ratio of (T_newell - T_magpy) over T_magpy.
    diff_block = T_newell - T_magpy  # (3, n, n, 3)
    block_diff_norm = np.sqrt((diff_block**2).sum(axis=(0, 3)))  # (n, n)
    block_norm = np.sqrt((T_magpy**2).sum(axis=(0, 3)))  # (n, n)
    rel = block_diff_norm[far] / np.maximum(block_norm[far], 1e-30)
    assert rel.max() < 5e-2, f"max far-field block-Frobenius rel diff = {rel.max():.3g}"


def test_demag_tensor_via_apply_demag_matches_existing():
    """Drive `apply_demag` end-to-end with the Newell path and verify the
    polarisations agree with the existing magpylib path within tolerance."""
    cube = magpy.magnet.Cuboid(dimension=(1.0, 1.0, 1.0), polarization=(0.0, 0.0, 1.0))
    cube.susceptibility = 0.5
    coll = magpy.Collection(mesh_Cuboid(cube, target_elems=27))

    # Path 1: default -- triggers Newell since orientations are identity.
    coll1 = apply_demag(coll)
    pol1 = np.array([s.polarization for s in coll1.sources_all])

    # Path 2: rotate everything by tiny angle -> falls back to magpylib path.
    coll_rot = coll.copy()
    rot = R.from_euler("z", 1e-9)
    for s in coll_rot.sources_all:
        s.orientation = rot
    coll2 = apply_demag(coll_rot)
    pol2 = np.array([s.polarization for s in coll2.sources_all])
    # rotate pol2 back to compare in cube frame.
    pol2 = rot.inv().apply(pol2)

    np.testing.assert_allclose(pol1, pol2, atol=2e-3)
