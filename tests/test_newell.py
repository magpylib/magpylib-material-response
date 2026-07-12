"""Unit tests for the Newell analytical demag tensor."""

from __future__ import annotations

import magpylib as magpy
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from magpylib_material_response.demag import apply_demag, demag_tensor
from magpylib_material_response.meshing import mesh_Cuboid
from magpylib_material_response.newell import (
    demag_block,
    demag_block_general,
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

    cells = [magpy.magnet.Cuboid(dimension=(a, a, a), position=p) for p in pos]
    # split=2 forces the legacy point-matched getH evaluation.
    T_magpy = demag_tensor(cells, split=2)

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
    polarisations agree with the legacy point-matching path within tolerance."""
    cube = magpy.magnet.Cuboid(dimension=(1.0, 1.0, 1.0), polarization=(0.0, 0.0, 1.0))
    cube.susceptibility = 0.5
    coll = magpy.Collection(mesh_Cuboid(cube, target_elems=27))

    # Path 1: default — the unified (Newell) assembly.
    coll1 = apply_demag(coll)
    pol1 = np.array([s.polarization for s in coll1.sources_all])

    # Path 2: split=2 forces the legacy point-matched tensor.
    coll2 = apply_demag(coll, split=2)
    pol2 = np.array([s.polarization for s in coll2.sources_all])

    np.testing.assert_allclose(pol1, pol2, atol=2e-3)


def _numeric_volume_avg_N(disp, dim_src, dim_obs, order=10):
    """N_mk = -mu0 * <H_m> over the observer volume for unit polarization e_k,
    integrated with Gauss-Legendre quadrature of magpylib's exact field."""
    x, w = leggauss(order)
    gx = 0.5 * dim_obs[0] * x
    gy = 0.5 * dim_obs[1] * x
    gz = 0.5 * dim_obs[2] * x
    XX, YY, ZZ = np.meshgrid(gx, gy, gz, indexing="ij")
    pts = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3) + np.asarray(disp)
    WW = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel() / 8.0
    N = np.empty((3, 3))
    for k in range(3):
        pol = np.zeros(3)
        pol[k] = 1.0
        src = magpy.magnet.Cuboid(polarization=pol, dimension=dim_src)
        H = src.getH(pts)
        N[:, k] = -magpy.mu_0 * (WW[:, None] * H).sum(axis=0)
    return N


@pytest.mark.parametrize(
    ("dim_src", "dim_obs", "disp", "rtol"),
    [
        # touching cells: quadrature converges slowly at the shared face
        ((1.0, 1.0, 1.0), (2.0, 1.0, 0.5), (1.5, 0.0, 0.0), 1e-3),
        ((1.0, 2.0, 3.0), (0.5, 0.5, 0.5), (1.0, 1.5, 2.0), 1e-6),
        ((2.0, 1.0, 1.0), (1.0, 3.0, 1.0), (-2.0, 1.0, -1.5), 1e-5),
        ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (8.0, -5.0, 3.0), 1e-6),
    ],
)
def test_demag_block_general_matches_numeric(dim_src, dim_obs, disp, rtol):
    """Generalized (different-size) Newell block equals the numerical volume
    average of magpylib's exact cuboid field."""
    Na = demag_block_general(np.array(disp), dim_src, dim_obs)
    Nn = _numeric_volume_avg_N(disp, dim_src, dim_obs)
    scale = np.abs(Nn).max()
    np.testing.assert_allclose(Na, Nn, atol=rtol * scale)


def test_demag_block_general_far_field_dipole():
    """Far apart, the volume-averaged tensor approaches the point dipole with
    moment M * V_src."""
    dim_s, dim_o = (1.0, 1.0, 1.0), (2.0, 1.0, 1.0)
    disp = np.array([50.0, 30.0, -20.0])
    Na = demag_block_general(disp, dim_s, dim_o)
    r = np.linalg.norm(disp)
    rh = disp / r
    D = (np.eye(3) - 3 * np.outer(rh, rh)) / (4 * np.pi * r**3) * np.prod(dim_s)
    np.testing.assert_allclose(Na, D, atol=1e-3 * np.abs(D).max())
