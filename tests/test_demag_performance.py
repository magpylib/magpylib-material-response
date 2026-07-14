"""Accuracy and (optional) timing tests for the iterative + FFT solver paths.

The central invariant: ``solver="direct"`` and ``solver="iterative"`` share
the same interaction model, so their polarizations must agree to solver
tolerance for *any* input — mixed bodies, rotations, anisotropy included.
"""

from __future__ import annotations

import copy
import time

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.demag import apply_demag, demag_tensor
from magpylib_material_response.demag_fft import (
    analyze_collection,
    analyze_structure,
    build_fft_kernel,
    demag_fft_matvec,
    detect_uniform_grid,
)
from magpylib_material_response.meshing import mesh_Cuboid
from magpylib_material_response.newell import demag_block_general, demag_tensor_newell


def _build_uniform_collection(target_elems, susceptibility=0.5, polarization=(0, 0, 1)):
    cube = magpy.magnet.Cuboid(dimension=(1.0, 1.0, 1.0), polarization=polarization)
    cube.susceptibility = susceptibility
    return magpy.Collection(mesh_Cuboid(cube, target_elems=target_elems))


def _pols(coll):
    return np.array([s.polarization for s in coll.sources_all])


def _assert_solvers_agree(coll, susceptibility=None, atol=1e-6, **iter_kwargs):
    coll1 = apply_demag(coll, susceptibility=susceptibility, solver="direct")
    coll2 = apply_demag(
        coll,
        susceptibility=susceptibility,
        solver="iterative",
        solver_tol=1e-10,
        **iter_kwargs,
    )
    np.testing.assert_allclose(_pols(coll1), _pols(coll2), atol=atol)


# ── structure detection ─────────────────────────────────────────────────────


def test_detect_uniform_grid_basic():
    coll = _build_uniform_collection(27)
    srcs = coll.sources_all
    pos = np.array([s.position for s in srcs])
    dim = np.array([s.dimension for s in srcs])
    rot = R.from_quat([s.orientation.as_quat() for s in srcs])
    info = detect_uniform_grid(pos, dim, rot)
    assert info is not None
    assert info["shape"] == (3, 3, 3)
    np.testing.assert_allclose(info["spacing"], (1 / 3, 1 / 3, 1 / 3))
    np.testing.assert_allclose(info["dim"], (1 / 3, 1 / 3, 1 / 3))


def test_detect_uniform_grid_negative():
    """Non-uniform spacing or mixed (non-common) rotations should not be detected."""
    # Random positions
    pos = np.random.default_rng(0).normal(size=(8, 3))
    dim = np.tile([1.0, 1.0, 1.0], (8, 1))
    rot = R.identity(8)
    assert detect_uniform_grid(pos, dim, rot) is None

    # Mixed (non-uniform) rotations — must still return None.
    coll = _build_uniform_collection(27)
    srcs = coll.sources_all
    pos = np.array([s.position for s in srcs])
    dim = np.array([s.dimension for s in srcs])
    # Different angle per cell → not a common rotation.
    angles = np.linspace(0.0, 0.5, len(srcs))
    mixed_rot = R.from_quat(np.array([R.from_euler("z", a).as_quat() for a in angles]))
    assert detect_uniform_grid(pos, dim, mixed_rot) is None


def test_detect_uniform_grid_common_rotation():
    """A uniformly-rotated grid is detected and r_common is set correctly."""
    coll = _build_uniform_collection(27)
    srcs = coll.sources_all
    pos = np.array([s.position for s in srcs])
    dim = np.array([s.dimension for s in srcs])

    q_rot = R.from_euler("z", 0.5).as_quat()
    rot = R.from_quat(np.tile(q_rot, (len(srcs), 1)))
    # Rotate positions to the global frame as mesh_Cuboid would produce.
    pos_rotated = rot[0].apply(pos)
    info = detect_uniform_grid(pos_rotated, dim, rot)
    assert info is not None, "Common-rotation grid should be detected"
    assert info["shape"] == (3, 3, 3)
    np.testing.assert_allclose(info["r_common"].as_quat(), q_rot, atol=1e-9)


def test_detect_uniform_grid_gapped():
    """Grid spacing may exceed the cell size (non-touching cells)."""
    dim = (1.0, 1.0, 1.0)
    pos = np.array(
        [
            (2.0 * i, 2.0 * j, 2.0 * k)
            for i in range(3)
            for j in range(3)
            for k in range(2)
        ]
    )
    rot = R.identity(len(pos))
    info = detect_uniform_grid(pos, np.tile(dim, (len(pos), 1)), rot)
    assert info is not None
    assert info["shape"] == (3, 3, 2)
    np.testing.assert_allclose(info["spacing"], (2.0, 2.0, 2.0))
    np.testing.assert_allclose(info["dim"], dim)


def test_analyze_structure_full_coverage():
    """Every cell lands in exactly one cluster, whatever the input mix."""
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    mesh = mesh_Cuboid(cube, 27)
    lone = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(2, 2, 2), position=(3, 0, 0)
    )
    cells = [*mesh.sources_all, lone]
    pos = np.array([s.position for s in cells])
    dims = np.array([s.dimension for s in cells])
    rots = R.from_quat([s.orientation.as_quat() for s in cells])
    clusters = analyze_structure(pos, dims, rots, np.ones(len(cells), dtype=bool))
    covered = np.sort(np.concatenate([c["indices"] for c in clusters]))
    np.testing.assert_array_equal(covered, np.arange(len(cells)))
    kinds = sorted(c["kind"] for c in clusters)
    assert kinds == ["generic", "grid"]


def test_analyze_collection_two_identical_meshes():
    """The collection-level wrapper detects two separate meshed bodies with
    identical cells as two grid clusters."""
    c1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    c2 = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1, 1, 1), position=(2.5, 0.3, 0)
    )
    cells = [*mesh_Cuboid(c1, 27).sources_all, *mesh_Cuboid(c2, 27).sources_all]
    positions, clusters = analyze_collection(cells)
    assert positions.shape == (len(cells), 3)
    assert [c["kind"] for c in clusters] == ["grid", "grid"]


# ── tensor-level checks ─────────────────────────────────────────────────────


def test_rotated_newell_tensor_is_rotation_of_axial():
    """demag_tensor on uniformly-rotated cells = R @ T_axial @ R^T to machine precision."""
    # Build axis-aligned collection, get local-frame positions and T_axial.
    coll_axial = _build_uniform_collection(27)
    srcs_axial = coll_axial.sources_all
    pos_axial = np.array([s.position for s in srcs_axial])

    T_axial = demag_tensor(srcs_axial)  # Newell, axis-aligned

    # Apply a 30° rotation around z to every cell.
    angle = 30.0
    r_rot = R.from_euler("z", np.deg2rad(angle))
    R_mat = r_rot.as_matrix()
    pos_rotated = r_rot.apply(pos_axial)

    # Build rotated collection manually (same dims, same susceptibility, rotated positions/orientations).
    srcs_rotated = copy.deepcopy(srcs_axial)
    for s, p in zip(srcs_rotated, pos_rotated, strict=True):
        s.position = p
        s.orientation = r_rot * s.orientation  # was identity, now r_rot

    T_rotated = demag_tensor(srcs_rotated)  # Newell with rotation

    # Expected: T_global = R @ T_axial @ R^T
    # T_global[k,i,j,m] = Σ_{a,b} R[m,a]*T_axial[b,i,j,a]*R[k,b]
    T_expected = np.einsum("ma,bija,kb->kijm", R_mat, T_axial, R_mat)

    np.testing.assert_allclose(T_rotated, T_expected, rtol=1e-12, atol=1e-6)


def test_fft_matvec_matches_dense_newell():
    """FFT matvec ``T @ M`` matches the dense Newell tensor matvec."""
    coll = _build_uniform_collection(125)
    srcs = coll.sources_all
    pos = np.array([s.position for s in srcs])
    dim = np.array([s.dimension for s in srcs])
    rot = R.from_quat([s.orientation.as_quat() for s in srcs])
    info = detect_uniform_grid(pos, dim, rot)
    assert info is not None
    Nx, Ny, Nz = info["shape"]

    kernel_fft = build_fft_kernel(info["shape"], info["spacing"], info["dim"])
    rng = np.random.default_rng(1)
    M_orig = rng.normal(size=(len(srcs), 3))

    # Reorder original->grid layout
    M_grid_flat = M_orig[info["order"]]
    M_grid = M_grid_flat.reshape((Nx, Ny, Nz, 3))
    H_grid = demag_fft_matvec(M_grid, kernel_fft, info["shape"])
    H_grid_flat = H_grid.reshape((Nx * Ny * Nz, 3))
    H_orig = np.empty_like(M_orig)
    H_orig[info["order"]] = H_grid_flat

    # Reference: dense T_newell @ v (post-mu_0).
    T = demag_tensor_newell(pos, info["dim"], magpy.mu_0) * magpy.mu_0
    # T axes: (k=3, i=src, j=obs, m=3); we need H_obs_m = sum_{k,i} T[k,i,j,m] * M_orig[i, k]
    H_ref = np.einsum("kijm,ik->jm", T, M_orig)

    np.testing.assert_allclose(H_orig, H_ref, atol=1e-10, rtol=1e-8)


def test_generalized_newell_reciprocity():
    """V_obs * N(d; src, obs) == V_src * N(d; obs, src).T for different dims."""
    dim_s, dim_o = (1.0, 2.0, 0.5), (0.7, 0.7, 1.3)
    disp = np.array([1.5, -0.8, 2.1])
    N1 = demag_block_general(disp, dim_s, dim_o)
    N2 = demag_block_general(disp, dim_o, dim_s)
    Vs, Vo = np.prod(dim_s), np.prod(dim_o)
    np.testing.assert_allclose(Vo * N1, Vs * N2.T, rtol=1e-10)


# ── solver equivalence (regression tests for the block/FFT paths) ───────────


def test_iterative_solver_matches_direct_small():
    _assert_solvers_agree(_build_uniform_collection(27), atol=1e-7)


def test_iterative_fft_solver_matches_direct():
    _assert_solvers_agree(_build_uniform_collection(125))


def test_iterative_fft_rotated_cuboid_matches_direct():
    cube = magpy.magnet.Cuboid(dimension=(1.0, 1.0, 1.0), polarization=(0, 0, 1))
    cube.rotate_from_angax(45, "y")
    cube.susceptibility = 0.5
    _assert_solvers_agree(magpy.Collection(mesh_Cuboid(cube, 64)))


def test_iterative_anisotropic_susceptibility():
    coll = _build_uniform_collection(27, susceptibility=None)
    _assert_solvers_agree(coll, susceptibility=(0.3, 0.2, 0.1))


def test_iterative_anisotropic_rotated_matches_direct():
    """Anisotropic susceptibility on a *rotated* grid: the solve must stay in
    the global frame where S is diagonal (regression: local-frame solve
    applied world-axis chi to local components)."""
    cube = magpy.magnet.Cuboid(dimension=(1.0, 1.0, 1.0), polarization=(0, 0, 1))
    cube.rotate_from_angax(35, (1, 1, 0))
    coll = magpy.Collection(mesh_Cuboid(cube, 27))
    _assert_solvers_agree(coll, susceptibility=(0.3, 0.1, 0.5))


def test_mixed_grid_plus_standalone_matches_direct():
    """A meshed body plus a standalone magnet: no cell may be silently
    excluded from the interaction system (regression: ungrouped cells
    received zero demag field)."""
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    lone = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(2, 2, 2), position=(2.2, 0, 0)
    )
    coll = magpy.Collection(mesh_Cuboid(cube, 27), lone)
    n = len(coll.sources_all)
    _assert_solvers_agree(coll, susceptibility=[3.0] * n)


def test_two_meshed_bodies_different_cells_match_direct():
    """Two meshed bodies with different cell sizes: self blocks (FFT/Newell)
    and cross blocks (generalized Newell) must match the dense assembly."""
    ca = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    cb = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(2, 1, 1), position=(2, 0, 0)
    )
    coll = magpy.Collection(mesh_Cuboid(ca, 27), mesh_Cuboid(cb, 27))
    n = len(coll.sources_all)
    _assert_solvers_agree(coll, susceptibility=[3.0] * n)


def test_180deg_rotated_group_matches_direct():
    """Quaternion w ~ 0: sign canonicalization must not split one body into
    fragments (regression: w<0 flip instability)."""
    cc = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    mc = mesh_Cuboid(cc, 27)
    mc.rotate_from_angax(180, (0, 0, 1))
    cd = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1.5, 1, 1), position=(2.5, 0, 0)
    )
    coll = magpy.Collection(mc, mesh_Cuboid(cd, 27))
    n = len(coll.sources_all)
    _assert_solvers_agree(coll, susceptibility=[3.0] * n)


def test_mixed_with_sphere_matches_direct():
    """Non-cuboid magnets participate via the point-matched generic cluster."""
    cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    sph = magpy.magnet.Sphere(polarization=(0, 0, 1), diameter=1.0, position=(0, 2, 0))
    coll = magpy.Collection(mesh_Cuboid(cube, 27), sph)
    n = len(coll.sources_all)
    _assert_solvers_agree(coll, susceptibility=[0.5] * n)


def test_h_ext_propagates_through_meshing():
    """susceptibility and H_ext set on a magnet carry over to its mesh cells
    and drive the material response."""
    soft = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(1, 1, 1))
    soft.susceptibility = 3999.0
    soft.H_ext = (0, 0, 0.1)
    mesh = mesh_Cuboid(soft, 27)
    for cell in mesh.sources_all:
        assert cell.susceptibility == 3999.0
        assert cell.H_ext == (0, 0, 0.1)
    out = apply_demag(mesh)
    pol = np.array([s.polarization for s in out.sources_all])
    # chi >> 1 soft cube in 0.1 T: J_mean ~ B_ext / N_eff with N_eff near the
    # uniform-magnetization value 1/3 (exact only for ellipsoids)
    assert 0.25 < pol[:, 2].mean() < 0.45


def test_gmres_nonconvergence_raises():
    """A partially-converged iterative result must never be returned silently."""
    coll = _build_uniform_collection(27, susceptibility=1000.0)
    with pytest.raises(RuntimeError, match="GMRES did not converge"):
        apply_demag(coll, solver="iterative", solver_tol=1e-14, max_iter=1)


def test_invalid_solver_kwarg():
    coll = _build_uniform_collection(8)
    with pytest.raises(ValueError, match="solver"):
        apply_demag(coll, solver="banana")


@pytest.mark.slow
def test_fft_faster_than_direct():
    """At n=3375, the FFT-iterative path clearly outperforms the direct
    solver (at small n the dedup-accelerated dense assembly + LAPACK solve
    is competitive, so the crossover sits in the low thousands)."""
    coll = _build_uniform_collection(3375)

    t0 = time.perf_counter()
    apply_demag(coll, solver="direct")
    t_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    apply_demag(coll, solver="iterative", solver_tol=1e-6)
    t_iter = time.perf_counter() - t0

    print(f"\nn=3375: direct={t_direct:.3f}s  iterative+FFT={t_iter:.3f}s")
    assert t_iter < t_direct, (
        f"FFT path ({t_iter:.3f}s) should be faster than direct ({t_direct:.3f}s)"
    )
