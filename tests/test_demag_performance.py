"""Accuracy and (optional) timing tests for the iterative + FFT solver paths."""

from __future__ import annotations

import copy
import time

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.demag import apply_demag, demag_tensor
from magpylib_material_response.demag_fft import (
    build_fft_kernel,
    demag_fft_matvec,
    detect_uniform_grid,
)
from magpylib_material_response.meshing import mesh_Cuboid
from magpylib_material_response.newell import demag_tensor_newell


def _build_uniform_collection(target_elems, susceptibility=0.5, polarization=(0, 0, 1)):
    cube = magpy.magnet.Cuboid(dimension=(1.0, 1.0, 1.0), polarization=polarization)
    cube.susceptibility = susceptibility
    return magpy.Collection(mesh_Cuboid(cube, target_elems=target_elems))


def test_detect_uniform_grid_basic():
    coll = _build_uniform_collection(27)
    srcs = coll.sources_all
    pos = np.array([s.position for s in srcs])
    dim = np.array([s.dimension for s in srcs])
    rot = R.from_quat([s.orientation.as_quat() for s in srcs])
    info = detect_uniform_grid(pos, dim, rot)
    assert info is not None
    assert info["shape"] == (3, 3, 3)
    np.testing.assert_allclose(info["cell"], (1 / 3, 1 / 3, 1 / 3))


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


def _build_rotated_cuboid_collection(angle_deg, axis, target_elems, susceptibility=0.5):
    """Single cuboid uniformly rotated by ``angle_deg`` around ``axis``, then meshed."""
    cube = magpy.magnet.Cuboid(dimension=(1.0, 1.0, 1.0), polarization=(0, 0, 1))
    cube.rotate_from_angax(angle_deg, axis)
    cube.susceptibility = susceptibility
    return magpy.Collection(mesh_Cuboid(cube, target_elems=target_elems))


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


def test_iterative_fft_rotated_cuboid_matches_direct():
    """Iterative + FFT path on a uniformly-rotated cuboid matches direct solver."""
    coll = _build_rotated_cuboid_collection(45, "y", 64)
    coll1 = apply_demag(coll, solver="direct")
    coll2 = apply_demag(coll, solver="iterative", solver_tol=1e-10)
    pol1 = np.array([s.polarization for s in coll1.sources_all])
    pol2 = np.array([s.polarization for s in coll2.sources_all])
    np.testing.assert_allclose(pol1, pol2, atol=1e-6)


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

    kernel_fft = build_fft_kernel(info["shape"], info["cell"])
    rng = np.random.default_rng(1)
    M_orig = rng.normal(size=(len(srcs), 3))

    # Reorder original->grid layout
    M_grid_flat = M_orig[info["order"]]
    M_grid = M_grid_flat.reshape((Nx, Ny, Nz, 3))
    H_grid = demag_fft_matvec(M_grid, kernel_fft, info["shape"], magpy.mu_0)
    H_grid_flat = H_grid.reshape((Nx * Ny * Nz, 3))
    H_orig = np.empty_like(M_orig)
    H_orig[info["order"]] = H_grid_flat

    # Reference: dense T_newell @ v (post-mu_0).
    T = demag_tensor_newell(pos, info["cell"], magpy.mu_0) * magpy.mu_0
    # T axes: (k=3, i=src, j=obs, m=3); we need H_obs_m = sum_{k,i} T[k,i,j,m] * M_orig[i, k]
    H_ref = np.einsum("kijm,ik->jm", T, M_orig)

    np.testing.assert_allclose(H_orig, H_ref, atol=1e-10, rtol=1e-8)


def test_iterative_solver_matches_direct_small():
    """Iterative GMRES (no FFT path) gives same polarisation as direct solve."""
    coll = _build_uniform_collection(27)
    coll1 = apply_demag(coll, solver="direct")
    coll2 = apply_demag(coll, solver="iterative", solver_tol=1e-10)
    pol1 = np.array([s.polarization for s in coll1.sources_all])
    pol2 = np.array([s.polarization for s in coll2.sources_all])
    np.testing.assert_allclose(pol1, pol2, atol=1e-7)


def test_iterative_fft_solver_matches_direct():
    """Iterative + FFT path matches direct solver on a uniform grid."""
    coll = _build_uniform_collection(125)
    coll1 = apply_demag(coll, solver="direct")
    coll2 = apply_demag(coll, solver="iterative", solver_tol=1e-10)
    pol1 = np.array([s.polarization for s in coll1.sources_all])
    pol2 = np.array([s.polarization for s in coll2.sources_all])
    np.testing.assert_allclose(pol1, pol2, atol=1e-6)


def test_iterative_anisotropic_susceptibility():
    """Iterative solver supports anisotropic susceptibility input."""
    coll = _build_uniform_collection(27, susceptibility=None)
    sus = (0.3, 0.2, 0.1)
    coll1 = apply_demag(coll, susceptibility=sus, solver="direct")
    coll2 = apply_demag(coll, susceptibility=sus, solver="iterative", solver_tol=1e-10)
    pol1 = np.array([s.polarization for s in coll1.sources_all])
    pol2 = np.array([s.polarization for s in coll2.sources_all])
    np.testing.assert_allclose(pol1, pol2, atol=1e-6)


def test_invalid_solver_kwarg():
    coll = _build_uniform_collection(8)
    with pytest.raises(ValueError, match="solver"):
        apply_demag(coll, solver="banana")


@pytest.mark.slow
def test_fft_faster_than_direct():
    """At n=512, the FFT-iterative path should outperform the direct solver."""
    coll = _build_uniform_collection(512)

    t0 = time.perf_counter()
    apply_demag(coll, solver="direct")
    t_direct = time.perf_counter() - t0

    t0 = time.perf_counter()
    apply_demag(coll, solver="iterative", solver_tol=1e-6)
    t_iter = time.perf_counter() - t0

    print(f"\nn=512: direct={t_direct:.3f}s  iterative+FFT={t_iter:.3f}s")
    assert t_iter < t_direct, (
        f"FFT path ({t_iter:.3f}s) should be faster than direct ({t_direct:.3f}s)"
    )
