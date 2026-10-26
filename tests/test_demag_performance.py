"""Accuracy and (optional) timing tests for the iterative + FFT solver paths.

The central invariant: ``solver="direct"`` and ``solver="iterative"`` share
the same interaction model, so their polarizations must agree to solver
tolerance for *any* input — mixed bodies, rotations, anisotropy included.
"""

from __future__ import annotations

import copy
import itertools
import time

import magpylib as magpy
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R

from magpylib_material_response import demag
from magpylib_material_response.demag import (
    _assemble_T_dense,
    _check_interaction,
    _dedup_blocks,
    _grid_pair_table,
    _pair_block,
    _reciprocity_factor,
    _row_sparsify,
    apply_demag,
    demag_tensor,
    get_susceptibilities,
)
from magpylib_material_response.demag_fft import (
    analyze_collection,
    analyze_structure,
    build_fft_kernel,
    demag_fft_matvec,
    detect_uniform_grid,
)
from magpylib_material_response.meshing import mesh_Cuboid, voxelize
from magpylib_material_response.newell import demag_block_general, demag_tensor_newell


def _build_uniform_collection(target_elems, susceptibility=0.5, polarization=(0, 0, 1)):
    cube = magpy.magnet.Cuboid(dimension=(1.0, 1.0, 1.0), polarization=polarization)
    cube.susceptibility = susceptibility
    return magpy.Collection(mesh_Cuboid(cube, target_elems=target_elems))


def _pols(coll):
    return np.array([s.polarization for s in coll.sources_all])


def _interactions(coll):
    """Every rule that applies to these cells: always point, Newell when all
    cells are parallel cuboids."""
    try:
        _check_interaction("newell", coll.sources_all)
    except ValueError:
        return ("point",)
    return ("point", "newell")


def _assert_solvers_agree(coll, susceptibility=None, atol=1e-6, **iter_kwargs):
    for interaction in _interactions(coll):
        coll1 = apply_demag(
            coll,
            susceptibility=susceptibility,
            solver="direct",
            interaction=interaction,
        )
        coll2 = apply_demag(
            coll,
            susceptibility=susceptibility,
            solver="iterative",
            solver_tol=1e-10,
            interaction=interaction,
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


@pytest.mark.parametrize("interaction", ["point", "newell"])
def test_rotated_tensor_is_rotation_of_axial(interaction):
    """demag_tensor on uniformly-rotated cells = R @ T_axial @ R^T to machine
    precision, under either rule."""
    # Build axis-aligned collection, get local-frame positions and T_axial.
    coll_axial = _build_uniform_collection(27)
    srcs_axial = coll_axial.sources_all
    pos_axial = np.array([s.position for s in srcs_axial])

    T_axial = demag_tensor(srcs_axial, interaction=interaction)  # axis-aligned

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

    T_rotated = demag_tensor(srcs_rotated, interaction=interaction)

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

    kernel_fft = build_fft_kernel(info["shape"], info["spacing"], info["dim"], "newell")
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
    """Anisotropic susceptibility on a *rotated* grid: each cell's
    susceptibility acts along its own axes (#59), so S has a 3x3 block per
    cell, and the iterative solve must apply those blocks as the direct one
    does."""
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


def test_gmres_nonconvergence_raises():
    """A partially-converged iterative result must never be returned silently."""
    coll = _build_uniform_collection(27, susceptibility=1000.0)
    with pytest.raises(RuntimeError, match="GMRES did not converge"):
        apply_demag(coll, solver="iterative", solver_tol=1e-14, max_iter=1)


def test_invalid_solver_kwarg():
    coll = _build_uniform_collection(8)
    with pytest.raises(ValueError, match="solver"):
        apply_demag(coll, solver="banana")


def _graded_cells():
    """125 cuboids graded toward every face, as ``mesh_Cuboid`` with grading
    would make them: no uniform grid."""
    w = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    c = np.cumsum(w) - w / 2
    return magpy.Collection(
        [
            magpy.magnet.Cuboid(
                dimension=w[[i, j, k]], position=c[[i, j, k]], polarization=(0, 0, 1)
            )
            for i, j, k in itertools.product(range(5), repeat=3)
        ]
    )


def _spheres(n_side):
    return magpy.Collection(
        [
            magpy.magnet.Sphere(diameter=0.2, position=p, polarization=(0, 0, 1))
            for p in itertools.product(range(n_side), repeat=3)
        ]
    )


def _small_bodies(k):
    """``k`` meshed cubes of 27 cells, apart: each a grid of its own."""
    cubes = []
    for i in range(k):
        cube = magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=(2 * i, 0, 0)
        )
        cubes.append(mesh_Cuboid(cube, 27))
    return magpy.Collection(*cubes)


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda: _build_uniform_collection(27), "iterative"),
        (
            lambda: magpy.Collection(_build_uniform_collection(27), _spheres(2)),
            "iterative",
        ),
        (lambda: magpy.Collection(_build_uniform_collection(8), _spheres(3)), "direct"),
        (lambda: _small_bodies(20), "direct"),
        (_graded_cells, "direct"),
    ],
    ids=[
        "grid",
        "grid-and-fewer-spheres",
        "grid-and-more-spheres",
        "small-bodies",
        "graded",
    ],
)
def test_auto_solver_follows_the_cell_structure(make, expected):
    """ "auto" goes by the share of the interaction matrix FFT products cover:
    a grid's cells among themselves, large grids of one spacing between them."""
    _, clusters = analyze_collection(make().sources_all)
    assert demag._auto_solver(clusters)[0] == expected


def test_iterative_preconditions_every_cell_type():
    """Spheres beside a soft grid converge about as fast as the grid alone:
    every cell is preconditioned, cells other than cuboids as cubes. Left
    unscaled, they took 219 iterations here, past this test's cap."""
    cube = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 0))
    spheres = [
        magpy.magnet.Sphere(
            diameter=0.3,
            position=np.array(p) * 0.4 + (1.0, -0.4, -0.4),
            polarization=(0, 0, 0),
        )
        for p in itertools.product(range(3), repeat=3)
    ]
    for src in (cube, *spheres):
        src.susceptibility = 1000.0
        src.H_ext = (0, 0, 1)
    coll = magpy.Collection(mesh_Cuboid(cube, 125), *spheres)
    direct = _pols(apply_demag(coll, solver="direct"))
    iterative = _pols(apply_demag(coll, solver="iterative", max_iter=100))
    np.testing.assert_allclose(iterative, direct, atol=1e-4 * np.abs(direct).max())


def test_auto_solver_matches_direct():
    """GMRES at solver_tol / 100 lands within about solver_tol of the direct
    polarization, at high susceptibility too."""
    coll = _build_uniform_collection(512, susceptibility=1000.0)
    direct = _pols(apply_demag(coll, solver="direct"))
    auto = _pols(apply_demag(coll))
    np.testing.assert_allclose(auto, direct, atol=2e-6 * np.abs(direct).max())


def test_auto_solver_falls_back_to_direct_only_when_it_fits():
    """When GMRES does not converge, "auto" solves directly up to 2 GB of dense
    matrix and raises beyond."""
    small = _build_uniform_collection(512, susceptibility=1000.0)
    np.testing.assert_allclose(
        _pols(apply_demag(small, solver_tol=1e-14, max_iter=1)),
        _pols(apply_demag(small, solver="direct")),
        atol=1e-12,
    )
    large = _build_uniform_collection(5832, susceptibility=1000.0)
    with pytest.raises(RuntimeError, match="direct solver would need"):
        apply_demag(large, solver_tol=1e-14, max_iter=1)


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


def test_three_magnet_scalar_susceptibilities_stay_isotropic():
    """Regression: a 3-magnet collection with per-object SCALAR
    susceptibilities was misread as one anisotropic 3-vector."""
    cubes = []
    for i, chi in enumerate([0.1, 0.2, 0.3]):
        c = magpy.magnet.Cuboid(
            polarization=(0, 0, 1), dimension=(1e-3,) * 3, position=(3e-3 * i, 0, 0)
        )
        c.susceptibility = chi
        cubes.append(c)
    per_cell = get_susceptibilities(cubes).reshape(3, 3).T  # (cell, xyz)
    np.testing.assert_allclose(per_cell, [[0.1] * 3, [0.2] * 3, [0.3] * 3])


def test_distant_bodies_interact_as_with_point_matching(getH_tensor):
    """A magnet and an iron cube 2,000 cells apart: the Newell blocks between
    them match magpylib's field, which the exact formulas alone missed by
    orders of magnitude at this distance, and so does the induced
    polarization."""
    a = 0.01

    def bodies():
        magnet = mesh_Cuboid(
            magpy.magnet.Cuboid(dimension=(a, a, a), polarization=(0, 0, 1.2)), 64
        )
        iron = mesh_Cuboid(
            magpy.magnet.Cuboid(
                dimension=(a, a, a), polarization=(0, 0, 0), position=(0, 0, 5.0)
            ),
            64,
        )
        return magpy.Collection(magnet, iron)

    cells = bodies().sources_all
    newell = demag_tensor(cells, interaction="newell")[:, :64, 64:]
    point = getH_tensor(cells)[:, :64, 64:]
    np.testing.assert_allclose(newell, point, rtol=0, atol=1e-2 * np.abs(point).max())

    chi = [0.05] * 64 + [1000.0] * 64
    jz = [
        np.mean([c.polarization[2] for c in res.children[1].sources_all])
        for res in (
            apply_demag(bodies(), susceptibility=chi, interaction=rule)
            for rule in ("newell", "point")
        )
    ]
    np.testing.assert_allclose(jz[0], jz[1], rtol=1e-2)


# ── the two interaction rules ───────────────────────────────────────────────


def _mixed_collection():
    """A grid (FFT), a loose cuboid group of another size, a small group below
    the cluster threshold, a turned cuboid group, a cylinder and a tetrahedron:
    every block path of the point rule."""
    a = 0.01
    grid = mesh_Cuboid(
        magpy.magnet.Cuboid(dimension=(a, a, a), polarization=(0, 0, 1)), 64
    )
    loose = mesh_Cuboid(
        magpy.magnet.Cuboid(
            dimension=(a, a, 2 * a), polarization=(0, 0, 1), position=(0.02, 0, 0)
        ),
        8,
    )
    small = mesh_Cuboid(
        magpy.magnet.Cuboid(
            dimension=(a, a, a), polarization=(1, 0, 0), position=(0, 0.02, 0)
        ),
        2,
    )
    turned = mesh_Cuboid(
        magpy.magnet.Cuboid(
            dimension=(a, a, a), polarization=(0, 1, 0), position=(0, -0.02, 0)
        ),
        27,
    ).rotate_from_angax(30, "z", anchor=(0, -0.02, 0))
    cylinder = magpy.magnet.Cylinder(
        dimension=(a, a), polarization=(0, 0, 1), position=(-0.02, 0, 0)
    )
    tetra = magpy.magnet.Tetrahedron(
        vertices=np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]) * a,
        polarization=(0, 0, 1),
        position=(0, 0, 0.02),
    )
    return magpy.Collection(grid, loose, small, turned, cylinder, tetra)


def test_default_is_point_matching(getH_tensor):
    """The default rule reproduces magpylib's point-matched field, through
    every fast path, up to the far-field expansion (here none: all close)."""
    cells = _mixed_collection().sources_all
    T = demag_tensor(cells) * magpy.mu_0
    np.testing.assert_allclose(T, getH_tensor(cells) * magpy.mu_0, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (_mixed_collection, "needs every cell to be a Cuboid"),
        (
            lambda: _build_uniform_collection(27).rotate_from_angax(
                1e-6, "x", anchor=None
            ),
            None,
        ),
        (
            lambda: magpy.Collection(
                _build_uniform_collection(8),
                _build_uniform_collection(8).rotate_from_angax(10, "x"),
            ),
            "turned against each other",
        ),
    ],
    ids=["cylinder-and-tetrahedron", "all-turned-alike", "turned-bodies"],
)
def test_newell_refuses_cells_it_cannot_cover(build, match):
    """The Newell rule applies to every pair or not at all: it raises rather
    than mixing rules. Cells all turned alike are parallel, so they pass."""
    coll = build()
    if match is None:
        apply_demag(coll, susceptibility=0.5, interaction="newell")
        return
    with pytest.raises(ValueError, match=match):
        apply_demag(coll, susceptibility=0.5, interaction="newell")


def test_unknown_interaction_is_refused():
    with pytest.raises(ValueError, match="interaction must be one of"):
        apply_demag(_build_uniform_collection(8), susceptibility=0.5, interaction="avg")


@pytest.mark.parametrize(
    ("dimension", "divisions"),
    [
        ((0.01, 0.01, 0.01), [(1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)]),
        ((0.02, 0.02, 0.001), [(5, 5, 1), (10, 10, 2), (20, 20, 4)]),
    ],
    ids=["cube", "thin-plate"],
)
def test_newell_moment_rises_under_refinement(dimension, divisions):
    """Galerkin: with the Newell rule, a soft body's induced moment in a uniform
    field rises with every nested refinement, towards the converged value from
    below. (The point rule overshoots on the thin plate.)"""
    moments = []
    for div in divisions:
        body = magpy.magnet.Cuboid(dimension=dimension, polarization=(0, 0, 0))
        body.H_ext = (0.01, 0, 0)
        res = apply_demag(
            mesh_Cuboid(body, div), susceptibility=1000.0, interaction="newell"
        )
        moments.append(
            sum(np.prod(c.dimension) * c.polarization[0] for c in res.sources_all)
        )
    assert np.all(np.diff(moments) > 0)


def test_point_rule_is_smooth_when_a_body_turns():
    """A touching magnet and iron block: turning the iron by a millionth of a
    degree changes the result by a matching amount, not by a jump."""
    a = 0.01

    def iron_jz(angle):
        magnet = mesh_Cuboid(
            magpy.magnet.Cuboid(dimension=(a, a, a), polarization=(0, 0, 1.2)), 27
        )
        iron = magpy.magnet.Cuboid(
            dimension=(a, a, a), polarization=(0, 0, 0), position=(0, 0, a)
        )
        iron = mesh_Cuboid(iron.rotate_from_angax(angle, "x"), 27)
        res = apply_demag(
            magpy.Collection(magnet, iron), susceptibility=[0.05] * 27 + [1000.0] * 27
        )
        return np.mean(
            [
                c.orientation.apply(c.polarization)[2]
                for c in res.children[1].sources_all
            ]
        )

    assert abs(iron_jz(1e-6) / iron_jz(0.0) - 1) < 1e-10


# ── touching bodies, large blocks, deduplication ────────────────────────────
def _u_core_cells():
    """A yoke and two legs of identical cells, touching: one connected component
    that is not a box, but three bodies that are."""
    yoke = magpy.magnet.Cuboid(dimension=(5.0, 1.0, 1.0), position=(0, 0, 0))
    legs = [
        magpy.magnet.Cuboid(dimension=(1.0, 1.0, 3.0), position=(x, 0, 2.0))
        for x in (-2.0, 2.0)
    ]
    core = magpy.Collection(
        mesh_Cuboid(yoke, (10, 2, 2)), *(mesh_Cuboid(leg, (2, 2, 6)) for leg in legs)
    )
    core.susceptibility = 1000.0
    return core


def test_touching_bodies_get_a_grid_each():
    core = _u_core_cells()
    _, clusters = analyze_collection(core.sources_all)
    assert sorted(c["kind"] for c in clusters) == ["grid", "grid", "grid"]
    assert sorted(len(c["indices"]) for c in clusters) == [24, 24, 40]
    _assert_solvers_agree(core)


@pytest.mark.parametrize(
    ("row_budget", "min_drop", "sparse"),
    [(1e-12, 0.6, False), (1e-2, None, False), (3e-2, None, True)],
    ids=[
        "keeps nearly all: dense",
        "drops 82 %, least next to the magnet: dense",
        "drops 96 %: sparse",
    ],
)
def test_large_block_is_sparsified_only_when_it_pays(
    monkeypatch, row_budget, min_drop, sparse
):
    # Every block may go sparse, in chunks of 32 plate cells. In the second
    # case the first chunk (an edge) would show 98 % droppable, and every
    # 32nd cell (the bottom layer) 99 %.
    monkeypatch.setattr(demag, "DENSE_BLOCK_MAX_ENTRIES", 1)
    monkeypatch.setattr(demag, "KERNEL_PAIRS_PER_CALL", 32 * 27)
    plate = magpy.magnet.Cuboid(dimension=(16, 16, 8), polarization=(0, 0, 1))
    magnet = magpy.magnet.Cuboid(
        dimension=(4, 4, 4), position=(0, 0, 6.5), polarization=(0, 0, 1)
    )
    coll = magpy.Collection(mesh_Cuboid(plate, 2048), mesh_Cuboid(magnet, 27))
    positions, clusters = analyze_collection(coll.sources_all)
    cl_plate, cl_magnet = sorted(clusters, key=lambda c: -len(c["indices"]))
    args = (cl_plate, cl_magnet, coll.sources_all, positions, "point")
    kwargs = {} if min_drop is None else {"min_drop": min_drop}
    block, full = _pair_block(*args, row_budget, **kwargs), _pair_block(*args)
    assert sp.issparse(block) is sparse
    kept = block.toarray() if sparse else block
    np.testing.assert_array_equal(kept[kept != 0], full[kept != 0])
    assert np.abs(full - kept).sum(axis=1).max() <= row_budget


def test_sparsifying_keeps_nan():
    n_kept, _, vals = _row_sparsify(np.array([[0.1, 0.2, np.nan, 5.0]]), 100)
    assert n_kept[0] == 4
    assert np.isnan(vals).any()


def test_dedup_blocks_matches_direct_evaluation():
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(*[np.arange(4.0)] * 3, indexing="ij"), -1).reshape(
        -1, 3
    )
    pos = np.vstack([grid, rng.uniform(0, 3, (10, 3))])
    disp = pos[:, None, :] - pos[None, :, :]  # many repeated displacements, some unique
    outer = lambda d: d[..., :, None] * d[..., None, :]
    np.testing.assert_allclose(
        _dedup_blocks(disp, 1.0, outer), outer(disp), rtol=0, atol=1e-15
    )


def test_grid_pair_block_matches_plain_dedup():
    """Two grids of one spacing, offset by a fraction of it: the sort-free path
    gives the same block as the general one, for both rules."""
    a = magpy.magnet.Cuboid(dimension=(2.0, 2.0, 2.0), position=(0, 0, 0))
    b = magpy.magnet.Cuboid(dimension=(1.5, 3.0, 2.0), position=(2.3, 0.7, -0.4))
    coll = magpy.Collection(mesh_Cuboid(a, (4, 4, 4)), mesh_Cuboid(b, (3, 6, 4)))
    positions, clusters = analyze_collection(coll.sources_all)
    assert [c["kind"] for c in clusters] == ["grid", "grid"]
    plain = [copy.deepcopy(c) | {"kind": "loose"} for c in clusters]
    for interaction in ("point", "newell"):
        for ca, cb, pa, pb in ((0, 1, 0, 1), (1, 1, 1, 1)):
            fast = _pair_block(
                clusters[ca], clusters[cb], coll.sources_all, positions, interaction
            )
            ref = _pair_block(
                plain[pa], plain[pb], coll.sources_all, positions, interaction
            )
            # the unique displacements are rebuilt from origin and spacing, a last-bit
            # difference the kernel turns into ~1e-12
            np.testing.assert_allclose(fast, ref, rtol=0, atol=1e-9 * np.abs(ref).max())


def _flattened(*colls):
    cells = [c for coll in colls for c in coll.sources_all]
    return magpy.Collection(*cells, override_parent=True)


def _voxel_ball(position=(0, 0, 0), target=400):
    ball = voxelize(magpy.magnet.Sphere(diameter=1.0, polarization=(0, 0, 0)), target)
    ball.position = position
    return ball


def _touching_voxel_balls():
    """Two balls whose cells sit 0.37 cell apart: two lattices, one component."""
    a = _voxel_ball(target=200)
    h = a.sources_all[0].dimension[0]
    reach = max(c.position[0] for c in a.sources_all) + h / 2
    return magpy.Collection(a, _voxel_ball((2 * reach + 0.37 * h, 0, 0), 200))


def _l_shape():
    """Two one-cell arms of 20 and 19 cells: one lattice, 10 % of its box."""
    arm_x = magpy.magnet.Cuboid(dimension=(20, 1, 1), polarization=(0, 0, 0))
    arm_y = magpy.magnet.Cuboid(
        dimension=(1, 19, 1), position=(-9.5, 10, 0), polarization=(0, 0, 0)
    )
    return _flattened(mesh_Cuboid(arm_x, (20, 1, 1)), mesh_Cuboid(arm_y, (1, 19, 1)))


def _gapped_far_away():
    """Two meshes 0.3 cell apart, 2e4 cells from the origin: off any lattice."""
    a, b = (
        magpy.magnet.Cuboid(
            dimension=(10, 4, 4), position=(2e4 + x, 0, 0), polarization=(0, 0, 0)
        )
        for x in (0, 10.3)
    )
    return _flattened(mesh_Cuboid(a, (10, 4, 4)), mesh_Cuboid(b, (10, 4, 4)))


def _T_from_magpylib(cells):
    """T from magpylib's fields of unit polarizations, component-major like
    _assemble_T_dense, without the structure analysis."""
    n = len(cells)
    pos = np.array([c.position for c in cells])
    T = np.empty((3 * n, 3 * n))
    for b in range(3):
        units = [
            magpy.magnet.Cuboid(
                dimension=c.dimension, position=c.position, polarization=np.eye(3)[b]
            )
            for c in cells
        ]
        H = magpy.getH(units, pos) * magpy.mu_0  # (source j, observer i, a)
        T[:, b * n : (b + 1) * n] = H.transpose(2, 1, 0).reshape(3 * n, n)
    return T


@pytest.mark.parametrize(
    ("make", "kinds"),
    [
        (lambda: _flattened(_u_core_cells()), ["grid"]),
        (_voxel_ball, ["grid"]),
        (_touching_voxel_balls, ["grid", "grid"]),
        (_l_shape, ["grid", "grid"]),
        (_gapped_far_away, ["loose"]),
    ],
    ids=[
        "flattened touching bodies: one grid",
        "voxelized ball: one grid",
        "touching voxelized balls: a grid each",
        "L-shape below an eighth of its box: full boxes",
        "off the lattice far from the origin: loose",
    ],
)
def test_lattice_with_holes_is_one_grid(monkeypatch, make, kinds):
    """Cells on one lattice that fill at least an eighth of their box run as
    one grid, zero in the holes; the dense T matches magpylib's own fields, and
    both solvers agree, cross-grid FFT included."""
    monkeypatch.setattr(demag, "FFT_CROSS_MIN_ENTRIES", 0)  # the test bodies are small
    coll = make()
    coll.susceptibility, coll.H_ext = 1000.0, (0.002, 0, 0.01)
    cells = coll.sources_all
    positions, clusters = analyze_collection(cells)
    assert sorted(c["kind"] for c in clusters) == kinds
    T = _assemble_T_dense(cells, positions, clusters, "point")
    ref = _T_from_magpylib(cells)
    np.testing.assert_allclose(T, ref, rtol=0, atol=1e-9 * np.abs(ref).max())
    _assert_solvers_agree(coll)


def test_cross_grid_blocks_use_fft(monkeypatch):
    monkeypatch.setattr(demag, "FFT_CROSS_MIN_ENTRIES", 0)  # the test bodies are small
    """Two grids of one spacing: the iterative solver's cross blocks are FFT
    convolutions that agree with the direct solver."""
    core = _u_core_cells()
    cells = core.sources_all
    positions, clusters = analyze_collection(cells)
    chi = np.zeros((len(cells), 3, 3))
    ops = demag._build_operator(cells, positions, clusters, chi, 1e-6, None, "point")
    kinds = sorted(op["kind"] for op in ops)
    assert kinds == ["fft"] * 3 + ["fft_cross"] * 6
    _assert_solvers_agree(core)


def test_tiny_grids_of_different_spacing_are_not_one_spacing():
    """Relative tolerances only: at nanometre scale an absolute 1e-8 would call
    different spacings, and different cell shapes, equal."""
    a = magpy.magnet.Cuboid(dimension=(3e-9,) * 3, position=(0, 0, 0))
    b = magpy.magnet.Cuboid(dimension=(5e-9,) * 3, position=(1e-8, 0, 0))
    coll = magpy.Collection(mesh_Cuboid(a, (3, 3, 3)), mesh_Cuboid(b, (3, 3, 3)))
    _, clusters = analyze_collection(coll.sources_all)
    assert [c["kind"] for c in clusters] == ["grid", "grid"]
    assert _grid_pair_table(clusters[0], clusters[1], "point") is None
    assert _reciprocity_factor(clusters[0], clusters[1], "point") is None


def test_dense_system_is_assembled_in_fortran_order():
    """What LAPACK factors in place; a C-ordered matrix is copied first,
    twice the matrix in memory."""
    core = _u_core_cells()
    cells = core.sources_all
    positions, clusters = analyze_collection(cells)
    T = _assemble_T_dense(cells, positions, clusters, "point")
    assert T.flags.f_contiguous
    assert not T.flags.c_contiguous
    n = len(cells)
    T2 = demag_tensor(cells).swapaxes(2, 3).reshape((3 * n, 3 * n)).T * magpy.mu_0
    np.testing.assert_allclose(T, T2, rtol=0, atol=1e-12 * np.abs(T2).max())
