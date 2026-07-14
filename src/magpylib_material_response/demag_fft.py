"""FFT-accelerated demagnetization matvec and cell-structure analysis.

For a regular grid of identical, identically-oriented cuboids, the demag
tensor is translation-invariant: ``T[i, j]`` depends only on the cell-index
displacement ``i - j``. The matvec ``H = T @ M`` therefore reduces to a 3-D
discrete convolution and is computed in :math:`O(n \\log n)` time using FFTs
on the zero-padded ("doubled") grid required for aperiodic boundary
conditions.

:func:`analyze_structure` partitions an arbitrary set of cells into
*clusters* covering **every** cell exactly once:

- ``"grid"``  : identical cuboids, common orientation, on a uniform grid
                (FFT-eligible self-interactions),
- ``"loose"`` : identical cuboids, common orientation, no grid structure
                (analytical Newell self-interactions, dense),
- ``"generic"``: everything else (point-matched ``magpy.getH``).

The kernel is built from the analytical Newell formula
(:mod:`magpylib_material_response.newell`); only the 6 independent
components ``(Nxx, Nyy, Nzz, Nxy, Nxz, Nyz)`` are stored. Grid *spacing* is
detected independently of the cell side lengths, so grids of non-touching
cells are supported.
"""

from __future__ import annotations

import numpy as np
from magpylib.magnet import Cuboid
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.newell import demag_block

__all__ = [
    "analyze_collection",
    "analyze_structure",
    "build_fft_kernel",
    "canonical_quats",
    "demag_fft_matvec",
    "detect_uniform_grid",
]

#: Quaternions are considered equal when all components match up to this
#: absolute tolerance (after sign canonicalization).
QUAT_ATOL = 1e-9

#: A geometry cluster below this size is folded into the point-matched
#: "generic" cluster: with e.g. one cluster per orientation (rotated-cell
#: meshes) the per-cluster bookkeeping would otherwise dominate.
MIN_CLUSTER_CELLS = 8

#: Minimum cells for a detected grid to be worth an FFT kernel; smaller
#: bodies are handled by the dense analytical path.
MIN_GRID_CELLS = 16


def canonical_quats(quats):
    """Return quaternions with a deterministic global sign.

    ``q`` and ``-q`` represent the same rotation. The sign is fixed such that
    the component with the largest magnitude is positive — stable for any
    quaternion (the largest component has magnitude >= 1/2), unlike flipping
    on ``w < 0`` which is unstable for 180-degree rotations where ``w ~ 0``.
    """
    quats = np.asarray(quats, dtype=float)
    lead = np.take_along_axis(
        quats, np.argmax(np.abs(quats), axis=-1, keepdims=True), axis=-1
    )
    return quats * np.where(lead < 0, -1.0, 1.0)


def same_rotation(quats, atol=QUAT_ATOL):
    """True when all quaternions represent one common rotation (up to sign)."""
    qc = canonical_quats(quats)
    return bool(np.allclose(qc, qc[0], atol=atol))


def _cluster_1d(coords, tol):
    """Collapse 1-D coordinates into cluster means; clusters separated > tol."""
    s = np.sort(np.asarray(coords, dtype=float))
    breaks = np.nonzero(np.diff(s) > tol)[0] + 1
    starts = np.concatenate(([0], breaks))
    sums = np.add.reduceat(s, starts)
    counts = np.diff(np.concatenate((starts, [s.size])))
    return sums / counts


def _detect_grid(positions, dim, rtol=1e-6):
    """Detect a uniform Cartesian grid in axis-aligned ``positions``.

    Grid spacing per axis is inferred from the data and may differ from the
    cell side lengths ``dim`` (non-touching cells). Returns ``None`` if the
    points do not fill a complete ``Nx x Ny x Nz`` box exactly once, else a
    dict with keys ``shape``, ``spacing``, ``origin``, ``order`` where
    ``order`` is the permutation such that grid-flat index
    ``ix*Ny*Nz + iy*Nz + iz`` corresponds to ``positions[order]``.
    """
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    if n == 0:
        return None
    tol = rtol * float(np.min(dim))

    centers = [_cluster_1d(positions[:, ax], tol) for ax in range(3)]
    shape = tuple(c.size for c in centers)
    if shape[0] * shape[1] * shape[2] != n:
        return None

    spacing = []
    for ax, c in enumerate(centers):
        if c.size > 1:
            d = np.diff(c)
            if not np.allclose(d, d[0], rtol=rtol, atol=tol):
                return None
            spacing.append(float(d[0]))
        else:
            # Singleton axis: spacing is arbitrary (never enters a nonzero
            # displacement); use the cell size.
            spacing.append(float(dim[ax]))

    idx3 = []
    for ax, c in enumerate(centers):
        ix = np.round((positions[:, ax] - c[0]) / spacing[ax]).astype(np.int64)
        if (ix < 0).any() or (ix >= c.size).any():
            return None
        if not np.allclose(positions[:, ax], c[0] + ix * spacing[ax], atol=tol):
            return None
        idx3.append(ix)

    flat = idx3[0] * (shape[1] * shape[2]) + idx3[1] * shape[2] + idx3[2]
    if np.unique(flat).size != n:
        return None
    order = np.empty(n, dtype=np.int64)
    order[flat] = np.arange(n)

    return {
        "shape": shape,
        "spacing": tuple(spacing),
        "origin": tuple(float(c[0]) for c in centers),
        "order": order,
    }


def detect_uniform_grid(positions, dimensions, rotations, atol=QUAT_ATOL):
    """Detect whether *all* cells form one uniform grid of identical cuboids.

    Convenience wrapper around :func:`_detect_grid` that first checks for
    identical dimensions and a single common rotation. For a non-identity
    common rotation, the grid is detected in the cells' local frame.

    Returns ``None`` on failure, else a dict with keys ``shape``,
    ``spacing``, ``origin``, ``order``, ``dim``, ``r_common``,
    ``is_identity``.
    """
    positions = np.asarray(positions, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    if positions.shape[0] == 0:
        return None

    dim0 = dimensions[0]
    if not np.allclose(dimensions, dim0, rtol=1e-9, atol=0.0):
        return None
    if not same_rotation(rotations.as_quat(), atol=atol):
        return None

    r_common = rotations[0]
    is_identity = bool(
        np.allclose(r_common.as_quat(), [0.0, 0.0, 0.0, 1.0], atol=atol)
        or np.allclose(r_common.as_quat(), [0.0, 0.0, 0.0, -1.0], atol=atol)
    )
    pos_local = positions if is_identity else r_common.inv().apply(positions)

    info = _detect_grid(pos_local, dim0)
    if info is None:
        return None
    info["dim"] = tuple(float(v) for v in dim0)
    info["r_common"] = r_common
    info["is_identity"] = is_identity
    return info


def _split_components(positions):
    """Split point indices into spatially connected components.

    Two cells are connected when their distance is below 1.5x the typical
    nearest-neighbour distance, so separate meshed bodies of identical cell
    geometry are analysed independently.
    """
    n = positions.shape[0]
    if n <= 1:
        return [np.arange(n)]
    tree = cKDTree(positions)
    d_nn = tree.query(positions, k=2)[0][:, 1]
    radius = 1.5 * float(np.median(d_nn))
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    return [np.nonzero(labels == c)[0] for c in range(n_comp)]


def analyze_collection(sources, atol=QUAT_ATOL):
    """Partition magnet objects into structure clusters (see
    :func:`analyze_structure`).

    Convenience wrapper that extracts positions (barycentre when available),
    dimensions, orientations and the cuboid mask from a sequence of magpylib
    magnet objects (e.g. ``collection.sources_all``) and returns the cluster
    list together with the extracted positions.

    Returns
    -------
    positions : ndarray (n, 3)
    clusters : list[dict]  — see :func:`analyze_structure`
    """
    sources = list(sources)
    positions = np.array(
        [getattr(src, "barycenter", src.position) for src in sources], dtype=float
    )
    is_cuboid = np.array([isinstance(s, Cuboid) for s in sources])
    dimensions = np.array(
        [s.dimension if isinstance(s, Cuboid) else (1.0, 1.0, 1.0) for s in sources],
        dtype=float,
    )
    rotations = R.from_quat([s.orientation.as_quat() for s in sources])
    clusters = analyze_structure(positions, dimensions, rotations, is_cuboid, atol)
    return positions, clusters


def analyze_structure(positions, dimensions, rotations, is_cuboid, atol=QUAT_ATOL):
    """Partition cells into structure clusters covering every cell exactly once.

    Parameters
    ----------
    positions : ndarray (n, 3)
        Cell barycentres, world frame.
    dimensions : ndarray (n, 3)
        Cuboid side lengths (rows for non-cuboid cells are ignored).
    rotations : scipy Rotation, length n
        Per-cell orientations.
    is_cuboid : ndarray (n,) of bool
        Which cells are ``magpylib.magnet.Cuboid`` instances.
    atol : float
        Quaternion comparison tolerance.

    Returns
    -------
    clusters : list[dict]
        Every cell index appears in exactly one cluster. Each dict has keys:

        - ``kind``      : ``"grid"`` | ``"loose"`` | ``"generic"``
        - ``indices``   : ndarray(int) — original cell indices
        - ``dim``       : tuple(3) cell side lengths (cuboid kinds only)
        - ``r_common``  : scipy Rotation — common orientation (cuboid kinds)
        - ``is_identity``: bool (cuboid kinds)
        - grid kind adds ``shape``, ``spacing``, ``origin``, ``order``
          (``order`` indexes *within* the cluster).
    """
    positions = np.asarray(positions, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    is_cuboid = np.asarray(is_cuboid, dtype=bool)
    n = positions.shape[0]
    quats = rotations.as_quat()

    clusters: list[dict] = []
    fold_to_generic: list[np.ndarray] = []

    cub_idx = np.nonzero(is_cuboid)[0]
    gen_idx = np.nonzero(~is_cuboid)[0]

    if cub_idx.size:
        # ── Geometry keying: (dims, canonical quat), robust two-stage ──────
        qc = canonical_quats(quats[cub_idx])
        dims_c = dimensions[cub_idx]
        dim_scale = max(float(np.max(np.abs(dims_c))), np.finfo(float).tiny)
        unit_d = 1e-9 * dim_scale
        feats = np.hstack([dims_c / unit_d, qc / max(atol, 1e-12)])
        # Stage 1: exact bucketing on rounded features (may over-split at
        # rounding boundaries).
        keys, inverse = np.unique(np.round(feats), axis=0, return_inverse=True)
        # Stage 2: merge buckets whose representatives are within tolerance
        # (a rounding boundary splits equal values by at most 1 unit).
        n_buckets = keys.shape[0]
        parent = np.arange(n_buckets)

        def _find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        merge_pairs = cKDTree(keys).query_pairs(
            r=2.0 * np.sqrt(keys.shape[1]), output_type="ndarray"
        )
        for i, j in merge_pairs:
            if np.all(np.abs(keys[i] - keys[j]) <= 2.0):
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[rj] = ri
        labels = np.array([_find(b) for b in inverse])

        for lab in np.unique(labels):
            members = cub_idx[labels == lab]  # original indices
            if members.size < MIN_CLUSTER_CELLS:
                fold_to_generic.append(members)
                continue
            dim0 = dimensions[members[0]]
            r_common = rotations[members[0]]
            is_identity = bool(
                np.allclose(r_common.as_quat(), [0, 0, 0, 1], atol=atol)
                or np.allclose(r_common.as_quat(), [0, 0, 0, -1], atol=atol)
            )
            pos_local = (
                positions[members]
                if is_identity
                else r_common.inv().apply(positions[members])
            )
            base = {
                "dim": tuple(float(v) for v in dim0),
                "r_common": r_common,
                "is_identity": is_identity,
            }

            # Try the whole geometry cluster as one grid first (covers a
            # single meshed body and aligned unions of bodies)...
            info = _detect_grid(pos_local, dim0)
            if info is not None and members.size >= MIN_GRID_CELLS:
                clusters.append({"kind": "grid", "indices": members, **base, **info})
                continue

            # ...then per spatially-connected component (separate meshed
            # bodies with identical cell geometry).
            loose: list[np.ndarray] = []
            for comp in _split_components(pos_local):
                info = _detect_grid(pos_local[comp], dim0)
                if info is not None and comp.size >= MIN_GRID_CELLS:
                    clusters.append(
                        {"kind": "grid", "indices": members[comp], **base, **info}
                    )
                else:
                    loose.append(comp)
            if loose:
                comp = np.concatenate(loose)
                clusters.append({"kind": "loose", "indices": members[comp], **base})

    if gen_idx.size or fold_to_generic:
        indices = np.concatenate([gen_idx, *fold_to_generic]).astype(np.int64)
        clusters.append({"kind": "generic", "indices": np.sort(indices)})

    covered = np.concatenate([c["indices"] for c in clusters]) if clusters else []
    assert np.array_equal(np.sort(covered), np.arange(n)), (
        "structure analysis must cover every cell exactly once"
    )
    return clusters


def build_fft_kernel(grid_shape, spacing, dim):
    """Build the 6-component Newell demag kernel on the doubled grid, FFTed.

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
    spacing : tuple (px, py, pz)
        Grid pitch per axis (>= cell size for non-overlapping cells).
    dim : tuple (sx, sy, sz)
        Cell side lengths.

    Returns
    -------
    kernel_fft : ndarray, shape (6, 2*Nx, 2*Ny, Nz+1), complex
        Components in order ``(Nxx, Nyy, Nzz, Nxy, Nxz, Nyz)``. The last axis
        is the half-spectrum produced by ``np.fft.rfftn`` applied to the
        doubled real-space kernel.
    """
    Nx, Ny, Nz = grid_shape
    px, py, pz = spacing

    # Real-space kernel on doubled grid with zero-padding for open BC.
    # Index convention: K[i, j, k] is N at displacement (di, dj, dk) with
    #   di = i      if i <  Nx else i - 2*Nx
    # Cell index 0 maps to displacement 0 (self).
    di = np.where(np.arange(2 * Nx) < Nx, np.arange(2 * Nx), np.arange(2 * Nx) - 2 * Nx)
    dj = np.where(np.arange(2 * Ny) < Ny, np.arange(2 * Ny), np.arange(2 * Ny) - 2 * Ny)
    dk = np.where(np.arange(2 * Nz) < Nz, np.arange(2 * Nz), np.arange(2 * Nz) - 2 * Nz)

    DX, DY, DZ = np.meshgrid(di * px, dj * py, dk * pz, indexing="ij")
    disp = np.stack([DX, DY, DZ], axis=-1)  # (2Nx, 2Ny, 2Nz, 3)
    N = demag_block(disp, dim)  # (2Nx, 2Ny, 2Nz, 3, 3)

    # Stack the 6 independent components: diagonal then upper triangle.
    idx = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    kernel = np.stack([N[..., a, b] for a, b in idx])

    # Real input -> half-spectrum FFT.
    return np.fft.rfftn(kernel, axes=(1, 2, 3))


def demag_fft_matvec(M, kernel_fft, grid_shape):
    """Compute the demag matvec ``H = T @ M`` via 3-D FFT convolution.

    Parameters
    ----------
    M : ndarray, shape (Nx, Ny, Nz, 3)
        Polarization per cell, in the grid's local (axis-aligned) frame.
    kernel_fft : ndarray
        Output of :func:`build_fft_kernel`.
    grid_shape : tuple (Nx, Ny, Nz)

    Returns
    -------
    H : ndarray, shape (Nx, Ny, Nz, 3)
        ``T @ M`` with ``T = -N`` (the demag field scaled by ``mu_0``,
        matching the dense demag matrix used in :func:`apply_demag`).
    """
    Nx, Ny, Nz = grid_shape
    Mp = np.zeros((2 * Nx, 2 * Ny, 2 * Nz, 3), dtype=float)
    Mp[:Nx, :Ny, :Nz, :] = M
    M_fft = np.fft.rfftn(Mp, axes=(0, 1, 2))  # (2Nx, 2Ny, Nz+1, 3)

    # Components: (Nxx, Nyy, Nzz, Nxy, Nxz, Nyz)
    Kxx, Kyy, Kzz, Kxy, Kxz, Kyz = kernel_fft  # each (2Nx, 2Ny, Nz+1)
    Mx_fft = M_fft[..., 0]
    My_fft = M_fft[..., 1]
    Mz_fft = M_fft[..., 2]

    # H_m = sum_k -N_mk * M_k, with the kernel symmetric (K_yx = K_xy etc.).
    Hx_fft = -(Kxx * Mx_fft + Kxy * My_fft + Kxz * Mz_fft)
    Hy_fft = -(Kxy * Mx_fft + Kyy * My_fft + Kyz * Mz_fft)
    Hz_fft = -(Kxz * Mx_fft + Kyz * My_fft + Kzz * Mz_fft)

    H = np.empty((2 * Nx, 2 * Ny, 2 * Nz, 3), dtype=float)
    H[..., 0] = np.fft.irfftn(Hx_fft, s=(2 * Nx, 2 * Ny, 2 * Nz), axes=(0, 1, 2))
    H[..., 1] = np.fft.irfftn(Hy_fft, s=(2 * Nx, 2 * Ny, 2 * Nz), axes=(0, 1, 2))
    H[..., 2] = np.fft.irfftn(Hz_fft, s=(2 * Nx, 2 * Ny, 2 * Nz), axes=(0, 1, 2))
    return H[:Nx, :Ny, :Nz, :]
