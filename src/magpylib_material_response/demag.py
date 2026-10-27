"""Demagnetization / material-response solver.

The self-consistent polarization of ``n`` interacting cells is the solution
of ``(I - S T) J = J0 + S B_ext`` where ``S`` is the block-diagonal
susceptibility matrix (one 3x3 block per cell, its susceptibility along its own axes) and ``T`` the demag interaction operator. The
``H_ext`` object attribute is interpreted as the applied flux density
``B_ext`` in Tesla (B-field convention, consistent with polarization).

Every entry of ``T`` follows one *interaction rule*, chosen per solve with
``interaction`` and applied to every pair of cells:

- ``"point"`` (default): the field of the source cell at the observer cell's
  centre (point matching, Chadebec 2006). Any cell type and orientation; the
  rule this package has always used. Cylinder-segment cells close together
  take each other's field averaged over the observer cell instead: on curved
  cells point matching makes the system indefinite at high susceptibility
  (#92, ``SEGMENT_AVERAGE_REACH``).
- ``"newell"``: the field of the source cell averaged over the observer cell,
  the volume-averaged Newell tensor (:mod:`magpylib_material_response.newell`).
  This is the Galerkin form of the problem: the induced moment of a body in a
  uniform field is then a lower bound that rises under refinement, so a mesh
  study knows which side the converged value is on. It needs every cell to be
  a cuboid, all with one orientation, and raises otherwise.

At equal cell count neither rule is more accurate in general: they agree for
cubes, and for flat cells each wins on different quantities. A mix of the two
on cuboids would jump when a body turns by any angle that makes its cells
non-parallel, and would lose the lower bound, so there is none.

Two evaluation strategies apply the chosen rule, so ``solver="direct"`` and
``solver="iterative"`` solve the same system:

- ``"direct"``   materialises the dense ``T`` and calls ``np.linalg.solve``;
- ``"iterative"`` applies ``T`` matrix-free in GMRES — FFT convolution for
  uniform-grid clusters (:mod:`magpylib_material_response.demag_fft`), dense
  or row-sum-bounded sparse blocks for everything else.

``solver="auto"`` (default) picks between them from the cell structure: the
iterative solver when FFT products cover at least a tenth of the interaction
matrix, else the direct one, which is then as fast or faster, since GMRES
applies the rest as blocks built pair by pair, as the direct solver builds them.

Cells sharing a cuboid shape and orientation make every block fast under
either rule: the tensor depends only on the displacement, so it is computed
once per distinct displacement, reverse blocks follow by reciprocity, and
uniform grids use FFT products. Other cells cost one field evaluation per pair.

All computations run in the global frame; per-cluster rotations are handled
inside the block builders. An anisotropic susceptibility acts along each cell's
own axes (#50), so it enters the solve as one 3x3 tensor per cell,
``R diag(chi) R^T``: ``S`` is block-diagonal.
"""
# pylint: disable=too-many-lines

from __future__ import annotations

import itertools
from collections import Counter

import magpylib as magpy
import numpy as np
import scipy.linalg
import scipy.sparse as sp
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent, BaseMagnet
from magpylib.magnet import Cuboid
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.demag_fft import (
    MIN_CLUSTER_CELLS,
    QUAT_ATOL,
    analyze_collection,
    build_fft_kernel,
    canonical_quats,
    demag_fft_matvec,
)
from magpylib_material_response.newell import (
    demag_block_general,
    point_block,
    self_demag_factors,
)
from magpylib_material_response.utils import timelog

#: A pair block with more dense entries than this may be stored sparse, with a
#: rigorous row-sum error budget, when that pays (``SPARSIFY_MIN_DROP``).
#: 2e7 float64 entries = 160 MB; dense blocks additionally allow deriving the
#: reverse block by reciprocity (halving the build cost).
DENSE_BLOCK_MAX_ENTRIES = 20_000_000
# A block above DENSE_BLOCK_MAX_ENTRIES is sparsified only when a sample of at
# least SPARSIFY_SAMPLE observers, drawn at random across the body, shows that
# at least SPARSIFY_MIN_DROP of the entries can go: a CSR product runs on one
# thread, and at 90 % dropped it takes about 1.7 times as long as the dense one,
# for a sixth of the memory. When the explicit blocks would pass
# DENSE_BLOCK_MAX_BYTES in all, memory comes first and SPARSIFY_MIN_DROP_LARGE
# suffices: a CSR takes 12 bytes per kept entry against 8 per dense entry, and
# about 20 while it is built, so at 60 % dropped it is no larger than the dense
# block even then. At high susceptibility the row budget drops almost nothing,
# and blocks stay dense at any size. The sample's rows are part of the block.
SPARSIFY_MIN_DROP = 0.9
SPARSIFY_MIN_DROP_LARGE = 0.6
SPARSIFY_SAMPLE = 64
#: Above this many bytes memory comes first: explicit blocks are sparsified
#: from ``SPARSIFY_MIN_DROP_LARGE`` when they would pass it in all, the
#: iterative solve warns when they do, and ``solver="auto"`` falls back to the
#: direct solver only when the dense T stays below it.
DENSE_BLOCK_MAX_BYTES = 2_000_000_000

#: Every block is evaluated in observer chunks covering at most this many
#: (observer, source) pairs per field evaluation. Memory grows with pairs, not
#: with output entries: about 0.8 kB per pair for cuboids and about 3 kB for
#: tetrahedra in magpylib's kernels, so one chunk stays below about 0.6 GB.
KERNEL_PAIRS_PER_CALL = 200_000
# A gather from a grid-pair table needs no kernel temporaries, but with its keys
# and transposed copy about 250 B per pair; larger chunks only add memory: a
# 4,096-cell cube's direct solve peaked at 2.0 GB with 2,000,000 pairs and 1.3 GB
# with 200,000, in the same time.
TABLE_PAIRS_PER_CHUNK = 200_000
# Between two grids of one spacing the block is an FFT convolution; below this
# many entries a dense block from the same table is cheaper to apply.
FFT_CROSS_MIN_ENTRIES = 1_000_000

#: ``solver="auto"`` takes the iterative solver when FFT products cover at least
#: this share of the interaction matrix's entries: the self-blocks of uniform
#: grids and the blocks between large grids of one spacing. Measured on 39
#: models at chi = 1000, GMRES was then as fast or faster and lighter (three
#: cubes of three cell sizes, 40 %: 8.3 s and 1.0 GB against 9.2 s and 1.9 GB;
#: a grid with 3,900 tetrahedra, 26 %: 99 s against 112 s). Below it, many small
#: bodies or cells off any grid made it slower (100 cubes of 27 cells, 1 %:
#: 6.2 s against 5.1 s; 1,728 cuboids of 216 sizes, 0 %: 61 s against 34 s).
AUTO_MIN_FFT_SHARE = 0.1
#: Under ``solver="auto"`` GMRES runs to ``solver_tol`` times this factor. At
#: chi = 1000 a GMRES polarization at ``solver_tol`` lies 160 times
#: ``solver_tol`` from the direct one on 4,096 grid cells and 580 times on
#: 32,768; a hundredfold tighter residual brings that to 1-10 times, for about
#: 20 % more time.
AUTO_TOL_FACTOR = 0.01


def _has_path(src):
    """True if the position, orientation or any path-capable property of `src`,
    such as `dimension` or `current`, varies along a path."""
    return src._get_path_len() > 1


def get_susceptibilities(sources, susceptibility=None):
    """Return a list of length (len(sources)) with susceptibility values
    Priority is given at the source level, however if value is not found, it is searched
    up the parent tree, if available. Raises an error if no value is found when reached
    the top level of the tree."""
    n = len(sources)

    if susceptibility is None:
        # Get susceptibilities from source attributes
        susceptibilities = []
        for src in sources:
            src_susceptibility = getattr(src, "susceptibility", None)
            if src_susceptibility is None:
                if src.parent is None:
                    msg = "No susceptibility defined in any parent collection"
                    raise ValueError(msg)
                src_susceptibility = _get_susceptibility_from_hierarchy(src.parent)
            susceptibilities.append(src_susceptibility)

        susis = _convert_to_array(susceptibilities, n, from_hierarchy=True)
    else:
        # Use function input susceptibility
        susis = _convert_to_array(susceptibility, n, from_hierarchy=False)

    return np.reshape(susis, 3 * n, order="F")


def _convert_to_array(susceptibility, n, from_hierarchy=False):
    """Convert susceptibility input(s) to (n, 3) array format"""
    # Handle single values (scalar or 3-vector) applied to all sources
    if np.isscalar(susceptibility) or (
        isinstance(susceptibility, np.ndarray) and susceptibility.ndim == 0
    ):
        return np.ones((n, 3)) * float(susceptibility)
    if (
        not from_hierarchy  # values read from the objects are one per object
        and hasattr(susceptibility, "__len__")
        and len(susceptibility) == 3
        and all(np.ndim(x) == 0 for x in susceptibility)
    ):
        # This is a 3-vector, not a list of 3 items
        susis = np.tile(susceptibility, (n, 1))
        if n == 3:
            msg = (
                "Apply_demag input susceptibility is ambiguous - either scalar list or vector single entry. "
                "Please choose different means of input or change the number of cells in the Collection."
            )
            raise ValueError(msg)
        return susis

    # Handle list of susceptibilities (one per source)
    susceptibility_list = (
        list(susceptibility) if not isinstance(susceptibility, list) else susceptibility
    )

    if len(susceptibility_list) != n:
        msg = "Apply_demag input susceptibility must be scalar, 3-vector, or same length as input Collection."
        raise ValueError(msg)

    # Convert each susceptibility to 3-tuple format
    susis = []
    for sus in susceptibility_list:
        if np.ndim(sus) == 0:
            susis.append((float(sus), float(sus), float(sus)))
        elif hasattr(sus, "__len__") and len(sus) == 3:
            try:
                sus_tuple = tuple(float(x) for x in sus)
            except Exception as e:
                msg = f"Each element of susceptibility 3-vector must be numeric. Got: {sus!r} ({e})"
                raise ValueError(msg) from e
            susis.append(sus_tuple)
        else:
            msg = "susceptibility is not scalar or array of length 3"
            raise ValueError(msg)

    return np.array(susis)


#: Turning ``T`` into ``I - S T`` for anisotropic cells rewrites three rows per
#: cell; at most this many matrix entries are copied at a time.
SUSCEPTIBILITY_CHUNK_ENTRIES = 20_000_000


def _cell_susceptibilities(sus, orientations):
    """Susceptibility tensor of each cell in the global frame, shape (n, 3, 3).

    `sus` is the flat output of `get_susceptibilities`: the values along each
    cell's own axes, ordered x1..xn, y1..yn, z1..zn like the polarization. Cell
    i contributes R_i diag(sus_i) R_i^T, so an anisotropic susceptibility turns
    with its cell, as its polarization does (#50). The susceptibility matrix
    ``S`` of the solve is block-diagonal: one such 3x3 block per cell."""
    n = len(orientations)
    rot = np.array([o.as_matrix() for o in orientations]).reshape((n, 3, 3))
    chi_own = np.reshape(sus, (n, 3), order="F")
    return np.einsum("nab,nb,ncb->nac", rot, chi_own, rot)


def _apply_susceptibility(chi, X):
    """``S @ X`` for the block-diagonal ``S`` given per cell by ``chi``, with
    ``X`` in the polarization layout x1..xn, y1..yn, z1..zn: shape (3n,) or
    (3n, m)."""
    n = len(chi)
    X3 = np.reshape(X, (3, n, -1))
    return np.einsum("iab,bim->aim", chi, X3).reshape(np.shape(X))


def _susceptibility_diagonal(chi):
    """The flat (3n,) diagonal of ``S`` when every cell's tensor is diagonal
    (isotropic, or anisotropic on an unrotated cell), else ``None``."""
    diag = np.einsum("iaa->ia", chi)
    off = chi - diag[:, :, None] * np.eye(3)
    if np.max(np.abs(off), initial=0.0) > 1e-12 * np.max(np.abs(chi), initial=0.0):
        return None
    return diag.T.ravel()


def _subtract_susceptibility_product(T, chi):
    """Turn the dense (3n, 3n) ``T`` into ``Q = I - S T`` in place.

    Diagonal ``S`` scales rows; otherwise each cell's three rows are replaced
    by their 3x3 combination, a slice of cells at a time, so no second
    (3n, 3n) array is ever made."""
    n = len(chi)
    diag = _susceptibility_diagonal(chi)
    if diag is not None:
        T *= -diag[:, None]
    else:
        step = max(1, SUSCEPTIBILITY_CHUNK_ENTRIES // (9 * n))
        for start in range(0, n, step):
            cells = np.arange(start, min(start + step, n))
            rows = (np.arange(3)[:, None] * n + cells).ravel()
            block = T[rows].reshape(3, len(cells), -1)
            mixed = np.einsum("iab,bix->aix", chi[cells], block)
            T[rows] = -mixed.reshape(len(rows), -1)
    T[np.diag_indices_from(T)] += 1.0
    return T


def _get_susceptibility_from_hierarchy(source):
    """Helper function to get susceptibility value from source or its parent hierarchy.
    Returns the raw susceptibility value (scalar or 3-tuple), not the reshaped array."""
    susceptibility = getattr(source, "susceptibility", None)
    if susceptibility is not None:
        return susceptibility
    if source.parent is None:
        msg = "No susceptibility defined in any parent collection"
        raise ValueError(msg)
    return _get_susceptibility_from_hierarchy(source.parent)


def get_H_ext(*sources):
    """Return a list of length (len(sources)) with H_ext values.

    `H_ext` is a uniform applied field given as mu_0 * H in tesla. A source's own
    value takes priority; otherwise it is searched up the parent tree, and it is
    zero if no parent sets it."""
    H_exts = []
    for src in sources:
        h_ext = getattr(src, "H_ext", None)
        if h_ext is None:
            if src.parent is None:
                H_exts.append((0.0, 0.0, 0.0))
            else:
                H_exts.extend(get_H_ext(src.parent))
        else:
            H_exts.append(h_ext)
    return H_exts


# ══════════════════════════════════════════════════════════════════════════
# Structure analysis and geometry helpers
# ══════════════════════════════════════════════════════════════════════════


#: The interaction rules ``apply_demag`` and ``demag_tensor`` accept.
INTERACTIONS = ("point", "newell")
SOLVERS = ("auto", "direct", "iterative")


def _check_interaction(interaction, magnets_list, atol=QUAT_ATOL):
    """Refuse an unknown rule, and the Newell rule on cells it cannot cover."""
    if interaction not in INTERACTIONS:
        msg = f"interaction must be one of {INTERACTIONS}; got {interaction!r}"
        raise ValueError(msg)
    if interaction == "point" or len(magnets_list) == 0:
        return
    others = Counter(
        type(m).__name__ for m in magnets_list if not isinstance(m, Cuboid)
    )
    if others:
        found = ", ".join(f"{k} {name}" for name, k in others.items())
        msg = (
            "interaction='newell' averages the field over each cell, which needs "
            f"every cell to be a Cuboid; found {found}. Use interaction='point' "
            "(the default), which handles any cell."
        )
        raise ValueError(msg)
    quats = canonical_quats(np.array([m.orientation.as_quat() for m in magnets_list]))
    if not np.allclose(quats, quats[0], atol=atol):
        msg = (
            "interaction='newell' needs all cuboid cells parallel, with one "
            "orientation; these cells are turned against each other. Use "
            "interaction='point' (the default), which handles any orientation."
        )
        raise ValueError(msg)


def _analyze_collection(magnets_list, interaction, atol=QUAT_ATOL):
    """Run :func:`analyze_collection` on a list of magnet objects, with logging.

    Under the Newell rule every cuboid stays in a geometry cluster, whatever the
    group size, because each pair needs both prisms' shapes.
    """
    min_cells = 1 if interaction == "newell" else MIN_CLUSTER_CELLS
    positions, clusters = analyze_collection(magnets_list, atol, min_cells)
    counts = Counter(c["kind"] for c in clusters)
    logger.info(
        "Cell structure: {n_grid} grid, {n_loose} loose, {n_generic} generic clusters",
        n_grid=counts.get("grid", 0),
        n_loose=counts.get("loose", 0),
        n_generic=counts.get("generic", 0),
    )
    for cl in clusters:
        if cl["kind"] == "generic":
            averaged = _segment_averages(magnets_list, cl["indices"])
            if averaged is not None:
                cl["averaged"] = averaged
    return positions, clusters


def _auto_solver(clusters):
    """The solver ``solver="auto"`` takes for these clusters, and the share of
    the interaction matrix's entries covered by FFT products it decided by,
    counted as :func:`_build_operator` builds them."""
    n = sum(len(cl["indices"]) for cl in clusters)
    grids = [cl for cl in clusters if cl["kind"] == "grid"]
    sizes = [len(cl["indices"]) for cl in grids]
    entries = sum(m * m for m in sizes)
    for (i, cl_a), (j, cl_b) in itertools.combinations(enumerate(grids), 2):
        if 9 * sizes[i] * sizes[j] > FFT_CROSS_MIN_ENTRIES and _same_lattice(
            cl_a, cl_b
        ):
            entries += 2 * sizes[i] * sizes[j]
    share = entries / (n * n) if n else 0.0
    return ("iterative" if share >= AUTO_MIN_FFT_SHARE else "direct"), share


def _solve_direct(T, chi, rhs):
    """LU solve of (I - S T) x = rhs. Q is formed in T's own buffer and factored
    in place: T is Fortran-ordered for that, and a C copy would cost twice the
    matrix."""
    Q = _subtract_susceptibility_product(T, chi)
    return scipy.linalg.solve(Q, rhs, overwrite_a=True, check_finite=False)


def _same_orientation(cl_a, cl_b, atol=QUAT_ATOL):
    """True when two cuboid clusters' prisms are parallel."""
    qa = canonical_quats(cl_a["r_common"].as_quat()[None])[0]
    qb = canonical_quats(cl_b["r_common"].as_quat()[None])[0]
    return bool(np.allclose(qa, qb, atol=atol))


def _reciprocity_factor(cl_a, cl_b, interaction):
    """``c`` with ``T_ba = c * T_ab.T`` exactly, or ``None`` to compute ``T_ba``.

    The volume-averaged tensor obeys ``V_a N_ab = V_b N_ba^T`` for any two
    parallel prisms. The point tensor is symmetric and even in the
    displacement, so ``N_ba = N_ab^T`` holds when both clusters share one prism
    shape and orientation.
    """
    if cl_a["kind"] == "generic" or cl_b["kind"] == "generic":
        return None
    if interaction == "newell":
        return float(np.prod(cl_a["dim"]) / np.prod(cl_b["dim"]))
    if np.allclose(cl_a["dim"], cl_b["dim"], rtol=1e-9, atol=0) and _same_orientation(
        cl_a, cl_b
    ):
        return 1.0
    return None


# ══════════════════════════════════════════════════════════════════════════
# T blocks — single source of truth for the interaction entries
#
# All blocks are (3*n_a, 3*n_b) in Fortran (component-major) layout:
# ``block[(m, j), (k, i)]`` is the ``m`` component of ``T @ v`` at observer
# cell ``j`` per unit polarization component ``k`` of source cell ``i``,
# i.e. ``-N_mk(pos_j - pos_i)`` post-``mu_0``.  Global frame.
# ══════════════════════════════════════════════════════════════════════════


def _dedup_blocks(disp, scale, block):
    """Evaluate ``block(disp)`` with duplicate displacements computed once.

    Cells on regular grids share displacements massively (two same-spacing
    grids have O(N) unique displacements out of N^2 pairs), so the tensor is
    gathered from the unique set when that pays off. ``scale`` sets the
    rounding (1e-9 of it) that decides when two displacements are the same.
    """
    flat = disp.reshape(-1, 3)
    keys = np.round(flat / (1e-9 * scale)).astype(np.int64)
    # One hashed key per row sorts several times faster than the rows; the
    # gather below checks that no two different rows hashed alike.
    with np.errstate(over="ignore"):
        h = keys[:, 0] * np.int64(-7046029254386353131)
        h ^= keys[:, 1] * np.int64(5591988499347399697)
        h ^= keys[:, 2]
    _, first, inverse = np.unique(h, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    if not np.array_equal(keys, keys[first[inverse]]):  # a hash collision
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        first = np.zeros(uniq.shape[0], dtype=np.int64)
        first[inverse[::-1]] = np.arange(flat.shape[0] - 1, -1, -1)
    if first.shape[0] > 0.25 * flat.shape[0]:
        return block(disp)
    N_u = block(flat[first])
    return N_u[inverse].reshape((*disp.shape[:-1], 3, 3))


def _newell_block_T(pos_a, pos_b, dim_a, dim_b, rot_mat):
    """Dense Newell T block: observers ``a``, sources ``b``, parallel prisms.

    ``rot_mat`` is the common rotation matrix (``None`` for identity); the
    displacement is evaluated in the prisms' local frame and the tensor
    rotated back to the global frame.
    """
    n_a, n_b = len(pos_a), len(pos_b)
    if rot_mat is not None:
        pos_a = pos_a @ rot_mat  # = rot.inv().apply(pos_a)
        pos_b = pos_b @ rot_mat
    disp = pos_a[:, None, :] - pos_b[None, :, :]  # (n_a, n_b, 3) obs - src
    scale = max(np.max(np.abs(dim_a)), np.max(np.abs(dim_b)))
    N = _dedup_blocks(  # (n_a, n_b, 3m, 3k), local
        disp, scale, lambda d: demag_block_general(d, dim_b, dim_a)
    )
    if rot_mat is not None:
        N = np.einsum("ma,ijab,kb->ijmk", rot_mat, N, rot_mat)
    return -N.transpose(2, 0, 3, 1).reshape(3 * n_a, 3 * n_b)


def _point_cuboid_block_T(pos_a, pos_b, dim_b, rot_mat):
    """Point-rule T block for cuboid sources sharing one shape and orientation.

    Observers ``a`` are any matching points; ``rot_mat`` is the sources' common
    rotation matrix (``None`` for identity). The tensor depends only on the
    displacement in the sources' frame, so it is deduplicated like the Newell
    blocks; it equals ``_point_block_T`` up to the far-field switch.
    """
    n_a, n_b = len(pos_a), len(pos_b)
    if rot_mat is not None:
        pos_a = pos_a @ rot_mat  # = rot.inv().apply(pos_a)
        pos_b = pos_b @ rot_mat
    disp = pos_a[:, None, :] - pos_b[None, :, :]  # (n_a, n_b, 3) obs - src
    N = _dedup_blocks(disp, np.max(np.abs(dim_b)), lambda d: point_block(d, dim_b))
    if rot_mat is not None:
        N = np.einsum("ma,ijab,kb->ijmk", rot_mat, N, rot_mat)
    return -N.transpose(2, 0, 3, 1).reshape(3 * n_a, 3 * n_b)


def _point_block_T(srcs_b, pos_a):
    """Point-matched T block via ``magpy.getH`` (any magnet types).

    Source polarizations are set to unit vectors during evaluation and
    restored afterwards.
    """
    n_a, n_b = len(pos_a), len(srcs_b)
    saved = [s.polarization for s in srcs_b]
    rot_b = R.from_quat([s.orientation.as_quat() for s in srcs_b])
    try:
        H_point = []
        for unit_pol in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
            pol_all = rot_b.inv().apply(unit_pol)
            for src, pol in zip(srcs_b, pol_all, strict=True):
                src.polarization = pol
            H = np.reshape(magpy.getH(srcs_b, pos_a), (n_b, n_a, 3))
            H_point.append(H)
    finally:
        # magpylib does not allow assigning None back; such cells keep the
        # last unit polarization (matches historical behaviour).
        for src, pol in zip(srcs_b, saved, strict=True):
            if pol is not None:
                src.polarization = pol
    T_raw = np.array(H_point) * magpy.mu_0  # (3k, n_b, n_a, 3m)
    return T_raw.swapaxes(2, 3).reshape(3 * n_b, 3 * n_a).T


#: Cylinder-segment cells closer than this many times the sum of their
#: half-diagonals take each other's field averaged over the observer cell, not
#: at its centroid. Point matching on curved cells gives T positive
#: eigenvalues, which no demagnetizing field has, and at high susceptibility
#: the system turns indefinite: a tube with a 1 mm wall at chi = 1000 swings
#: from -10 to -68 to +2 % with 1, 2 and 3 cells across the wall (#92). The
#: average makes T symmetric with every eigenvalue negative. Beyond the reach
#: the centroid stands in for the average. On that tube, in cells 16 around and
#: 16 along, a reach of 2 gives the moment of averaging every pair to 0.13
#: points at 1, 2 and 3 cells across, 1.5 to 0.7 points; at 1 a positive
#: eigenvalue comes back with 3 across, and at 0.75 the system is indefinite.
SEGMENT_AVERAGE_REACH = 2.0
#: Gauss points per direction (radius, angle, height) of a cell average: 3
#: instead of 2 moves that tube's moment by 0.5 points, 4 % of its error.
SEGMENT_AVERAGE_POINTS = 2


def _segment_quadrature(segments):
    """Gauss points ``(n, q, 3)`` in the global frame and their weights
    ``(n, q)``, summing to 1 over each cylinder-segment cell."""
    x, w = np.polynomial.legendre.leggauss(SEGMENT_AVERAGE_POINTS)
    x, w = (x + 1) / 2, w / 2
    r1, r2, h, phi1, phi2 = np.array([s.dimension for s in segments]).T
    r = r1[:, None] + (r2 - r1)[:, None] * x  # (n, g)
    phi = np.deg2rad(phi1[:, None] + (phi2 - phi1)[:, None] * x)
    z = h[:, None] * (x - 0.5)
    local = np.stack(
        np.broadcast_arrays(
            r[:, :, None, None] * np.cos(phi[:, None, :, None]),
            r[:, :, None, None] * np.sin(phi[:, None, :, None]),
            z[:, None, None, :],
        ),
        axis=-1,
    ).reshape(len(segments), -1, 3)
    weights = (r * w)[:, :, None, None] * w[:, None] * w  # dV = r dr dphi dz
    weights = weights.reshape(len(segments), -1)
    weights /= weights.sum(axis=1, keepdims=True)
    mats = R.from_quat([s.orientation.as_quat() for s in segments]).as_matrix()
    positions = np.array([s.position for s in segments], dtype=float)
    points = np.einsum("nij,nqj->nqi", mats, local) + positions[:, None]
    return points, weights


def _segment_averages(magnets_list, indices):
    """The T blocks between cylinder-segment cells within
    ``SEGMENT_AVERAGE_REACH`` of each other, averaged over the observer cell,
    or ``None`` when there are none.

    Returns ``(rows, cols, blocks)``: observer and source positions within
    ``indices``, and the ``(m, k)`` blocks, symmetrized by reciprocity
    (``V_i T_ij = V_j T_ji^T`` holds exactly for the average; the quadrature
    keeps it only approximately).
    """
    local = np.flatnonzero(
        [isinstance(magnets_list[i], magpy.magnet.CylinderSegment) for i in indices]
    )
    if len(local) == 0:
        return None
    segments = [magnets_list[indices[i]] for i in local]
    r1, r2, h, phi1, phi2 = np.array([s.dimension for s in segments]).T
    arc = (r1 + r2) / 2 * np.deg2rad(phi2 - phi1)
    rho = 0.5 * np.sqrt((r2 - r1) ** 2 + arc**2 + h**2)
    centroids = np.array([s.centroid for s in segments])
    tree = cKDTree(centroids)
    near = tree.query_ball_point(centroids, SEGMENT_AVERAGE_REACH * (rho + rho.max()))
    i = np.repeat(np.arange(len(segments)), [len(p) for p in near])
    j = np.concatenate(near).astype(int)
    keep = np.linalg.norm(centroids[i] - centroids[j], axis=1) < (
        SEGMENT_AVERAGE_REACH * (rho[i] + rho[j])
    )
    i, j = i[keep], j[keep]

    with timelog(f"Cell averages of {len(i):,} cylinder-segment pairs"):
        points, weights = _segment_quadrature(segments)
        q = points.shape[1]
        dims = np.array([s.dimension for s in segments], dtype=float)
        positions = np.array([s.position for s in segments], dtype=float)
        quats = np.array([s.orientation.as_quat() for s in segments])
        mats = R.from_quat(quats).as_matrix()
        blocks = np.empty((len(i), 3, 3))
        step = max(1, KERNEL_PAIRS_PER_CALL // q)
        for start in range(0, len(i), step):
            ii, jj = i[start : start + step], j[start : start + step]
            src = np.repeat(jj, q)
            for k in range(3):
                # the unit polarization along global k, in each source's frame
                H = magpy.func.cylinder_segment_field(
                    "H",
                    points[ii].reshape(-1, 3),
                    dims[src],
                    mats[src, k, :],
                    positions[src],
                    R.from_quat(quats[src]),
                ).reshape(len(ii), q, 3)
                blocks[start : start + step, :, k] = magpy.mu_0 * np.einsum(
                    "pqm,pq->pm", H, weights[ii]
                )

    volumes = np.array([s.volume for s in segments])
    order = np.lexsort((j, i))
    i, j, blocks = i[order], j[order], blocks[order]
    reverse = np.searchsorted(i * len(segments) + j, j * len(segments) + i)
    moments = volumes[i, None, None] * blocks
    blocks = (moments + moments[reverse].transpose(0, 2, 1)) / 2
    blocks /= volumes[i, None, None]
    return local[i], local[j], blocks


def _average_segments(block, averaged, observers, n_b):
    """``block`` of the generic self-block, observers ``observers`` and all its
    ``n_b`` sources, with the averaged cylinder-segment pairs written in."""
    rows, cols, blocks = averaged
    at = np.full(n_b, -1)
    at[observers] = np.arange(len(observers))
    sel = at[rows] >= 0
    if not sel.any():
        return block
    out = np.ascontiguousarray(block).reshape(3, len(observers), 3, n_b)
    out[:, at[rows[sel]], :, cols[sel]] = blocks[sel]
    return out.reshape(3 * len(observers), 3 * n_b)


def _row_floor(mag, row_budget):
    """Each row's smallest kept magnitude: the entries below it can go with
    their ``|.|`` sum within ``row_budget`` (exact, entry-based bound)."""
    srt = np.sort(mag, axis=1)
    n_drop = (np.cumsum(srt, axis=1) <= row_budget).sum(axis=1)  # smallest-k
    n = mag.shape[1]
    floor = np.take_along_axis(srt, np.minimum(n_drop, n - 1)[:, None], axis=1)
    floor[n_drop == n] = np.inf
    return floor


def _droppable(block, row_budget):
    """How many of a block's entries :func:`_row_sparsify` drops."""
    mag = np.abs(block)
    return np.count_nonzero(mag < _row_floor(mag, row_budget))


def _row_sparsify(block, row_budget):
    """CSR rows of a block within the per-row budget: the kept count of each
    row, and the kept columns (int32) and values in row order. Ties at a row's
    floor stay, and so does NaN, to show in the solve."""
    mag = np.abs(block)
    keep = ~(mag < _row_floor(mag, row_budget))
    cols = np.flatnonzero(keep) % block.shape[1]
    return keep.sum(axis=1), cols.astype(np.int32), block[keep]


def _same_lattice(cl_a, cl_b, rtol=1e-6):
    """Whether two clusters are grids of one spacing and orientation, so that
    the block between them depends on the index difference only. A relative
    tolerance only: at nanometre scale an absolute one would call different
    spacings equal."""
    return (
        cl_a["kind"] == "grid"
        and cl_b["kind"] == "grid"
        and _same_orientation(cl_a, cl_b)
        and np.allclose(cl_a["spacing"], cl_b["spacing"], rtol=rtol, atol=0)
    )


def _grid_pair_table(cl_a, cl_b, interaction, rtol=1e-6):
    """T between two grid clusters of one spacing and orientation, tabulated
    over the index differences, in the grids' local frame.

    Observer ``i`` at grid index ``p`` and source ``j`` at ``q`` are
    ``origin_a - origin_b + spacing * (p - q)`` apart, so the block has at most
    ``prod(Na + Nb - 1)`` distinct entries: ``table[p - q + Nb - 1]`` (per axis)
    is the (3, 3) tensor of pair ``(i, j)``. Returns ``(table, index_a, index_b)``
    with ``table`` of shape ``(*(Na + Nb - 1), 3, 3)`` and the grid indices of
    the clusters' members, or ``None`` when the clusters do not share spacing
    and orientation (:func:`_same_lattice`).
    """
    if not _same_lattice(cl_a, cl_b, rtol):
        return None
    sa, sb = np.array(cl_a["shape"]), np.array(cl_b["shape"])
    spacing = np.array(cl_a["spacing"])

    def _index3(cl, shape):
        flat = cl["flat"]  # cluster member -> grid-flat index
        return np.stack(
            [
                flat // (shape[1] * shape[2]),
                (flat // shape[2]) % shape[1],
                flat % shape[2],
            ],
            axis=1,
        )

    K = sa + sb - 1  # index differences span [-(Nb-1), Na-1] per axis
    du = np.stack(np.meshgrid(*[np.arange(k) for k in K], indexing="ij"), axis=-1) - (
        sb - 1
    )
    disp = (np.array(cl_a["origin"]) - np.array(cl_b["origin"])) + spacing * du
    if interaction == "newell":
        N = demag_block_general(disp, cl_b["dim"], cl_a["dim"])
    else:
        N = point_block(disp, cl_b["dim"])
    return -N, _index3(cl_a, sa), _index3(cl_b, sb)


def _grid_pair_keys(index_a, index_b, shape_b, K):
    """Flat table indices of every (observer, source) pair: (n_a, n_b)."""
    delta = index_a[:, None, :] - index_b[None, :, :] + (np.asarray(shape_b) - 1)
    return (delta[..., 0] * K[1] + delta[..., 1]) * K[2] + delta[..., 2]


def _pair_block(
    cl_a,
    cl_b,
    magnets_list,
    positions,
    interaction,
    row_budget=None,
    sink=None,
    min_drop=SPARSIFY_MIN_DROP,
):
    """Build the T block for a cluster pair (observers ``a``, sources ``b``)
    under the solve's interaction rule.

    Returns a dense ndarray when the block is small, when ``row_budget`` is
    None, or when the budget would drop less than ``min_drop`` of the entries
    (at high susceptibility it keeps nearly all of them); else a CSR matrix
    with per-row dropped sums below ``row_budget``. With ``sink``, a dense
    block is not built: each chunk goes to ``sink(start, stop, part)``,
    ``part`` of shape ``(3, stop - start, 3 n_b)`` over the observers
    ``start:stop``, and nothing is returned.
    """
    ia, ib = cl_a["indices"], cl_b["indices"]
    pos_a, pos_b = positions[ia], positions[ib]
    n_a, n_b = len(ia), len(ib)
    sparsify = row_budget is not None and 9 * n_a * n_b > DENSE_BLOCK_MAX_ENTRIES
    cuboid_sources = cl_b["kind"] != "generic"
    # Newell: every cluster is cuboid and all share one orientation (checked
    # up front). Point: the sources' orientation sets the frame.
    rot_mat = None
    if cuboid_sources and not cl_b["is_identity"]:
        rot_mat = cl_b["r_common"].as_matrix()

    grid_pair = _grid_pair_table(cl_a, cl_b, interaction)
    if grid_pair is not None:
        # Two grids of one spacing: the kernel once per index difference for
        # the whole block, every chunk a gather by its own keys.
        table, index_a, index_b = grid_pair
        K = np.array(table.shape[:3])
        if rot_mat is not None:
            table = np.einsum("ma,...ab,kb->...mk", rot_mat, table, rot_mat)
        table = table.reshape(-1, 3, 3)

    def _dense_chunk(sl):
        if grid_pair is not None:
            keys = _grid_pair_keys(index_a[sl], index_b, cl_b["shape"], K)
            N = table[keys]  # (n_chunk, n_b, 3m, 3k), global frame
            return N.transpose(2, 0, 3, 1).reshape(3 * N.shape[0], 3 * n_b)
        if interaction == "newell":
            return _newell_block_T(pos_a[sl], pos_b, cl_a["dim"], cl_b["dim"], rot_mat)
        if cuboid_sources:
            return _point_cuboid_block_T(pos_a[sl], pos_b, cl_b["dim"], rot_mat)
        block = _point_block_T([magnets_list[i] for i in ib], pos_a[sl])
        if cl_a is cl_b and "averaged" in cl_a:
            block = _average_segments(block, cl_a["averaged"], np.arange(n_a)[sl], n_b)
        return block

    # Observer chunks bounded by kernel pairs, which is what costs memory;
    # a single call over all n_a * n_b pairs of 5,000 tetrahedra needs ~97 GB.
    pairs = TABLE_PAIRS_PER_CHUNK if grid_pair is not None else KERNEL_PAIRS_PER_CALL
    chunk = max(1, pairs // max(n_b, 1))

    sampled, sample = np.empty(0, dtype=int), None
    if sparsify:
        # Sparsify only when it pays, judged on observers drawn at random across
        # the whole body, one per stretch of it: consecutive cells are one slab
        # of it, and a fixed stride can fall on one layer.
        n_s = min(n_a, max(chunk, SPARSIFY_SAMPLE))
        edges = np.arange(n_s + 1) * n_a // n_s
        offsets = np.random.default_rng(0).random(n_s) * np.diff(edges)
        sampled = edges[:-1] + offsets.astype(int)
        sample = np.empty((3, n_s, 3 * n_b))  # rows m-major, as in the block
        dropped = 0
        for k in range(0, n_s, chunk):
            part = _dense_chunk(sampled[k : k + chunk])
            dropped += _droppable(part, row_budget)
            sample[:, k : k + chunk] = part.reshape(3, -1, 3 * n_b)
        sparsify = dropped >= min_drop * sample.size

    def _chunks():
        # Consecutive observers; the rows the sample holds are not evaluated again.
        for start in range(0, n_a, chunk):
            stop = min(start + chunk, n_a)
            lo, hi = np.searchsorted(sampled, [start, stop])
            if lo == hi:
                part = _dense_chunk(slice(start, stop))
                yield start, stop, part.reshape(3, stop - start, 3 * n_b)
                continue
            part = np.empty((3, stop - start, 3 * n_b))
            part[:, sampled[lo:hi] - start] = sample[:, lo:hi]
            fresh = np.setdiff1d(np.arange(start, stop), sampled[lo:hi])
            if len(fresh):
                rows = _dense_chunk(fresh).reshape(3, len(fresh), 3 * n_b)
                part[:, fresh - start] = rows
            yield start, stop, part

    if not sparsify:
        # Rows of a block are m-major over its observers: (3, n_a, 3 n_b).
        if sink is not None:
            for start, stop, part in _chunks():
                sink(start, stop, part)
            return None
        block = np.empty((3, n_a, 3 * n_b))
        for start, stop, part in _chunks():
            block[:, start:stop] = part
        return block.reshape(3 * n_a, 3 * n_b)

    # Each chunk straight into CSR pieces, one per field component m: a chunk's
    # rows are m-major over its own observers, the block's over all of them.
    counts, cols, vals = [[], [], []], [[], [], []], [[], [], []]
    for start, stop, part in _chunks():
        n_kept, c, v = _row_sparsify(part.reshape(-1, 3 * n_b), row_budget)
        n_kept = n_kept.reshape(3, stop - start)
        ends = np.cumsum(n_kept.sum(axis=1))
        for m, lo, hi in zip(range(3), [0, *ends[:2]], ends, strict=True):
            counts[m].append(n_kept[m])
            cols[m].append(c[lo:hi])
            vals[m].append(v[lo:hi])

    def _join(pieces):
        # Free the pieces once joined: the CSR peaks at 20 bytes per entry.
        out = np.concatenate([p for per_m in pieces for p in per_m])
        pieces.clear()
        return out

    indptr = np.concatenate([[0], np.cumsum(_join(counts))])
    data = _join(vals)
    return sp.csr_matrix((data, _join(cols), indptr), shape=(3 * n_a, 3 * n_b))


def _assemble_T_dense(
    magnets_list, positions, clusters, interaction, min_log_time=None
):
    """Assemble the full dense (3n, 3n) T, global frame, in Fortran order.

    Fortran order is what LAPACK factors in place: `scipy.linalg.solve` on a
    C-ordered matrix copies it first, twice the matrix in memory. Every chunk
    of a block is written straight into T, and reverse cross-blocks come from
    the forward chunk by reciprocity wherever it holds exactly (see
    :func:`_reciprocity_factor`), so no block is ever held on its own.
    """
    n = len(magnets_list)
    T = np.zeros((3 * n, 3 * n), order="F")

    def _flat(cl):
        return np.arange(3)[:, None] * n + cl["indices"][None, :]  # (3, n_cl)

    def _write(rows, cols, factor):
        def sink(start, stop, part):
            r = rows[:, start:stop].ravel()
            T[np.ix_(r, cols.ravel())] = part.reshape(len(r), -1)
            if factor is not None:
                T[np.ix_(cols.ravel(), r)] = factor * part.reshape(len(r), -1).T

        return sink

    with timelog("Demagnetization tensor assembly", min_log_time=min_log_time):
        for idx_a, cl_a in enumerate(clusters):
            rows = _flat(cl_a)
            _pair_block(
                cl_a,
                cl_a,
                magnets_list,
                positions,
                interaction,
                sink=_write(rows, rows, None),
            )
            for cl_b in clusters[idx_a + 1 :]:
                cols = _flat(cl_b)
                factor = _reciprocity_factor(cl_a, cl_b, interaction)
                _pair_block(
                    cl_a,
                    cl_b,
                    magnets_list,
                    positions,
                    interaction,
                    sink=_write(rows, cols, factor),
                )
                if factor is None:
                    _pair_block(
                        cl_b,
                        cl_a,
                        magnets_list,
                        positions,
                        interaction,
                        sink=_write(cols, rows, None),
                    )
    return T


# ══════════════════════════════════════════════════════════════════════════
# Matrix-free operator for the iterative solver
# ══════════════════════════════════════════════════════════════════════════


def _to_box(M, flat, shape):
    """Cell values ``(n, 3)`` placed in their grid box ``(*shape, 3)``, zero in
    the holes."""
    box = np.zeros((shape[0] * shape[1] * shape[2], 3))
    box[flat] = M
    return box.reshape((*shape, 3))


def _fft_apply(op, v_flat):
    """Apply the grid self-block T to component-major ``(3n_g,)`` global-frame
    polarizations; returns the same layout. Rotation handled internally."""
    shape = op["shape"]
    rot_mat = op["rot_mat"]
    M = v_flat.reshape(3, -1).T  # (n_g, 3) global
    if rot_mat is not None:
        M = M @ rot_mat  # to local frame
    H_grid = demag_fft_matvec(_to_box(M, op["flat"], shape), op["kernel_fft"], shape)
    H = H_grid.reshape(-1, 3)[op["flat"]]
    if rot_mat is not None:
        H = H @ rot_mat.T  # back to global frame
    return H.T.ravel()


def _fft_cross_op(cl_a, cl_b, interaction):
    """The block of two grid clusters of one spacing as a zero-padded FFT
    convolution on a box of ``Na + Nb - 1`` per axis, or ``None``.

    Like the self-block kernel, but the index difference ``d`` runs over
    ``[-(Nb - 1), Na - 1]``: position ``i < Na`` of the padded axis holds
    ``d = i``, position ``i >= Na`` holds ``d = i - L``. The nine components
    are kept (no symmetry assumed) in the grids' local frame.
    """
    grid_pair = _grid_pair_table(cl_a, cl_b, interaction)
    if grid_pair is None:
        return None
    table, _, _ = grid_pair
    sa, sb = np.array(cl_a["shape"]), np.array(cl_b["shape"])
    L = sa + sb - 1
    circ = [
        np.where(np.arange(l) < na, np.arange(l), np.arange(l) - l) + (nb - 1)
        for l, na, nb in zip(L, sa, sb, strict=True)
    ]
    kernel = table[np.ix_(*circ)]  # (Lx, Ly, Lz, 3, 3)
    return {
        "kind": "fft_cross",
        "ia": cl_a["indices"],
        "ib": cl_b["indices"],
        "shape_a": tuple(int(v) for v in sa),
        "shape_b": tuple(int(v) for v in sb),
        "flat_a": cl_a["flat"],
        "flat_b": cl_b["flat"],
        "rot_mat": None if cl_b["is_identity"] else cl_b["r_common"].as_matrix(),
        "L": tuple(int(v) for v in L),
        "kernel_fft": np.fft.rfftn(kernel, axes=(0, 1, 2)),
    }


def _fft_cross_apply(op, v_flat):
    """Apply a cross-grid block to component-major ``(3 n_b,)`` global-frame
    polarizations; returns ``(3 n_a,)`` in the same layout."""
    L, rot_mat = op["L"], op["rot_mat"]
    M = v_flat.reshape(3, -1).T  # (n_b, 3) global
    if rot_mat is not None:
        M = M @ rot_mat  # to local frame
    Mp = np.zeros((*L, 3))
    sb = op["shape_b"]
    Mp[: sb[0], : sb[1], : sb[2]] = _to_box(M, op["flat_b"], sb)
    M_fft = np.fft.rfftn(Mp, axes=(0, 1, 2))  # (Lx, Ly, Lz//2+1, 3)
    H_fft = np.einsum("...mk,...k->...m", op["kernel_fft"], M_fft)
    H_grid = np.fft.irfftn(H_fft, s=L, axes=(0, 1, 2))
    sa = op["shape_a"]
    H = H_grid[: sa[0], : sa[1], : sa[2]].reshape(-1, 3)[op["flat_a"]]
    if rot_mat is not None:
        H = H @ rot_mat.T  # back to global frame
    return H.T.ravel()


def _build_operator(
    magnets_list, positions, clusters, chi, solver_tol, min_log_time, interaction
):
    """Build the list of block operators representing T for the matvec.

    Self-blocks of grid clusters are FFT convolutions (exact); every other
    block is dense, or sparse where a per-row error budget drops at least
    ``SPARSIFY_MIN_DROP`` of it (``SPARSIFY_MIN_DROP_LARGE`` when these blocks
    would pass ``DENSE_BLOCK_MAX_BYTES`` dense). The budget keeps the total
    operator perturbation below ``0.1 * solver_tol``:
    ``|S (T - T~)|_inf <= chi_eff * K * row_budget``, with ``chi_eff`` the
    largest row sum of ``|S|`` (at least 1). A reverse block given by
    reciprocity is the forward one transposed, a view with a ``scale``.
    """
    K = len(clusters)
    chi_eff = max(1.0, float(np.abs(chi).sum(axis=2).max()) if len(chi) else 1.0)
    row_budget = 0.1 * solver_tol / (chi_eff * max(K, 1))
    n = sum(len(cl["indices"]) for cl in clusters)
    _, fft_share = _auto_solver(clusters)
    large = 72 * n * n * (1 - fft_share) > DENSE_BLOCK_MAX_BYTES
    budget = {
        "row_budget": row_budget,
        "min_drop": SPARSIFY_MIN_DROP_LARGE if large else SPARSIFY_MIN_DROP,
    }

    ops = []
    with timelog("Demag operator build", min_log_time=min_log_time):
        for idx_a, cl_a in enumerate(clusters):
            if cl_a["kind"] == "grid":
                rot_mat = None if cl_a["is_identity"] else cl_a["r_common"].as_matrix()
                ops.append(
                    {
                        "kind": "fft",
                        "ia": cl_a["indices"],
                        "ib": cl_a["indices"],
                        "shape": cl_a["shape"],
                        "flat": cl_a["flat"],
                        "rot_mat": rot_mat,
                        "kernel_fft": build_fft_kernel(
                            cl_a["shape"], cl_a["spacing"], cl_a["dim"], interaction
                        ),
                    }
                )
            else:
                ops.append(
                    {
                        "kind": "mat",
                        "self": cl_a["kind"],
                        "ia": cl_a["indices"],
                        "ib": cl_a["indices"],
                        "M": _pair_block(
                            cl_a, cl_a, magnets_list, positions, interaction, **budget
                        ),
                    }
                )
            for idx_b, cl_b in enumerate(clusters):
                if idx_b <= idx_a:
                    continue
                n_ab = 9 * len(cl_a["indices"]) * len(cl_b["indices"])
                cross = (
                    _fft_cross_op(cl_a, cl_b, interaction)
                    if n_ab > FFT_CROSS_MIN_ENTRIES
                    else None
                )
                if cross is not None:
                    ops.append(cross)
                    ops.append(
                        _fft_cross_op(cl_a=cl_b, cl_b=cl_a, interaction=interaction)
                    )
                    continue
                M_ab = _pair_block(
                    cl_a, cl_b, magnets_list, positions, interaction, **budget
                )
                ops.append(
                    {
                        "kind": "mat",
                        "ia": cl_a["indices"],
                        "ib": cl_b["indices"],
                        "M": M_ab,
                    }
                )
                op_ba = {"kind": "mat", "ia": cl_b["indices"], "ib": cl_a["indices"]}
                factor = _reciprocity_factor(cl_a, cl_b, interaction)
                if factor is not None and not sp.issparse(M_ab):
                    op_ba.update(M=M_ab.T, scale=factor)
                else:
                    op_ba["M"] = _pair_block(
                        cl_a=cl_b,
                        cl_b=cl_a,
                        magnets_list=magnets_list,
                        positions=positions,
                        interaction=interaction,
                        **budget,
                    )
                ops.append(op_ba)
    held = sum(
        _stored_bytes(op["M"])
        for op in ops
        if op["kind"] == "mat" and "scale" not in op
    )
    if held > DENSE_BLOCK_MAX_BYTES:
        logger.warning(
            "The iterative operator holds {gb:.1f} GB of explicit blocks: cells that "
            "are not cuboids on one grid, and pairs of bodies whose grids differ in "
            "spacing. Cuboid cells of one spacing in every body make FFT products of "
            "these blocks; fewer cells also help.",
            gb=held / 1e9,
        )
    return ops


def _stored_bytes(M):
    """Memory of a dense or CSR block."""
    if sp.issparse(M):
        return M.data.nbytes + M.indices.nbytes + M.indptr.nbytes
    return M.nbytes


def _build_matvec(n, chi, ops):
    """Return the Fortran-flat matvec ``v -> (I - S T) @ v``."""

    def matvec(v_flat):
        v3 = v_flat.reshape(3, n)
        Tv = np.zeros((3, n))
        for op in ops:
            v_b = v3[:, op["ib"]].ravel()
            if op["kind"] == "fft":
                h = _fft_apply(op, v_b)
            elif op["kind"] == "fft_cross":
                h = _fft_cross_apply(op, v_b)
            else:
                h = np.asarray(op["M"] @ v_b).ravel()
                if "scale" in op:
                    h *= op["scale"]
            Tv[:, op["ia"]] += h.reshape(3, -1)
        return v_flat - _apply_susceptibility(chi, Tv.ravel())

    return matvec


def _diagonal_blocks(M, m):
    """The (m, 3, 3) diagonal blocks of a component-major (3m, 3m) block of T,
    dense or sparse: each cell's own field per unit of its own polarization."""
    k = np.arange(m)
    out = np.empty((m, 3, 3))
    for a in range(3):
        for b in range(3):
            out[:, a, b] = np.asarray(M[a * m + k, b * m + k]).ravel()
    return out


def _self_blocks(n, clusters, interaction, ops):
    """Each cell's own (3, 3) block of T in the global frame, shape (n, 3, 3),
    for the block-Jacobi preconditioner. Cuboid clusters take their cells'
    self-demagnetizing factors under the solve's rule, rotated with the cluster.
    Other cells take the isotropic part of their stored self-block, ``tr/3``
    times the identity, which is ``-I/3`` for any cell whose centroid lies
    inside it (the demagnetizing tensor has trace 1 inside a body; see the
    method page): their full point-matched blocks vary with the cell's shape, and
    scaling tetrahedra by them triples the iterations (35 to 220 on 1,931
    cells at chi = 1000), while leaving them unscaled puts them about chi away
    from the scaled cuboids, where restarted GMRES stalls."""
    T_self = np.zeros((n, 3, 3))
    for cl in clusters:
        if cl["kind"] == "generic":
            continue
        if interaction == "newell":
            Ns = self_demag_factors(cl["dim"])  # (3,) local frame
        else:
            Ns = np.diag(point_block(np.zeros(3), cl["dim"]))
        block = -np.diag(Ns)
        if not cl["is_identity"]:
            rot_mat = cl["r_common"].as_matrix()
            block = rot_mat @ block @ rot_mat.T
        T_self[cl["indices"]] = block
    for op in ops:
        if op.get("self") == "generic":
            own = _diagonal_blocks(op["M"], len(op["ia"]))
            T_self[op["ia"]] = (
                np.trace(own, axis1=1, axis2=2)[:, None, None] / 3 * np.eye(3)
            )
    return T_self


#: GMRES restart length: iterations per cycle before the Krylov space is rebuilt.
#: Against 20, 100 takes 15 to 55 % fewer iterations on soft iron (388 instead of
#: 835 on 262,144 cells, 61 s instead of 101 s) and lets a 1 mm mu-metal sheet
#: (chi = 1e4) converge within max_iter, for about 2 kB more memory per cell (0.6 GB
#: at 262,144 cells).
GMRES_RESTART = 100


def _solve_iterative(*, n, chi, rhs, rhs_shape, matvec, T_self, solver_tol, max_iter):
    """GMRES solve of (I - S T) x = rhs with block-Jacobi preconditioning.

    Raises ``RuntimeError`` when GMRES does not reach ``solver_tol`` — a
    partially-converged result would silently be wrong.
    """
    Q_op = LinearOperator((3 * n, 3 * n), matvec=matvec, dtype=float)

    # Block-Jacobi preconditioner: each cell's own 3x3 block of Q,
    # I - chi_i T_ii, exact for every cell type, orientation and anisotropy.
    # Leaving some cells unscaled would put their part of the spectrum about
    # chi away from the rest, where restarted GMRES stalls.
    blocks = np.eye(3) - chi @ T_self
    singular = np.abs(np.linalg.det(blocks)) < 1e-12
    blocks[singular] = np.eye(3)
    blocks_inv = np.linalg.inv(blocks)

    def _precondition(v):
        return np.einsum("iab,bi->ai", blocks_inv, v.reshape(3, n)).ravel()

    M_inv = LinearOperator((3 * n, 3 * n), matvec=_precondition, dtype=float)

    x, info = gmres(
        Q_op,
        rhs,
        M=M_inv,
        rtol=solver_tol,
        atol=0.0,
        restart=GMRES_RESTART,
        # scipy counts restart cycles; max_iter counts iterations
        maxiter=max(1, -(-max_iter // GMRES_RESTART)),
        x0=rhs.copy(),  # warm start at rhs
    )
    if info > 0:
        msg = (
            f"GMRES did not converge to tol={solver_tol} within {max_iter} "
            "iterations. Increase max_iter, loosen solver_tol, or use "
            "solver='direct'."
        )
        raise RuntimeError(msg)
    if info < 0:
        msg = f"GMRES illegal input or breakdown (info={info})"
        raise RuntimeError(msg)
    return x.reshape(rhs_shape)


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════


def demag_tensor(
    src_list,
    *,
    min_log_time=None,
    interaction="point",
):
    """
    Compute the demagnetization tensor T for n sources.

    Every pair follows ``interaction``, the same rule as in
    :func:`apply_demag`: the field at each cell's centre (``"point"``, the
    default) or averaged over each cell (``"newell"``).

    Parameters
    ----------
    src_list: sequence of magpylib magnet sources
        Each source is treated as a magnetic cell.

    min_log_time:
        Minimum logging time in seconds. If computation time is below this value, step
        will not be logged.

    interaction: {"point", "newell"}
        Interaction rule, see :func:`apply_demag`. ``"newell"`` needs every
        source to be a Cuboid, all with one orientation.

    Returns
    -------
    Demagnetization tensor: ndarray, shape (3,n,n,3), pre-``mu_0``:
        ``T[k, i, j, m] = -N_mk(pos_j - pos_i) / mu_0``

    Notes
    -----
    Cells whose field is taken from ``magpy.getH`` (cells that are not
    cuboids, and small groups of cuboids) are given unit polarizations during
    the evaluation. Sources whose ``polarization`` is ``None`` keep the last unit
    polarization afterwards (magpylib does not allow assigning ``None``
    back); all other sources are restored. ``apply_demag`` is unaffected —
    it operates on a copy and overwrites every polarization with the
    solution.

    TODO: allow multi-point matching
    """
    _check_interaction(interaction, list(src_list))
    n = len(src_list)
    if n == 0:
        return np.zeros((3, 0, 0, 3))
    positions, clusters = _analyze_collection(list(src_list), interaction)
    T2 = _assemble_T_dense(
        list(src_list), positions, clusters, interaction, min_log_time=min_log_time
    )
    # T2[(m, j), (k, i)] -> T[k, i, j, m], pre-mu_0.
    return T2.reshape((3, n, 3, n)).transpose((2, 3, 1, 0)) / magpy.mu_0


def apply_demag(
    collection,
    susceptibility=None,
    inplace=False,
    *,
    min_log_time=None,
    style=None,
    solver="auto",
    solver_tol=1e-6,
    max_iter=1000,
    interaction="point",
):
    """
    Computes the interaction between all collection magnets and fixes their
    polarization.

    Each magnet is a cell with one uniform polarization. How strongly the
    cells act on each other follows ``interaction``, one rule for every pair:

    - ``"point"`` (default): the field of each cell taken at the centre of the
      other. Works for any cell type and orientation, and gives the results
      this package has always given.
    - ``"newell"``: the field averaged over the whole of the other cell. Needs
      every cell to be a Cuboid, all with one orientation (a mesh from
      :func:`~magpylib_material_response.meshing.mesh_Cuboid` of unrotated or
      equally rotated bodies), and raises otherwise. Its induced moment in a
      uniform field never overshoots: it rises towards the converged value as
      the mesh is refined, so a convergence study knows which side the answer
      is on. Choose it for mesh studies of cuboid models.

    At equal cell count neither rule is more accurate in general: they agree
    for cubic cells, and for flat or long cells each is closer on different
    quantities. Both are equally fast for cuboid meshes.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    susceptibility: scalar, 3-vector, or array_like, shape (n,) or (n, 3)
        Magnetic susceptibility of the cells: one scalar or one 3-vector applied to
        all cells, or one entry (scalar or 3-vector) per cell. A 3-vector gives an
        anisotropic susceptibility along the cell's own x, y and z axes, which turn
        with its orientation like its polarization. If not defined, values are
        searched at object level or parent level if needed.

    inplace: bool
        If False, applies demagnetization on a copy of the input collection and returns
        the demagnetized collection

    min_log_time:
        Minimum logging time in seconds. If computation time is below this value, step
        will not be logged. If ``None`` (default), the value set by
        :func:`magpylib_material_response.configure_logging` is used
        (``1.0`` s by default).

    style: dict
        Set collection style. If `inplace=False` only affects the copied collection

    solver: {"auto", "direct", "iterative"}
        Linear solver to use. All apply the same ``interaction`` rule and give the
        same polarization to about ``solver_tol``, so this sets only time and
        memory.
        ``"auto"`` (default) takes ``"iterative"`` when FFT products cover at
        least a tenth of the interaction matrix: the cells of each uniform grid
        (meshed cuboids of one size and orientation) among themselves, and large
        grids of one spacing between each other. There it is as fast or faster
        and needs less memory, often far less. Otherwise it takes ``"direct"``.
        Its log says which it took and why. It runs GMRES to
        ``solver_tol / 100``, so the polarization agrees with the direct solver
        to about ``solver_tol`` on a few thousand cells and ``10 * solver_tol``
        on 32,768 (measured at ``chi = 1000``). When GMRES does not converge, it
        falls back to the direct solver if the dense matrix takes at most 2 GB
        (about 5,000 cells), and raises otherwise.
        ``"direct"`` builds the dense ``Q`` matrix and solves it by LU
        decomposition, exact within floating-point precision.
        ``"iterative"`` solves matrix-free with :func:`scipy.sparse.linalg.gmres`
        and a block-Jacobi preconditioner (one 3x3 block per cell);
        uniform-grid clusters of cells
        use an FFT-accelerated :math:`O(n \\log n)` matvec. GMRES stops when the
        residual is below ``solver_tol`` relative to the right-hand side. The
        polarization then agrees with the direct solver to about ``solver_tol``
        for hard magnets; for soft ones the gap grows with the cell count, to
        160 times ``solver_tol`` on 4,096 cells and 580 times on 32,768
        (measured at ``chi = 1000``), so tighten ``solver_tol`` for high
        susceptibilities.

    solver_tol: float
        Relative residual tolerance of GMRES when ``solver="iterative"``;
        ``solver="auto"`` runs GMRES to ``solver_tol / 100``.

    max_iter: int
        Maximum number of GMRES iterations, rounded up to whole restart cycles of
        100. Non-convergence raises ``RuntimeError`` under ``solver="iterative"``.

    interaction: {"point", "newell"}
        How cells act on each other, see above.

    Returns
    -------
    magpylib.Collection or None
        The demagnetized copy of `collection` if `inplace=False`, else None.

    Notes
    -----
    Each cell responds to a uniform applied field set by its `H_ext` attribute (or
    a parent's), given as mu_0 * H in tesla, and to the field of the current
    sources in `collection` at its centroid.
    """
    if solver not in SOLVERS:
        msg = f"solver must be one of {SOLVERS}; got {solver!r}"
        raise ValueError(msg)
    if interaction not in INTERACTIONS:
        msg = f"interaction must be one of {INTERACTIONS}; got {interaction!r}"
        raise ValueError(msg)
    # The copy has no parent, so material attributes a cell inherits from above
    # `collection` are read from the original.
    original = collection
    if not inplace:
        collection = collection.copy()
    if style is not None:
        collection.style = style
    srcs = collection.sources_all
    src_with_paths = [src for src in srcs if _has_path(src)]
    if src_with_paths:
        msg = (
            f"{len(src_with_paths)} objects with paths, found. Demagnetization of "
            "objects with paths (in position, orientation or any other property, "
            "e.g. dimension, polarization or current) is not yet supported"
        )
        raise ValueError(msg)
    magnets_list = [src for src in srcs if isinstance(src, BaseMagnet)]
    originals = [src for src in original.sources_all if isinstance(src, BaseMagnet)]
    currents_list = [src for src in srcs if isinstance(src, BaseCurrent)]
    others_list = [
        src
        for src in srcs
        if not isinstance(src, BaseMagnet | BaseCurrent | magpy.Sensor)
    ]
    if others_list:
        counts_others = Counter(s.__class__.__name__ for s in others_list)
        counts_str = ", ".join(
            f"{count} {name}" for name, count in counts_others.items()
        )
        msg = (
            "Only Magnet and Current sources supported. "
            f"Incompatible objects found: {counts_str}"
        )
        raise TypeError(msg)
    n = len(magnets_list)
    if n == 0:
        msg = "Apply_demag input collection contains no magnet sources."
        raise ValueError(msg)
    counts = Counter(s.__class__.__name__ for s in magnets_list)
    inplace_str = f"""{" (inplace)" if inplace else ""}"""
    lbl = collection.style.label
    coll_str = lbl or str(collection)
    counts_str = ", ".join(f"{count} {name}" for name, count in counts.items())
    demag_msg = (
        f"Demagnetization{inplace_str} of {coll_str} with {n} cells ({counts_str})"
    )
    with timelog(demag_msg, min_log_time=min_log_time):
        # set up mr
        pol_magnets = [
            src.orientation.apply(
                (0.0, 0.0, 0.0) if src.polarization is None else (src.polarization)
            )
            for src in magnets_list
        ]  # ROTATION CHECK
        pol_magnets = np.reshape(
            pol_magnets, (3 * n, 1), order="F"
        )  # shape ii = x1, ... xn, y1, ... yn, z1, ... zn

        # set up S: block-diagonal, one 3x3 tensor per cell in the global
        # frame, R diag(chi) R^T with chi along the cell's own axes (#50).
        sus = get_susceptibilities(originals, susceptibility)
        chi = _cell_susceptibilities(sus, [src.orientation for src in magnets_list])

        # set up H_ext
        H_ext = get_H_ext(*originals)
        H_ext = np.array(H_ext)
        if len(H_ext) != n:
            msg = "Apply_demag input collection and H_ext must have same length."
            raise ValueError(msg)
        H_ext = np.reshape(H_ext, (3 * n, 1), order="F")

        pol_total = pol_magnets

        if currents_list:
            with timelog(
                "Add current sources contributions", min_log_time=min_log_time
            ):
                pos = np.array([src.centroid for src in magnets_list])
                pol_currents = magpy.getB(currents_list, pos, sumup=True)
                pol_currents = np.reshape(pol_currents, (3 * n, 1), order="F")
                pol_total = pol_total + _apply_susceptibility(chi, pol_currents)

        rhs = pol_total + _apply_susceptibility(chi, H_ext)  # shape (3n, 1)

        # ── demag operator under the chosen rule ────────────────────────────
        _check_interaction(interaction, magnets_list)
        positions, clusters = _analyze_collection(magnets_list, interaction)
        auto = solver == "auto"
        if auto:
            solver, share = _auto_solver(clusters)
            logger.info(
                "solver='auto' chose {solver}: FFT covers {share:.0%} of the "
                "interactions",
                share=share,
                solver=solver,
            )
        T = None
        if solver == "direct":
            T = _assemble_T_dense(
                magnets_list,
                positions,
                clusters,
                interaction,
                min_log_time=min_log_time,
            )

        with timelog("Solving of linear system", min_log_time=min_log_time):
            if solver == "direct":
                pol_new = _solve_direct(T, chi, rhs)
            else:
                tol = solver_tol * AUTO_TOL_FACTOR if auto else solver_tol
                ops = _build_operator(
                    magnets_list,
                    positions,
                    clusters,
                    chi,
                    tol,
                    min_log_time,
                    interaction,
                )
                matvec = _build_matvec(n, chi, ops)
                T_self = _self_blocks(n, clusters, interaction, ops)
                gmres_failure = None
                try:
                    pol_new = _solve_iterative(
                        n=n,
                        chi=chi,
                        rhs=rhs.ravel(),
                        rhs_shape=rhs.shape,
                        matvec=matvec,
                        T_self=T_self,
                        solver_tol=tol,
                        max_iter=max_iter,
                    )
                except RuntimeError as err:
                    if not auto:
                        raise
                    if 72 * n * n > DENSE_BLOCK_MAX_BYTES:
                        msg = (
                            f"{err} solver='auto' ran GMRES to solver_tol / "
                            f"{1 / AUTO_TOL_FACTOR:g}, and the direct solver would "
                            f"need {72 * n * n / 1e9:.1f} GB."
                        )
                        raise RuntimeError(msg) from err
                    # Leave the except block first: its traceback holds the
                    # GMRES frames, and with them the operator.
                    gmres_failure = str(err)
                if gmres_failure is not None:
                    logger.warning(
                        "solver='auto': {err} Falling back to the direct solver.",
                        err=gmres_failure,
                    )
                    ops = matvec = None  # free the operator before the dense T
                    T = _assemble_T_dense(
                        magnets_list,
                        positions,
                        clusters,
                        interaction,
                        min_log_time=min_log_time,
                    )
                    pol_new = _solve_direct(T, chi, rhs)

        pol_new = np.reshape(pol_new, (n, 3), order="F")

        for src, pol in zip(magnets_list, pol_new, strict=True):
            src.polarization = src.orientation.inv().apply(pol)  # ROTATION CHECK

    if not inplace:
        return collection
    return None
