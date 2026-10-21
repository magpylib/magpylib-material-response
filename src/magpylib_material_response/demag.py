"""Demagnetization / material-response solver.

The self-consistent polarization of ``n`` interacting cells is the solution
of ``(I - S T) J = J0 + S B_ext`` where ``S`` is the (diagonal)
susceptibility matrix and ``T`` the demag interaction operator. The
``H_ext`` object attribute is interpreted as the applied flux density
``B_ext`` in Tesla (B-field convention, consistent with polarization).

Every entry of ``T`` follows one *interaction rule*, chosen per solve with
``interaction`` and applied to every pair of cells, never mixed:

- ``"point"`` (default): the field of the source cell at the observer cell's
  centre (point matching, Chadebec 2006). Any cell type and orientation; the
  rule this package has always used.
- ``"newell"``: the field of the source cell averaged over the observer cell,
  the volume-averaged Newell tensor (:mod:`magpylib_material_response.newell`).
  This is the Galerkin form of the problem: the induced moment of a body in a
  uniform field is then a lower bound that rises under refinement, so a mesh
  study knows which side the converged value is on. It needs every cell to be
  a cuboid, all with one orientation, and raises otherwise.

At equal cell count neither rule is more accurate in general: they agree for
cubes, and for flat cells each wins on different quantities. A mix of the two
would jump when a body turns by any angle that makes its cells non-parallel,
and would lose the lower bound, so there is none.

Two evaluation strategies apply the chosen rule, so ``solver="direct"`` and
``solver="iterative"`` solve the same system:

- ``"direct"``   materialises the dense ``T`` and calls ``np.linalg.solve``;
- ``"iterative"`` applies ``T`` matrix-free in GMRES — FFT convolution for
  uniform-grid clusters (:mod:`magpylib_material_response.demag_fft`), dense
  or row-sum-bounded sparse blocks for everything else.

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

#: A pair block with more dense entries than this is built in observer
#: chunks and stored sparse (with a rigorous row-sum error budget) instead
#: of dense. 2e7 float64 entries = 160 MB; dense blocks additionally allow
#: deriving the reverse block by reciprocity (halving the build cost).
DENSE_BLOCK_MAX_ENTRIES = 20_000_000
# A block above DENSE_BLOCK_MAX_ENTRIES is sparsified only when its first chunk
# shows that at least SPARSIFY_MIN_DROP of the entries can go; at high
# susceptibility the row budget drops almost nothing, and a CSR holding most of
# a block is slower and larger than the dense array. Above DENSE_BLOCK_MAX_BYTES
# it is sparsified regardless.
SPARSIFY_MIN_DROP = 0.5
DENSE_BLOCK_MAX_BYTES = 2_000_000_000

#: Every block is evaluated in observer chunks covering at most this many
#: (observer, source) pairs per field evaluation. Memory grows with pairs, not
#: with output entries: about 0.8 kB per pair for cuboids and about 3 kB for
#: tetrahedra in magpylib's kernels, so one chunk stays below about 0.6 GB.
KERNEL_PAIRS_PER_CALL = 200_000


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


def _check_interaction(interaction, magnets_list, atol=QUAT_ATOL):
    """Refuse an unknown rule, and the Newell rule on cells it cannot cover."""
    if interaction not in INTERACTIONS:
        msg = f"interaction must be one of {INTERACTIONS}; got {interaction!r}"
        raise ValueError(msg)
    if interaction == "point":
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
    return positions, clusters


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


def _row_sparsify(block, row_budget):
    """Per-row triplets keeping enough entries that the dropped ``|.|`` sum
    stays below ``row_budget`` in every row (exact, entry-based bound)."""
    A = np.abs(block)
    order = np.argsort(A, axis=1)
    srt = np.take_along_axis(A, order, axis=1)
    csum = np.cumsum(srt, axis=1)
    n_drop = (csum <= row_budget).sum(axis=1)  # smallest-k droppable per row
    rank = np.argsort(order, axis=1)  # rank of each entry in its row
    keep = rank >= n_drop[:, None]
    rows, cols = np.nonzero(keep)
    return rows, cols, block[rows, cols]


def _grid_pair_table(cl_a, cl_b, interaction, rtol=1e-6):
    """T between two grid clusters of one spacing and orientation, tabulated
    over the index differences, in the grids' local frame.

    Observer ``i`` at grid index ``p`` and source ``j`` at ``q`` are
    ``origin_a - origin_b + spacing * (p - q)`` apart, so the block has at most
    ``prod(Na + Nb - 1)`` distinct entries: ``table[p - q + Nb - 1]`` (per axis)
    is the (3, 3) tensor of pair ``(i, j)``. Returns ``(table, index_a, index_b)``
    with ``table`` of shape ``(*(Na + Nb - 1), 3, 3)`` and the grid indices of
    the clusters' members, or ``None`` when the clusters do not share spacing
    and orientation. A relative tolerance only: at nanometre scale an absolute
    one would call different spacings equal.
    """
    if not (
        cl_a["kind"] == "grid"
        and cl_b["kind"] == "grid"
        and _same_orientation(cl_a, cl_b)
        and np.allclose(cl_a["spacing"], cl_b["spacing"], rtol=rtol, atol=0)
    ):
        return None
    sa, sb = np.array(cl_a["shape"]), np.array(cl_b["shape"])
    spacing = np.array(cl_a["spacing"])

    def _index3(cl, shape):
        n = cl["indices"].size
        flat = np.empty(n, dtype=np.int64)
        flat[cl["order"]] = np.arange(n)  # cluster member -> grid-flat index
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


def _droppable_fraction(block, row_budget):
    """Fraction of a block's entries that the per-row budget allows to drop."""
    csum = np.cumsum(np.sort(np.abs(block), axis=1), axis=1)
    return (csum <= row_budget).sum() / block.size


def _pair_block(cl_a, cl_b, magnets_list, positions, interaction, row_budget=None):
    """Build the T block for a cluster pair (observers ``a``, sources ``b``)
    under the solve's interaction rule.

    Returns a dense ndarray when the block is small, when ``row_budget`` is
    None, or when the budget would keep most of the entries anyway (the case
    at high susceptibility); else a CSR matrix with per-row dropped sums below
    ``row_budget``.
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
        return _point_block_T([magnets_list[i] for i in ib], pos_a[sl])

    def _chunks():
        # Observer chunks bounded by kernel pairs, which is what costs memory;
        # a single call over all n_a * n_b pairs of 5,000 tetrahedra needs ~97 GB.
        chunk = max(1, KERNEL_PAIRS_PER_CALL // max(n_b, 1))
        for start in range(0, n_a, chunk):
            stop = min(start + chunk, n_a)
            yield start, stop, _dense_chunk(slice(start, stop))

    chunks = _chunks()
    first = next(chunks, None)
    # Sparsify only when the first chunk shows it pays: at high susceptibility
    # the row budget keeps nearly every entry.
    if (
        sparsify
        and first is not None
        and 72 * n_a * n_b <= DENSE_BLOCK_MAX_BYTES
        and _droppable_fraction(first[2], row_budget) < SPARSIFY_MIN_DROP
    ):
        sparsify = False
    all_chunks = itertools.chain([first], chunks) if first is not None else iter(())

    if not sparsify:
        # Rows of a block are m-major over its observers: (3, n_a, 3 n_b).
        block = np.empty((3, n_a, 3 * n_b))
        for start, stop, part in all_chunks:
            block[:, start:stop] = part.reshape(3, stop - start, 3 * n_b)
        return block.reshape(3 * n_a, 3 * n_b)

    rows_all, cols_all, vals_all = [], [], []
    for start, stop, block in all_chunks:
        r, c, v = _row_sparsify(block, row_budget)
        # Local rows are m-major over the chunk; map to m-major over n_a.
        n_ch = stop - start
        m, i_loc = r // n_ch, r % n_ch
        rows_all.append(m * n_a + start + i_loc)
        cols_all.append(c)
        vals_all.append(v)
    return sp.csr_matrix(
        (
            np.concatenate(vals_all),
            (np.concatenate(rows_all), np.concatenate(cols_all)),
        ),
        shape=(3 * n_a, 3 * n_b),
    )


def _assemble_T_dense(
    magnets_list, positions, clusters, interaction, min_log_time=None
):
    """Assemble the full dense (3n, 3n) T in Fortran layout, global frame.

    Reverse cross-blocks are derived from the forward block by reciprocity
    wherever it holds exactly (see :func:`_reciprocity_factor`) instead of a
    second evaluation, the same shortcut the iterative operator build uses.
    """
    n = len(magnets_list)
    T = np.zeros((3 * n, 3 * n))

    def _flat(cl):
        return (np.arange(3)[:, None] * n + cl["indices"][None, :]).ravel()

    with timelog("Demagnetization tensor assembly", min_log_time=min_log_time):
        for idx_a, cl_a in enumerate(clusters):
            rows = _flat(cl_a)
            T[np.ix_(rows, rows)] = _pair_block(
                cl_a, cl_a, magnets_list, positions, interaction
            )
            for cl_b in clusters[idx_a + 1 :]:
                cols = _flat(cl_b)
                M_ab = _pair_block(cl_a, cl_b, magnets_list, positions, interaction)
                T[np.ix_(rows, cols)] = M_ab
                factor = _reciprocity_factor(cl_a, cl_b, interaction)
                if factor is not None:
                    M_ba = factor * M_ab.T
                else:
                    M_ba = _pair_block(cl_b, cl_a, magnets_list, positions, interaction)
                T[np.ix_(cols, rows)] = M_ba
    return T


# ══════════════════════════════════════════════════════════════════════════
# Matrix-free operator for the iterative solver
# ══════════════════════════════════════════════════════════════════════════


def _fft_apply(op, v_flat):
    """Apply the grid self-block T to component-major ``(3n_g,)`` global-frame
    polarizations; returns the same layout. Rotation handled internally."""
    Nx, Ny, Nz = op["shape"]
    rot_mat = op["rot_mat"]
    M = v_flat.reshape(3, -1).T  # (n_g, 3) global
    if rot_mat is not None:
        M = M @ rot_mat  # to local frame
    M_grid = M[op["order"]].reshape((Nx, Ny, Nz, 3))
    H_grid = demag_fft_matvec(M_grid, op["kernel_fft"], (Nx, Ny, Nz))
    H = np.empty_like(M)
    H[op["order"]] = H_grid.reshape((Nx * Ny * Nz, 3))
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
        "order_a": cl_a["order"],
        "order_b": cl_b["order"],
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
    Mp[: sb[0], : sb[1], : sb[2]] = M[op["order_b"]].reshape((*sb, 3))
    M_fft = np.fft.rfftn(Mp, axes=(0, 1, 2))  # (Lx, Ly, Lz//2+1, 3)
    H_fft = np.einsum("...mk,...k->...m", op["kernel_fft"], M_fft)
    H_grid = np.fft.irfftn(H_fft, s=L, axes=(0, 1, 2))
    sa = op["shape_a"]
    H = np.empty((np.prod(sa), 3))
    H[op["order_a"]] = H_grid[: sa[0], : sa[1], : sa[2]].reshape(-1, 3)
    if rot_mat is not None:
        H = H @ rot_mat.T  # back to global frame
    return H.T.ravel()


def _build_operator(
    magnets_list, positions, clusters, chi, solver_tol, min_log_time, interaction
):
    """Build the list of block operators representing T for the matvec.

    Self-blocks of grid clusters are FFT convolutions (exact); every other
    block is dense when small, else sparsified with a per-row error budget
    chosen so the total operator perturbation stays below
    ``0.1 * solver_tol``:  ``|S (T - T~)|_inf <= chi_eff * K * row_budget``,
    with ``chi_eff`` the largest row sum of ``|S|`` (at least 1).
    """
    K = len(clusters)
    chi_eff = max(1.0, float(np.abs(chi).sum(axis=2).max()) if len(chi) else 1.0)
    row_budget = 0.1 * solver_tol / (chi_eff * max(K, 1))

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
                        "order": cl_a["order"],
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
                        "ia": cl_a["indices"],
                        "ib": cl_a["indices"],
                        "M": _pair_block(
                            cl_a, cl_a, magnets_list, positions, interaction, row_budget
                        ),
                    }
                )
            for idx_b, cl_b in enumerate(clusters):
                if idx_b <= idx_a:
                    continue
                cross = _fft_cross_op(cl_a, cl_b, interaction)
                if cross is not None:
                    ops.append(cross)
                    ops.append(_fft_cross_op(cl_b, cl_a, interaction))
                    continue
                M_ab = _pair_block(
                    cl_a, cl_b, magnets_list, positions, interaction, row_budget
                )
                ops.append(
                    {
                        "kind": "mat",
                        "ia": cl_a["indices"],
                        "ib": cl_b["indices"],
                        "M": M_ab,
                    }
                )
                factor = _reciprocity_factor(cl_a, cl_b, interaction)
                if factor is not None and not sp.issparse(M_ab):
                    M_ba = factor * M_ab.T
                else:
                    M_ba = _pair_block(
                        cl_a=cl_b,
                        cl_b=cl_a,
                        magnets_list=magnets_list,
                        positions=positions,
                        interaction=interaction,
                        row_budget=row_budget,
                    )
                ops.append(
                    {
                        "kind": "mat",
                        "ia": cl_b["indices"],
                        "ib": cl_a["indices"],
                        "M": M_ba,
                    }
                )
    return ops


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
            Tv[:, op["ia"]] += h.reshape(3, -1)
        return v_flat - _apply_susceptibility(chi, Tv.ravel())

    return matvec


def _build_dense_matvec(chi, T):
    """Return matvec ``v -> (I - S T) @ v`` for dense T."""

    def matvec(v_flat):
        Tv = T @ v_flat
        return v_flat - _apply_susceptibility(chi, Tv)

    return matvec


def _self_demag_diagonal(n, clusters, interaction):
    """Fortran-flat (3n,) diagonal of the cells' own demag factors under the
    solve's rule, rotated to the global frame per cluster (zero for generic
    cells): the Jacobi preconditioner."""
    Nself_flat = np.zeros(3 * n)
    for cl in clusters:
        if cl["kind"] == "generic":
            continue
        if interaction == "newell":
            Ns = self_demag_factors(cl["dim"])  # (3,) local frame
        else:
            Ns = np.diag(point_block(np.zeros(3), cl["dim"]))
        if cl["is_identity"]:
            diag_m = Ns
        else:
            rot_mat = cl["r_common"].as_matrix()
            diag_m = (rot_mat**2) @ Ns  # diag of R diag(Ns) R^T
        for m in range(3):
            Nself_flat[m * n + cl["indices"]] = diag_m[m]
    return Nself_flat


#: GMRES restart length: iterations per cycle before the Krylov space is rebuilt.
GMRES_RESTART = 20


def _solve_iterative(
    *, n, chi, rhs, rhs_shape, matvec, Nself_flat, solver_tol, max_iter
):
    """GMRES solve of (I - S T) x = rhs with block-Jacobi preconditioning.

    Raises ``RuntimeError`` when GMRES does not reach ``solver_tol`` — a
    partially-converged result would silently be wrong.
    """
    Q_op = LinearOperator((3 * n, 3 * n), matvec=matvec, dtype=float)

    # Block-Jacobi preconditioner: each cell's 3x3 block of Q from its own
    # demag factors, I + chi_i diag(N_self,i) (identity for generic cells),
    # so a rotated anisotropic susceptibility is preconditioned exactly.
    N_self = Nself_flat.reshape(3, n).T  # (n, 3)
    blocks = np.eye(3) + chi * N_self[:, None, :]
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
    pairs_matching=False,
    split=False,
    max_dist=0,
    min_log_time=None,
    interaction="point",
):
    """
    Compute the demagnetization tensor T for n sources.

    Every pair follows ``interaction``, the same rule as in
    :func:`apply_demag`: the field at each cell's centre (``"point"``, the
    default) or averaged over each cell (``"newell"``). The legacy options
    ``pairs_matching``, ``split`` and ``max_dist`` take the historical
    point-matching path, which gives the same tensor up to the far-field
    expansion of :func:`~magpylib_material_response.newell.point_block`.

    Parameters
    ----------
    src_list: sequence of magpylib magnet sources
        Each source is treated as a magnetic cell.

    pairs_matching: bool
        If True, equivalent pair of interactions are identified and unique pairs are
        calculated only once and copied to duplicates. Implies point matching.

    split: int
        Number of times the sources list is split before getH calculation ind demag
        tensor calculation. Implies point matching.

    max_dist: float
        Leaves out the interaction of two cells whose centroids lie `max_dist` or more
        times the larger side of the two cells apart. If `max_dist=0`, all
        interactions are calculated.

    min_log_time:
        Minimum logging time in seconds. If computation time is below this value, step
        will not be logged.

    interaction: {"point", "newell"}
        Interaction rule, see :func:`apply_demag`. ``"newell"`` needs every
        source to be a Cuboid, all with one orientation, and does not combine
        with the legacy options.

    Returns
    -------
    Demagnetization tensor: ndarray, shape (3,n,n,3), pre-``mu_0``:
        ``T[k, i, j, m] = -N_mk(pos_j - pos_i) / mu_0``

    Notes
    -----
    Point-matched evaluation temporarily assigns unit polarizations to the
    sources. Sources whose ``polarization`` is ``None`` keep the last unit
    polarization afterwards (magpylib does not allow assigning ``None``
    back); all other sources are restored. ``apply_demag`` is unaffected —
    it operates on a copy and overwrites every polarization with the
    solution.

    TODO: allow multi-point matching
    """
    nof_src = len(src_list)

    splitting = split > 1
    if max_dist < 0:
        msg = f"max_dist must be 0 (no filtering) or positive, got {max_dist}."
        raise ValueError(msg)
    if pairs_matching and splitting:
        msg = "Pairs matching does not support splitting"
        raise ValueError(msg)
    if max_dist != 0 and (pairs_matching or splitting):
        msg = "max_dist does not combine with pairs_matching or split."
        raise ValueError(msg)

    _check_interaction(interaction, list(src_list))
    legacy = pairs_matching or max_dist != 0 or splitting
    if legacy and interaction == "newell":
        msg = (
            "pairs_matching, max_dist and split take the legacy point-matching "
            "path and do not combine with interaction='newell'."
        )
        raise ValueError(msg)
    if not legacy:
        # Assembly under the chosen rule, returned in the legacy
        # (3, n, n, 3) pre-mu_0 layout.
        positions, clusters = _analyze_collection(list(src_list), interaction)
        T2 = _assemble_T_dense(
            list(src_list), positions, clusters, interaction, min_log_time=min_log_time
        )
        n = nof_src
        # T2[(m, j), (k, i)] -> legacy T[k, i, j, m], pre-mu_0.
        return T2.reshape((3, n, 3, n)).transpose((2, 3, 1, 0)) / magpy.mu_0

    mask_inds = None
    getH_params = {}
    if max_dist != 0:
        mask_inds, getH_params, pos0, rot0 = filter_distance(
            src_list, max_dist, return_params=True, return_base_geo=True
        )
    elif pairs_matching:
        getH_params, mask_inds, unique_inv_inds, pos0, rot0 = match_pairs(src_list)
    else:
        pos0 = np.array([src.centroid for src in src_list])
        rotQ0 = [src.orientation.as_quat() for src in src_list]
        rot0 = R.from_quat(rotQ0)

    saved_pols = [src.polarization for src in src_list]
    try:
        H_point = []
        for unit_pol in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            pol_all = rot0.inv().apply(unit_pol)
            # point matching field and demag tensor
            with timelog(f"getH with unit_pol={unit_pol}", min_log_time=min_log_time):
                if pairs_matching or max_dist != 0:
                    polarization = np.repeat(pol_all, len(src_list), axis=0)
                    if mask_inds is not None:
                        polarization = polarization[mask_inds]
                    # magpy.func replaces the deprecated string-based
                    # functional interface (magpy.getH("Cuboid", ...))
                    H_unique = magpy.func.cuboid_field(
                        "H",
                        observers=getH_params["observers"],
                        dimensions=getH_params["dimension"],
                        polarizations=polarization,
                        positions=getH_params["position"],
                        orientations=getH_params["orientation"],
                        squeeze=False,
                    )
                    if max_dist != 0:
                        H_temp = np.zeros((len(src_list) ** 2, 3))
                        H_temp[mask_inds] = H_unique
                        H_unit_pol = H_temp
                    else:
                        H_unit_pol = H_unique[unique_inv_inds]
                else:
                    for src, pol in zip(src_list, pol_all, strict=False):
                        src.polarization = pol
                    src_list_split = np.array_split(src_list, split)
                    with logger.contextualize(
                        task="Splitting field calculation", split=split
                    ):
                        H_unit_pol = []
                        for split_ind, src_list_subset in enumerate(src_list_split):
                            logger.info(
                                "Sources subset {subset_num}/{total_subsets}",
                                subset_num=split_ind + 1,
                                total_subsets=len(src_list_split),
                            )
                            if src_list_subset.size > 0:
                                # getH squeezes the source axis of a one-source
                                # chunk
                                H_sub = np.reshape(
                                    magpy.getH(src_list_subset.tolist(), pos0),
                                    (len(src_list_subset), len(pos0), 3),
                                )
                                H_unit_pol.append(H_sub)
                        H_unit_pol = np.concatenate(H_unit_pol, axis=0)
                H_point.append(H_unit_pol)  # shape (n_cells, n_pos, 3_xyz)
    finally:
        for src, pol in zip(src_list, saved_pols, strict=False):
            if pol is not None:
                src.polarization = pol

    # shape (3_unit_pol, n_cells, n_pos, 3_xyz)
    return np.array(H_point).reshape((3, nof_src, nof_src, 3))


def filter_distance(
    src_list,
    max_dist,
    min_log_time=None,
    return_params=False,
    return_base_geo=False,
):
    """filter indices by distance parameter"""
    with timelog("Distance filter", min_log_time=min_log_time):
        all_cuboids = all(isinstance(src, Cuboid) for src in src_list)
        if not all_cuboids:
            msg = "filter_distance only implemented if all sources are Cuboids"
            raise ValueError(msg)
        pos0 = np.array([src.centroid for src in src_list])
        rotQ0 = [src.orientation.as_quat() for src in src_list]
        rot0 = R.from_quat(rotQ0)
        dim0 = [src.dimension for src in src_list]

        pos2 = np.tile(pos0, (len(pos0), 1)) - np.repeat(pos0, len(pos0), axis=0)
        dist2 = np.linalg.norm(pos2, axis=1)
        dim2 = np.tile(dim0, (len(dim0), 1)), np.repeat(dim0, len(dim0), axis=0)
        maxdim2 = np.concatenate(dim2, axis=1).max(axis=1)
        mask = (dist2 / maxdim2) < max_dist
        if return_params:
            params = {
                "observers": np.tile(pos0, (len(src_list), 1))[mask],
                "position": np.repeat(pos0, len(src_list), axis=0)[mask],
                "orientation": R.from_quat(np.repeat(rotQ0, len(src_list), axis=0))[
                    mask
                ],
                "dimension": np.repeat(dim0, len(src_list), axis=0)[mask],
            }
        dsf = sum(mask) / len(mask) * 100
    if dsf == 0:
        logger.warning(
            "No interaction pairs left after distance factor filtering",
            percentage=f"{dsf:.2f}%",
        )
    else:
        logger.info(
            "Interaction pairs left after distance factor filtering",
            percentage=f"{dsf:.2f}%",
        )
    out = [mask]
    if return_params:
        out.append(params)
    if return_base_geo:
        out.extend([pos0, rot0])
    if len(out) == 1:
        return out[0]
    return tuple(out)


def match_pairs(src_list, min_log_time=None):
    """match all pairs of sources from `src_list`"""
    with timelog("Pairs matching", min_log_time=min_log_time):
        all_cuboids = all(isinstance(src, Cuboid) for src in src_list)
        if not all_cuboids:
            msg = "Pairs matching only implemented if all sources are Cuboids"
            raise ValueError(msg)
        pos0 = np.array([src.centroid for src in src_list])
        rotQ0 = [src.orientation.as_quat() for src in src_list]
        rot0 = R.from_quat(rotQ0)
        dim0 = [src.dimension for src in src_list]
        len_src = len(src_list)
        num_of_pairs = len_src**2
        with logger.contextualize(task="Match interactions pairs"):
            logger.debug("Computing position differences")
            pos2 = np.tile(pos0, (len_src, 1)) - np.repeat(pos0, len_src, axis=0)
            logger.debug("Computing orientation differences")
            rotQ2a = np.tile(rotQ0, (len_src, 1)).reshape((num_of_pairs, -1))
            rotQ2b = np.repeat(rotQ0, len_src, axis=0).reshape((num_of_pairs, -1))
            # The reused field depends on the source cell's own dimensions, not
            # on a difference with the observer's (2 -> 4 and 3 -> 5 collide).
            dim_src = np.repeat(dim0, len_src, axis=0)
            # Round relative to the geometry's size: an absolute 1e-8 merges
            # every pair of a mesh with cells below 1e-8 m.
            geo_scale = max(
                float(np.max(np.abs(pos2), initial=0.0)),
                float(np.max(np.abs(dim0))),
                np.finfo(float).tiny,
            )
            prop = np.concatenate(
                [pos2 / geo_scale, rotQ2a, rotQ2b, dim_src / geo_scale], axis=1
            )
            prop = (prop + 1e-9).round(8)
            logger.debug("Finding unique interaction pairs")
            _, unique_inds, unique_inv_inds = np.unique(
                prop, return_index=True, return_inverse=True, axis=0
            )
            perc = len(unique_inds) / len(unique_inv_inds) * 100
            logger.info(
                "Interaction pairs left after pair matching filtering",
                percentage=f"{perc:.2f}%",
            )

        params = {
            "observers": np.tile(pos0, (len(src_list), 1))[unique_inds],
            "position": np.repeat(pos0, len(src_list), axis=0)[unique_inds],
            "orientation": R.from_quat(rotQ2b)[unique_inds],
            "dimension": dim_src[unique_inds],
        }
    return params, unique_inds, unique_inv_inds, pos0, rot0


def apply_demag(
    collection,
    susceptibility=None,
    inplace=False,
    pairs_matching=False,
    max_dist=0,
    split=1,
    min_log_time=None,
    style=None,
    solver="direct",
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

    pairs_matching: bool
        If True, equivalent pair of interactions are identified and unique pairs are
        calculated only once and copied to duplicates. This parameter is not compatible
        with `max_dist` or `split` and applies only cuboid cells. Forces the legacy
        point-matched tensor for all pairs.

    max_dist: float
        Leaves out the interaction of two cells whose centroids lie `max_dist` or more
        times the larger side of the two cells apart, an approximation. If
        `max_dist=0`, all interactions are calculated. This parameter is not
        compatible with `pairs_matching` or `split` and applies only cuboid cells. Forces
        the legacy point-matched tensor for all pairs.

    split: int
        Number of times the sources list is split before getH calculation ind demag
        tensor calculation. This parameter is not compatible with `pairs_matching` or
        `max_dist`. Forces the legacy point-matched tensor for all pairs.

    min_log_time:
        Minimum logging time in seconds. If computation time is below this value, step
        will not be logged. If ``None`` (default), the value set by
        :func:`magpylib_material_response.configure_logging` is used
        (``1.0`` s by default).

    style: dict
        Set collection style. If `inplace=False` only affects the copied collection

    solver: {"direct", "iterative"}
        Linear solver to use. Both apply the same ``interaction`` rule.
        ``"direct"`` (default) builds the dense ``Q`` matrix and calls
        :func:`numpy.linalg.solve` — exact within floating-point precision.
        ``"iterative"`` solves matrix-free with :func:`scipy.sparse.linalg.gmres`
        and a diagonal Jacobi preconditioner; uniform-grid clusters of cells
        use an FFT-accelerated :math:`O(n \\log n)` matvec. GMRES stops when the
        residual is below ``solver_tol`` relative to the right-hand side. The
        polarization then agrees with the direct solver to about ``solver_tol``
        for hard magnets; for soft ones the gap can grow up to about
        ``chi * solver_tol / 3`` (35 times ``solver_tol`` measured at
        ``chi = 1000``), so tighten ``solver_tol`` for high susceptibilities.

    solver_tol: float
        Relative residual tolerance passed to GMRES when ``solver="iterative"``.

    max_iter: int
        Maximum number of GMRES iterations when ``solver="iterative"``, rounded up
        to whole restart cycles of 20. Non-convergence raises ``RuntimeError``.

    interaction: {"point", "newell"}
        How cells act on each other, see above. ``"newell"`` does not combine
        with ``pairs_matching``, ``max_dist`` or ``split``.

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
    if solver not in ("direct", "iterative"):
        msg = f"solver must be 'direct' or 'iterative'; got {solver!r}"
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

        # ── demag operator: legacy point-matching or the chosen rule ────────
        _check_interaction(interaction, magnets_list)
        no_split = split is False or split == 1
        legacy = pairs_matching or max_dist != 0 or not no_split
        if legacy and interaction == "newell":
            msg = (
                "pairs_matching, max_dist and split take the legacy point-matching "
                "path and do not combine with interaction='newell'."
            )
            raise ValueError(msg)
        if legacy:
            with timelog(
                "Demagnetization tensor calculation", min_log_time=min_log_time
            ):
                T = demag_tensor(
                    magnets_list,
                    split=split,
                    pairs_matching=pairs_matching,
                    max_dist=max_dist,
                )
                T *= magpy.mu_0
                T = T.swapaxes(2, 3).reshape((3 * n, 3 * n)).T  # shape ii, jj
            positions, clusters = None, None
        else:
            positions, clusters = _analyze_collection(magnets_list, interaction)
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
                # In-place assembly of Q = I - S T reuses T's buffer (T is
                # not needed afterwards), cutting peak memory ~2-3x compared
                # to np.eye(3n) - S @ T which materializes three matrices.
                Q = _subtract_susceptibility_product(np.ascontiguousarray(T), chi)
                pol_new = scipy.linalg.solve(Q, rhs, overwrite_a=True)
            else:
                if T is not None:
                    matvec = _build_dense_matvec(chi, T)
                    Nself_flat = np.zeros(3 * n)
                else:
                    ops = _build_operator(
                        magnets_list,
                        positions,
                        clusters,
                        chi,
                        solver_tol,
                        min_log_time,
                        interaction,
                    )
                    matvec = _build_matvec(n, chi, ops)
                    Nself_flat = _self_demag_diagonal(n, clusters, interaction)
                pol_new = _solve_iterative(
                    n=n,
                    chi=chi,
                    rhs=rhs.ravel(),
                    rhs_shape=rhs.shape,
                    matvec=matvec,
                    Nself_flat=Nself_flat,
                    solver_tol=solver_tol,
                    max_iter=max_iter,
                )

        pol_new = np.reshape(pol_new, (n, 3), order="F")

        for src, pol in zip(magnets_list, pol_new, strict=True):
            src.polarization = src.orientation.inv().apply(pol)  # ROTATION CHECK

    if not inplace:
        return collection
    return None
