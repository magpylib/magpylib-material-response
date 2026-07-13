"""Demagnetization / material-response solver.

The self-consistent polarization of ``n`` interacting cells is the solution
of ``(I - S T) J = J0 + S mu_0 H_ext`` where ``S`` is the (diagonal)
susceptibility matrix and ``T`` the demag interaction operator.

A single *pair rule* defines every entry of ``T`` (single source of truth):

- both cells are Cuboids sharing one orientation → analytical volume-averaged
  Newell tensor (:mod:`magpylib_material_response.newell`), generalized to
  different cell sizes;
- anything else → point-matched ``magpy.getH`` (field evaluated at the
  observer barycentre, see Chadebec 2006).

Two evaluation strategies share that rule, so ``solver="direct"`` and
``solver="iterative"`` agree to solver tolerance by construction:

- ``"direct"``   materialises the dense ``T`` and calls ``np.linalg.solve``;
- ``"iterative"`` applies ``T`` matrix-free in GMRES — FFT convolution for
  uniform-grid clusters (:mod:`magpylib_material_response.demag_fft`), dense
  or row-sum-bounded sparse blocks for everything else.

All computations run in the global frame; per-cluster rotations are handled
inside the block builders, so anisotropic susceptibility (diagonal in the
global frame) is always applied to the correct components.
"""

from __future__ import annotations

from collections import Counter

import magpylib as magpy
import numpy as np
import scipy.sparse as sp
from loguru import logger
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent, BaseMagnet
from magpylib.magnet import Cuboid
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.demag_fft import (
    QUAT_ATOL,
    analyze_collection,
    build_fft_kernel,
    canonical_quats,
    demag_fft_matvec,
)
from magpylib_material_response.newell import (
    demag_block_general,
    self_demag_factors,
)
from magpylib_material_response.utils import timelog

#: A pair block with more dense entries than this is built in observer
#: chunks and stored sparse (with a rigorous row-sum error budget) instead
#: of dense. 2e7 float64 entries = 160 MB; dense blocks additionally allow
#: deriving the reverse block by reciprocity (halving the build cost).
DENSE_BLOCK_MAX_ENTRIES = 20_000_000


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
    if np.isscalar(susceptibility):
        return np.ones((n, 3)) * susceptibility
    if (
        hasattr(susceptibility, "__len__")
        and len(susceptibility) == 3
        and all(not isinstance(x, list | tuple | np.ndarray) for x in susceptibility)
    ):
        # This is a 3-vector, not a list of 3 items
        susis = np.tile(susceptibility, (n, 1))
        # Only check for ambiguity when susceptibility comes from user input, not from hierarchy
        if n == 3 and not from_hierarchy:
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
        if np.isscalar(sus):
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


def get_H_ext(*sources, H_ext=None):
    """Return a list of length (len(sources)) with H_ext values
    Priority is given at the source level, however if value is not found, it is searched up the
    the parent tree, if available. Sets H_ext to zero if no value is found when reached the top
    level of the tree"""
    H_exts = []
    for src in sources:
        H_ext = getattr(src, "H_ext", None)
        if H_ext is None:
            if src.parent is None:
                # print("Warning: No value for H_ext defined in any parent collection. H_ext set to zero.")
                H_exts.append((0.0, 0.0, 0.0))
            else:
                H_exts.extend(get_H_ext(src.parent))
        else:
            H_exts.append(H_ext)
    return H_exts


# ══════════════════════════════════════════════════════════════════════════
# Structure analysis and geometry helpers
# ══════════════════════════════════════════════════════════════════════════


def _cell_positions(srcs):
    """Barycentre (fallback: position) of each source as an (n, 3) array."""
    return np.array(
        [getattr(src, "barycenter", src.position) for src in srcs], dtype=float
    )


def _analyze_collection(magnets_list, atol=QUAT_ATOL):
    """Run :func:`analyze_collection` on a list of magnet objects, with logging."""
    positions, clusters = analyze_collection(magnets_list, atol)
    counts = Counter(c["kind"] for c in clusters)
    logger.info(
        "Cell structure: {n_grid} grid, {n_loose} loose, {n_generic} generic clusters",
        n_grid=counts.get("grid", 0),
        n_loose=counts.get("loose", 0),
        n_generic=counts.get("generic", 0),
    )
    return positions, clusters


def _pair_uses_newell(cl_a, cl_b, atol=QUAT_ATOL):
    """True when the analytical Newell tensor applies to this cluster pair:
    both cuboid clusters, prisms parallel (same orientation up to sign)."""
    if cl_a["kind"] == "generic" or cl_b["kind"] == "generic":
        return False
    qa = canonical_quats(cl_a["r_common"].as_quat()[None])[0]
    qb = canonical_quats(cl_b["r_common"].as_quat()[None])[0]
    return bool(np.allclose(qa, qb, atol=atol))


# ══════════════════════════════════════════════════════════════════════════
# T blocks — single source of truth for the interaction entries
#
# All blocks are (3*n_a, 3*n_b) in Fortran (component-major) layout:
# ``block[(m, j), (k, i)]`` is the ``m`` component of ``T @ v`` at observer
# cell ``j`` per unit polarization component ``k`` of source cell ``i``,
# i.e. ``-N_mk(pos_j - pos_i)`` post-``mu_0``.  Global frame.
# ══════════════════════════════════════════════════════════════════════════


def _dedup_demag_blocks(disp, dim_src, dim_obs):
    """Evaluate Newell blocks with duplicate displacements computed once.

    Cells on regular grids share displacements massively (two same-spacing
    grids have O(N) unique displacements out of N^2 pairs), so the analytical
    evaluation is gathered from the unique set when that pays off.
    """
    flat = disp.reshape(-1, 3)
    scale = max(np.max(np.abs(dim_src)), np.max(np.abs(dim_obs)))
    keys = np.round(flat / (1e-9 * scale)).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    if uniq.shape[0] > 0.25 * flat.shape[0]:
        return demag_block_general(disp, dim_src, dim_obs)
    # Representative displacement per unique key (first occurrence).
    first = np.zeros(uniq.shape[0], dtype=np.int64)
    first[inverse[::-1]] = np.arange(flat.shape[0] - 1, -1, -1)
    N_u = demag_block_general(flat[first], dim_src, dim_obs)
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
    N = _dedup_demag_blocks(disp, dim_b, dim_a)  # (n_a, n_b, 3m, 3k), local
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


def _pair_block(cl_a, cl_b, magnets_list, positions, row_budget=None):
    """Build the T block for a cluster pair via the pair rule.

    Returns a dense ndarray when the block is small (or ``row_budget`` is
    None), else a CSR matrix with per-row dropped sums below ``row_budget``.
    """
    ia, ib = cl_a["indices"], cl_b["indices"]
    pos_a, pos_b = positions[ia], positions[ib]
    n_a, n_b = len(ia), len(ib)
    use_newell = _pair_uses_newell(cl_a, cl_b)
    sparsify = row_budget is not None and 9 * n_a * n_b > DENSE_BLOCK_MAX_ENTRIES

    rot_mat = None
    if use_newell and not cl_a["is_identity"]:
        rot_mat = cl_a["r_common"].as_matrix()

    def _dense_chunk(sl):
        if use_newell:
            return _newell_block_T(pos_a[sl], pos_b, cl_a["dim"], cl_b["dim"], rot_mat)
        return _point_block_T([magnets_list[i] for i in ib], pos_a[sl])

    if not sparsify:
        return _dense_chunk(slice(None))

    chunk = max(1, DENSE_BLOCK_MAX_ENTRIES // max(9 * n_b, 1))
    rows_all, cols_all, vals_all = [], [], []
    for start in range(0, n_a, chunk):
        stop = min(start + chunk, n_a)
        block = _dense_chunk(slice(start, stop))
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


def _assemble_T_dense(magnets_list, positions, clusters, min_log_time=None):
    """Assemble the full dense (3n, 3n) T in Fortran layout, global frame."""
    n = len(magnets_list)
    T = np.zeros((3 * n, 3 * n))
    with timelog("Demagnetization tensor assembly", min_log_time=min_log_time):
        for cl_a in clusters:
            rows = (np.arange(3)[:, None] * n + cl_a["indices"][None, :]).ravel()
            for cl_b in clusters:
                cols = (np.arange(3)[:, None] * n + cl_b["indices"][None, :]).ravel()
                T[np.ix_(rows, cols)] = _pair_block(cl_a, cl_b, magnets_list, positions)
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


def _build_operator(magnets_list, positions, clusters, sus, solver_tol, min_log_time):
    """Build the list of block operators representing T for the matvec.

    Self-blocks of grid clusters are FFT convolutions (exact); every other
    block is dense when small, else sparsified with a per-row error budget
    chosen so the total operator perturbation stays below
    ``0.1 * solver_tol``:  ``|S (T - T~)|_inf <= chi_eff * K * row_budget``.
    """
    K = len(clusters)
    chi_eff = max(1.0, float(np.max(np.abs(sus))) if len(sus) else 1.0)
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
                            cl_a["shape"], cl_a["spacing"], cl_a["dim"]
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
                            cl_a, cl_a, magnets_list, positions, row_budget
                        ),
                    }
                )
            for idx_b, cl_b in enumerate(clusters):
                if idx_b <= idx_a:
                    continue
                M_ab = _pair_block(cl_a, cl_b, magnets_list, positions, row_budget)
                ops.append(
                    {
                        "kind": "mat",
                        "ia": cl_a["indices"],
                        "ib": cl_b["indices"],
                        "M": M_ab,
                    }
                )
                if _pair_uses_newell(cl_a, cl_b) and not sp.issparse(M_ab):
                    # Volume-weighted reciprocity: V_a * N_ab = V_b * N_ba^T,
                    # exact for the volume-averaged tensor.
                    v_ratio = float(np.prod(cl_a["dim"]) / np.prod(cl_b["dim"]))
                    M_ba = v_ratio * M_ab.T
                else:
                    M_ba = _pair_block(cl_b, cl_a, magnets_list, positions, row_budget)
                ops.append(
                    {
                        "kind": "mat",
                        "ia": cl_b["indices"],
                        "ib": cl_a["indices"],
                        "M": M_ba,
                    }
                )
    return ops


def _build_matvec(n, sus, ops):
    """Return the Fortran-flat matvec ``v -> (I - S T) @ v``."""

    def matvec(v_flat):
        v3 = v_flat.reshape(3, n)
        Tv = np.zeros((3, n))
        for op in ops:
            v_b = v3[:, op["ib"]].ravel()
            if op["kind"] == "fft":
                h = _fft_apply(op, v_b)
            else:
                h = np.asarray(op["M"] @ v_b).ravel()
            Tv[:, op["ia"]] += h.reshape(3, -1)
        return v_flat - sus * Tv.ravel()

    return matvec


def _build_dense_matvec(sus, T):
    """Return matvec ``v -> (I - S T) @ v`` for dense T."""

    def matvec(v_flat):
        Tv = T @ v_flat
        return v_flat - sus * Tv

    return matvec


def _self_demag_diagonal(n, clusters):
    """Fortran-flat (3n,) diagonal of the analytical self-demag factors,
    rotated to the global frame per cluster (zero for generic cells)."""
    Nself_flat = np.zeros(3 * n)
    for cl in clusters:
        if cl["kind"] == "generic":
            continue
        Ns = self_demag_factors(cl["dim"])  # (3,) local frame
        if cl["is_identity"]:
            diag_m = Ns
        else:
            rot_mat = cl["r_common"].as_matrix()
            diag_m = (rot_mat**2) @ Ns  # diag of R diag(Ns) R^T
        for m in range(3):
            Nself_flat[m * n + cl["indices"]] = diag_m[m]
    return Nself_flat


def _solve_iterative(
    *, n, sus, rhs, rhs_shape, matvec, Nself_flat, solver_tol, max_iter
):
    """GMRES solve of (I - S T) x = rhs with diagonal Jacobi preconditioning.

    Raises ``RuntimeError`` when GMRES does not reach ``solver_tol`` — a
    partially-converged result would silently be wrong.
    """
    Q_op = LinearOperator((3 * n, 3 * n), matvec=matvec, dtype=float)

    # Diagonal Jacobi preconditioner: Q_diag = 1 - sus * (-N_self) = 1 + sus * N_self.
    diag_Q = 1.0 + sus * Nself_flat
    safe = np.where(np.abs(diag_Q) > 1e-12, diag_Q, 1.0)
    M_inv = LinearOperator((3 * n, 3 * n), matvec=lambda v: v / safe, dtype=float)

    x, info = gmres(
        Q_op,
        rhs,
        M=M_inv,
        rtol=solver_tol,
        atol=0.0,
        maxiter=max_iter,
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
):
    """
    Compute the demagnetization tensor T for n sources.

    By default the tensor is assembled with the unified pair rule: the
    analytical volume-averaged Newell tensor for parallel Cuboid cells
    (generalized to different sizes) and point matching (see Chadebec 2006)
    otherwise. The legacy options ``pairs_matching``, ``split`` and
    ``max_dist`` force the historical point-matching evaluation for all
    pairs.

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
        Maximum distance-to-dimension ratio; farther interactions are dropped.
        Implies point matching.

    min_log_time:
        Minimum logging time in seconds. If computation time is below this value, step
        will not be logged.

    Returns
    -------
    Demagnetization tensor: ndarray, shape (3,n,n,3), pre-``mu_0``:
        ``T[k, i, j, m] = -N_mk(pos_j - pos_i) / mu_0``

    TODO: allow multi-point matching
    TODO: allow current sources
    TODO: allow external stray fields
    """
    nof_src = len(src_list)

    if pairs_matching and split != 1 and split is not False:
        msg = "Pairs matching does not support splitting"
        raise ValueError(msg)

    no_split = split is False or split == 1
    if not pairs_matching and max_dist == 0 and no_split:
        # Unified assembly (Newell where applicable), returned in the legacy
        # (3, n, n, 3) pre-mu_0 layout.
        positions, clusters = _analyze_collection(list(src_list))
        T2 = _assemble_T_dense(
            list(src_list), positions, clusters, min_log_time=min_log_time
        )
        n = nof_src
        # T2[(m, j), (k, i)] -> legacy T[k, i, j, m], pre-mu_0.
        return T2.reshape(3, n, 3, n).transpose(2, 3, 1, 0) / magpy.mu_0

    mask_inds = None
    getH_params = {}
    if max_dist != 0:
        mask_inds, getH_params, pos0, rot0 = filter_distance(
            src_list, max_dist, return_params=False, return_base_geo=True
        )
    elif pairs_matching:
        getH_params, mask_inds, unique_inv_inds, pos0, rot0 = match_pairs(src_list)
    else:
        pos0 = _cell_positions(src_list)
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
                    H_unique = magpy.getH(
                        "Cuboid", polarization=polarization, **getH_params
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
                                H_unit_pol.append(
                                    magpy.getH(src_list_subset.tolist(), pos0)
                                )
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
        pos0 = _cell_positions(src_list)
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
        pos0 = _cell_positions(src_list)
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
            logger.debug("Computing dimension differences")
            dim2 = np.tile(dim0, (len_src, 1)) - np.repeat(dim0, len_src, axis=0)
            logger.debug("Concatenating properties for comparison")
            prop = (np.concatenate([pos2, rotQ2a, rotQ2b, dim2], axis=1) + 1e-9).round(
                8
            )
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
            "dimension": np.repeat(dim0, len(src_list), axis=0)[unique_inds],
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
    max_iter=50,
):
    """
    Computes the interaction between all collection magnets and fixes their
    polarization.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    susceptibility: array_like, shape (n,)
        Vector of n magnetic susceptibilities of the cells. If not defined, values are
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
        Positive number representing the max_dimension to distance ratio for each pair
        of interacting cells. This filters out far interactions. If `max_dist=0`, all
        interactions are calculated. This parameter is not compatible with
        `pairs_matching` or `split` and applies only cuboid cells. Forces the legacy
        point-matched tensor for all pairs.

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
        Linear solver to use. Both solvers share the same interaction model
        (analytical volume-averaged Newell tensor for parallel cuboid cells,
        point matching otherwise) and agree to ``solver_tol``.
        ``"direct"`` (default) builds the dense ``Q`` matrix and calls
        :func:`numpy.linalg.solve` — exact within floating-point precision.
        ``"iterative"`` solves matrix-free with :func:`scipy.sparse.linalg.gmres`
        and a diagonal Jacobi preconditioner; uniform-grid clusters of cells
        use an FFT-accelerated :math:`O(n \\log n)` matvec.

    solver_tol: float
        Relative residual tolerance passed to GMRES when ``solver="iterative"``.

    max_iter: int
        Maximum number of GMRES iterations when ``solver="iterative"``.
        Non-convergence raises ``RuntimeError``.

    Returns
    -------
    demagnetized collection or None (if ``inplace=True``)
    """
    if solver not in ("direct", "iterative"):
        msg = f"solver must be 'direct' or 'iterative'; got {solver!r}"
        raise ValueError(msg)
    if not inplace:
        collection = collection.copy()
    if style is not None:
        collection.style = style
    srcs = collection.sources_all
    src_with_paths = [src for src in srcs if src.position.ndim != 1]
    if src_with_paths:
        msg = (
            f"{len(src_with_paths)} objects with paths, found. Demagnetization of "
            "objects with paths is not yet supported"
        )
        raise ValueError(msg)
    magnets_list = [src for src in srcs if isinstance(src, BaseMagnet)]
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

        # set up S: 1-D Fortran-flat (3n,) diagonal of the susceptibility
        # matrix, applied via broadcasting (sus[:, None] * X == diag(sus) @ X).
        sus = get_susceptibilities(magnets_list, susceptibility)

        # set up H_ext
        H_ext = get_H_ext(*magnets_list)
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
                pos = _cell_positions(magnets_list)
                pol_currents = magpy.getB(currents_list, pos, sumup=True)
                pol_currents = np.reshape(pol_currents, (3 * n, 1), order="F")
                # use elementwise multiply because S is diagonal
                pol_total = pol_total + (sus[:, None] * pol_currents)

        rhs = pol_total + (sus[:, None] * H_ext)  # shape (3n, 1)

        # ── demag operator: legacy point-matching or unified pair rule ──────
        no_split = split is False or split == 1
        legacy = pairs_matching or max_dist != 0 or not no_split
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
            positions, clusters = _analyze_collection(magnets_list)
            T = None
            if solver == "direct":
                T = _assemble_T_dense(
                    magnets_list, positions, clusters, min_log_time=min_log_time
                )

        with timelog("Solving of linear system", min_log_time=min_log_time):
            if solver == "direct":
                Q = np.eye(3 * n) - (sus[:, None] * T)
                pol_new = np.linalg.solve(Q, rhs)
            else:
                if T is not None:
                    matvec = _build_dense_matvec(sus, T)
                    Nself_flat = np.zeros(3 * n)
                else:
                    ops = _build_operator(
                        magnets_list,
                        positions,
                        clusters,
                        sus,
                        solver_tol,
                        min_log_time,
                    )
                    matvec = _build_matvec(n, sus, ops)
                    Nself_flat = _self_demag_diagonal(n, clusters)
                pol_new = _solve_iterative(
                    n=n,
                    sus=sus,
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
