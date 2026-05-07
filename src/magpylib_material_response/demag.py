"""demag_functions"""

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
    build_fft_kernel,
    demag_fft_matvec,
    detect_grid_groups,
    detect_uniform_grid,
)
from magpylib_material_response.newell import (
    demag_tensor_newell,
    self_demag_factors,
)
from magpylib_material_response.utils import timelog


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


def demag_tensor(
    src_list,
    pairs_matching=False,
    split=False,
    max_dist=0,
    min_log_time=None,
):
    """
    Compute the demagnetization tensor T based on point matching (see Chadbec 2006)
    for n sources in the input collection.

    Parameters
    ----------
    collection: magpylib.Collection object with n magnet sources
        Each magnet source in collection is treated as a magnetic cell.

    pairs_matching: bool
        If True, equivalent pair of interactions are identified and unique pairs are
        calculated only once and copied to duplicates.

    split: int
        Number of times the sources list is split before getH calculation ind demag
        tensor calculation

    min_log_time:
        Minimum logging time in seconds. If computation time is below this value, step
        will not be logged.

    Returns
    -------
    Demagnetization tensor: ndarray, shape (3,n,n,3)

    TODO: allow multi-point matching
    TODO: allow current sources
    TODO: allow external stray fields
    TODO: status bar when n>1000
    TODO: Speed up with direct interface for field computation
    TODO: Use newell formulas for cube-cube interactions
    """
    nof_src = len(src_list)

    if pairs_matching and split != 1:
        msg = "Pairs matching does not support splitting"
        raise ValueError(msg)

    # Fast Newell path: identical cuboids with a common rotation, no pairs matching / max_dist
    no_split = split is False or split == 1
    if not pairs_matching and max_dist == 0 and no_split:
        all_cuboids = all(isinstance(s, Cuboid) for s in src_list)
        if all_cuboids and nof_src > 0:
            dims = np.array([s.dimension for s in src_list])
            quats = np.array([s.orientation.as_quat() for s in src_list])
            same_dim = np.allclose(dims, dims[0])
            q0 = quats[0]
            same_rot = np.allclose(quats, q0, atol=1e-9) or np.allclose(
                quats, -q0, atol=1e-9
            )
            if same_dim and same_rot:
                with timelog(
                    "Newell analytical demag tensor", min_log_time=min_log_time
                ):
                    pos0 = np.array(
                        [getattr(s, "barycenter", s.position) for s in src_list]
                    )
                    r_common = R.from_quat(q0)
                    is_identity = np.allclose(
                        q0, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9
                    )
                    if is_identity:
                        return demag_tensor_newell(pos0, dims[0], magpy.mu_0)
                    # Rotated grid: transform positions to local frame, compute
                    # Newell tensor there, then rotate back to global frame.
                    # Convention: T[K,i,j,M] = field M at j due to unit pol K at i
                    # H_global[M,j] = R[M,A]*H_local[A,j], pol_local[B] = R[K,B]*pol_global[K]
                    # → T_global[K,i,j,M] = Σ_{A,B} R[M,A]*T_local[B,i,j,A]*R[K,B]
                    pos_local = r_common.inv().apply(pos0)
                    T_local = demag_tensor_newell(pos_local, dims[0], magpy.mu_0)
                    R_mat = r_common.as_matrix()
                    return np.einsum("ma,bija,kb->kijm", R_mat, T_local, R_mat)

    mask_inds = None
    getH_params = {}
    if max_dist != 0:
        mask_inds, getH_params, pos0, rot0 = filter_distance(
            src_list, max_dist, return_params=False, return_base_geo=True
        )
    elif pairs_matching:
        getH_params, mask_inds, unique_inv_inds, pos0, rot0 = match_pairs(src_list)
    else:
        pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
        rotQ0 = [src.orientation.as_quat() for src in src_list]
        rot0 = R.from_quat(rotQ0)

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
                if split > 1:
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
                else:
                    H_unit_pol = magpy.getH(src_list, pos0)
            H_point.append(H_unit_pol)  # shape (n_cells, n_pos, 3_xyz)

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
        pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
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
        pos0 = np.array([getattr(src, "barycenter", src.position) for src in src_list])
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


def _rotate_fortran_flat(v_flat, n, rot):
    """Apply rotation ``rot`` to each 3-vector in a Fortran-flat (3n,) array.

    The Fortran-flat layout stores component k of cell i at index ``k*n + i``.
    Equivalent to reshaping to (3, n), transposing to (n, 3), applying rotation,
    then packing back.
    """
    v_array = v_flat.reshape((3, n)).T  # (n, 3)
    return rot.apply(v_array).T.ravel()  # (3n,)


def _build_fft_matvec(n, sus, fft_info):
    """Return a Fortran-flat matvec ``v -> (I - S T) @ v`` using the FFT path."""
    Nx, Ny, Nz = fft_info["shape"]
    order = fft_info["order"]
    kernel_fft = fft_info["kernel_fft"]

    # Inverse permutation: order maps grid_flat_index -> original_cell_index.
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)

    def matvec(v_flat):
        # v_flat shape (3n,); component-major layout [x1..xn, y1..yn, z1..zn].
        # Reshape (3, n) in C-order so v[k, i] = v_flat[k*n + i] (component k,
        # original cell i). Note Fortran-order would interleave components.
        v = v_flat.reshape((3, n))
        # Build polarisation per grid cell (Nx, Ny, Nz, 3).
        M_orig = v.T  # (n, 3) per original cell
        M_grid_flat = M_orig[order]  # reorder to grid-flat layout
        M_grid = M_grid_flat.reshape((Nx, Ny, Nz, 3))
        H_grid = demag_fft_matvec(M_grid, kernel_fft, (Nx, Ny, Nz), magpy.mu_0)
        # H_grid (Nx, Ny, Nz, 3) -> per-original-cell ordering.
        H_grid_flat = H_grid.reshape((Nx * Ny * Nz, 3))
        H_orig = np.empty_like(M_orig)
        H_orig[order] = H_grid_flat
        # Tv flat (component-major): H_orig[i, m] -> flat[m*n + i].
        Tv = H_orig.T  # (3, n)
        Tv_flat = Tv.reshape(3 * n)
        return v_flat - sus * Tv_flat

    return matvec


def _build_dense_matvec(sus, T):
    """Return matvec ``v -> (I - S T) @ v`` for dense T."""

    def matvec(v_flat):
        Tv = T @ v_flat
        return v_flat - sus * Tv

    return matvec


def _fft_block_h(v_g_flat, n_g, g):
    """Apply T_{GG} (self-block) to group G polarizations and return H, both global frame.

    Both input ``v_g_flat`` and returned array are Fortran-flat ``(3*n_g,)``
    in component-major layout.  Per-group rotation is handled internally.
    """
    Nx, Ny, Nz = g["shape"]
    order = g["order"]
    kernel_fft = g["kernel_fft"]
    r_frame = g["r_common"]
    is_rotated = not np.allclose(r_frame.as_quat(), [0.0, 0.0, 0.0, 1.0], atol=1e-9)

    v_local = (
        _rotate_fortran_flat(v_g_flat, n_g, r_frame.inv()) if is_rotated else v_g_flat
    )

    M_orig = v_local.reshape((3, n_g)).T  # (n_g, 3)
    M_grid_flat = M_orig[order]
    M_grid = M_grid_flat.reshape((Nx, Ny, Nz, 3))
    H_grid = demag_fft_matvec(M_grid, kernel_fft, (Nx, Ny, Nz), magpy.mu_0)
    H_grid_flat = H_grid.reshape((Nx * Ny * Nz, 3))
    H_local_arr = np.empty_like(M_orig)
    H_local_arr[order] = H_grid_flat
    H_local_flat = H_local_arr.T.reshape(3 * n_g)

    return (
        _rotate_fortran_flat(H_local_flat, n_g, r_frame) if is_rotated else H_local_flat
    )


def _compute_cross_block(srcs_b, pos_a, dims_b, sus_a, sus_b, interaction_tol):
    """Build a sparse cross-block T_{AB} in Fortran-flat CSR form (global frame).

    Uses a χ-weighted solid-angle triage:

        keep(i, j)  iff  max(χ_i, χ_j) * V_j / |r_ij|³  >  interaction_tol

    Source cells (group B) with no significant influence on any observer in A
    are pruned before calling ``magpy.getH``; observer rows with no significant
    source are similarly pruned.  The result is returned as a
    ``scipy.sparse.csr_matrix`` so each GMRES matvec costs O(nnz) instead of
    O(n_a * n_b).

    Parameters
    ----------
    srcs_b : list of Cuboid, length n_b
    pos_a  : ndarray (n_a, 3)  — observer positions (barycentres of group A)
    dims_b : ndarray (n_b, 3)  — cell dimensions of group B
    sus_a  : ndarray (n_a,)   — per-cell susceptibility of group A (scalar part)
    sus_b  : ndarray (n_b,)   — per-cell susceptibility of group B
    interaction_tol : float   — threshold ε for the triage criterion

    Returns
    -------
    T_AB : scipy.sparse.csr_matrix, shape (3*n_a, 3*n_b)
    active_a : ndarray(int) — row indices of observers with ≥1 active source
    active_b : ndarray(int) — column indices of sources with ≥1 active observer
    """
    n_a = pos_a.shape[0]
    n_b = len(srcs_b)
    pos_b = np.array(
        [getattr(s, "barycenter", s.position) for s in srcs_b], dtype=float
    )

    # ── Triage: χ-weighted solid-angle criterion ──────────────────────────────
    # influence(i, j) ≈ max(χ_i, χ_j) * V_j / |r_ij|³
    r_vec = pos_a[:, None, :] - pos_b[None, :, :]  # (n_a, n_b, 3)
    r_sq = np.einsum("ijk,ijk->ij", r_vec, r_vec)  # (n_a, n_b)
    # Avoid division by zero for coincident cells (shouldn't happen cross-block,
    # but guard anyway).
    r_sq = np.where(r_sq > 0, r_sq, np.inf)
    V_b = np.prod(dims_b, axis=1)  # (n_b,)
    chi_max = np.maximum(sus_a[:, None], sus_b[None, :])  # (n_a, n_b)
    influence = chi_max * V_b[None, :] / r_sq**1.5  # (n_a, n_b)
    mask = influence > interaction_tol  # (n_a, n_b) bool

    # Prune axes: keep only sources/observers that matter to at least one partner.
    active_b = np.where(mask.any(axis=0))[0]  # source columns to keep
    active_a = np.where(mask.any(axis=1))[0]  # observer rows to keep

    # If nothing survives triage, return an explicit zero sparse matrix.
    if active_b.size == 0 or active_a.size == 0:
        return sp.csr_matrix((3 * n_a, 3 * n_b), dtype=float), active_a, active_b

    srcs_b_active = [srcs_b[i] for i in active_b]
    pos_a_active = pos_a[active_a]  # (|active_a|, 3)
    mask_sub = mask[np.ix_(active_a, active_b)]  # (|active_a|, |active_b|)
    n_a_sub, n_b_sub = len(active_a), len(active_b)

    # ── Build T for the active sub-block ─────────────────────────────────────
    rot_b_active = R.from_quat(
        [srcs_b_active[i].orientation.as_quat() for i in range(n_b_sub)]
    )
    H_point = []
    for unit_pol in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        pol_all = rot_b_active.inv().apply(unit_pol)
        for src, pol in zip(srcs_b_active, pol_all, strict=False):
            src.polarization = pol
        H = magpy.getH(srcs_b_active, pos_a_active)  # (n_b_sub, n_a_sub, 3)
        H_point.append(H)
    T_raw_sub = np.array(H_point) * magpy.mu_0  # (3, n_b_sub, n_a_sub, 3)
    # T_sub[row_a, col_b] in Fortran-flat: rows = (m, j), cols = (k, i)
    T_sub_dense = (
        T_raw_sub.swapaxes(2, 3).reshape((3 * n_b_sub, 3 * n_a_sub)).T
    )  # (3*n_a_sub, 3*n_b_sub)

    # ── Zero out below-threshold entries within the sub-block ─────────────────
    # Expand the cell-level mask to the 3x3 component blocks.
    mask3 = np.repeat(
        np.repeat(mask_sub, 3, axis=0), 3, axis=1
    )  # (3*n_a_sub, 3*n_b_sub)
    T_sub_dense[~mask3] = 0.0

    # ── Scatter back into the full (3*n_a, 3*n_b) sparse matrix ──────────────
    # Build (row, col, val) triplets.
    T_sub_sparse = sp.csr_matrix(T_sub_dense)
    rows_sub, cols_sub = T_sub_sparse.nonzero()
    vals = np.asarray(T_sub_sparse[rows_sub, cols_sub]).ravel()

    # Map sub-block indices to full-block indices.
    # Row m*n_a_sub + j  →  m*n_a + active_a[j]
    def _expand_idx(flat_sub, n_sub, n_full, active):
        comp = flat_sub // n_sub
        local = flat_sub % n_sub
        return comp * n_full + active[local]

    rows_full = _expand_idx(rows_sub, n_a_sub, n_a, active_a)
    cols_full = _expand_idx(cols_sub, n_b_sub, n_b, active_b)

    T_AB = sp.csr_matrix(
        (vals, (rows_full, cols_full)), shape=(3 * n_a, 3 * n_b), dtype=float
    )
    return T_AB, active_a, active_b


def _build_block_fft_matvec(n, sus, groups, cross_blocks):
    """Return matvec ``v -> (I - S T) @ v`` for a multi-group block structure.

    Diagonal blocks (same group) use the FFT kernel via :func:`_fft_block_h`;
    off-diagonal blocks use precomputed sparse T_{AB} matrices
    (scipy.sparse.csr_matrix), so the matvec cost is O(nnz) rather than
    O(n_a * n_b).
    """

    def matvec(v_flat):
        Tv = np.zeros(3 * n)
        v3 = v_flat.reshape(3, n)  # component-major view

        # Self-blocks via FFT
        for g in groups:
            ia = g["indices"]
            n_g = len(ia)
            v_g = v3[:, ia].ravel()
            h_g = _fft_block_h(v_g, n_g, g)
            Tv.reshape(3, n)[:, ia] += h_g.reshape(3, n_g)

        # Cross-blocks via sparse matrix-vector multiply (O(nnz))
        for (idx_a, idx_b), T_AB in cross_blocks.items():
            ia = groups[idx_a]["indices"]
            ib = groups[idx_b]["indices"]
            n_a = len(ia)
            v_b = v3[:, ib].ravel()
            h_a = np.asarray(T_AB @ v_b).ravel()  # (3*n_a,)
            Tv.reshape(3, n)[:, ia] += h_a.reshape(3, n_a)

        return v_flat - sus * Tv

    return matvec


def _solve_iterative(
    *,
    n,
    sus,
    rhs,
    rhs_shape,
    T,
    fft_info,
    magnets_list,
    solver_tol,
    max_iter,
    groups=None,
    cross_blocks=None,
):
    """GMRES solve of (I - S T) x = rhs with diagonal Jacobi preconditioning.

    Selects the FFT matvec when ``fft_info`` is provided, otherwise falls back
    to a dense T matvec.
    """
    if fft_info is not None:
        matvec = _build_fft_matvec(n, sus, fft_info)
    elif groups is not None:
        matvec = _build_block_fft_matvec(n, sus, groups, cross_blocks)
    else:
        matvec = _build_dense_matvec(sus, T)

    Q_op = LinearOperator((3 * n, 3 * n), matvec=matvec, dtype=float)

    # Diagonal Jacobi preconditioner using analytical self-demag factors.
    # T_post = -N (post-mu_0). Q_diag = 1 - sus * (-N_self)_kk = 1 + sus * N_self_kk.
    # Cell ordering of sus is Fortran (m-major): [Nxx_1..Nxx_n, Nyy_1..Nyy_n, Nzz_1..Nzz_n].
    all_cuboids = all(isinstance(s, Cuboid) for s in magnets_list)
    if all_cuboids and n > 0:
        dims = np.array([s.dimension for s in magnets_list])
        # Same dim across all cells? Use one factor; else per-cell.
        if np.allclose(dims, dims[0]):
            Nself = self_demag_factors(dims[0])  # (3,)
            Nself_flat = np.repeat(Nself, n)  # (3n,) Fortran flat
        else:
            Nself_per_cell = np.array([self_demag_factors(d) for d in dims])  # (n, 3)
            Nself_flat = Nself_per_cell.T.ravel()  # (3n,) m-major
        diag_Q = 1.0 + sus * Nself_flat
    else:
        diag_Q = np.ones(3 * n)

    # Avoid division by zero in degenerate cases.
    safe = np.where(np.abs(diag_Q) > 1e-12, diag_Q, 1.0)
    M_inv = LinearOperator((3 * n, 3 * n), matvec=lambda v: v / safe, dtype=float)

    x0 = rhs.copy()  # warm start at rhs
    x, info = gmres(
        Q_op,
        rhs,
        M=M_inv,
        rtol=solver_tol,
        atol=0.0,
        maxiter=max_iter,
        x0=x0,
    )
    if info > 0:
        logger.warning(
            "GMRES did not converge after {iters} iterations (tol={tol})",
            iters=info,
            tol=solver_tol,
        )
    elif info < 0:
        msg = f"GMRES illegal input or breakdown (info={info})"
        raise RuntimeError(msg)
    return x.reshape(rhs_shape)


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
        with `max_dist` or `split` and applies only cuboid cells.

    max_dist: float
        Posivive number representing the max_dimension to distance ratio for each pair
        of interacting cells. This filters out far interactions. If `max_dist=0`, all
        interactions are calculated. This parameter is not compatible with
        `pairs_matching` or `split` and applies only cuboid cells.

    split: int
        Number of times the sources list is split before getH calculation ind demag
        tensor calculation. This parameter is not compatible with `pairs_matching` or
        `max_dist`.

    min_log_time:
        Minimum logging time in seconds. If computation time is below this value, step
        will not be logged. If ``None`` (default), the value set by
        :func:`magpylib_material_response.configure_logging` is used
        (``1.0`` s by default).

    style: dict
        Set collection style. If `inplace=False` only affects the copied collection

    solver: {"direct", "iterative"}
        Linear solver to use. ``"direct"`` (default) builds the dense ``Q`` matrix
        and calls :func:`numpy.linalg.solve` -- exact within floating-point
        precision. ``"iterative"`` solves with :func:`scipy.sparse.linalg.gmres`
        and a diagonal Jacobi preconditioner; if all cells are identical
        axis-aligned Cuboids on a uniform Cartesian grid, an FFT-accelerated
        :math:`O(n \\log n)` matvec is used. Defaults to ``"direct"`` to
        preserve backward-compatible behaviour.

    solver_tol: float
        Relative residual tolerance passed to GMRES when ``solver="iterative"``.

    max_iter: int
        Maximum number of GMRES iterations when ``solver="iterative"``.

    Returns
    -------
    None
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

        # set up S
        sus = get_susceptibilities(magnets_list, susceptibility)
        # ``sus`` is a 1-D Fortran-flat (3n,) vector representing the
        # diagonal of the susceptibility matrix S. We exploit this everywhere
        # via broadcasting (sus[:, None] * X == np.diag(sus) @ X) to avoid
        # materialising the (3n, 3n) dense diagonal matrix.

        # set up H_ext
        H_ext = get_H_ext(*magnets_list)
        H_ext = np.array(H_ext)
        if len(H_ext) != n:
            msg = "Apply_demag input collection and H_ext must have same length."
            raise ValueError(msg)
        H_ext = np.reshape(H_ext, (3 * n, 1), order="F")

        # set up T (3 pol unit, n cells, n positions, 3 Bxyz)
        # Try FFT path first when iterative solver is requested.
        fft_info = None
        groups = None
        cross_blocks = {}
        if solver == "iterative" and not pairs_matching and max_dist == 0:
            all_cuboids = all(isinstance(s, Cuboid) for s in magnets_list)
            if all_cuboids and n > 0:
                positions = np.array(
                    [getattr(s, "barycenter", s.position) for s in magnets_list]
                )
                dimensions = np.array([s.dimension for s in magnets_list])
                rotations = R.from_quat([s.orientation.as_quat() for s in magnets_list])
                fft_info = detect_uniform_grid(positions, dimensions, rotations)
                if fft_info is None:
                    # Try multi-group block-FFT (each meshed cuboid gets its own FFT kernel)
                    groups = detect_grid_groups(positions, dimensions, rotations)

        # interaction_tol: threshold for the χ·V/r³ triage in cross-block construction.
        # Tied to solver_tol with a safety margin so dropped interactions stay below
        # the GMRES residual target.
        _interaction_tol = solver_tol * 0.1

        T = None
        if fft_info is not None:
            with timelog("FFT demag kernel build", min_log_time=min_log_time):
                fft_info["kernel_fft"] = build_fft_kernel(
                    fft_info["shape"], fft_info["cell"]
                )
        elif groups is not None:
            with timelog("Block FFT kernel build", min_log_time=min_log_time):
                for g in groups:
                    g["kernel_fft"] = build_fft_kernel(g["shape"], g["cell"])
            with timelog(
                "Cross-block demag tensor calculation", min_log_time=min_log_time
            ):
                for idx_a, g_a in enumerate(groups):
                    for idx_b, g_b in enumerate(groups):
                        if idx_a != idx_b:
                            ia, ib = g_a["indices"], g_b["indices"]
                            srcs_b = [magnets_list[i] for i in ib]
                            dims_b = dimensions[ib]
                            pos_a = positions[ia]
                            # Conservative per-cell χ: max over x/y/z components.
                            # sus is Fortran-flat (3n,): block k is sus[k*n:(k+1)*n].
                            sus_a_grp = np.maximum.reduce(
                                [sus[k * n : (k + 1) * n][ia] for k in range(3)]
                            )
                            sus_b_grp = np.maximum.reduce(
                                [sus[k * n : (k + 1) * n][ib] for k in range(3)]
                            )
                            T_AB, _, _ = _compute_cross_block(
                                srcs_b,
                                pos_a,
                                dims_b,
                                sus_a_grp,
                                sus_b_grp,
                                _interaction_tol,
                            )
                            cross_blocks[(idx_a, idx_b)] = T_AB
        else:
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

        pol_total = pol_magnets

        if currents_list:
            with timelog(
                "Add current sources contributions", min_log_time=min_log_time
            ):
                pos = np.array([src.position for src in magnets_list])
                pol_currents = magpy.getB(currents_list, pos, sumup=True)
                pol_currents = np.reshape(pol_currents, (3 * n, 1), order="F")
                # use elementwise multiply because S is diagonal
                pol_total = pol_total + (sus[:, None] * pol_currents)

        rhs = pol_total + (sus[:, None] * H_ext)  # shape (3n, 1)

        # For the FFT path with a non-identity common rotation the solve runs in
        # the cuboid's local frame.  Rotate rhs there and rotate result back.
        r_frame = fft_info["r_common"] if fft_info is not None else None
        is_rotated_frame = r_frame is not None and not np.allclose(
            r_frame.as_quat(), [0.0, 0.0, 0.0, 1.0], atol=1e-9
        )

        rhs_solve = rhs.ravel()
        if is_rotated_frame:
            rhs_solve = _rotate_fortran_flat(rhs_solve, n, r_frame.inv())

        with timelog("Solving of linear system", min_log_time=min_log_time):
            if solver == "direct":
                Q = np.eye(3 * n) - (sus[:, None] * T)
                pol_new = np.linalg.solve(Q, rhs)
            else:
                pol_new = _solve_iterative(
                    n=n,
                    sus=sus,
                    rhs=rhs_solve,
                    rhs_shape=rhs.shape,
                    T=T,
                    fft_info=fft_info,
                    magnets_list=magnets_list,
                    solver_tol=solver_tol,
                    max_iter=max_iter,
                    groups=groups,
                    cross_blocks=cross_blocks,
                )
                if is_rotated_frame:
                    pol_new = _rotate_fortran_flat(pol_new, n, r_frame)

        pol_new = np.reshape(pol_new, (n, 3), order="F")
        # pol_new *= .4*np.pi

        for s, pol in zip(collection.sources_all, pol_new, strict=False):
            s.polarization = s.orientation.inv().apply(pol)  # ROTATION CHECK

    if not inplace:
        return collection
    return None
