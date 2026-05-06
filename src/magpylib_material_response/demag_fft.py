"""FFT-accelerated demagnetisation matvec for uniform Cartesian Cuboid grids.

For a regular axis-aligned grid of identical cuboids, the demag tensor is
translation-invariant: ``T[i, j]`` depends only on the cell-index displacement
``i - j``. The matvec ``H = T @ M`` therefore reduces to a 3-D discrete
convolution and is computed in :math:`O(n \\log n)` time using FFTs on the
zero-padded ("doubled") grid required for aperiodic boundary conditions.

The kernel is built from the analytical Newell formula
(:mod:`magpylib_material_response.newell`); only the 6 independent
components ``(Nxx, Nyy, Nzz, Nxy, Nxz, Nyz)`` are stored.
"""

from __future__ import annotations

import numpy as np

from magpylib_material_response.newell import demag_block

__all__ = [
    "build_fft_kernel",
    "demag_fft_matvec",
    "detect_uniform_grid",
]


def detect_uniform_grid(positions, dimensions, rotations, atol=1e-9):
    """Detect whether ``positions`` form a uniform axis-aligned Cartesian grid.

    Parameters
    ----------
    positions : ndarray, shape (n, 3)
        Cell centre positions in world frame.
    dimensions : ndarray, shape (n, 3)
        Cell side lengths.
    rotations : scipy Rotation, length n
        Per-cell orientations.
    atol : float
        Absolute tolerance for grouping floating-point coordinates.

    Returns
    -------
    info : dict | None
        ``None`` if positions are not on a uniform Cartesian grid of identical,
        identically-oriented cells. Otherwise a dict with keys:
            ``shape``  : (Nx, Ny, Nz)
            ``cell``   : (sx, sy, sz)  -- cell side lengths
            ``origin`` : (x0, y0, z0)  -- centre of cell index (0,0,0)
            ``order``  : ndarray of shape (n,) of int -- permutation such that
                          flat ``ix*Ny*Nz + iy*Nz + iz`` ordering corresponds
                          to ``positions[order]``.
    """
    positions = np.asarray(positions, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    n = positions.shape[0]
    if n == 0:
        return None

    # All dimensions identical?
    dim0 = dimensions[0]
    if not np.allclose(dimensions, dim0, atol=atol):
        return None

    # All rotations identical (any common rotation is accepted).
    quats = rotations.as_quat()
    q0 = quats[0]
    # Rotations are equivalent if quaternions match up to a global sign.
    if not (np.allclose(quats, q0, atol=atol) or np.allclose(quats, -q0, atol=atol)):
        return None
    # Canonical representative: positive w component.
    r_common = rotations[0]
    is_identity = np.allclose(
        r_common.as_quat(), np.array([0.0, 0.0, 0.0, 1.0]), atol=atol
    )

    # For a non-identity common rotation, transform positions to the local
    # (cuboid) frame where the grid is axis-aligned.
    if not is_identity:
        positions = r_common.inv().apply(positions)

    sx, sy, sz = float(dim0[0]), float(dim0[1]), float(dim0[2])

    # Cluster coordinates along each axis to discrete grid indices.
    def _axis_grid(coords, spacing):
        sorted_unique = np.sort(coords)
        # collapse near-equal values
        kept = [sorted_unique[0]]
        for v in sorted_unique[1:]:
            if v - kept[-1] > 0.5 * spacing:
                kept.append(v)
        kept = np.asarray(kept)
        if kept.size > 1:
            diffs = np.diff(kept)
            if not np.allclose(diffs, spacing, atol=max(atol, 1e-6 * spacing)):
                return None
        return kept

    gx = _axis_grid(positions[:, 0], sx)
    gy = _axis_grid(positions[:, 1], sy)
    gz = _axis_grid(positions[:, 2], sz)
    if gx is None or gy is None or gz is None:
        return None

    Nx, Ny, Nz = gx.size, gy.size, gz.size
    if Nx * Ny * Nz != n:
        return None

    # Build index of each cell into (ix, iy, iz) by nearest-grid-coord lookup.
    def _nearest_idx(coords, grid):
        idx = np.empty(coords.size, dtype=np.int64)
        for k, c in enumerate(coords):
            j = int(np.argmin(np.abs(grid - c)))
            if abs(grid[j] - c) > max(atol, 1e-6 * max(abs(c), 1.0)) * 10:
                idx[k] = -1
            else:
                idx[k] = j
        return idx

    ix = _nearest_idx(positions[:, 0], gx)
    iy = _nearest_idx(positions[:, 1], gy)
    iz = _nearest_idx(positions[:, 2], gz)
    if (ix < 0).any() or (iy < 0).any() or (iz < 0).any():
        return None

    # Verify uniqueness: every (ix,iy,iz) must appear exactly once.
    flat_idx = ix * (Ny * Nz) + iy * Nz + iz
    if np.unique(flat_idx).size != n:
        return None

    # Permutation: order[flat_idx[k]] = k, i.e. order such that the resulting
    # array ordered by flat_idx contains the original index of each cell.
    order = np.empty(n, dtype=np.int64)
    order[flat_idx] = np.arange(n)

    return {
        "shape": (Nx, Ny, Nz),
        "cell": (sx, sy, sz),
        "origin": (float(gx[0]), float(gy[0]), float(gz[0])),
        "order": order,
        "r_common": r_common,
    }


def build_fft_kernel(grid_shape, cell):
    """Build the 6-component Newell demag kernel on the doubled grid, FFTed.

    Parameters
    ----------
    grid_shape : tuple (Nx, Ny, Nz)
    cell : tuple (sx, sy, sz)

    Returns
    -------
    kernel_fft : ndarray, shape (6, 2*Nx, 2*Ny, Nz+1), complex
        Components in order ``(Nxx, Nyy, Nzz, Nxy, Nxz, Nyz)``. The last axis
        is the half-spectrum produced by ``np.fft.rfftn`` applied to the
        doubled real-space kernel.
    """
    Nx, Ny, Nz = grid_shape
    sx, sy, sz = cell

    # Real-space kernel on doubled grid with zero-padding for open BC.
    # Index convention: K[i, j, k] is N at displacement (di, dj, dk) with
    #   di = i      if i <  Nx else i - 2*Nx
    #   dj = j      if j <  Ny else j - 2*Ny
    #   dk = k      if k <  Nz else k - 2*Nz
    # Cell index 0 maps to displacement 0 (self).
    di = np.where(np.arange(2 * Nx) < Nx, np.arange(2 * Nx), np.arange(2 * Nx) - 2 * Nx)
    dj = np.where(np.arange(2 * Ny) < Ny, np.arange(2 * Ny), np.arange(2 * Ny) - 2 * Ny)
    dk = np.where(np.arange(2 * Nz) < Nz, np.arange(2 * Nz), np.arange(2 * Nz) - 2 * Nz)

    DX, DY, DZ = np.meshgrid(di * sx, dj * sy, dk * sz, indexing="ij")
    disp = np.stack([DX, DY, DZ], axis=-1)  # (2Nx, 2Ny, 2Nz, 3)
    N = demag_block(disp, (sx, sy, sz))  # (2Nx, 2Ny, 2Nz, 3, 3)

    # Stack 6 independent components.
    kernel = np.empty((6, 2 * Nx, 2 * Ny, 2 * Nz), dtype=float)
    kernel[0] = N[..., 0, 0]
    kernel[1] = N[..., 1, 1]
    kernel[2] = N[..., 2, 2]
    kernel[3] = N[..., 0, 1]
    kernel[4] = N[..., 0, 2]
    kernel[5] = N[..., 1, 2]

    # Real input -> half-spectrum FFT.
    return np.fft.rfftn(kernel, axes=(1, 2, 3))


def demag_fft_matvec(M, kernel_fft, grid_shape, mu_0):
    """Compute the demag matvec ``H = T @ M`` via 3-D FFT convolution.

    Parameters
    ----------
    M : ndarray, shape (Nx, Ny, Nz, 3)
        Polarisation per cell, world frame.
    kernel_fft : ndarray
        Output of :func:`build_fft_kernel`.
    grid_shape : tuple (Nx, Ny, Nz)
    mu_0 : float
        Magnetic constant.

    Returns
    -------
    H : ndarray, shape (Nx, Ny, Nz, 3)
        Demag field times ``mu_0`` (B-like units), matching the post-``mu_0``
        scaling of the dense demag matrix used in :func:`apply_demag`.
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

    # T[k, i, j, m] = -(1/mu_0) * N_mk(disp_ji)  (post-mu_0 we want -N_mk).
    # Convolution: (T @ M)_m at j = sum_i sum_k T[k,i,j,m] * M[k,i]
    #            = sum_i [-N(p_j-p_i)]_mk * M[k,i]
    # In Fourier:  H_m_fft = - (Kmx * Mx_fft + Kmy * My_fft + Kmz * Mz_fft)
    # with kernel symmetric: K_yx = K_xy etc.
    Hx_fft = -(Kxx * Mx_fft + Kxy * My_fft + Kxz * Mz_fft)
    Hy_fft = -(Kxy * Mx_fft + Kyy * My_fft + Kyz * Mz_fft)
    Hz_fft = -(Kxz * Mx_fft + Kyz * My_fft + Kzz * Mz_fft)

    H = np.empty((2 * Nx, 2 * Ny, 2 * Nz, 3), dtype=float)
    H[..., 0] = np.fft.irfftn(Hx_fft, s=(2 * Nx, 2 * Ny, 2 * Nz), axes=(0, 1, 2))
    H[..., 1] = np.fft.irfftn(Hy_fft, s=(2 * Nx, 2 * Ny, 2 * Nz), axes=(0, 1, 2))
    H[..., 2] = np.fft.irfftn(Hz_fft, s=(2 * Nx, 2 * Ny, 2 * Nz), axes=(0, 1, 2))
    # ``mu_0`` cancels: T_post = -N (no mu_0 factor) — see demag_tensor_newell
    # docstring; ``mu_0`` argument retained for future tensor variants.
    del mu_0
    return H[:Nx, :Ny, :Nz, :]
