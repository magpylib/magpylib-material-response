"""CuboidResponse: a cuboid's full linear response, precomputed once per shape.

Offline (once per shape, any chi): face mesh, P1 Galerkin operator G of K',
and its eigen-decomposition G v = lam M v. Online, for any susceptibility chi,
remanence and external field:

    c = sum_k v_k (u_k . F) / (1/chi + 1/2 - lam_k)

i.e. two dense mat-vecs. chi enters only through the scalar factors: the
"demagnetization spectrum" n_k = 1/2 - lam_k is a property of the shape.
"""

import time

import magpylib as magpy
import numpy as np
import scipy.linalg

from bem import layer_panels
from p1 import P1

MU0 = 4e-7 * np.pi


class CuboidResponse:
    def __init__(self, dimension, m=4, g=3.0, nq=4, nq_rhs=3, verbose=False):
        t0 = time.perf_counter()
        self.dimension = np.asarray(dimension, float)
        half = self.dimension / 2
        self.pan = layer_panels(half, m, g)
        self.s = P1(self.pan, nq=nq)
        self.nq_rhs = nq_rhs
        G = self.s.assemble()
        t1 = time.perf_counter()
        d = np.sqrt(self.s.mass)
        Gh = G / d[:, None] / d[None, :]
        # complex Schur Gh = Z T Z^H, equilibrium-charge mode (lam ~ 1/2) sorted first
        T, Z, _ = scipy.linalg.schur(Gh, output="complex", sort=lambda x: x.real > 0.5 - 1e-3)
        if abs(T[0, 0] - 0.5) > 1e-2 or (T.diagonal()[1:].real > 0.5 - 1e-3).any():
            raise RuntimeError("equilibrium mode not isolated")
        self.d = d
        self.T22 = T[1:, 1:]
        self.Z2 = Z[:, 1:]
        self.Z2H = Z[:, 1:].conj().T
        z1 = (Z[:, 0] / d).real if np.abs((Z[:, 0]).imag).max() < 1e-10 else None
        if z1 is None:
            raise RuntimeError("complex equilibrium vector")
        self.z1 = z1
        self.charge = np.zeros(self.s.Nd)
        self.charge[0::3] = self.pan.area
        self.n = 0.5 - T.diagonal()[1:].real  # demagnetization spectrum
        self.max_imag = np.abs(T.diagonal().imag).max()
        self._diag = np.arange(len(self.T22))
        # sample points for the load (fixed per shape, local frame)
        pts, wts, S, T = self.s.gauss_points(nq_rhs)
        self._pts, self._wts, self._S, self._T = pts, wts, S, T
        self.t_build = (t1 - t0, time.perf_counter() - t1)
        if verbose:
            print(f"shape {self.dimension}: {self.s.Nd} dofs, assembly {self.t_build[0]:.2f}s, schur {self.t_build[1]:.2f}s, max|Im lam| {self.max_imag:.1e}, n in [{self.n.min():.3f}, {self.n.max():.3f}]")

    # ---- online ---------------------------------------------------------
    def load(self, Hn):
        """Galerkin load from H.n at the sample points, shape (nq^2, N)."""
        w = self._wts
        F = np.stack([(w * Hn).sum(0), (w * Hn * self._S[:, None]).sum(0), (w * Hn * self._T[:, None]).sum(0)], 1)
        return F.ravel()

    def sample_points(self, position=(0, 0, 0), orientation=None):
        """Global coordinates of the load sample points for a placed body."""
        p = self._pts.reshape(-1, 3)
        if orientation is not None:
            p = orientation.apply(p)
        return p + np.asarray(position)

    def solve(self, chi, H_samples=None, J0=None, orientation=None):
        """Coefficients for susceptibility chi, external H (A/m) at the
        sample points (global frame), remanent polarization J0 (T, body frame)."""
        F = np.zeros(self.s.Nd)
        if H_samples is not None:
            H = np.asarray(H_samples).reshape(len(self._S), self.pan.N, 3)
            if orientation is not None:
                H = orientation.inv().apply(H.reshape(-1, 3)).reshape(H.shape)
            Hn = np.einsum("gnd,nd->gn", H, self.pan.normal)
            F += self.load(Hn)
        if J0 is not None:
            F += self.s.rhs_const_normal(self.pan.normal @ (np.asarray(J0) / MU0)) / chi
        A = -self.T22
        A[self._diag, self._diag] += 1 / chi + 0.5
        y = scipy.linalg.solve_triangular(A, self.Z2H @ (F / self.d), check_finite=False)
        c = (self.Z2 @ y).real / self.d
        return c - (self.charge @ c) / (self.charge @ self.z1) * self.z1

    def B(self, obs, c, position=(0, 0, 0), orientation=None):
        """B (T) of the body at observers outside it (global frame)."""
        o = np.atleast_2d(obs) - np.asarray(position)
        if orientation is not None:
            o = orientation.inv().apply(o)
        H = self.s.field(o, c)
        if orientation is not None:
            H = orientation.apply(H)
        return MU0 * H

    def moment(self, c):
        return self.s.moment(c)


def solve_in_field(resp, chi, sources, J0=None, position=(0, 0, 0), orientation=None):
    """One soft/hard cuboid with susceptibility chi in the field of magpylib sources."""
    H = magpy.getH(sources, resp.sample_points(position, orientation), sumup=True) if sources else None
    return resp.solve(chi, H, J0=J0, orientation=orientation)


class SymCuboidResponse(CuboidResponse):
    """Same response, split into the 8 mirror-parity blocks: one Schur
    decomposition per block (offline / 64), triangular solves per block."""

    def __init__(self, dimension, m=4, g=2.5, nq=4, nq_rhs=3, verbose=False):
        from sym import irrep_bases

        t0 = time.perf_counter()
        self.dimension = np.asarray(dimension, float)
        self.pan = layer_panels(self.dimension / 2, m, g)
        self.s = P1(self.pan, nq=nq)
        self.nq_rhs = nq_rhs
        G = self.s.assemble()
        t1 = time.perf_counter()
        d = np.sqrt(self.s.mass)
        self.d = d
        Gh = G / d[:, None] / d[None, :]
        self.blocks = []
        self._fixed = {}
        ns = []
        for e, Bs in enumerate(irrep_bases(self.pan)):
            Bd = Bs.toarray()
            Ge = Bd.T @ Gh @ Bd
            eq_first = e == 0
            if eq_first:
                T, Z, _ = scipy.linalg.schur(Ge, output="complex", sort=lambda x: x.real > 0.5 - 1e-3)
            else:
                T, Z = scipy.linalg.schur(Ge, output="complex")
            k = 1 if eq_first else 0
            if eq_first and abs(T[0, 0] - 0.5) > 1e-2:
                raise RuntimeError("equilibrium mode not isolated")
            if (T.diagonal()[k:].real > 0.5 - 1e-3).any():
                raise RuntimeError("stray eigenvalue near 1/2")
            if eq_first:
                self.z1 = (Bd @ Z[:, 0]).real / d
            self.blocks.append((Bs.T.tocsr(), Bs.tocsr(), Z[:, k:].conj().T.copy(), T[k:, k:], Z[:, k:].copy()))
            ns.append(0.5 - T.diagonal()[k:].real)
        self.n = np.concatenate(ns)
        self.charge = np.zeros(self.s.Nd)
        self.charge[0::3] = self.pan.area
        pts, wts, S, T_ = self.s.gauss_points(nq_rhs)
        self._pts, self._wts, self._S, self._T = pts, wts, S, T_
        self.t_build = (t1 - t0, time.perf_counter() - t1)
        if verbose:
            print(f"shape {self.dimension}: {self.s.Nd} dofs in 8 blocks of {self.s.Nd // 8}, assembly {self.t_build[0]:.2f}s, block Schur {self.t_build[1]:.2f}s, n in [{self.n.min():.3f}, {self.n.max():.3f}]")

    def _apply(self, chi, F):
        x = F / self.d
        c = np.zeros(self.s.Nd)
        mu = 1 / chi + 0.5
        R = self._fixed.get(chi)
        for e, (Bt, B, Zh, T, Z) in enumerate(self.blocks):
            xe = Bt @ x
            if R is not None:  # pre-inverted block for this chi
                c += B @ (R[e] @ xe)
                continue
            A = -T
            A.flat[:: len(T) + 1] += mu
            y = scipy.linalg.solve_triangular(A, Zh @ xe, check_finite=False)
            c += B @ (Z @ y).real
        c /= self.d
        return c - (self.charge @ c) / (self.charge @ self.z1) * self.z1

    def fix_chi(self, chi):
        """Pre-invert every block for one susceptibility: then each new
        external field costs 8 small real mat-vecs."""
        R = []
        for Bt, B, Zh, T, Z in self.blocks:
            A = -T
            A.flat[:: len(T) + 1] += 1 / chi + 0.5
            R.append((Z @ scipy.linalg.solve_triangular(A, Zh, check_finite=False)).real)
        self._fixed[chi] = R

    def solve(self, chi, H_samples=None, J0=None, orientation=None):
        F = np.zeros(self.s.Nd)
        if H_samples is not None:
            H = np.asarray(H_samples).reshape(len(self._S), self.pan.N, 3)
            if orientation is not None:
                H = orientation.inv().apply(H.reshape(-1, 3)).reshape(H.shape)
            F += self.load(np.einsum("gnd,nd->gn", H, self.pan.normal))
        if J0 is not None:
            F += self.s.rhs_const_normal(self.pan.normal @ (np.asarray(J0) / MU0)) / chi
        return self._apply(chi, F)
