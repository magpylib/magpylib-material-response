"""Face-charge BEM for a homogeneous linear cuboid (reference solver).

Units: H and M in A/m, sigma (surface charge) in A/m. Material law
M = M0 + chi H. Inside a homogeneous linear body div M = 0, so the body is
fully described by the charge sigma = M.n on its six faces, solving

    (1 + chi/2) sigma - chi K' sigma = chi H_ext.n + M0.n

with K' the adjoint double-layer (Neumann-Poincare) operator. Collocation at
panel centres, piecewise-constant sigma on graded panels, exact panel
integrals (closed form for a uniformly charged rectangle).
"""

import numpy as np

FOUR_PI = 4 * np.pi


def _ln_y_plus_r(x2w2, y, r):
    # ln(y + r), stable for y < 0
    return np.where(y >= 0, np.log(y + r), np.log(x2w2) - np.log(r - y))


def rect_field(P, Q, W, p1, p2, q1, q2):
    """Field per unit charge of the rectangle [p1,p2]x[q1,q2] in its plane,
    at local observer coords (P, Q, W). Broadcasts. Returns (Hp, Hq, Hw)."""
    xs = (P - p1, P - p2)
    ys = (Q - q1, Q - q2)
    W2 = W * W
    sW = np.where(W < 0, -1.0, 1.0)  # W = 0: limit from the +normal side
    aW = np.abs(W)
    hp = hq = hw = 0.0
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            s = 1.0 if ix == iy else -1.0
            r = np.sqrt(x * x + y * y + W2)
            hp = hp - s * _ln_y_plus_r(x * x + W2, y, r)
            hq = hq - s * _ln_y_plus_r(y * y + W2, x, r)
            hw = hw + s * sW * np.arctan2(x * y, aW * r)
    return hp / FOUR_PI, hq / FOUR_PI, hw / FOUR_PI


def graded_edges(n, kind="cheb", g=2.0):
    """Panel edges on [-1, 1], refined toward both ends."""
    t = np.linspace(-1, 1, n + 1)
    if kind == "cheb":
        return -np.cos(np.pi * (t + 1) / 2)
    if kind == "uniform":
        return t
    if kind == "power":  # symmetric power grading, exponent g
        return np.sign(t) * (1 - (1 - np.abs(t)) ** g)
    raise ValueError(kind)


def geo_edges(a, h, L, r):
    """Edges on [-a, a]: uniform panels of about h, then the two end panels
    split geometrically L times with ratio r (an hp-style edge mesh)."""
    n = max(2, int(np.ceil(2 * a / h)))
    e = np.linspace(-a, a, n + 1)
    d = (e[1] - e[0]) * r ** np.arange(1, L + 1)
    return np.unique(np.concatenate([-a + d, e, a - d]))


def geo_panels(half, h, L=4, r=0.25):
    return Panels(half, 0, axis_edges=[geo_edges(a, h, L, r) for a in half])


class Panels:
    """Panels on the six faces of a cuboid centred at the origin."""

    def __init__(self, half, n, kind="cheb", g=2.0, axis_edges=None):
        half = np.asarray(half, float)
        self.half = half
        if axis_edges is None:
            e = graded_edges(n, kind, g)
            axis_edges = [e * h for h in half]
        self.axis_edges = axis_edges
        rows = []
        for k in range(3):
            i, j = [a for a in range(3) if a != k]
            ei, ej = axis_edges[i], axis_edges[j]
            for s in (1.0, -1.0):
                P1, Q1 = np.meshgrid(ei[:-1], ej[:-1], indexing="ij")
                P2, Q2 = np.meshgrid(ei[1:], ej[1:], indexing="ij")
                for a in zip(P1.ravel(), P2.ravel(), Q1.ravel(), Q2.ravel()):
                    rows.append((k, s, i, j, s * half[k], *a))
        r = np.array(rows)
        self.k = r[:, 0].astype(int)
        self.sign = r[:, 1]
        self.i = r[:, 2].astype(int)
        self.j = r[:, 3].astype(int)
        self.w0 = r[:, 4]
        self.p1, self.p2, self.q1, self.q2 = r[:, 5], r[:, 6], r[:, 7], r[:, 8]
        self.N = len(r)
        self.area = (self.p2 - self.p1) * (self.q2 - self.q1)
        c = np.zeros((self.N, 3))
        idx = np.arange(self.N)
        c[idx, self.k] = self.w0
        c[idx, self.i] = 0.5 * (self.p1 + self.p2)
        c[idx, self.j] = 0.5 * (self.q1 + self.q2)
        self.center = c
        nrm = np.zeros((self.N, 3))
        nrm[idx, self.k] = self.sign
        self.normal = nrm
        self.face = 2 * self.k + (self.sign < 0)  # 0..5

    def field(self, obs, sigma, chunk=2_000_000):
        """H (A/m) at obs (M,3) from charges sigma (N,) or (N, m)."""
        obs = np.atleast_2d(obs)
        sig = sigma if sigma.ndim == 2 else sigma[:, None]
        out = np.zeros((len(obs), 3, sig.shape[1]))
        step = max(1, chunk // self.N)
        for a in range(0, len(obs), step):
            o = obs[a : a + step]
            G = self.field_matrix(o)  # (m, 3, N)
            out[a : a + step] = G @ sig
        return out if sigma.ndim == 2 else out[..., 0]

    def field_matrix(self, obs):
        """(M, 3, N) field per unit charge of each panel."""
        obs = np.atleast_2d(obs)
        idx = np.arange(self.N)
        P = obs[:, self.i]
        Q = obs[:, self.j]
        W = obs[:, self.k] - self.w0
        hp, hq, hw = rect_field(P, Q, W, self.p1, self.p2, self.q1, self.q2)
        G = np.zeros((len(obs), 3, self.N))
        G[:, self.i, idx] = hp
        G[:, self.j, idx] = hq
        G[:, self.k, idx] = hw
        return G

    def kprime(self):
        """K' matrix: normal field at panel centres from other faces."""
        N = self.N
        K = np.zeros((N, N))
        for fa in range(6):
            ta = np.flatnonzero(self.face == fa)
            ka = self.k[ta[0]]
            na = self.sign[ta[0]]
            for fb in range(6):
                if fb == fa:
                    continue
                sb = np.flatnonzero(self.face == fb)
                ib, jb, kb = self.i[sb[0]], self.j[sb[0]], self.k[sb[0]]
                Pp = self.center[ta, ib][:, None]
                Qq = self.center[ta, jb][:, None]
                Ww = self.center[ta, kb][:, None] - self.w0[sb][None, :]
                hp, hq, hw = rect_field(
                    Pp, Qq, Ww, self.p1[sb], self.p2[sb], self.q1[sb], self.q2[sb]
                )
                comp = {ib: hp, jb: hq, kb: hw}[ka]
                K[np.ix_(ta, sb)] = na * comp
        return K


def kprime_galerkin(pan, nq=3):
    """Galerkin K': panel-averaged normal flux. Entry (i, j) is the flux of
    panel j's unit charge through panel i, divided by A_i. Computed as the
    integral over panel j of the solid angle of panel i, which is bounded
    where the field itself is log-singular (the shared edge)."""
    g, w = np.polynomial.legendre.leggauss(nq)
    N = pan.N
    K = np.zeros((N, N))
    for fa in range(6):
        ta = np.flatnonzero(pan.face == fa)
        ka, ia, ja = pan.k[ta[0]], pan.i[ta[0]], pan.j[ta[0]]
        sa = pan.sign[ta[0]]
        for fb in range(6):
            if fb == fa:
                continue
            sb = np.flatnonzero(pan.face == fb)
            ib, jb, kb = pan.i[sb[0]], pan.j[sb[0]], pan.k[sb[0]]
            acc = np.zeros((len(ta), len(sb)))
            for gp, wp in zip(g, w):
                for gq, wq in zip(g, w):
                    y = np.zeros((len(sb), 3))
                    y[:, ib] = 0.5 * (pan.p1[sb] + pan.p2[sb]) + 0.5 * gp * (pan.p2[sb] - pan.p1[sb])
                    y[:, jb] = 0.5 * (pan.q1[sb] + pan.q2[sb]) + 0.5 * gq * (pan.q2[sb] - pan.q1[sb])
                    y[:, kb] = pan.w0[sb]
                    _, _, hw = rect_field(
                        y[None, :, ia], y[None, :, ja], y[None, :, ka] - pan.w0[ta][:, None],
                        pan.p1[ta][:, None], pan.p2[ta][:, None],
                        pan.q1[ta][:, None], pan.q2[ta][:, None],
                    )
                    acc += (0.25 * wp * wq) * hw
            # mean over panel j times its area = integral; flux along n_i = -sign_i hw
            K[np.ix_(ta, sb)] = -sa * acc * pan.area[sb][None, :] / pan.area[ta][:, None]
    return K


def system(pan, chi, K):
    """Bordered matrix of (1/chi + 1/2) sigma - K' sigma, total charge = 0.

    Dividing by chi keeps chi -> inf finite; the border removes the
    equilibrium-charge null mode that K' has at eigenvalue 1/2."""
    N = pan.N
    A = np.empty((N + 1, N + 1))
    A[:N, :N] = -K
    A[np.arange(N), np.arange(N)] += 1 / chi + 0.5
    A[:N, N] = 1.0
    A[N, :N] = pan.area / pan.area.sum()
    A[N, N] = 0.0
    return A


def solve(pan, chi, f, m0n=None, K=None):
    """Charges for applied normal field f (N,) or (N,m), remanent M0.n."""
    if K is None:
        K = pan.kprime()
    rhs = np.array(f, float)
    if m0n is not None:
        rhs = rhs + (m0n / chi if rhs.ndim == 1 else m0n[:, None] / chi)
    rhs = np.concatenate([rhs, np.zeros((1,) + rhs.shape[1:])])
    return np.linalg.solve(system(pan, chi, K), rhs)[: pan.N]


def moment(pan, sigma):
    """Magnetic moment (A m^2) of the charges: sum sigma A c."""
    return (pan.area * pan.center.T) @ sigma if sigma.ndim == 1 else np.einsum(
        "n,nd,nm->dm", pan.area, pan.center, sigma
    )


def solid(P, Q, W, p1, p2, q1, q2):
    """Normal field per unit charge of a rectangle (the hw part of rect_field)."""
    sW = np.where(W < 0, -1.0, 1.0)
    aW = np.abs(W)
    W2 = W * W
    out = 0.0
    for ix, x in enumerate((P - p1, P - p2)):
        x2 = x * x + W2
        for iy, y in enumerate((Q - q1, Q - q2)):
            r = np.sqrt(x2 + y * y)
            t = np.arctan2(x * y, aW * r)
            out = out + t if ix == iy else out - t
    return sW * out / FOUR_PI


def kprime_gal(pan, nq=3):
    """Same as kprime_galerkin, faster: only the solid-angle term."""
    g, w = np.polynomial.legendre.leggauss(nq)
    N = pan.N
    K = np.zeros((N, N))
    for fa in range(6):
        ta = np.flatnonzero(pan.face == fa)
        ka, ia, ja = pan.k[ta[0]], pan.i[ta[0]], pan.j[ta[0]]
        sa = pan.sign[ta[0]]
        P1, P2 = pan.p1[ta][:, None], pan.p2[ta][:, None]
        Q1, Q2 = pan.q1[ta][:, None], pan.q2[ta][:, None]
        W0 = pan.w0[ta][:, None]
        for fb in range(6):
            if fb == fa:
                continue
            sb = np.flatnonzero(pan.face == fb)
            ib, jb, kb = pan.i[sb[0]], pan.j[sb[0]], pan.k[sb[0]]
            cp = 0.5 * (pan.p1[sb] + pan.p2[sb]); hp = 0.5 * (pan.p2[sb] - pan.p1[sb])
            cq = 0.5 * (pan.q1[sb] + pan.q2[sb]); hq = 0.5 * (pan.q2[sb] - pan.q1[sb])
            acc = np.zeros((len(ta), len(sb)))
            for gp, wp in zip(g, w):
                for gq, wq in zip(g, w):
                    y = np.empty((len(sb), 3))
                    y[:, ib] = cp + gp * hp
                    y[:, jb] = cq + gq * hq
                    y[:, kb] = pan.w0[sb]
                    acc += (0.25 * wp * wq) * solid(
                        y[None, :, ia], y[None, :, ja], y[None, :, ka] - W0, P1, P2, Q1, Q2
                    )
            K[np.ix_(ta, sb)] = -sa * acc * pan.area[sb][None, :] / pan.area[ta][:, None]
    return K


def face_average(pan, Hfun, nq=2):
    """Panel averages of H.n for a field function Hfun((M,3)) -> (M,3)."""
    g, w = np.polynomial.legendre.leggauss(nq)
    pts, wts = [], []
    for gp, wp in zip(g, w):
        for gq, wq in zip(g, w):
            y = pan.center.copy()
            idx = np.arange(pan.N)
            y[idx, pan.i] = 0.5 * (pan.p1 + pan.p2) + 0.5 * gp * (pan.p2 - pan.p1)
            y[idx, pan.j] = 0.5 * (pan.q1 + pan.q2) + 0.5 * gq * (pan.q2 - pan.q1)
            pts.append(y); wts.append(0.25 * wp * wq)
    H = Hfun(np.concatenate(pts)).reshape(len(wts), pan.N, 3)
    return np.einsum("q,qnd,nd->n", np.array(wts), H, pan.normal)


def layer_edges(a, ell, m, g):
    """Edges on [-a, a]: m power-g graded panels in a layer of length ell at
    each end, uniform panels of the layer's largest size in between."""
    ell = min(ell, a)
    t = np.linspace(0, 1, m + 1)
    lay = ell * t ** g  # distance from the end, fine at the end
    lay = -a + lay
    hmax = lay[-1] - lay[-2]
    gap = 2 * (a - ell)
    k = int(np.ceil(gap / hmax - 1e-9)) if gap > 1e-12 * a else 0
    mid = np.linspace(-a + ell, a - ell, k + 1) if k else np.array([-a + ell])
    e = np.concatenate([lay, mid, -lay[::-1]])
    return np.unique(np.round(e / a, 13)) * a


def layer_panels(half, m, g=3.0, kappa=2.0):
    """Mesh sized by the smallest half-side c: end layers of length kappa*c."""
    c = np.min(half)
    return Panels(half, 0, axis_edges=[layer_edges(a, kappa * c, m, g) for a in half])
