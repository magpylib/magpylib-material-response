//! The field of a homogeneously polarized cylinder (magpylib's `Cylinder`):
//! `_array_api/cylinder.py`'s `cylinder_field`, one observer at a time. The
//! cylinder is centered on the origin, its axis along `z`; the formulas, the
//! series and where each is taken are in that module's docstrings.

use std::f64::consts::PI;

use crate::cuboid::MU0;

type V3 = [f64; 3];

/// `_array_api/cylinder.RTOL_EDGE`.
const RTOL_EDGE: f64 = 1e-15;
/// `_array_api.ERRTOL`: Bulirsch's stop in doubles.
const ERRTOL: f64 = 1e-8;
/// `_array_api/cylinder.SERIES_RADIUS`.
const SERIES_RADIUS: f64 = 0.25;
/// `_array_api/cylinder.series_terms` in doubles.
const TERMS: usize = 15;

/// Bulirsch's `cel(kc, p, c, s)` for each `(c, s)` at one `kc` and `p`,
/// either branch: the iterations share every variable but `c` and `s`, and
/// the stop. A NaN stops it; `kc` must not be 0.
#[inline(always)]
fn cel<const N: usize>(kc: f64, p: f64, pairs: [(f64, f64); N]) -> [f64; N] {
    let f = kc * kc;
    let (mut p, mut cs) = if p > 0.0 {
        let root = p.sqrt();
        (root, pairs.map(|(c, s)| (c, s / root)))
    } else {
        let g = 1.0 - p;
        let pn = ((f - p) / g).sqrt();
        let cs = pairs.map(|(c, s)| {
            let cn = (c - s) / g;
            (cn, -(1.0 - f) * (s - c * p) / (g * g * pn) + cn * pn)
        });
        (pn, cs)
    };
    let (mut em, mut qc) = (1.0, kc.abs());
    let mut kk = qc;
    loop {
        let g = kk / p;
        for (c, s) in cs.iter_mut() {
            (*c, *s) = (*c + *s / p, 2.0 * (*s + *c * g));
        }
        p += g;
        let g = em;
        em += qc;
        // `!(a > b)`, not `a <= b`: a NaN stops the loop
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !((g - qc).abs() > g * ERRTOL) {
            break;
        }
        qc = 2.0 * kk.sqrt();
        kk = em * qc;
    }
    let den = em * (em + p);
    cs.map(|(c, s)| (PI / 2.0) * (s + c * em) / den)
}

/// `B_r` and `B_z` per unit `J_z` (Derby).
fn axial(z0: f64, r: f64, z: f64) -> (f64, f64) {
    let (zph, zmh) = (z + z0, z - z0);
    let (dpr, dmr) = (1.0 + r, 1.0 - r);
    let sq0 = (zmh * zmh + dpr * dpr).sqrt();
    let sq1 = (zph * zph + dpr * dpr).sqrt();
    let k0 = (zmh * zmh + dmr * dmr).sqrt() / sq0;
    let k1 = (zph * zph + dmr * dmr).sqrt() / sq1;
    let gamma = dmr / dpr;
    let [br1] = cel(k1, 1.0, [(1.0, -1.0)]);
    let [br0] = cel(k0, 1.0, [(1.0, -1.0)]);
    let [bz1] = cel(k1, gamma * gamma, [(1.0, gamma)]);
    let [bz0] = cel(k0, gamma * gamma, [(1.0, gamma)]);
    let br = (br1 / sq1 - br0 / sq0) / PI;
    let bz = (zph * bz1 / sq1 - zmh * bz0 / sq0) / dpr / PI;
    (br, bz)
}

/// `S_f`, `D` and `S_z / r` near the axis: `_transverse_series`.
fn series(z0: f64, r: f64, z: f64) -> (f64, f64, f64) {
    let side = |zs: f64| {
        let rho2 = 1.0 + zs * zs;
        let rho = rho2.sqrt();
        let x = -zs / rho;
        let u2 = r * r / rho2;
        let (mut c_prev, mut c_odd) = (1.0, 3.0 * x);
        let mut w = -1.0 / 8.0;
        let (mut sf, mut dr, mut sz) = (0.0, 0.0, 0.0);
        for k in 1..TERMS {
            let k = k as f64;
            let c_even = ((4.0 * k + 1.0) * x * c_odd - (2.0 * k + 1.0) * c_prev) / (2.0 * k);
            sf += w * u2 * c_odd;
            dr += (2.0 * k) * w * c_odd;
            sz += (2.0 * k) * w * u2 * c_even;
            w = w * (-(2.0 * k) * (2.0 * k + 1.0) * u2) / (4.0 * (k + 1.0) * (k + 2.0));
            c_prev = c_even;
            c_odd = ((4.0 * k + 3.0) * x * c_even - (2.0 * k + 2.0) * c_odd) / (2.0 * k + 1.0);
        }
        (rho, sf / rho2, dr / (rho2 * rho2), sz / (rho2 * rho))
    };
    let (zp, zm) = (z + z0, z - z0);
    let (rp, fp, dp, sp) = side(zp);
    let (rm, fm, dm, sm) = side(zm);
    // the leading terms without the cancellation far along the axis
    let d2 = 4.0 * z * z0;
    let f0 = if zp * zm > 0.0 {
        d2 / (rp * rm * (zp * rm + zm * rp))
    } else {
        zp / rp - zm / rm
    };
    let s0 = -d2 / (rm + rp) * (rm * rm + rm * rp + rp * rp) / (rp * rp * rp * rm * rm * rm);
    (f0 + fp - fm, dp - dm, s0 + sp - sm)
}

/// `S_f`, `D` and `S_z / r` off the axis: `_transverse_general`.
fn general(z0: f64, r: f64, z: f64) -> (f64, f64, f64) {
    let (rp, rm) = (r + 1.0, r - 1.0);
    let r2 = r * r;
    let hull = rm == 0.0;
    let rm_s = if hull { 1.0 } else { rm };
    let p_pi = 1.0 + 4.0 * r / (rm_s * rm_s);
    let side = |zs: f64| {
        let a2 = zs * zs + rm * rm;
        let a = a2.sqrt();
        let kc = (1.0 + 4.0 * r / a2).sqrt();
        let [k, e] = cel(kc, 1.0, [(1.0, 1.0), (1.0, kc * kc)]);
        let [p] = cel(kc, p_pi, [(1.0, 1.0)]);
        (a, k, e, if hull { 0.0 } else { p })
    };
    let (zp, zm) = (z + z0, z - z0);
    let (ap, kp, ep, pp) = side(zp);
    let (am, km, em, pm) = side(zm);
    let xr = -zm * am * em + zp * ap * ep + zm / am * (2.0 + zm * zm) * km
        - zp / ap * (2.0 + zp * zp) * kp
        + (zm / am * pm - zp / ap * pp) * rp * (r2 + 1.0) / rm_s;
    let xf = zm * am * em - zp * ap * ep - zm / am * (2.0 + zm * zm + 2.0 * r2) * km
        + zp / ap * (2.0 + zp * zp + 2.0 * r2) * kp
        + zm / am * rp * rp * pm
        - zp / ap * rp * rp * pp;
    let xz = am * em - ap * ep - (1.0 + zm * zm + r2) / am * km + (1.0 + zp * zp + r2) / ap * kp;
    (
        xf / (PI * r2),
        (xr - xf) / (PI * r2 * r2),
        2.0 * xz / (PI * r2),
    )
}

/// `mu_0 H` in T and whether `x` is inside (and the field general), of a
/// cylinder of `dimension` (diameter, height) in m, signs ignored, and
/// `polarization` in T.
fn mu0_h(x: V3, dimension: [f64; 2], pol: V3) -> (V3, bool) {
    let radius = dimension[0].abs() / 2.0;
    let half = dimension[1].abs() / 2.0;
    let rad = (x[0] * x[0] + x[1] * x[1]).sqrt();
    let inside = x[2].abs() <= half && rad <= radius;
    if radius == 0.0 || half == 0.0 || pol == [0.0; 3] {
        return ([0.0; 3], false);
    }
    let r = rad / radius;
    let zz = x[2] / radius;
    let z0 = half / radius;
    let gap_edge = zz.abs() - z0;
    if (r - 1.0).abs() <= RTOL_EDGE && gap_edge.abs() <= RTOL_EDGE * z0 {
        return ([0.0; 3], false);
    }
    // so near the edge that the distance squares below the smallest normal
    // number; a NaN stays general
    if (r - 1.0) * (r - 1.0) + gap_edge * gap_edge < f64::MIN_POSITIVE {
        return ([0.0; 3], false);
    }
    let (xs, ys) = (x[0] / radius, x[1] / radius);
    let to_hull = 1.0
        + if gap_edge > 0.0 {
            gap_edge * gap_edge
        } else {
            0.0
        };
    let (sf, d, sz, br_r, bz) = if r * r < SERIES_RADIUS * SERIES_RADIUS * to_hull {
        let (sf, d, sz) = series(z0, r, zz);
        (sf, d, sz, -sz / 4.0, (2.0 * sf + r * r * d) / 4.0)
    } else {
        let (sf, d, sz) = general(z0, r, zz);
        let (br, bz) = axial(z0, r, zz);
        (sf, d, sz, br / r, bz)
    };
    let [jx, jy, jz] = pol;
    let jr = jx * xs + jy * ys;
    let inside_jz = if inside { jz } else { 0.0 };
    (
        [
            jz * br_r * xs - (sf * jx + d * jr * xs) / 4.0,
            jz * br_r * ys - (sf * jy + d * jr * ys) / 4.0,
            jz * bz - inside_jz - sz * jr / 4.0,
        ],
        inside,
    )
}

/// B in T at `x` of a cylinder of `dimension` (diameter, height) in m and
/// `polarization` in T: B and H 0 on the edge, for a zero polarization or
/// dimension, and so near the edge that the distance squares below the
/// smallest normal number; J added inside, the surface included.
pub fn cylinder_b(x: V3, dimension: [f64; 2], polarization: V3) -> V3 {
    let (h, inside) = mu0_h(x, dimension, polarization);
    if inside {
        [
            h[0] + polarization[0],
            h[1] + polarization[1],
            h[2] + polarization[2],
        ]
    } else {
        h
    }
}

/// H in A/m: `cylinder_b`'s `mu_0 H` over `mu_0`.
pub fn cylinder_h(x: V3, dimension: [f64; 2], polarization: V3) -> V3 {
    mu0_h(x, dimension, polarization).0.map(|v| v / MU0)
}
