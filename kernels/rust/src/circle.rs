//! The field of a circular current loop (magpylib's `Circle`):
//! `_array_api/circle.py`'s `circle_field`, one observer at a time, as
//! magpylib's scalar path, its two integrals in one loop. The loop lies in
//! the `z = 0` plane, centered on the origin; the formula (Ortner et al.,
//! Magnetism 2023) is in that module's docstring.

use std::f64::consts::PI;

type V3 = [f64; 3];

/// `_array_api/circle.RTOL_WIRE`: distance from the wire in the loop's
/// plane, in radii, taken as on it (H 0 there, as magpylib).
const RTOL_WIRE: f64 = 1e-15;
/// `_array_api/circle.ERRTOL`: Bulirsch's stop in doubles.
const ERRTOL: f64 = 1e-8;

/// Bulirsch's `cel` from the state after its first step (`qc = kk = q`,
/// `p = em = p0`, `g = 1`), for two pairs `(cc, ss)` at once: their
/// iterations share every other variable and the stop, so each result is
/// bit for bit the one a loop of its own gives. A NaN stops it.
#[inline(always)]
fn cel_pair(q: f64, p0: f64, cs1: (f64, f64), cs2: (f64, f64)) -> (f64, f64) {
    let (mut qc, mut kk, mut p, mut em, mut g) = (q, q, p0, p0, 1.0);
    let ((mut c1, mut s1), (mut c2, mut s2)) = (cs1, cs2);
    while (g - qc).abs() > g * ERRTOL {
        qc = 2.0 * kk.sqrt();
        kk = em * qc;
        g = kk / p;
        (c1, s1) = (c1 + s1 / p, 2.0 * (s1 + c1 * g));
        (c2, s2) = (c2 + s2 / p, 2.0 * (s2 + c2 * g));
        p += g;
        g = em;
        em += qc;
    }
    let den = em * (em + p);
    (
        (PI / 2.0) * (s1 + c1 * em) / den,
        (PI / 2.0) * (s2 + c2 * em) / den,
    )
}

/// H in A/m at `x` of a loop of `diameter` m (its sign ignored) carrying
/// `current` A counter-clockwise about `+z`; 0 for a loop without diameter
/// and on the wire, or so near it that the distance squares to 0.
pub fn circle_h(x: V3, diameter: f64, current: f64) -> V3 {
    let r0 = diameter.abs() / 2.0;
    let r = (x[0] * x[0] + x[1] * x[1]).sqrt();
    if r0 == 0.0 || ((r - r0).abs() < RTOL_WIRE * r0 && x[2] == 0.0) {
        return [0.0; 3];
    }
    let rho = r / r0;
    let zeta = x[2] / r0;
    let z2 = zeta * zeta;
    let x0 = z2 + (rho + 1.0) * (rho + 1.0);
    let k2 = 4.0 * rho / x0;
    let q2 = (z2 + (rho - 1.0) * (rho - 1.0)) / x0;
    if q2 == 0.0 {
        // on the wire to the precision: the iteration never stops at q = 0
        return [0.0; 3];
    }
    let q = q2.sqrt();
    let p = 1.0 + q;
    let pf = current / (4.0 * PI * r0 * x0.sqrt() * q2);
    let cc_r = k2 * 4.0 * zeta / x0;
    let ss_r = 2.0 * cc_r * q / p;
    let k4 = k2 * k2;
    let cc_z = k4 - (q2 + 1.0) * (4.0 / x0);
    let ss_z = 2.0 * q * (k4 / p - (4.0 / x0) * p);
    let (cel_r, cel_z) = cel_pair(q, p, (cc_r, ss_r), (cc_z, ss_z));
    let hr = pf * cel_r;
    let hz = -pf * cel_z;
    if r == 0.0 {
        [0.0, 0.0, hz]
    } else {
        [hr * x[0] / r, hr * x[1] / r, hz]
    }
}
