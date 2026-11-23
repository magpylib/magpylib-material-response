//! The fields of a magnetic dipole and of a homogeneously polarized sphere
//! (magpylib's `Dipole` and `Sphere`): `_array_api/sphere.py`'s
//! `dipole_field` and `sphere_field`, operation for operation.

use crate::cuboid::MU0;

type V3 = [f64; 3];

#[inline(always)]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// H in A/m at `x` of a dipole of `moment` A m^2 at the origin; NaN at the
/// dipole, as magpylib's `getB` and `func.dipole_field`.
pub fn dipole_h(x: V3, moment: V3) -> V3 {
    let r2 = dot(x, x);
    if r2 == 0.0 {
        return [f64::NAN; 3];
    }
    let r = r2.sqrt();
    let mr = dot(moment, x);
    let (r3, r5) = (r * r * r, r * r * r * r * r);
    let s = 3.0 * (mr / r5);
    [0, 1, 2].map(|k| (s * x[k] - moment[k] / r3) / (4.0 * std::f64::consts::PI))
}

/// B in T at `x` of a sphere of `diameter` m and `polarization` T at the
/// origin: `2 J / 3` inside, the surface included; the dipole `J V / mu_0`
/// outside.
pub fn sphere_b(x: V3, diameter: f64, polarization: V3) -> V3 {
    let radius = diameter.abs() / 2.0;
    let r2 = dot(x, x);
    let r = r2.sqrt();
    if r > radius {
        let jr = dot(polarization, x);
        let r5 = r * r * r * r * r;
        let s = radius * radius * radius / 3.0;
        [0, 1, 2].map(|k| (3.0 * jr * x[k] - polarization[k] * r2) / r5 * s)
    } else {
        polarization.map(|j| 2.0 * j / 3.0)
    }
}

/// H in A/m: `(B - J) / mu_0`, `J` the polarization inside, 0 outside.
pub fn sphere_h(x: V3, diameter: f64, polarization: V3) -> V3 {
    let b = sphere_b(x, diameter, polarization);
    // not outside: a NaN observer counts inside, as in Python and magpylib
    let inside = !(dot(x, x).sqrt() > diameter.abs() / 2.0);
    [0, 1, 2].map(|k| (b[k] - if inside { polarization[k] } else { 0.0 }) / MU0)
}

/// B in T of a dipole: `mu_0 H`.
pub fn dipole_b(x: V3, moment: V3) -> V3 {
    dipole_h(x, moment).map(|h| MU0 * h)
}
