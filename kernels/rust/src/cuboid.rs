//! magpylib's homogeneously magnetized cuboid, in its own frame: a port of
//! `magnet_cuboid_Bfield` and `_BHJM_magnet_cuboid` from magpylib's main
//! branch (with #1066's stable logarithms), operation for operation. The
//! cuboid is centered at the origin, its sides along the axes.

use std::f64::consts::PI;

/// scipy's `mu_0` (CODATA 2022), as magpylib uses it.
pub const MU0: f64 = 1.25663706127e-06;

/// `sqrt(q^2 + t2) + q` without cancellation.
#[inline(always)]
fn sqrt_plus_signed(q: f64, radius: f64, t2: f64) -> f64 {
    if q < 0.0 {
        t2 / (radius - q)
    } else {
        radius + q
    }
}

const FLIP_X: [[f64; 3]; 3] = [[1., -1., -1.], [-1., 1., 1.], [-1., 1., 1.]];
const FLIP_Y: [[f64; 3]; 3] = [[1., -1., 1.], [-1., 1., -1.], [1., -1., 1.]];
const FLIP_Z: [[f64; 3]; 3] = [[1., 1., -1.], [1., 1., -1.], [-1., -1., 1.]];

/// `magnet_cuboid_Bfield` for one observer: B in units of the polarization.
#[inline]
pub fn cuboid_b_core(obs: [f64; 3], dim: [f64; 3], pol: [f64; 3]) -> [f64; 3] {
    let (a, b, c) = (dim[0] / 2.0, dim[1] / 2.0, dim[2] / 2.0);
    let [mut x, mut y, mut z] = obs;
    let mut qs = [[1.0f64; 3]; 3];
    let mut flip = |f: &[[f64; 3]; 3]| {
        for i in 0..3 {
            for j in 0..3 {
                qs[i][j] *= f[i][j];
            }
        }
    };
    // to the bottom quadrant, where no indeterminate form arises
    if x < 0.0 {
        x = -x;
        flip(&FLIP_X);
    }
    if y > 0.0 {
        y = -y;
        flip(&FLIP_Y);
    }
    if z > 0.0 {
        z = -z;
        flip(&FLIP_Z);
    }
    let (xma, xpa) = (x - a, x + a);
    let (ymb, ypb) = (y - b, y + b);
    let (zmc, zpc) = (z - c, z + c);
    let (xma2, xpa2) = (xma * xma, xpa * xpa);
    let (ymb2, ypb2) = (ymb * ymb, ypb * ypb);
    let (zmc2, zpc2) = (zmc * zmc, zpc * zpc);

    let mmm = (xma2 + ymb2 + zmc2).sqrt();
    let pmp = (xpa2 + ymb2 + zpc2).sqrt();
    let pmm = (xpa2 + ymb2 + zmc2).sqrt();
    let mmp = (xma2 + ymb2 + zpc2).sqrt();
    let mpm = (xma2 + ypb2 + zmc2).sqrt();
    let ppp = (xpa2 + ypb2 + zpc2).sqrt();
    let ppm = (xpa2 + ypb2 + zmc2).sqrt();
    let mpp = (xma2 + ypb2 + zpc2).sqrt();

    let sp = sqrt_plus_signed;
    let ff2x = (sp(xma, mmm, ymb2 + zmc2) * (xpa + ppm) * (xpa + pmp) * sp(xma, mpp, ypb2 + zpc2))
        .ln()
        - ((xpa + pmm) * sp(xma, mpm, ypb2 + zmc2) * sp(xma, mmp, ymb2 + zpc2) * (xpa + ppp)).ln();
    let ff2y = ((-ymb + mmm)
        * sp(-ypb, ppm, xpa2 + zmc2)
        * (-ymb + pmp)
        * sp(-ypb, mpp, xma2 + zpc2))
    .ln()
        - ((-ymb + pmm) * sp(-ypb, mpm, xma2 + zmc2) * (-ymb + mmp) * sp(-ypb, ppp, xpa2 + zpc2))
            .ln();
    let ff2z = ((-zmc + mmm)
        * (-zmc + ppm)
        * sp(-zpc, pmp, xpa2 + ymb2)
        * sp(-zpc, mpp, xma2 + ypb2))
    .ln()
        - ((-zmc + pmm) * (-zmc + mpm) * sp(-zpc, mmp, xma2 + ymb2) * sp(-zpc, ppp, xpa2 + ypb2))
            .ln();

    let at = f64::atan2;
    let ff1x = at(ymb * zmc, xma * mmm) - at(ymb * zmc, xpa * pmm) - at(ypb * zmc, xma * mpm)
        + at(ypb * zmc, xpa * ppm)
        - at(ymb * zpc, xma * mmp)
        + at(ymb * zpc, xpa * pmp)
        + at(ypb * zpc, xma * mpp)
        - at(ypb * zpc, xpa * ppp);
    let ff1y = at(xma * zmc, ymb * mmm) - at(xpa * zmc, ymb * pmm) - at(xma * zmc, ypb * mpm)
        + at(xpa * zmc, ypb * ppm)
        - at(xma * zpc, ymb * mmp)
        + at(xpa * zpc, ymb * pmp)
        + at(xma * zpc, ypb * mpp)
        - at(xpa * zpc, ypb * ppp);
    let ff1z = at(xma * ymb, zmc * mmm) - at(xpa * ymb, zmc * pmm) - at(xma * ypb, zmc * mpm)
        + at(xpa * ypb, zmc * ppm)
        - at(xma * ymb, zpc * mmp)
        + at(xpa * ymb, zpc * pmp)
        + at(xma * ypb, zpc * mpp)
        - at(xpa * ypb, zpc * ppp);

    let [px, py, pz] = pol;
    let bx = px * ff1x * qs[0][0] + py * ff2z * qs[1][0] + pz * ff2y * qs[2][0];
    let by = px * ff2z * qs[0][1] + py * ff1y * qs[1][1] - pz * ff2x * qs[2][1];
    let bz = px * ff2y * qs[0][2] - py * ff2x * qs[1][2] + pz * ff1z * qs[2][2];
    [bx / (4.0 * PI), by / (4.0 * PI), bz / (4.0 * PI)]
}

/// Whether `obs` is inside the cuboid (faces included), and whether the
/// field is computed there at all: not on edges and corners, nor for a zero
/// polarization or dimension, all (0, 0, 0) by convention (a non-finite
/// polarization's component stays NaN: a source parameter out of the
/// reference files' scope).
#[inline]
fn classify(obs: [f64; 3], dim: [f64; 3], pol: [f64; 3]) -> (bool, bool) {
    const RTOL_SURFACE: f64 = 1e-15;
    let half = dim.map(|d| d.abs() / 2.0);
    let dist = [0, 1, 2].map(|k| obs[k].abs() - half[k]);
    let surf = [0, 1, 2].map(|k| dist[k].abs() < RTOL_SURFACE * half[k]);
    let ins = [0, 1, 2].map(|k| dist[k] < RTOL_SURFACE * half[k]);
    let inside = ins[0] && ins[1] && ins[2];
    let edge = (surf[1] && surf[2] && ins[0])
        || (surf[0] && surf[2] && ins[1])
        || (surf[0] && surf[1] && ins[2]);
    let pol_not_null = !(pol[0] == 0.0 && pol[1] == 0.0 && pol[2] == 0.0);
    let dim_not_null = half[0] * half[1] * half[2] != 0.0;
    (inside, pol_not_null && dim_not_null && !edge)
}

/// B in T of one cuboid at one observer, as `_BHJM_magnet_cuboid("B", ...)`.
#[inline]
pub fn cuboid_b(obs: [f64; 3], dim: [f64; 3], pol: [f64; 3]) -> [f64; 3] {
    let (_, general) = classify(obs, dim, pol);
    if general {
        cuboid_b_core(obs, dim, pol)
    } else {
        pol.map(|p| p * 0.0) // numpy's `BHJM *= 0`: NaN stays NaN
    }
}

/// H in A/m of one cuboid at one observer, as `_BHJM_magnet_cuboid("H", ...)`.
#[inline]
pub fn cuboid_h(obs: [f64; 3], dim: [f64; 3], pol: [f64; 3]) -> [f64; 3] {
    let (inside, _) = classify(obs, dim, pol);
    let mut b = cuboid_b(obs, dim, pol);
    if inside {
        for k in 0..3 {
            b[k] -= pol[k];
        }
    }
    b.map(|v| v / MU0)
}
