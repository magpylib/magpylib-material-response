//! Field of flat triangles whose surface charge varies linearly across them:
//! `face_charge/_triangle.py`'s `frames`, `charge_field`, `far_points` and
//! `charge_field_far`, operation for operation. The formulas and their
//! references are in that module's docstring.
//!
//! Arrays are flat and row-major, as numpy's C order: observers `x` (M, 3),
//! triangles `v` (N, 3, 3), coefficients `c` (N, 3) of `1, u, v`, far points
//! `y` (6, N, 3) and their charges (6, N).

type V3 = [f64; 3];

#[inline(always)]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

#[inline(always)]
pub(crate) fn row(a: &[f64], i: usize) -> V3 {
    [a[3 * i], a[3 * i + 1], a[3 * i + 2]]
}

/// One triangle's frame, as `frames` gives it, and what `_in_plane` derives
/// from it once per triangle: its corners in plane coordinates and the first
/// corner's coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub centroid: V3,
    pub n: V3,
    pub e1: V3,
    pub e2: V3,
    pub s: f64,
    pub area: f64,
    ub: f64,
    uc: f64,
    vc: f64,
    a_dot: V3, // a.e1, a.e2, a.n
}

/// The frames of all triangles in `v` (N, 3, 3).
pub fn frames(v: &[f64]) -> Vec<Frame> {
    v.chunks_exact(9)
        .map(|t| {
            let (a, b, c) = (row(t, 0), row(t, 1), row(t, 2));
            let cr = cross(sub(b, a), sub(c, a));
            let area = 0.5 * norm(cr);
            let n = cr.map(|x| x / (2.0 * area));
            let ab = sub(b, a);
            let lab = norm(ab);
            let e1 = ab.map(|x| x / lab);
            let e2 = cross(n, e1);
            let centroid = [0, 1, 2].map(|d| (a[d] + b[d] + c[d]) / 3.0);
            Frame {
                centroid,
                n,
                e1,
                e2,
                s: area.sqrt(),
                area,
                ub: dot(sub(b, a), e1),
                uc: dot(sub(c, a), e1),
                vc: dot(sub(c, a), e2),
                a_dot: [dot(a, e1), dot(a, e2), dot(a, n)],
            }
        })
        .collect()
}

/// `R + l`, stable where `l < 0`: there it is `R0^2 / (R - l)`.
#[inline(always)]
fn r_plus_l(r: f64, l: f64, r02: f64) -> f64 {
    if l >= 0.0 { r + l } else { r02 / (r - l) }
}

const EDGES: [(usize, usize, usize); 3] = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]; // i, j, opposite k

/// One triangle's `charge_field` at one observer: 4 pi times the field of
/// charge `c0 + c1 u + c2 v` on it.
#[inline(always)]
pub fn charge_pair(x: V3, f: &Frame, c: V3) -> V3 {
    let u = dot(x, f.e1) - f.a_dot[0];
    let v = dot(x, f.e2) - f.a_dot[1];
    let mut w = dot(x, f.n) - f.a_dot[2];
    if w == 0.0 {
        w = 1e-12 * f.s; // lifted off the plane, as numpy does
    }
    let w2 = w * w;
    let cu = [0.0, f.ub, f.uc];
    let cv = [0.0, 0.0, f.vc];
    let du = [cu[0] - u, cu[1] - u, cu[2] - u];
    let dv = [cv[0] - v, cv[1] - v, cv[2] - v];
    let r = [0, 1, 2].map(|k| (du[k] * du[k] + dv[k] * dv[k] + w2).sqrt());
    let mut den = r[0] * r[1] * r[2];
    for (i, j, k) in EDGES {
        den += (du[i] * du[j] + dv[i] * dv[j] + w2) * r[k];
    }
    let omega = 2.0 * (2.0 * f.area * w.abs()).atan2(den); // sum beta
    let (c0, c1, c2) = (c[0], c[1], c[2]);
    let (mut e0u, mut e0v, mut phi) = (0.0, 0.0, 0.0);
    let (mut slu, mut slv, mut slw) = (0.0, 0.0, 0.0);
    for (i, j, _) in EDGES {
        let (pu, pv, qu, qv) = (cu[i], cv[i], cu[j], cv[j]);
        let l = ((qu - pu) * (qu - pu) + (qv - pv) * (qv - pv)).sqrt();
        let (tu, tv) = ((qu - pu) / l, (qv - pv) / l);
        let l_minus = du[i] * tu + dv[i] * tv;
        let l_plus = du[j] * tu + dv[j] * tv;
        let p0 = du[i] * tv - dv[i] * tu;
        let r02 = p0 * p0 + w2;
        let fe = (r_plus_l(r[j], l_plus, r02) / r_plus_l(r[i], l_minus, r02)).ln();
        let p0f = p0 * fe;
        let dr = r[j] - r[i];
        let gm = c1 * tv - c2 * tu;
        e0u += tv * fe;
        e0v -= tu * fe;
        phi += p0f;
        slu -= gm * (p0f * tv + dr * tu);
        slv += gm * (p0f * tu - dr * tv);
        slw += gm * fe;
    }
    phi -= w.abs() * omega;
    let s = f.s;
    let psi = c0 + (c1 * (u - (f.ub + f.uc) / 3.0) + c2 * (v - f.vc / 3.0)) / s;
    let fu = psi * e0u - (slu + c1 * phi) / s;
    let fv = psi * e0v - (slv + c2 * phi) / s;
    let fw = psi * sign(w) * omega - w * slw / s;
    [0, 1, 2].map(|d| fu * f.e1[d] + fv * f.e2[d] + fw * f.n[d])
}

/// numpy's `sign`: -1, 0 or 1, NaN for NaN.
#[inline(always)]
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x // 0 or NaN
    }
}

/// `charge_field` at one observer: summed over all triangles.
#[inline]
pub fn charge_at(x: V3, frames: &[Frame], c: &[f64]) -> V3 {
    let mut acc = [0.0; 3];
    for (t, f) in frames.iter().enumerate() {
        let g = charge_pair(x, f, row(c, t));
        acc[0] += g[0];
        acc[1] += g[1];
        acc[2] += g[2];
    }
    acc
}

// the six points of `charge_field_far`, exact to degree 4 (Dunavant 1985)
const FAR_A: f64 = 0.816847572980459;
const FAR_B: f64 = 0.091576213509771;
const FAR_C: f64 = 0.108103018168070;
const FAR_D: f64 = 0.445948490915965;
const FAR_POINTS: [V3; 6] = [
    [FAR_A, FAR_B, FAR_B],
    [FAR_B, FAR_A, FAR_B],
    [FAR_B, FAR_B, FAR_A],
    [FAR_C, FAR_D, FAR_D],
    [FAR_D, FAR_C, FAR_D],
    [FAR_D, FAR_D, FAR_C],
];
const FAR_W1: f64 = 0.109951743655322;
const FAR_W2: f64 = 0.223381589678011;

fn far_weights() -> [f64; 6] {
    let sum = 3.0 * FAR_W1 + 3.0 * FAR_W2;
    [FAR_W1, FAR_W1, FAR_W1, FAR_W2, FAR_W2, FAR_W2].map(|w| w / sum)
}

/// `far_points`: the points `y` (6, N, 3) and the charge each stands for (6, N).
pub fn far_points(v: &[f64], frames: &[Frame], c: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = frames.len();
    let weights = far_weights();
    let mut y = vec![0.0; 6 * n * 3];
    let mut charge = vec![0.0; 6 * n];
    for (g, (bary, wg)) in FAR_POINTS.iter().zip(weights).enumerate() {
        for (t, f) in frames.iter().enumerate() {
            let tri = &v[9 * t..9 * t + 9];
            let p =
                [0, 1, 2].map(|d| bary[0] * tri[d] + bary[1] * tri[3 + d] + bary[2] * tri[6 + d]);
            let d = sub(p, f.centroid);
            let (u, w) = (dot(d, f.e1), dot(d, f.e2));
            let q = row(c, t);
            let density = q[0] + (q[1] * u + q[2] * w) / f.s;
            y[3 * (g * n + t)..3 * (g * n + t) + 3].copy_from_slice(&p);
            charge[g * n + t] = wg * density * f.area;
        }
    }
    (y, charge)
}

/// `charge_field_far` at one observer: point charges `charge` at `y`.
#[inline]
pub fn far_at(x: V3, y: &[f64], charge: &[f64]) -> V3 {
    let mut acc = [0.0; 3];
    for (p, q) in y.chunks_exact(3).zip(charge) {
        let d = [x[0] - p[0], x[1] - p[1], x[2] - p[2]];
        let r2 = dot(d, d);
        let scale = q / (r2 * r2.sqrt());
        acc[0] += scale * d[0];
        acc[1] += scale * d[1];
        acc[2] += scale * d[2];
    }
    acc
}

/// The far rule's point charges at one observer, summed: `add` takes the
/// entries `range` (an even count) of `rule` (x, y, z, q arrays), each adding
/// `q (x - p) / |x - p|^3`; `sum` gives the total.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
struct FarSum {
    x: V3,
    acc: V3,
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
impl FarSum {
    #[inline]
    fn new(x: V3) -> Self {
        FarSum { x, acc: [0.0; 3] }
    }

    #[inline]
    fn add(&mut self, rule: &[Vec<f64>; 4], range: std::ops::Range<usize>) {
        let x = self.x;
        for i in range {
            let r = [x[0] - rule[0][i], x[1] - rule[1][i], x[2] - rule[2][i]];
            let r2 = dot(r, r);
            let scale = rule[3][i] / (r2 * r2.sqrt());
            self.acc[0] += scale * r[0];
            self.acc[1] += scale * r[1];
            self.acc[2] += scale * r[2];
        }
    }

    #[inline]
    fn sum(self) -> V3 {
        self.acc
    }
}

/// [`FarSum`] two charges at a time, in WebAssembly's 128-bit lanes; the
/// lanes meet once, in `sum`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
struct FarSum {
    x: [core::arch::wasm32::v128; 3],
    acc: [core::arch::wasm32::v128; 3],
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl FarSum {
    #[inline]
    fn new(x: V3) -> Self {
        use core::arch::wasm32::f64x2_splat;
        FarSum {
            x: x.map(|v| f64x2_splat(v)),
            acc: [f64x2_splat(0.0); 3],
        }
    }

    #[inline]
    fn add(&mut self, rule: &[Vec<f64>; 4], range: std::ops::Range<usize>) {
        use core::arch::wasm32::*;
        let [xs, ys, zs, qs] = rule;
        assert!(range.end <= xs.len().min(ys.len()).min(zs.len()).min(qs.len()));
        assert_eq!(range.len() % 2, 0, "far-rule charges come in pairs");
        let [x0, x1, x2] = self.x;
        let [mut a0, mut a1, mut a2] = self.acc;
        for i in range.step_by(2) {
            // SAFETY: i + 1 < range.end <= every array's length, checked
            // above; v128_load needs no alignment
            let (px, py, pz, q) = unsafe {
                (
                    v128_load(xs.as_ptr().add(i).cast()),
                    v128_load(ys.as_ptr().add(i).cast()),
                    v128_load(zs.as_ptr().add(i).cast()),
                    v128_load(qs.as_ptr().add(i).cast()),
                )
            };
            let (rx, ry, rz) = (f64x2_sub(x0, px), f64x2_sub(x1, py), f64x2_sub(x2, pz));
            let r2 = f64x2_add(
                f64x2_add(f64x2_mul(rx, rx), f64x2_mul(ry, ry)),
                f64x2_mul(rz, rz),
            );
            let scale = f64x2_div(q, f64x2_mul(r2, f64x2_sqrt(r2)));
            a0 = f64x2_add(a0, f64x2_mul(scale, rx));
            a1 = f64x2_add(a1, f64x2_mul(scale, ry));
            a2 = f64x2_add(a2, f64x2_mul(scale, rz));
        }
        self.acc = [a0, a1, a2];
    }

    #[inline]
    fn sum(self) -> V3 {
        use core::arch::wasm32::f64x2_extract_lane;
        self.acc
            .map(|a| f64x2_extract_lane::<0>(a) + f64x2_extract_lane::<1>(a))
    }
}

/// A solved `ChargeSheet`: what `ChargeSheet._field` needs, set up once.
pub struct Sheet {
    pub frames: Vec<Frame>,
    pub c: Vec<f64>,
    pub center: V3,
    pub far_from: f64,
    /// per triangle: its longest edge squared, and its six far-rule points
    /// and charges, triangle by triangle, as x, y, z and q arrays
    pub longest2: Vec<f64>,
    pub rule: [Vec<f64>; 4],
}

/// `_triangle.PAIR_RULE`: (bound, k^2) rows; a tolerance takes the first row
/// whose bound it reaches. From `k` longest edges of a triangle's centroid a
/// pair takes the far rule, off by at most `bound` times the field its charge
/// would make all of one sign.
pub const PAIR_RULE: [(f64, f64); 12] = [
    (1.3e-3, 2.25),
    (4.1e-4, 4.0),
    (8.1e-5, 9.0),
    (2.6e-5, 16.0),
    (5.1e-6, 36.0),
    (1.6e-6, 64.0),
    (3.2e-7, 144.0),
    (1.0e-7, 256.0),
    (2.0e-8, 576.0),
    (6.2e-9, 1024.0),
    (1.3e-9, 2304.0),
    (3.9e-10, 4096.0),
];

/// `_triangle.pair_k2`: the squared distance in squared longest edges from
/// which a pair takes the far rule; infinity for the closed form throughout.
pub fn pair_k2(tolerance: f64) -> f64 {
    PAIR_RULE
        .iter()
        .find(|(bound, _)| *bound <= tolerance)
        .map_or(f64::INFINITY, |row| row.1)
}

impl Sheet {
    pub fn new(v: &[f64], c: &[f64], center: V3, far_from: f64) -> Self {
        let frames = frames(v);
        let (y, charge) = far_points(v, &frames, c);
        let n = frames.len();
        let longest2 = v
            .chunks_exact(9)
            .map(|t| {
                let (a, b, c) = (row(t, 0), row(t, 1), row(t, 2));
                let [e0, e1, e2] = [sub(b, a), sub(c, b), sub(a, c)]
                    .map(|e| e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
                e0.max(e1).max(e2)
            })
            .collect();
        let rule = std::array::from_fn(|k| {
            (0..n)
                .flat_map(|t| (0..6).map(move |g| g * n + t))
                .map(|i| if k < 3 { y[3 * i + k] } else { charge[i] })
                .collect()
        });
        Sheet {
            frames,
            c: c.to_vec(),
            center,
            far_from,
            longest2,
            rule,
        }
    }

    /// H in A/m at one observer: the far rule at or beyond `far_from` from
    /// the center; closer, the closed form, or with `k2` finite
    /// (`pair_k2`) the far rule for each triangle whose centroid is at
    /// least `k2` squared longest edges away.
    #[inline]
    pub fn h_at(&self, x: V3, k2: f64) -> V3 {
        let d = sub(x, self.center);
        let g = if norm(d) >= self.far_from {
            let mut far = FarSum::new(x);
            far.add(&self.rule, 0..6 * self.frames.len());
            far.sum()
        } else if k2.is_infinite() {
            charge_at(x, &self.frames, &self.c)
        } else {
            self.mixed_at(x, k2)
        };
        g.map(|k| k / (4.0 * std::f64::consts::PI))
    }

    /// The per-pair rule at one observer, the squares summed x, y, z as
    /// `_triangle.far_pairs`.
    #[inline]
    fn mixed_at(&self, x: V3, k2: f64) -> V3 {
        let (mut near, mut far) = ([0.0; 3], FarSum::new(x));
        for (t, f) in self.frames.iter().enumerate() {
            let d = sub(x, f.centroid);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 >= k2 * self.longest2[t] {
                far.add(&self.rule, 6 * t..6 * t + 6);
            } else {
                let g = charge_pair(x, f, row(&self.c, t));
                near[0] += g[0];
                near[1] += g[1];
                near[2] += g[2];
            }
        }
        let far = far.sum();
        [near[0] + far[0], near[1] + far[1], near[2] + far[2]]
    }
}
