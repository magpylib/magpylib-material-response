//! The field of a straight current segment (magpylib's `Polyline`):
//! `_array_api/polyline.py`'s `polyline_field`, operation for operation. The
//! formula, and why it is rationalized beyond the ends, are in that module's
//! docstring.

type V3 = [f64; 3];

/// `_array_api/polyline.RTOL_LINE`: distance from the line, in segment
/// lengths, taken as on it (H 0 there, as magpylib).
const RTOL_LINE: f64 = 1e-15;

#[inline(always)]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn sum3(a: V3) -> f64 {
    a[0] + a[1] + a[2]
}

#[inline(always)]
fn mul(a: V3, b: V3) -> V3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

/// H in A/m at `x` of a segment from `a` to `b` carrying `current` A.
pub fn polyline_h(x: V3, a: V3, b: V3, current: f64) -> V3 {
    let lv = sub(b, a);
    let l = sum3(mul(lv, lv)).sqrt();
    if l == 0.0 {
        return [0.0; 3];
    }
    let e = lv.map(|v| v / l);
    let r1 = sub(x, a);
    let r2 = sub(x, b);
    let big_r1 = sum3(mul(r1, r1)).sqrt();
    let big_r2 = sum3(mul(r2, r2)).sqrt();
    let t1 = sum3(mul(e, r1));
    let t2 = sum3(mul(e, r2));
    let c = [
        e[1] * r1[2] - e[2] * r1[1],
        e[2] * r1[0] - e[0] * r1[2],
        e[0] * r1[1] - e[1] * r1[0],
    ];
    let d2 = sum3(mul(c, c));
    if d2 <= (RTOL_LINE * l) * (RTOL_LINE * l) {
        return [0.0; 3];
    }
    let k = if t1 * t2 > 0.0 {
        l * (t1 + t2) / (big_r1 * big_r2 * (t1 * big_r2 + t2 * big_r1))
    } else {
        (t1 / big_r1 - t2 / big_r2) / d2
    };
    let s = current * k / (4.0 * std::f64::consts::PI);
    c.map(|v| s * v)
}
