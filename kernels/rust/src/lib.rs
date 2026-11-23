//! Field kernels for magpylib-material-response's solved face charge and
//! magpylib's cuboid, straight current segment, sphere and dipole, for the browser
//! (WebAssembly, `wasm` feature). Checked against `kernels/reference/*.json`
//! (issue #208).
//!
//! All arrays are flat f64 slices in numpy's C order. Every batch function
//! fills `out` (M, 3), one row per observer; with `threads`, and the
//! `parallel` feature, observers are split over rayon's threads.

pub mod cuboid;
pub mod polyline;
pub mod sphere;
pub mod triangle;

#[cfg(feature = "wasm")]
mod wasm;

use triangle::{Frame, Sheet, row};

/// Observers handed to one thread at a time.
const CHUNK: usize = 64;

fn each_observer<F>(x: &[f64], out: &mut [f64], threads: bool, f: F)
where
    F: Fn(usize, [f64; 3]) -> [f64; 3] + Sync,
{
    assert_eq!(x.len(), out.len(), "observers and output differ in length");
    assert_eq!(x.len() % 3, 0, "observers come in rows of 3");
    let run = |(ci, block): (usize, &mut [f64])| {
        for (k, o) in block.chunks_exact_mut(3).enumerate() {
            let i = ci * CHUNK + k;
            o.copy_from_slice(&f(i, row(x, i)));
        }
    };
    #[cfg(feature = "parallel")]
    if threads {
        use rayon::prelude::*;
        out.par_chunks_mut(3 * CHUNK).enumerate().for_each(run);
        return;
    }
    let _ = threads;
    out.chunks_mut(3 * CHUNK).enumerate().for_each(run);
}

/// `charge_field`: 4 pi times the field of triangles `v` (N, 3, 3) with
/// charge coefficients `c` (N, 3), summed, at observers `x` (M, 3).
pub fn charge_field(x: &[f64], v: &[f64], c: &[f64], out: &mut [f64], threads: bool) {
    let frames = triangle::frames(v);
    charge_field_frames(x, &frames, c, out, threads);
}

/// [`charge_field`] with the frames computed once.
pub fn charge_field_frames(x: &[f64], frames: &[Frame], c: &[f64], out: &mut [f64], threads: bool) {
    assert_eq!(
        c.len(),
        3 * frames.len(),
        "one coefficient row per triangle"
    );
    each_observer(x, out, threads, |_, p| triangle::charge_at(p, frames, c));
}

/// A solved sheet's H in A/m in its frame, as `_triangle.charge_sheet_field`:
/// `tolerance` 0 the closed form below `far_from`, as `ChargeSheet._field`;
/// above, per pair the far, middle or closed rule (`triangle::PAIR_RULE`,
/// `triangle::PAIR_RULE_MIDDLE`).
pub fn sheet_h(sheet: &Sheet, x: &[f64], tolerance: f64, out: &mut [f64], threads: bool) {
    let k2_far = triangle::pair_k2(tolerance, &triangle::PAIR_RULE);
    let k2_middle = triangle::pair_k2(tolerance, &triangle::PAIR_RULE_MIDDLE);
    each_observer(x, out, threads, |_, p| sheet.h_at(p, k2_far, k2_middle));
}

/// B in T (`field` b'B') or H in A/m (b'H') of cuboids in their own frames,
/// as `magpylib.func.cuboid_field`: `dimension` and `polarization` one row
/// (3,) for every observer, or one per observer (M, 3).
pub fn cuboid_field(
    field: u8,
    x: &[f64],
    dimension: &[f64],
    polarization: &[f64],
    out: &mut [f64],
    threads: bool,
) {
    for a in [dimension, polarization] {
        assert!(
            a.len() == 3 || a.len() == x.len(),
            "one cuboid for all observers, or one per observer"
        );
    }
    let kernel = match field {
        b'B' => cuboid::cuboid_b,
        b'H' => cuboid::cuboid_h,
        _ => panic!("field must be 'B' or 'H'"),
    };
    let at = |a: &[f64], i: usize| if a.len() == 3 { row(a, 0) } else { row(a, i) };
    each_observer(x, out, threads, |i, p| {
        kernel(p, at(dimension, i), at(polarization, i))
    });
}

/// B in T (`field` b'B') or H in A/m (b'H') of straight current segments in
/// their own frames, as `magpylib.func.polyline_field`: `start`, `end` one
/// row (3,) for every observer or one per observer (M, 3), `current` one
/// value or one per observer (M,), flowing from start to end.
pub fn polyline_field(
    field: u8,
    x: &[f64],
    start: &[f64],
    end: &[f64],
    current: &[f64],
    out: &mut [f64],
    threads: bool,
) {
    for a in [start, end] {
        assert!(
            a.len() == 3 || a.len() == x.len(),
            "one segment for all observers, or one per observer"
        );
    }
    assert!(
        current.len() == 1 || 3 * current.len() == x.len(),
        "one current for all observers, or one per observer"
    );
    let scale = match field {
        b'B' => cuboid::MU0,
        b'H' => 1.0,
        _ => panic!("field must be 'B' or 'H'"),
    };
    let at = |a: &[f64], i: usize| if a.len() == 3 { row(a, 0) } else { row(a, i) };
    each_observer(x, out, threads, |i, p| {
        let c = if current.len() == 1 {
            current[0]
        } else {
            current[i]
        };
        polyline::polyline_h(p, at(start, i), at(end, i), c).map(|v| scale * v)
    });
}

/// B in T (`field` b'B') or H in A/m (b'H') of spheres at the origin, as
/// `magpylib.func.sphere_field`: `diameter` one value or one per observer
/// (M,), `polarization` one row (3,) or one per observer (M, 3).
pub fn sphere_field(
    field: u8,
    x: &[f64],
    diameter: &[f64],
    polarization: &[f64],
    out: &mut [f64],
    threads: bool,
) {
    assert!(
        diameter.len() == 1 || 3 * diameter.len() == x.len(),
        "one diameter for all observers, or one per observer"
    );
    assert!(
        polarization.len() == 3 || polarization.len() == x.len(),
        "one polarization for all observers, or one per observer"
    );
    let kernel = match field {
        b'B' => sphere::sphere_b,
        b'H' => sphere::sphere_h,
        _ => panic!("field must be 'B' or 'H'"),
    };
    each_observer(x, out, threads, |i, p| {
        let d = if diameter.len() == 1 {
            diameter[0]
        } else {
            diameter[i]
        };
        let j = if polarization.len() == 3 {
            row(polarization, 0)
        } else {
            row(polarization, i)
        };
        kernel(p, d, j)
    });
}

/// B in T (`field` b'B') or H in A/m (b'H') of dipoles at the origin, as
/// `magpylib.func.dipole_field`: `moment` one row (3,) or one per observer
/// (M, 3), in A m^2.
pub fn dipole_field(field: u8, x: &[f64], moment: &[f64], out: &mut [f64], threads: bool) {
    assert!(
        moment.len() == 3 || moment.len() == x.len(),
        "one moment for all observers, or one per observer"
    );
    let kernel = match field {
        b'B' => sphere::dipole_b,
        b'H' => sphere::dipole_h,
        _ => panic!("field must be 'B' or 'H'"),
    };
    each_observer(x, out, threads, |i, p| {
        kernel(
            p,
            if moment.len() == 3 {
                row(moment, 0)
            } else {
                row(moment, i)
            },
        )
    });
}
