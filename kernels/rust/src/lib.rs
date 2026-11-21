//! Field kernels for magpylib-material-response's solved face charge and
//! magpylib's cuboid, one source for Python (maturin), C and the browser
//! (WebAssembly). Checked against `kernels/reference/*.json`, the files the
//! generators next to them write (issue #208).
//!
//! All arrays are flat f64 slices in numpy's C order. Every batch function
//! fills `out` (M, 3), one row per observer; with `threads`, and the
//! `parallel` feature, observers are split over rayon's threads.

pub mod cuboid;
pub mod triangle;

mod ffi;
#[cfg(feature = "python")]
mod python;
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

/// `far_points`: points (6, N, 3) and charges (6, N).
pub fn far_points(v: &[f64], c: &[f64]) -> (Vec<f64>, Vec<f64>) {
    triangle::far_points(v, &triangle::frames(v), c)
}

/// `charge_field_far`: point charges `charge` (6, N) at `y` (6, N, 3), summed.
pub fn charge_field_far(x: &[f64], y: &[f64], charge: &[f64], out: &mut [f64], threads: bool) {
    assert_eq!(y.len(), 3 * charge.len(), "one point per charge");
    each_observer(x, out, threads, |_, p| triangle::far_at(p, y, charge));
}

/// A solved sheet's H in A/m in its frame, as `_triangle.charge_sheet_field`:
/// `tolerance` 0 the closed form below `far_from`, as `ChargeSheet._field`;
/// above, the per-pair far rule (`triangle::PAIR_RULE`).
pub fn sheet_h(sheet: &Sheet, x: &[f64], tolerance: f64, out: &mut [f64], threads: bool) {
    let k2 = triangle::pair_k2(tolerance);
    each_observer(x, out, threads, |_, p| sheet.h_at(p, k2));
}

/// B in T (`field` b'B') or H in A/m (b'H') of cuboids, one per observer
/// row: `dimension` and `polarization` (M, 3), each in its own frame.
pub fn cuboid_field(
    field: u8,
    x: &[f64],
    dimension: &[f64],
    polarization: &[f64],
    out: &mut [f64],
    threads: bool,
) {
    assert_eq!(x.len(), dimension.len(), "one cuboid per observer");
    assert_eq!(x.len(), polarization.len(), "one cuboid per observer");
    let kernel = match field {
        b'B' => cuboid::cuboid_b,
        b'H' => cuboid::cuboid_h,
        _ => panic!("field must be 'B' or 'H'"),
    };
    each_observer(x, out, threads, |i, p| {
        kernel(p, row(dimension, i), row(polarization, i))
    });
}
