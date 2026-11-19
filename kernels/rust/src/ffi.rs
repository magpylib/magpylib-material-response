//! The C ABI: one function per kernel, raw pointers and counts, single
//! threaded. For any language that calls C; Python and the browser have
//! their own bindings. Returns 0, or -1 for a null pointer.

use std::slice::{from_raw_parts, from_raw_parts_mut};

unsafe fn slices<'a>(
    inputs: &[(*const f64, usize)],
    out: *mut f64,
    n_out: usize,
) -> Option<(Vec<&'a [f64]>, &'a mut [f64])> {
    if out.is_null() || inputs.iter().any(|(p, n)| p.is_null() && *n > 0) {
        return None;
    }
    let ins = inputs
        .iter()
        .map(|&(p, n)| {
            if n == 0 {
                &[][..]
            } else {
                unsafe { from_raw_parts(p, n) }
            }
        })
        .collect();
    Some((ins, unsafe { from_raw_parts_mut(out, n_out) }))
}

/// 4 pi times the field of `n` charged triangles at `m` observers.
///
/// # Safety
/// `x` and `out` hold `3 m` values, `v` `9 n` and `c` `3 n`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mmr_charge_field(
    x: *const f64,
    m: usize,
    v: *const f64,
    c: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    match unsafe { slices(&[(x, 3 * m), (v, 9 * n), (c, 3 * n)], out, 3 * m) } {
        Some((s, o)) => {
            crate::charge_field(s[0], s[1], s[2], o, false);
            0
        }
        None => -1,
    }
}

/// The far rule: `k` point charges at `y` (`3 k` values), at `m` observers.
///
/// # Safety
/// `x` and `out` hold `3 m` values, `y` `3 k`, `charge` `k`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mmr_charge_field_far(
    x: *const f64,
    m: usize,
    y: *const f64,
    charge: *const f64,
    k: usize,
    out: *mut f64,
) -> i32 {
    match unsafe { slices(&[(x, 3 * m), (y, 3 * k), (charge, k)], out, 3 * m) } {
        Some((s, o)) => {
            crate::charge_field_far(s[0], s[1], s[2], o, false);
            0
        }
        None => -1,
    }
}

/// B (`field` 'B') or H ('H') of `m` cuboids, one per observer.
///
/// # Safety
/// `x`, `dimension`, `polarization` and `out` hold `3 m` values each.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mmr_cuboid_field(
    field: u8,
    x: *const f64,
    dimension: *const f64,
    polarization: *const f64,
    m: usize,
    out: *mut f64,
) -> i32 {
    if field != b'B' && field != b'H' {
        return -1;
    }
    match unsafe {
        slices(
            &[(x, 3 * m), (dimension, 3 * m), (polarization, 3 * m)],
            out,
            3 * m,
        )
    } {
        Some((s, o)) => {
            crate::cuboid_field(field, s[0], s[1], s[2], o, false);
            0
        }
        None => -1,
    }
}
