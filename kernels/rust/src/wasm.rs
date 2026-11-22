//! The browser build, by wasm-pack: Float64Arrays in and out. A saved page
//! builds one `Sheet` per solved sheet once, then asks it for H at the
//! field-line grid every frame.

use wasm_bindgen::prelude::*;

// The far rule runs two charges at a time in WebAssembly's 128-bit SIMD;
// without it the build would be silently slower.
#[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
compile_error!(
    "the browser build needs SIMD: run wasm-pack in kernels/rust, whose .cargo/config.toml sets it"
);

/// A solved `ChargeSheet`, from its public data: `triangles` (N, 3, 3), the
/// corners of each face in its own frame, and `charge_densities` (N, 3), the
/// charge density at those corners in A/m. The kernel derives the rest.
#[wasm_bindgen]
pub struct Sheet(crate::triangle::Sheet);

#[wasm_bindgen]
impl Sheet {
    #[wasm_bindgen(constructor)]
    pub fn new(triangles: &[f64], charge_densities: &[f64]) -> Result<Sheet, JsError> {
        if triangles.len() % 9 != 0 || charge_densities.len() * 3 != triangles.len() {
            return Err(JsError::new(
                "shapes: triangles (N, 3, 3), charge_densities (N, 3)",
            ));
        }
        Ok(Sheet(crate::triangle::Sheet::from_corners(
            triangles,
            charge_densities,
        )))
    }

    /// A sheet from what [`Sheet::new`] derives: coefficients `c` (N, 3) of
    /// `1, u, v`, `center` (3,) and `far_from`. For the reference files'
    /// `charge_sheet` kind, whose switch points need a chosen `far_from`; a
    /// page takes `new`.
    pub fn from_coefficients(
        v: &[f64],
        c: &[f64],
        center: &[f64],
        far_from: f64,
    ) -> Result<Sheet, JsError> {
        if v.len() % 9 != 0 || c.len() * 3 != v.len() || center.len() != 3 {
            return Err(JsError::new("shapes: v (N, 3, 3), c (N, 3), center (3,)"));
        }
        Ok(Sheet(crate::triangle::Sheet::new(
            v,
            c,
            [center[0], center[1], center[2]],
            far_from,
        )))
    }

    /// H in A/m at observers `x` (M, 3), in the sheet's own frame;
    /// `tolerance` 0 exact, above it the per-pair far rule (coarser and
    /// faster as it grows: 1e-3 while dragging, 1e-6 at rest, say).
    pub fn h(&self, x: &[f64], tolerance: f64) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        crate::sheet_h(&self.0, x, tolerance, &mut out, false);
        out
    }
}

/// 4 pi times the field of triangles `v` with coefficients `c`, at `x`.
#[wasm_bindgen]
pub fn charge_field(x: &[f64], v: &[f64], c: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    crate::charge_field(x, v, c, &mut out, false);
    out
}

/// B in T of cuboids in their own frames, one per observer row.
#[wasm_bindgen]
pub fn cuboid_b(x: &[f64], dimension: &[f64], polarization: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    crate::cuboid_field(b'B', x, dimension, polarization, &mut out, false);
    out
}

/// H in A/m of cuboids in their own frames, as `cuboid_b`.
#[wasm_bindgen]
pub fn cuboid_h(x: &[f64], dimension: &[f64], polarization: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    crate::cuboid_field(b'H', x, dimension, polarization, &mut out, false);
    out
}
