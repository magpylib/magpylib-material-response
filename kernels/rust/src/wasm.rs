//! The browser build, by wasm-pack: Float64Arrays in and out. A saved page
//! builds one `Sheet` per solved sheet once, then asks it for H at the
//! field-line grid every frame.

use wasm_bindgen::prelude::*;

/// A solved `ChargeSheet`: triangles, coefficients, its center and the
/// distance from which the far rule takes over.
#[wasm_bindgen]
pub struct Sheet(crate::triangle::Sheet);

#[wasm_bindgen]
impl Sheet {
    #[wasm_bindgen(constructor)]
    pub fn new(v: &[f64], c: &[f64], center: &[f64], far_from: f64) -> Result<Sheet, JsError> {
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

    /// H in A/m at observers `x` (M, 3), in the sheet's own frame.
    pub fn h(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; x.len()];
        crate::sheet_h(&self.0, x, &mut out, false);
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
