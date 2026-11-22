//! The browser build, by wasm-pack: Float64Arrays in and out, every array
//! flat in numpy's C order, every source in its own frame. A saved page
//! builds one `ChargeSheet` per solved sheet once, then asks it for H at the
//! field-line grid every frame. Names follow magpylib's and
//! magpylib-material-response's Python; a wrong argument throws.

use wasm_bindgen::prelude::*;

// The far rule runs two charges at a time in WebAssembly's 128-bit SIMD;
// without it the build would be silently slower.
#[cfg(all(target_arch = "wasm32", not(target_feature = "simd128")))]
compile_error!(
    "the browser build needs SIMD: run wasm-pack in kernels/rust, whose .cargo/config.toml sets it"
);

/// The version of this API, raised by every change a caller must follow; a
/// page refuses a build whose version is not the one it was written for.
#[wasm_bindgen]
pub fn api_version() -> u32 {
    2
}

fn check_observers(x: &[f64]) -> Result<(), JsError> {
    if x.len().is_multiple_of(3) {
        Ok(())
    } else {
        Err(JsError::new("observers: (M, 3)"))
    }
}

/// B in T (`field` "B") or H in A/m ("H") of cuboids at `observers` (M, 3),
/// as `magpylib.func.cuboid_field` at the origin, unturned: `dimensions` and
/// `polarizations` one cuboid (3,) for every observer, or one per observer
/// (M, 3).
#[wasm_bindgen]
pub fn cuboid_field(
    field: &str,
    observers: &[f64],
    dimensions: &[f64],
    polarizations: &[f64],
) -> Result<Vec<f64>, JsError> {
    let field = match field {
        "B" => b'B',
        "H" => b'H',
        _ => return Err(JsError::new("field: \"B\" or \"H\"")),
    };
    check_observers(observers)?;
    for a in [dimensions, polarizations] {
        if a.len() != 3 && a.len() != observers.len() {
            return Err(JsError::new(
                "dimensions, polarizations: (3,) for every observer, or (M, 3)",
            ));
        }
    }
    let mut out = vec![0.0; observers.len()];
    crate::cuboid_field(field, observers, dimensions, polarizations, &mut out, false);
    Ok(out)
}

/// A solved `ChargeSheet`, from its public data, as `ChargeSheet.from_triangles`
/// takes it: `triangles` (N, 3, 3), the corners of each face, and
/// `charge_densities` (N, 3), the charge density at those corners in A/m.
/// The kernel derives the rest (`_triangle.sheet_parameters`).
#[wasm_bindgen]
pub struct ChargeSheet(crate::triangle::Sheet);

#[wasm_bindgen]
impl ChargeSheet {
    #[wasm_bindgen(constructor)]
    pub fn new(triangles: &[f64], charge_densities: &[f64]) -> Result<ChargeSheet, JsError> {
        if !triangles.len().is_multiple_of(9) || charge_densities.len() * 3 != triangles.len() {
            return Err(JsError::new(
                "triangles: (N, 3, 3), charge_densities: (N, 3)",
            ));
        }
        Ok(ChargeSheet(crate::triangle::Sheet::from_corners(
            triangles,
            charge_densities,
        )))
    }

    /// For the reference files' checker only, not a page: a sheet from what
    /// the constructor derives, coefficients (N, 3) of `1, u, v`, `center`
    /// (3,) and `far_from`, for the `charge_sheet` kind's chosen switch points.
    pub fn from_coefficients(
        triangles: &[f64],
        coefficients: &[f64],
        center: &[f64],
        far_from: f64,
    ) -> Result<ChargeSheet, JsError> {
        if !triangles.len().is_multiple_of(9)
            || coefficients.len() * 3 != triangles.len()
            || center.len() != 3
        {
            return Err(JsError::new(
                "triangles: (N, 3, 3), coefficients: (N, 3), center: (3,)",
            ));
        }
        Ok(ChargeSheet(crate::triangle::Sheet::new(
            triangles,
            coefficients,
            [center[0], center[1], center[2]],
            far_from,
        )))
    }

    /// H in A/m at `observers` (M, 3): `tolerance` 0 exact, above it the
    /// per-pair rules (coarser and faster as it grows: 1e-3 while dragging,
    /// 1e-6 at rest, say). H only: B is mu_0 H outside the body, and inside
    /// it lacks the magnetization, which the caller knows and the sheet not.
    pub fn h(&self, observers: &[f64], tolerance: f64) -> Result<Vec<f64>, JsError> {
        check_observers(observers)?;
        let mut out = vec![0.0; observers.len()];
        crate::sheet_h(&self.0, observers, tolerance, &mut out, false);
        Ok(out)
    }
}

/// For the reference files' checker only: 4 pi times H in A/m of triangles
/// (N, 3, 3) with charge coefficients (N, 3), summed, at `observers` (M, 3).
#[wasm_bindgen]
pub fn charge_field(
    observers: &[f64],
    triangles: &[f64],
    coefficients: &[f64],
) -> Result<Vec<f64>, JsError> {
    check_observers(observers)?;
    if !triangles.len().is_multiple_of(9) || coefficients.len() * 3 != triangles.len() {
        return Err(JsError::new("triangles: (N, 3, 3), coefficients: (N, 3)"));
    }
    let mut out = vec![0.0; observers.len()];
    crate::charge_field(observers, triangles, coefficients, &mut out, false);
    Ok(out)
}
