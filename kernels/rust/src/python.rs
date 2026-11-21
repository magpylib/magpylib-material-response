//! The Python extension `mmr_kernels`, built by maturin: numpy arrays in,
//! numpy arrays out, the GIL released while computing. `threads=True`
//! splits observers over rayon's threads.

use numpy::{PyArray2, PyReadonlyArrayDyn, ToPyArray, ndarray::Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn flat<'a>(a: &'a PyReadonlyArrayDyn<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous float64")))
}

/// [`flat`], with the array's last axes checked: an (M, 3) array, not (3, M).
fn shaped<'a>(
    a: &'a PyReadonlyArrayDyn<'_, f64>,
    name: &str,
    tail: &[usize],
) -> PyResult<&'a [f64]> {
    if !a.as_array().shape().ends_with(tail) {
        return Err(PyValueError::new_err(format!(
            "{name}: its last axes must be {tail:?}, got shape {:?}",
            a.as_array().shape()
        )));
    }
    flat(a, name)
}

fn rows(out: Vec<f64>, py: Python<'_>) -> Bound<'_, PyArray2<f64>> {
    let m = out.len() / 3;
    Array2::from_shape_vec((m, 3), out).unwrap().to_pyarray(py)
}

/// 4 pi times the field of triangles V (N, 3, 3) with coefficients c (N, 3)
/// of 1, u, v, summed at observers X (M, 3): `_triangle.charge_field`.
#[pyfunction]
#[pyo3(signature = (X, V, c, threads=false))]
#[allow(non_snake_case)]
fn charge_field<'py>(
    py: Python<'py>,
    X: PyReadonlyArrayDyn<'py, f64>,
    V: PyReadonlyArrayDyn<'py, f64>,
    c: PyReadonlyArrayDyn<'py, f64>,
    threads: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (x, v, c) = (
        shaped(&X, "X", &[3])?,
        shaped(&V, "V", &[3, 3])?,
        shaped(&c, "c", &[3])?,
    );
    if x.len() % 3 != 0 || v.len() % 9 != 0 || c.len() * 3 != v.len() {
        return Err(PyValueError::new_err(
            "shapes: X (M, 3), V (N, 3, 3), c (N, 3)",
        ));
    }
    let mut out = vec![0.0; x.len()];
    py.detach(|| crate::charge_field(x, v, c, &mut out, threads));
    Ok(rows(out, py))
}

/// The far rule: charges (6, N) at points y (6, N, 3), at observers X (M, 3):
/// `_triangle.charge_field_far`.
#[pyfunction]
#[pyo3(signature = (X, y, charge, threads=false))]
#[allow(non_snake_case)]
fn charge_field_far<'py>(
    py: Python<'py>,
    X: PyReadonlyArrayDyn<'py, f64>,
    y: PyReadonlyArrayDyn<'py, f64>,
    charge: PyReadonlyArrayDyn<'py, f64>,
    threads: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (x, y, q) = (
        shaped(&X, "X", &[3])?,
        shaped(&y, "y", &[3])?,
        flat(&charge, "charge")?,
    );
    if x.len() % 3 != 0 || y.len() != 3 * q.len() {
        return Err(PyValueError::new_err(
            "shapes: X (M, 3), y (6, N, 3), charge (6, N)",
        ));
    }
    let mut out = vec![0.0; x.len()];
    py.detach(|| crate::charge_field_far(x, y, q, &mut out, threads));
    Ok(rows(out, py))
}

/// B in T ('B') or H in A/m ('H') of cuboids in their own frames, one per
/// observer row: magpylib's `_BHJM_magnet_cuboid`.
#[pyfunction]
#[pyo3(signature = (field, X, dimension, polarization, threads=false))]
#[allow(non_snake_case)]
fn cuboid_field<'py>(
    py: Python<'py>,
    field: &str,
    X: PyReadonlyArrayDyn<'py, f64>,
    dimension: PyReadonlyArrayDyn<'py, f64>,
    polarization: PyReadonlyArrayDyn<'py, f64>,
    threads: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let f = match field {
        "B" => b'B',
        "H" => b'H',
        _ => return Err(PyValueError::new_err("field must be 'B' or 'H'")),
    };
    let (x, d, p) = (
        shaped(&X, "X", &[3])?,
        shaped(&dimension, "dimension", &[3])?,
        shaped(&polarization, "polarization", &[3])?,
    );
    if x.len() % 3 != 0 || d.len() != x.len() || p.len() != x.len() {
        return Err(PyValueError::new_err(
            "X, dimension and polarization: (M, 3) each",
        ));
    }
    let mut out = vec![0.0; x.len()];
    py.detach(|| crate::cuboid_field(f, x, d, p, &mut out, threads));
    Ok(rows(out, py))
}

#[pymodule]
fn mmr_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(charge_field, m)?)?;
    m.add_function(wrap_pyfunction!(charge_field_far, m)?)?;
    m.add_function(wrap_pyfunction!(cuboid_field, m)?)?;
    Ok(())
}
