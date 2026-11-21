//! Every case of `kernels/reference/*.json`, the files the generator writes,
//! to its stated tolerance: per row `r`, `|got_r - expected_r| <= rtol
//! |expected_r| + atol[r]`, non-finite components matched exactly. A case
//! whose kernel this crate lacks is skipped and reported.

use mmr_kernels as k;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    fn walk(v: &Value, out: &mut Vec<f64>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Number(n) => out.push(n.as_f64().unwrap()),
            Value::String(s) => out.push(match s.as_str() {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                other => panic!("unknown float {other}"),
            }),
            other => panic!("not a float: {other}"),
        }
    }
    let mut out = Vec::new();
    walk(v, &mut out);
    out
}

/// A case's input, or the shared array it names.
fn input(doc: &Value, case: &Value, name: &str) -> Vec<f64> {
    let v = &case["inputs"][name];
    match v.get("array") {
        Some(id) => floats(&doc["arrays"][id.as_str().unwrap()]),
        None => floats(v),
    }
}

/// The kernel's output, flat, or None for a kernel this crate lacks.
fn run(doc: &Value, case: &Value, threads: bool) -> Option<Vec<f64>> {
    let get = |name: &str| input(doc, case, name);
    let kernel = case["kernel"].as_str().unwrap();
    let field = |x: &[f64]| vec![0.0; x.len()];
    Some(match kernel {
        "charge_field" => {
            let x = get("X");
            let mut out = field(&x);
            k::charge_field(&x, &get("V"), &get("c"), &mut out, threads);
            out
        }
        "far_points" => k::far_points(&get("V"), &get("c")).0,
        "far_charges" => k::far_points(&get("V"), &get("c")).1,
        "charge_field_far" => {
            let x = get("X");
            let mut out = field(&x);
            k::charge_field_far(&x, &get("y"), &get("charge"), &mut out, threads);
            out
        }
        "sheet_H" => {
            let (x, c) = (get("X"), get("center"));
            let sheet = k::triangle::Sheet::new(
                &get("V"),
                &get("c"),
                [c[0], c[1], c[2]],
                get("far_from")[0],
            );
            let mut out = field(&x);
            k::sheet_h(&sheet, &x, &mut out, threads);
            out
        }
        "cuboid_B" | "cuboid_H" => {
            let x = get("X");
            let mut out = field(&x);
            let f = kernel.as_bytes()[7];
            k::cuboid_field(
                f,
                &x,
                &get("dimension"),
                &get("polarization"),
                &mut out,
                threads,
            );
            out
        }
        _ => return None,
    })
}

fn same_non_finite(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}

/// The files' rule for one row: non-finite components exactly, the rest within
/// `rtol |expected| + atol`. Ok holds the share of the tolerance used.
fn within(got: &[f64], expected: &[f64], rtol: f64, atol: f64) -> Result<f64, String> {
    if expected.iter().any(|v| !v.is_finite()) {
        return if got
            .iter()
            .zip(expected)
            .all(|(a, b)| same_non_finite(*a, *b))
        {
            Ok(0.0)
        } else {
            Err(format!("got {got:?}, expected {expected:?}"))
        };
    }
    let err = got
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let bound = rtol * expected.iter().map(|b| b * b).sum::<f64>().sqrt() + atol;
    if err <= bound {
        Ok(if err > 0.0 { err / bound } else { 0.0 })
    } else {
        Err(format!(
            "error {err:.3e} over {bound:.3e}; got {got:?}, expected {expected:?}"
        ))
    }
}

/// `cuboid.json`: every point, B and H, one thread.
#[test]
fn cuboid_points() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../reference/cuboid.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(
        k::cuboid::MU0,
        doc["constants"]["mu_0"].as_f64().unwrap(),
        "mu_0"
    );
    let rtol = doc["rtol"].as_f64().unwrap();
    let (mut failures, mut worst) = (Vec::new(), (0.0f64, String::new()));
    for p in doc["points"].as_array().unwrap() {
        let id = p["id"].as_str().unwrap();
        let src = &doc["sources"][p["source"].as_str().unwrap()];
        let v3 = |v: &Value| {
            let f = floats(v);
            [f[0], f[1], f[2]]
        };
        let (x, d, j) = (
            v3(&p["observer"]),
            v3(&src["dimension"]),
            v3(&src["polarization"]),
        );
        for (field, got) in [
            ("B", k::cuboid::cuboid_b(x, d, j)),
            ("H", k::cuboid::cuboid_h(x, d, j)),
        ] {
            let atol = p
                .get(format!("atol_{field}"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            match within(&got, &floats(&p[field]), rtol, atol) {
                Ok(r) if r > worst.0 => worst = (r, format!("{id} {field}")),
                Ok(_) => {}
                Err(msg) => failures.push(format!("{id} {field}: {msg}")),
            }
        }
    }
    println!(
        "cuboid points: worst {} at {:.3} of its tolerance",
        worst.1, worst.0
    );
    assert!(
        failures.is_empty(),
        "{} off:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn reference_cases() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../reference");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no reference files in {dir}");
    let (mut failures, mut skipped) = (Vec::new(), Vec::new());
    for path in files {
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        if let Some(mu0) = doc["constants"]["mu_0"].as_f64() {
            assert_eq!(k::cuboid::MU0, mu0, "mu_0 differs from the reference's");
        }
        let Some(cases) = doc["cases"].as_array() else {
            continue; // a file of points: see cuboid_points
        };
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let rows: Vec<Vec<f64>> = case["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(floats)
                .collect();
            let atol = floats(&case["atol"]);
            let rtol = case["rtol"].as_f64().unwrap();
            let width = rows[0].len();
            let mut worst = 0.0f64;
            for threads in [false, true] {
                let Some(got) = run(&doc, case, threads) else {
                    skipped.push(name.to_string());
                    break;
                };
                assert_eq!(got.len(), rows.len() * width, "{name}: output length");
                for (r, (g, e)) in got.chunks(width).zip(&rows).enumerate() {
                    match within(g, e, rtol, atol[r]) {
                        Ok(ratio) => worst = worst.max(ratio),
                        Err(msg) => failures.push(format!("{name} row {r}: {msg}")),
                    }
                }
            }
            println!("{name:26} worst row at {worst:.3} of its tolerance");
        }
    }
    if !skipped.is_empty() {
        println!("skipped, kernel not in this crate: {}", skipped.join(", "));
    }
    assert!(
        failures.is_empty(),
        "{} rows off:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
