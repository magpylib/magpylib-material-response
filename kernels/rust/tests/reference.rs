//! Every case of `kernels/reference/kernels.json`, the file the numpy
//! reference writes, to its stated tolerance: per row `r`,
//! `|got_r - expected_r| <= rtol_r |expected_r| + atol`, and non-finite
//! components matched exactly.

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

fn run(kernel: &str, inp: &Value, threads: bool) -> Vec<f64> {
    let get = |name: &str| floats(&inp[name]);
    let x = get("X");
    let mut out = vec![0.0; x.len()];
    match kernel {
        "charge_field" => k::charge_field(&x, &get("V"), &get("c"), &mut out, threads),
        "charge_field_far" => k::charge_field_far(&x, &get("y"), &get("charge"), &mut out, threads),
        "cuboid_B" | "cuboid_H" => {
            let f = kernel.as_bytes()[7];
            k::cuboid_field(
                f,
                &x,
                &get("dimension"),
                &get("polarization"),
                &mut out,
                threads,
            )
        }
        "sheet_H" => {
            let c = get("center");
            let sheet = k::triangle::Sheet::new(
                &get("V"),
                &get("c"),
                [c[0], c[1], c[2]],
                get("far_from")[0],
            );
            k::sheet_h(&sheet, &x, &mut out, threads);
        }
        other => panic!("no kernel {other}"),
    }
    out
}

fn far_points_rows(inp: &Value) -> Vec<f64> {
    let (v, c) = (floats(&inp["V"]), floats(&inp["c"]));
    let n = c.len() / 3;
    let (y, q) = k::far_points(&v, &c);
    let mut rows = Vec::with_capacity(24 * n);
    for t in 0..n {
        for g in 0..6 {
            rows.extend_from_slice(&y[3 * (g * n + t)..3 * (g * n + t) + 3]);
            rows.push(q[g * n + t]);
        }
    }
    rows
}

fn same_non_finite(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}

#[test]
fn reference_cases() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../reference/kernels.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mu0 = doc["constants"]["mu_0"].as_f64().unwrap();
    assert_eq!(k::cuboid::MU0, mu0, "mu_0 differs from the reference's");
    let mut failures = Vec::new();
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let kernel = case["kernel"].as_str().unwrap();
        let expected = floats(&case["expected"]);
        let rtol = floats(&case["rtol"]);
        let atol = case["atol"].as_f64().unwrap();
        let width = expected.len() / rtol.len();
        for threads in [false, true] {
            let got = if kernel == "far_points" {
                far_points_rows(&case["inputs"])
            } else {
                run(kernel, &case["inputs"], threads)
            };
            assert_eq!(got.len(), expected.len(), "{name}: output length");
            let mut worst = (0.0f64, 0usize);
            for (r, (g, e)) in got.chunks(width).zip(expected.chunks(width)).enumerate() {
                if e.iter().any(|v| !v.is_finite()) {
                    if !g.iter().zip(e).all(|(a, b)| same_non_finite(*a, *b)) {
                        failures.push(format!("{name} row {r}: got {g:?}, expected {e:?}"));
                    }
                    continue;
                }
                let err = g
                    .iter()
                    .zip(e)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                let size = e.iter().map(|b| b * b).sum::<f64>().sqrt();
                let bound = rtol[r] * size + atol;
                let ratio = if err == 0.0 { 0.0 } else { err / bound };
                if !(err <= bound) {
                    failures.push(format!(
                        "{name} row {r}: error {err:.3e}, {ratio:.2} times the tolerance; got {g:?}, expected {e:?}"
                    ));
                }
                if ratio > worst.0 {
                    worst = (ratio, r);
                }
            }
            if !threads {
                println!(
                    "{name:26} worst row {:4} at {:.3} of its tolerance",
                    worst.1, worst.0
                );
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} rows off:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
