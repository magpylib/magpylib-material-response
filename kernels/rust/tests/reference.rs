//! Every point and relation of `kernels/reference/*.json` to its tolerance, by
//! the rules of `kernels/reference/SPEC.md`: per field,
//! `|got - expected| <= rtol |expected| + atol`, non-finite components matched
//! exactly. On one thread, and on all with `--features parallel` (without it
//! the crate has no threads).

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

fn v3(v: &Value) -> [f64; 3] {
    let f = floats(v);
    [f[0], f[1], f[2]]
}

/// The files' rule for one row; Ok holds the share of the tolerance used.
fn within(got: &[f64], expected: &[f64], rtol: f64, atol: f64) -> Result<f64, String> {
    if expected.iter().any(|v| !v.is_finite()) {
        let same = got
            .iter()
            .zip(expected)
            .all(|(a, b)| (a.is_nan() && b.is_nan()) || a == b);
        return if same {
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

/// One field at one observer of one source, or None for a field this crate
/// lacks.
fn field(name: &str, src: &Value, x: [f64; 3], threads: bool) -> Option<Vec<f64>> {
    let mut out = vec![0.0; 3];
    match (src["kind"].as_str().unwrap(), name) {
        ("cuboid", "B" | "H") => {
            let (d, p) = (floats(&src["dimension"]), floats(&src["polarization"]));
            k::cuboid_field(name.as_bytes()[0], &x, &d, &p, &mut out, threads);
        }
        ("triangle_charge", "H") => {
            k::charge_field(
                &x,
                &floats(&src["vertices"]),
                &floats(&src["charge"]),
                &mut out,
                threads,
            );
            out.iter_mut()
                .for_each(|v| *v /= 4.0 * std::f64::consts::PI);
        }
        ("charge_sheet", "H") => {
            let far_from = floats(&src["far_from"])[0];
            let sheet = k::triangle::Sheet::new(
                &floats(&src["vertices"]),
                &floats(&src["charge"]),
                v3(&src["center"]),
                far_from,
            );
            k::sheet_h(&sheet, &x, &mut out, threads);
        }
        _ => return None,
    }
    Some(out)
}

/// The passes: one thread, and all where the crate is built with threads.
const PASSES: &[bool] = if cfg!(feature = "parallel") {
    &[false, true]
} else {
    &[false]
};

#[test]
fn reference_points() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../reference");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no reference files in {dir}");
    let (mut failures, mut skipped) = (Vec::new(), std::collections::BTreeSet::new());
    let mut skipped_relations = std::collections::BTreeSet::new();
    for path in files {
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let rtol = doc["rtol"].as_f64().unwrap();
        let mut worst = (0.0f64, String::new());
        for p in doc["points"].as_array().unwrap() {
            let id = p["id"].as_str().unwrap();
            let src = &doc["sources"][p["source"].as_str().unwrap()];
            let x = v3(&p["observer"]);
            for name in ["B", "H", "J", "M"]
                .into_iter()
                .filter(|n| p.get(*n).is_some())
            {
                let atol = p
                    .get(format!("atol_{name}"))
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                for &threads in PASSES {
                    let Some(got) = field(name, src, x, threads) else {
                        skipped.insert(format!("{} {name}", src["kind"].as_str().unwrap()));
                        break;
                    };
                    match within(&got, &floats(&p[name]), rtol, atol) {
                        Ok(r) if r > worst.0 => worst = (r, format!("{id} {name}")),
                        Ok(_) => {}
                        Err(msg) => failures.push(format!("{id} {name} ({}): {msg}", p["why"])),
                    }
                }
            }
        }
        // relations: the sum of weight @ field over the terms, against the target
        for r in doc["relations"].as_array().into_iter().flatten() {
            let id = r["id"].as_str().unwrap();
            for &threads in PASSES {
                let mut total = Vec::new();
                for t in r["terms"].as_array().unwrap() {
                    let name = t["field"].as_str().unwrap();
                    let src = &doc["sources"][t["source"].as_str().unwrap()];
                    let Some(f) = field(name, src, v3(&t["observer"]), threads) else {
                        skipped.insert(format!("{} {name}", src["kind"].as_str().unwrap()));
                        skipped_relations
                            .insert(format!("{}/{id}", file.trim_end_matches(".json")));
                        total.clear();
                        break;
                    };
                    let w = floats(&t["weight"]);
                    let part: Vec<f64> = if w.len() == 1 {
                        f.iter().map(|x| w[0] * x).collect()
                    } else {
                        w.chunks(3)
                            .map(|row| row.iter().zip(&f).map(|(a, b)| a * b).sum())
                            .collect()
                    };
                    if total.is_empty() {
                        total = part;
                    } else {
                        total.iter_mut().zip(part).for_each(|(a, b)| *a += b);
                    }
                }
                if total.is_empty() {
                    break;
                }
                let atol = r["atol"].as_f64().unwrap();
                match within(&total, &floats(&r["target"]), rtol, atol) {
                    Ok(x) if x > worst.0 => worst = (x, id.to_string()),
                    Ok(_) => {}
                    Err(msg) => failures.push(format!("{id} ({}): {msg}", r["why"])),
                }
            }
        }
        println!(
            "{file}: worst {} at {:.3} of its tolerance",
            worst.1, worst.0
        );
    }
    if !skipped_relations.is_empty() {
        println!(
            "relations skipped whole: {}",
            skipped_relations.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    if !PASSES.contains(&true) {
        println!("threads not built: run with --features parallel for that pass");
    }
    if !skipped.is_empty() {
        println!(
            "not in this crate: {}",
            skipped.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    assert!(
        failures.is_empty(),
        "{} off:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
