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

/// One field at observers `x` (M, 3), the sources one per row for a cuboid
/// and one for all rows otherwise, or None for a field this crate lacks.
fn field(name: &str, srcs: &[&Value], x: &[f64], threads: bool) -> Option<Vec<f64>> {
    let mut out = vec![0.0; x.len()];
    let src = srcs[0];
    match (src["kind"].as_str().unwrap(), name) {
        ("cuboid", "B" | "H") => {
            let rows = |key: &str| -> Vec<f64> {
                (0..x.len() / 3)
                    .flat_map(|i| floats(&srcs[i % srcs.len()][key]))
                    .collect()
            };
            let (d, p) = (rows("dimension"), rows("polarization"));
            k::cuboid_field(name.as_bytes()[0], x, &d, &p, &mut out, threads);
        }
        ("triangle_linear_charge", "H") => {
            k::charge_field(
                x,
                &floats(&src["vertices"]),
                &floats(&src["charge"]),
                &mut out,
                threads,
            );
            out.iter_mut()
                .for_each(|v| *v /= 4.0 * std::f64::consts::PI);
        }
        ("charge_sheet", "H") => {
            let sheet = k::triangle::Sheet::new(
                &floats(&src["vertices"]),
                &floats(&src["charge"]),
                v3(&src["center"]),
                floats(&src["far_from"])[0],
            );
            k::sheet_h(&sheet, x, &mut out, threads);
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
/// Rows in a batch check: past two of the crate's chunks of 64 observers.
const BATCH: usize = 130;

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
    let mut checked = 0usize;
    for path in files {
        let doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let family = file.trim_end_matches(".json").to_string();
        let rtol = doc["rtol"].as_f64().unwrap();
        let mut worst = (0.0f64, String::new());
        let points = doc["points"].as_array().unwrap();
        let source = |p: &Value| &doc["sources"][p["source"].as_str().unwrap()];
        let atol = |p: &Value, name: &str| {
            p.get(format!("atol_{name}"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
        };
        for p in points {
            let id = p["id"].as_str().unwrap();
            let src = source(p);
            for name in ["B", "H", "J", "M"]
                .into_iter()
                .filter(|n| p.get(*n).is_some())
            {
                for &threads in PASSES {
                    let Some(got) = field(name, &[src], &v3(&p["observer"]), threads) else {
                        skipped.insert(format!("{} {name}", src["kind"].as_str().unwrap()));
                        break;
                    };
                    checked += 1;
                    match within(&got, &floats(&p[name]), rtol, atol(p, name)) {
                        Ok(r) if r > worst.0 => worst = (r, format!("{id} {name}")),
                        Ok(_) => {}
                        Err(msg) => failures.push(format!("{id} {name} ({}): {msg}", p["why"])),
                    }
                }
            }
        }
        // batches: a cuboid's points in one call, their sources mixed row by
        // row; another kind's points per source; tiled past two chunks
        let mut groups: Vec<Vec<&Value>> = Vec::new();
        for p in points {
            let kind = source(p)["kind"].as_str().unwrap();
            let same = |q: &Value| {
                let qk = source(q)["kind"].as_str().unwrap();
                qk == kind && (kind == "cuboid" || q["source"] == p["source"])
            };
            if !groups.iter().any(|g| same(g[0])) {
                groups.push(points.iter().filter(|q| same(q)).collect());
            }
        }
        for group in &groups {
            let rows: Vec<&Value> = group
                .iter()
                .cycle()
                .take(BATCH.max(group.len()))
                .copied()
                .collect();
            let x: Vec<f64> = rows.iter().flat_map(|p| v3(&p["observer"])).collect();
            let srcs: Vec<&Value> = rows.iter().map(|p| source(p)).collect();
            for name in ["B", "H", "J", "M"] {
                if rows.iter().any(|p| p.get(name).is_none()) {
                    continue;
                }
                for &threads in PASSES {
                    let Some(got) = field(name, &srcs, &x, threads) else {
                        break;
                    };
                    for (p, g) in rows.iter().zip(got.chunks(3)) {
                        checked += 1;
                        if let Err(msg) = within(g, &floats(&p[name]), rtol, atol(p, name)) {
                            failures.push(format!("{} {name} in a batch: {msg}", p["id"]));
                        }
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
                    let Some(f) = field(name, &[src], &v3(&t["observer"]), threads) else {
                        skipped.insert(format!("{} {name}", src["kind"].as_str().unwrap()));
                        skipped_relations.insert(format!("{family}/{id}"));
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
                checked += 1;
                match within(
                    &total,
                    &floats(&r["target"]),
                    rtol,
                    r["atol"].as_f64().unwrap(),
                ) {
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
    println!("{checked} checks");
    assert!(checked > 0, "nothing was checked");
    assert!(
        failures.is_empty(),
        "{} off:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
