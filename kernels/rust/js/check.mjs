// The WebAssembly build against every point and relation of
// kernels/reference/*.json, by the rules of kernels/reference/SPEC.md, then H
// of studio's solved plate timed at a field-line grid (the build uses SIMD:
// kernels/rust/.cargo/config.toml). Node runs the same engine as Chrome (V8):
//
//     wasm-pack build --release --target web --out-dir pkg -- --features wasm
//     node js/check.mjs

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import init, {
  ChargeSheet,
  api_version,
  charge_field,
  circle_field,
  cuboid_field,
  cylinder_field,
  dipole_field,
  polyline_field,
  sphere_field,
} from "../pkg/mmr_kernels.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
await init({
  module_or_path: readFileSync(here("../pkg/mmr_kernels_bg.wasm")),
});

// the API's version, and a wrong argument throws, leaving the instance usable
// (a Rust panic would trap it)
if (api_version() !== 1) throw new Error(`api_version ${api_version()}, not 1`);
const one = Float64Array.of(0.01, 0.02, 0.03);
const unit = Float64Array.of(0.01, 0.01, 0.01);
for (const [what, call] of [
  ["field J", () => cuboid_field("J", one, unit, unit)],
  ["observers (4,)", () => cuboid_field("B", one.subarray(0, 2), unit, unit)],
  ["dimensions (2,)", () => cuboid_field("B", one, unit.subarray(0, 2), unit)],
  [
    "corners (N, 2)",
    () => new ChargeSheet(new Float64Array(9), one.subarray(0, 2)),
  ],
  [
    "coefficients",
    () => charge_field(one, new Float64Array(9), one.subarray(0, 1)),
  ],
  [
    "segments_start (2,)",
    () =>
      polyline_field("B", one, unit.subarray(0, 2), one, one.subarray(0, 1)),
  ],
  [
    "currents (3,), one observer",
    () => polyline_field("B", one, unit, one, one),
  ],
  ["diameters (2,)", () => sphere_field("B", one, one.subarray(0, 2), unit)],
  ["moments (2,)", () => dipole_field("H", one, one.subarray(0, 2))],
  [
    "circle diameters (2,)",
    () => circle_field("B", one, one.subarray(0, 2), one.subarray(0, 1)),
  ],
  [
    "circle currents (2,)",
    () => circle_field("H", one, one.subarray(0, 1), one.subarray(0, 2)),
  ],
  [
    "circle field J",
    () => circle_field("J", one, one.subarray(0, 1), one.subarray(0, 1)),
  ],
  ["cylinder dimensions (3,)", () => cylinder_field("B", one, unit, unit)],
  [
    "cylinder polarizations (2,)",
    () => cylinder_field("H", one, unit.subarray(0, 2), unit.subarray(0, 2)),
  ],
  [
    "cylinder field J",
    () => cylinder_field("J", one, unit.subarray(0, 2), unit),
  ],
  ["dipole field J", () => dipole_field("J", one, unit)],
  ["sphere field J", () => sphere_field("J", one, one.subarray(0, 1), unit)],
  [
    "polarizations (2,)",
    () => sphere_field("B", one, one.subarray(0, 1), unit.subarray(0, 2)),
  ],
  [
    "currents (2,)",
    () => polyline_field("B", one, unit, one, one.subarray(0, 2)),
  ],
]) {
  let threw = false;
  try {
    call();
  } catch (e) {
    // a JsError; a trap (a Rust panic) is a RuntimeError, and fails
    if (e instanceof WebAssembly.RuntimeError) throw e;
    threw = true;
  }
  if (!threw) throw new Error(`${what}: no error`);
}
cuboid_field("B", one, unit, unit); // still usable

// one source given once, (3,) and (1,), equals the same given per row, past
// two chunks of observers (the reference points pass per-row parameters)
{
  const m = 130;
  const x = Float64Array.from(
    { length: 3 * m },
    (_, i) => 0.003 * Math.sin(i * 0.618) + 0.001 * (i % 3),
  );
  const tile = (row) =>
    Float64Array.from(
      { length: row.length * m },
      (_, i) => row[i % row.length],
    );
  const [d, p] = [
    Float64Array.of(0.004, 0.002, 0.003),
    Float64Array.of(0.3, -0.5, 1.2),
  ];
  const [a, b, c] = [
    Float64Array.of(0.001, -0.002, 0.0005),
    Float64Array.of(0.004, 0.003, -0.001),
    Float64Array.of(-2.5),
  ];
  for (const f of ["B", "H"]) {
    const pairs = [
      [
        "cuboid",
        cuboid_field(f, x, d, p),
        cuboid_field(f, x, tile(d), tile(p)),
      ],
      [
        "polyline",
        polyline_field(f, x, a, b, c),
        polyline_field(f, x, tile(a), tile(b), tile(c)),
      ],
      [
        "sphere",
        sphere_field(f, x, Float64Array.of(0.004), p),
        sphere_field(f, x, tile(Float64Array.of(0.004)), tile(p)),
      ],
      ["dipole", dipole_field(f, x, p), dipole_field(f, x, tile(p))],
      [
        "circle",
        circle_field(f, x, Float64Array.of(0.004), c),
        circle_field(f, x, tile(Float64Array.of(0.004)), tile(c)),
      ],
      [
        "cylinder",
        cylinder_field(f, x, Float64Array.of(0.004, 0.003), p),
        cylinder_field(f, x, tile(Float64Array.of(0.004, 0.003)), tile(p)),
      ],
    ];
    for (const [what, once, rows] of pairs)
      if (!once.every((v, i) => v === rows[i]))
        throw new Error(`${what} ${f}: shared parameters differ from rows`);
  }
}

const dir = here("../../reference/");
const docs = readdirSync(dir)
  .filter((f) => f.endsWith(".json"))
  .sort()
  .map((f) => ({ file: f, ...JSON.parse(readFileSync(dir + f, "utf8")) }));

const num = (v) =>
  v === "nan" ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v;
const flat = (a) => Float64Array.from([a].flat(Infinity), num);

// a cuboid's parameter: one row for every observer where the batch has one
// source (magpylib's broadcast), else one per observer row, the sources cycled
const rows = (srcs, x, key) =>
  srcs.length === 1
    ? flat(srcs[0][key])
    : flat(
        Array.from(
          { length: x.length / 3 },
          (_, i) => srcs[i % srcs.length][key],
        ),
      );

// the fields this build has, by source kind, at observers x (M, 3): a cuboid
// with one row of parameters per observer (PER_ROW) takes the sources mixed,
// another kind one source for all; any other field
// is skipped and reported
const PER_ROW = new Set([
  "cuboid",
  "polyline",
  "circle",
  "cylinder",
  "sphere",
  "dipole",
]);
const fields = {
  cuboid: {
    B: (srcs, x) =>
      cuboid_field(
        "B",
        x,
        rows(srcs, x, "dimension"),
        rows(srcs, x, "polarization"),
      ),
    H: (srcs, x) =>
      cuboid_field(
        "H",
        x,
        rows(srcs, x, "dimension"),
        rows(srcs, x, "polarization"),
      ),
  },
  polyline: {
    B: (srcs, x) =>
      polyline_field(
        "B",
        x,
        rows(srcs, x, "segment_start"),
        rows(srcs, x, "segment_end"),
        rows(srcs, x, "current"),
      ),
    H: (srcs, x) =>
      polyline_field(
        "H",
        x,
        rows(srcs, x, "segment_start"),
        rows(srcs, x, "segment_end"),
        rows(srcs, x, "current"),
      ),
  },
  circle: {
    B: (srcs, x) =>
      circle_field("B", x, rows(srcs, x, "diameter"), rows(srcs, x, "current")),
    H: (srcs, x) =>
      circle_field("H", x, rows(srcs, x, "diameter"), rows(srcs, x, "current")),
  },
  cylinder: {
    B: (srcs, x) =>
      cylinder_field(
        "B",
        x,
        rows(srcs, x, "dimension"),
        rows(srcs, x, "polarization"),
      ),
    H: (srcs, x) =>
      cylinder_field(
        "H",
        x,
        rows(srcs, x, "dimension"),
        rows(srcs, x, "polarization"),
      ),
  },
  sphere: {
    B: (srcs, x) =>
      sphere_field(
        "B",
        x,
        rows(srcs, x, "diameter"),
        rows(srcs, x, "polarization"),
      ),
    H: (srcs, x) =>
      sphere_field(
        "H",
        x,
        rows(srcs, x, "diameter"),
        rows(srcs, x, "polarization"),
      ),
  },
  dipole: {
    B: (srcs, x) => dipole_field("B", x, rows(srcs, x, "moment")),
    H: (srcs, x) => dipole_field("H", x, rows(srcs, x, "moment")),
  },
  triangle_linear_charge: {
    H: ([s], x) =>
      charge_field(x, flat(s.vertices), flat(s.charge)).map(
        (v) => v / (4 * Math.PI),
      ),
  },
  charge_sheet: {
    H: ([s], x) => {
      const sheet = ChargeSheet.from_coefficients(
        flat(s.vertices),
        flat(s.charge),
        flat(s.center),
        num(s.far_from),
      );
      const out = sheet.h(x, num(s.tolerance));
      sheet.free();
      return out;
    },
  },
  charge_sheet_corners: {
    H: ([s], x) => {
      const sheet = new ChargeSheet(flat(s.vertices), flat(s.charge_densities));
      const out = sheet.h(x, num(s.tolerance));
      sheet.free();
      return out;
    },
  },
};

// the files' rule for one field; the share of the tolerance used, or -1
function within(g, e, rtol, atol) {
  if (!e.every(Number.isFinite))
    return e.every((v, k) => v === g[k] || (isNaN(v) && isNaN(g[k]))) ? 0 : -1;
  const err = Math.hypot(...e.map((v, k) => g[k] - v));
  const bound = rtol * Math.hypot(...e) + atol;
  return err <= bound ? (err > 0 ? err / bound : 0) : -1;
}

const BATCH = 130; // rows in a batch check: past two of the crate's chunks of 64
let failed = 0;
let checked = 0;
const skipped = new Set();
const skippedRelations = [];
for (const doc of docs) {
  let worst = [0, ""];
  for (const p of doc.points) {
    const src = doc.sources[p.source];
    for (const name of ["B", "H", "J", "M"].filter((n) => n in p)) {
      const kernel = fields[src.kind]?.[name];
      if (!kernel) {
        skipped.add(`${src.kind} ${name}`);
        continue;
      }
      const got = kernel([src], flat(p.observer));
      checked++;
      const r = within(got, flat(p[name]), doc.rtol, p[`atol_${name}`] ?? 0);
      if (r < 0) {
        failed++;
        console.log(
          `${p.id} ${name} off (${p.why}): got ${got}, expected ${p[name]}`,
        );
      } else if (r > worst[0]) worst = [r, `${p.id} ${name}`];
    }
  }
  // batches: a cuboid's points in one call, their sources mixed row by row;
  // another kind's points per source; tiled past two chunks
  const kindOf = (p) => doc.sources[p.source].kind;
  const groups = new Map();
  for (const p of doc.points) {
    const key = PER_ROW.has(kindOf(p)) ? kindOf(p) : `${kindOf(p)} ${p.source}`;
    groups.set(key, [...(groups.get(key) ?? []), p]);
  }
  for (const group of groups.values()) {
    const rows = Array.from(
      { length: Math.max(BATCH, group.length) },
      (_, i) => group[i % group.length],
    );
    const x = flat(rows.map((p) => p.observer));
    const srcs = rows.map((p) => doc.sources[p.source]);
    for (const name of ["B", "H", "J", "M"]) {
      const kernel = fields[kindOf(rows[0])]?.[name];
      if (!kernel || rows.some((p) => !(name in p))) continue;
      const got = kernel(srcs, x);
      rows.forEach((p, i) => {
        checked++;
        const g = got.slice(3 * i, 3 * i + 3);
        if (within(g, flat(p[name]), doc.rtol, p[`atol_${name}`] ?? 0) < 0) {
          failed++;
          console.log(`${p.id} ${name} off in a batch: got ${g}`);
        }
      });
    }
  }
  // relations: the sum of weight @ field over the terms, against the target
  for (const r of doc.relations ?? []) {
    let total = [0, 0, 0];
    let ok = true;
    for (const t of r.terms) {
      const src = doc.sources[t.source];
      const kernel = fields[src.kind]?.[t.field];
      if (!kernel) {
        skipped.add(`${src.kind} ${t.field}`);
        skippedRelations.push(`${doc.file.replace(".json", "")}/${r.id}`);
        ok = false;
        break;
      }
      const f = kernel([src], flat(t.observer));
      const w = flat(t.weight);
      const part =
        w.length === 1
          ? [...f].map((x) => w[0] * x)
          : [0, 1, 2].map(
              (i) =>
                w[3 * i] * f[0] + w[3 * i + 1] * f[1] + w[3 * i + 2] * f[2],
            );
      total = total.map((a, i) => a + part[i]);
    }
    if (!ok) continue;
    checked++;
    const x = within(total, flat(r.target), doc.rtol, r.atol);
    if (x < 0) {
      failed++;
      console.log(`${r.id} off (${r.why}): got ${total}, expected ${r.target}`);
    } else if (x > worst[0]) worst = [x, r.id];
  }
  console.log(
    `${doc.file}: worst ${worst[1]} at ${worst[0].toFixed(3)} of its tolerance`,
  );
}
console.log(
  failed ? `${failed} off` : `every point within tolerance, ${checked} checks`,
);
if (!checked) {
  console.log("nothing was checked");
  failed++;
}
if (skipped.size) console.log(`not in this build: ${[...skipped].join(", ")}`);
// what the build leaves to Python (README: the browser computes B and H); any
// other kind or field the files hold and the build lacks fails
const PYTHON_ONLY = new Set([
  "cuboid J",
  "cuboid M",
  "cylinder J",
  "cylinder M",
  "sphere J",
  "sphere M",
]);
const missing = [...skipped].filter((s) => !PYTHON_ONLY.has(s));
if (missing.length) {
  console.log(`MISSING from this build: ${missing.join(", ")}`);
  failed++;
}
if (skippedRelations.length)
  console.log(`relations skipped whole: ${skippedRelations.join(", ")}`);

// timing: the solved plate's sheet (plate.json, a timing input without
// expected values) at a grid round it, as studio's field lines ask
const plate = JSON.parse(readFileSync(here("./plate.json"), "utf8"));
const sheet = new ChargeSheet(
  flat(plate.vertices),
  flat(plate.charge_densities),
);
for (const tolerance of [0, 1e-6, 1e-3]) {
  for (const n of [2500, 8000]) {
    const side = Math.round(Math.cbrt(n));
    const X = new Float64Array(3 * side ** 3);
    let k = 0;
    for (let i = 0; i < side; i++)
      for (let j = 0; j < side; j++)
        for (let l = 0; l < side; l++) {
          X[k++] = -0.04 + (0.08 * i) / (side - 1);
          X[k++] = -0.025 + (0.05 * j) / (side - 1);
          X[k++] = -0.02 + (0.04 * l) / (side - 1);
        }
    sheet.h(X, tolerance); // warm up
    const times = [];
    for (let r = 0; r < 10; r++) {
      const t0 = performance.now();
      sheet.h(X, tolerance);
      times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    const m = X.length / 3;
    const ns = (times[0] * 1e6) / (m * plate.vertices.length);
    console.log(
      `plate, tolerance ${tolerance}, ${m} points: ${times[0].toFixed(1)} ms ` +
        `best, ${times[5].toFixed(1)} ms median (${ns.toFixed(1)} ns per pair, one thread)`,
    );
  }
}
sheet.free();
if (failed) process.exit(1);
