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
  cuboid_field,
} from "../pkg/mmr_kernels.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
await init({
  module_or_path: readFileSync(here("../pkg/mmr_kernels_bg.wasm")),
});

// the API's version, and a wrong argument throws, leaving the instance usable
// (a Rust panic would trap it)
if (api_version() !== 2) throw new Error(`api_version ${api_version()}, not 2`);
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
]) {
  let threw = false;
  try {
    call();
  } catch {
    threw = true;
  }
  if (!threw) throw new Error(`${what}: no error`);
}
cuboid_field("B", one, unit, unit); // still usable

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
// takes one source per row, another kind one source for all; any other field
// is skipped and reported
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
    const key = kindOf(p) === "cuboid" ? "cuboid" : `${kindOf(p)} ${p.source}`;
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
