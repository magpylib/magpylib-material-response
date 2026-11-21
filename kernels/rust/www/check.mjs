// The WebAssembly build against kernels/reference/*.json, then H of studio's
// solved plate timed at a field-line grid. Node runs the same engine as
// Chrome (V8):
//
//     wasm-pack build --release --target web --out-dir pkg -- --features wasm
//     node www/check.mjs

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import init, { Sheet, charge_field, cuboid_b } from "../pkg/mmr_kernels.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
await init({
  module_or_path: readFileSync(here("../pkg/mmr_kernels_bg.wasm")),
});
const dir = here("../../reference/");
const docs = readdirSync(dir)
  .filter((f) => f.endsWith(".json"))
  .sort()
  .map((f) => JSON.parse(readFileSync(dir + f, "utf8")));

const num = (v) =>
  v === "nan" ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v;
const flat = (a) => Float64Array.from(a.flat(Infinity), num);

// the kernels this build has; a case needing another one is skipped
const run = {
  charge_field: (i) => charge_field(i.X, i.V, i.c),
  cuboid_B: (i) => cuboid_b(i.X, i.dimension, i.polarization),
  sheet_H: (i) => {
    const s = new Sheet(i.V, i.c, i.center, i.far_from[0]);
    const out = s.h(i.X);
    s.free();
    return out;
  },
};

let failed = 0;
const skipped = [];
// the files' rule for one row; returns the share of the tolerance used, or -1
function within(g, e, rtol, atol) {
  if (!e.every(Number.isFinite))
    return e.every((v, k) => Object.is(v, g[k]) || (isNaN(v) && isNaN(g[k])))
      ? 0
      : -1;
  const err = Math.hypot(...e.map((v, k) => g[k] - v));
  const bound = rtol * Math.hypot(...e) + atol;
  return err <= bound ? (err > 0 ? err / bound : 0) : -1;
}

// a file of points: the cuboid, B only (the build has no H)
for (const doc of docs.filter((d) => d.points)) {
  let worst = [0, ""];
  for (const p of doc.points) {
    const src = doc.sources[p.source];
    const got = cuboid_b(
      flat(p.observer),
      flat(src.dimension),
      flat(src.polarization),
    );
    const r = within(got, Float64Array.from(p.B, num), doc.rtol, p.atol_B ?? 0);
    if (r < 0) {
      failed++;
      console.log(`${p.id} B off: got ${got}, expected ${p.B}`);
    } else if (r > worst[0]) worst = [r, p.id];
  }
  console.log(
    `${doc.family}: ${doc.points.length} points, B; worst ${worst[1]} at ` +
      `${worst[0].toFixed(3)} of its tolerance (H not in this build)`,
  );
}

for (const doc of docs.filter((d) => d.cases)) {
  for (const c of doc.cases) {
    if (!run[c.kernel]) {
      skipped.push(c.name);
      continue;
    }
    const inputs = Object.fromEntries(
      Object.entries(c.inputs).map(([k, v]) => [
        k,
        flat(v.array ? doc.arrays[v.array] : v),
      ]),
    );
    const got = run[c.kernel](inputs);
    const width = c.expected[0].length;
    let worst = 0;
    c.expected.forEach((row, r) => {
      const e = Float64Array.from(row, num);
      const g = got.subarray(width * r, width * (r + 1));
      if (!e.every(Number.isFinite)) {
        if (!e.every((v, k) => Object.is(v, g[k]) || (isNaN(v) && isNaN(g[k]))))
          failed++;
        return;
      }
      const err = Math.hypot(...e.map((v, k) => g[k] - v));
      const bound = c.rtol * Math.hypot(...e) + c.atol[r];
      if (!(err <= bound)) failed++;
      else if (err > 0) worst = Math.max(worst, err / bound);
    });
    console.log(
      `${c.name.padEnd(26)} worst row at ${worst.toFixed(3)} of its tolerance`,
    );
  }
}
console.log(failed ? `${failed} rows off` : "all rows within tolerance");
if (skipped.length)
  console.log(`skipped, not in this build: ${skipped.join(", ")}`);

// timing: the solved plate's sheet at a grid round it, as studio's field
// lines ask; only with face_charge.json, which holds the plate
function timePlate(face) {
  const plate = face.cases.find((c) => c.name === "sheet_solved_plate").inputs;
  const pV = face.arrays[plate.V.array];
  const sheet = new Sheet(
    flat(pV),
    flat(face.arrays[plate.c.array]),
    flat(plate.center),
    plate.far_from[0],
  );
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
    sheet.h(X); // warm up
    const times = [];
    for (let r = 0; r < 10; r++) {
      const t0 = performance.now();
      sheet.h(X);
      times.push(performance.now() - t0);
    }
    times.sort((a, b) => a - b);
    const m = X.length / 3;
    const ns = (times[0] * 1e6) / (m * (pV.length || 1));
    console.log(
      `plate, ${pV.length} triangles, ${m} points: ${times[0].toFixed(1)} ms best, ` +
        `${times[5].toFixed(1)} ms median (${ns.toFixed(0)} ns per pair, one thread)`,
    );
  }
  sheet.free();
}
const face = docs.find((d) => d.family === "face_charge");
if (face) timePlate(face);
else console.log("plate timing skipped: no face_charge.json");
if (failed) process.exit(1);
