// The WebAssembly build against kernels/reference/kernels.json, then H of
// studio's solved plate timed at a field-line grid. Node runs the same
// engine as Chrome (V8):
//
//     wasm-pack build --release --target web --out-dir pkg -- --features wasm
//     node www/check.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import init, { Sheet, charge_field, cuboid_b } from "../pkg/mmr_kernels.js";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
await init({
  module_or_path: readFileSync(here("../pkg/mmr_kernels_bg.wasm")),
});
const doc = JSON.parse(
  readFileSync(here("../../reference/kernels.json"), "utf8"),
);

const num = (v) =>
  v === "nan" ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v;
const flat = (a) => Float64Array.from(a.flat(Infinity), num);

const run = {
  charge_field: (i) => charge_field(flat(i.X), flat(i.V), flat(i.c)),
  cuboid_B: (i) => cuboid_b(flat(i.X), flat(i.dimension), flat(i.polarization)),
  sheet_H: (i) => {
    const s = new Sheet(flat(i.V), flat(i.c), flat(i.center), i.far_from[0]);
    const out = s.h(flat(i.X));
    s.free();
    return out;
  },
};

let failed = 0;
for (const c of doc.cases) {
  if (!run[c.kernel]) continue;
  const got = run[c.kernel](c.inputs);
  const exp = flat(c.expected);
  let worst = 0;
  for (let r = 0; r < exp.length / 3; r++) {
    const e = exp.subarray(3 * r, 3 * r + 3);
    const g = got.subarray(3 * r, 3 * r + 3);
    if (!e.every(Number.isFinite)) {
      if (!e.every((v, k) => Object.is(v, g[k]) || (isNaN(v) && isNaN(g[k]))))
        failed++;
      continue;
    }
    const err = Math.hypot(g[0] - e[0], g[1] - e[1], g[2] - e[2]);
    const bound = c.rtol[r] * Math.hypot(...e) + c.atol;
    if (!(err <= bound)) failed++;
    if (err > 0) worst = Math.max(worst, err / bound);
  }
  console.log(
    `${c.name.padEnd(26)} worst row at ${worst.toFixed(3)} of its tolerance`,
  );
}
console.log(failed ? `${failed} rows off` : "all rows within tolerance");

// timing: the plate's sheet at a grid round it, as studio's field lines ask
const plate = doc.cases.find((c) => c.name === "sheet_solved_plate").inputs;
const sheet = new Sheet(
  flat(plate.V),
  flat(plate.c),
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
  const ns = (times[0] * 1e6) / (m * (plate.V.length || 1));
  console.log(
    `plate, ${plate.V.length} triangles, ${m} points: ${times[0].toFixed(1)} ms best, ` +
      `${times[5].toFixed(1)} ms median (${ns.toFixed(0)} ns per pair, one thread)`,
  );
}
sheet.free();
if (failed) process.exit(1);
