// The WebAssembly build against every point and relation of
// kernels/reference/*.json, by the rules of kernels/reference/SPEC.md, then H
// of studio's solved plate timed at a field-line grid. Node runs the same engine as Chrome (V8):
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
  .map((f) => ({ file: f, ...JSON.parse(readFileSync(dir + f, "utf8")) }));

const num = (v) =>
  v === "nan" ? NaN : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v;
const flat = (a) => Float64Array.from([a].flat(Infinity), num);

// the fields this build has, by source kind; any other is skipped and reported
const fields = {
  cuboid: {
    B: (s, x) => cuboid_b(x, flat(s.dimension), flat(s.polarization)),
  },
  triangles: {
    F: (s, x) => charge_field(x, flat(s.V), flat(s.c)),
  },
  sheet: {
    H: (s, x) => {
      const sheet = new Sheet(flat(s.V), flat(s.c), flat(s.center), s.far_from);
      const out = sheet.h(x);
      sheet.free();
      return out;
    },
  },
};

// the files' rule for one field; the share of the tolerance used, or -1
function within(g, e, rtol, atol) {
  if (!e.every(Number.isFinite))
    return e.every((v, k) => Object.is(v, g[k]) || (isNaN(v) && isNaN(g[k])))
      ? 0
      : -1;
  const err = Math.hypot(...e.map((v, k) => g[k] - v));
  const bound = rtol * Math.hypot(...e) + atol;
  return err <= bound ? (err > 0 ? err / bound : 0) : -1;
}

let failed = 0;
const skipped = new Set();
for (const doc of docs) {
  let worst = [0, ""];
  for (const p of doc.points) {
    const src = doc.sources[p.source];
    for (const name of ["B", "H", "J", "M", "F", "F_far"].filter(
      (n) => n in p,
    )) {
      const kernel = fields[src.kind]?.[name];
      if (!kernel) {
        skipped.add(`${src.kind} ${name}`);
        continue;
      }
      const got = kernel(src, flat(p.observer));
      const r = within(got, flat(p[name]), doc.rtol, p[`atol_${name}`] ?? 0);
      if (r < 0) {
        failed++;
        console.log(
          `${p.id} ${name} off (${p.why}): got ${got}, expected ${p[name]}`,
        );
      } else if (r > worst[0]) worst = [r, `${p.id} ${name}`];
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
        ok = false;
        break;
      }
      const f = kernel(src, flat(t.observer));
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
console.log(failed ? `${failed} off` : "every point within tolerance");
if (skipped.size) console.log(`not in this build: ${[...skipped].join(", ")}`);

// timing: the solved plate's sheet at a grid round it, as studio's field
// lines ask
const plate = docs.find((d) => d.file === "face_charge.json")?.sources
  .solved_plate;
if (plate) {
  const sheet = new Sheet(
    flat(plate.V),
    flat(plate.c),
    flat(plate.center),
    plate.far_from,
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
    const ns = (times[0] * 1e6) / (m * plate.V.length);
    console.log(
      `plate, ${plate.V.length} triangles, ${m} points: ${times[0].toFixed(1)} ms ` +
        `best, ${times[5].toFixed(1)} ms median (${ns.toFixed(0)} ns per pair, one thread)`,
    );
  }
  sheet.free();
} else console.log("plate timing skipped: no face_charge.json");
if (failed) process.exit(1);
