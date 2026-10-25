"""Step 1: the reference, and uniform levels against it.

Uniform levels 3 to 9 and 11, and levels 3 and 5 with the magnet's outline
graded like a contact footprint (a different mesh family). Writes
results/step1.json; skips the meshes already saved there.
"""

import json
import os

import numpy as np
from common import FOOTPRINT, HERE, NAMES, NEAR, PlateBody, corners, errors, indicator, jumps, solve

OUT = os.path.join(HERE, "results", "step1.json")
runs = json.load(open(OUT)) if os.path.exists(OUT) else {}
plan = [(f"uniform m={m}", m, None) for m in (3, 4, 5, 6, 7, 8, 9, 11)] + [(f"outline m={m}", m, FOOTPRINT) for m in (3, 5)]
for name, m, lines in plan:
    if name in runs:
        continue
    body = PlateBody(m, lines=lines)
    c, B, secs = solve(body)
    eta = indicator(body, c)
    runs[name] = {"unknowns": int(body.Nd), "seconds": secs, "jump": float(jumps(body, c).max()),
                  "indicator": float(eta.max()), "indicator_at_mm": (corners(body, c)[0].mean(1)[np.argmax(eta)] * 1e3).round(2).tolist(), "B": B.tolist()}
    print(f"{name}: {body.Nd} unknowns, {secs:.1f} s, largest jump {runs[name]['jump']:.1%}, size-weighted {eta.max():.2%}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(runs, open(OUT, "w"), indent=1)

ref = np.array(runs["uniform m=11"]["B"])
print("\n|B| error against uniform m=11:")
print(f"{'':16s}{'unknowns':>9s} {'jump':>6s} {'weighted':>9s} " + " ".join(f"{n:>16s}" for n in NAMES))
for name, r in runs.items():
    e = errors(np.array(r["B"]), ref)
    print(f"{name:16s}{r['unknowns']:9d} {r['jump']:6.1%} {r['indicator']:9.2%} " + " ".join(f"{x:16.3%}" for x in e)
          + f"   worst near / weighted {e[NEAR].max() / r['indicator']:.1f}")
