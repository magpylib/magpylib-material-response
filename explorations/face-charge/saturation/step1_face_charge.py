"""Step 1: face charge on slabs of uniform chi, iterated on the B-H curve.

Whole cuboids, then slabs about 20 and 10 mm long, at mesh level 2 (and 3 on
the 20 mm slabs, to see the panels' share; and the 20 mm slabs with their
cross-section cut 2 x 2), at 25, 100 and 400 A per winding.
Writes results/step1.json; skips what is saved.
"""

import json
import os

import numpy as np
from common import HERE, LINE, face_charge

OUT = os.path.join(HERE, "results", "step1.json")
CURRENTS = (25.0, 100.0, 400.0)
runs = json.load(open(OUT)) if os.path.exists(OUT) else {}
for size, m, across in ((1.0, 2, 1), (0.02, 2, 1), (0.01, 2, 1), (0.02, 3, 1), (0.02, 2, 2)):
    name = f"slabs {size * 1e3:.0f} mm, m={m}" if size < 1 else f"whole cuboids, m={m}"
    if across > 1:
        name += f", cross-section cut {across} x {across}"
    if name in runs:
        continue
    runs[name] = face_charge(CURRENTS, size, m, across)
    r = runs[name][0]
    print(f"{name}: {r['bodies']} bodies, {r['unknowns']} unknowns, built in {r['build_s']:.1f} s", flush=True)
    for r in runs[name]:
        lin, sat = np.array(r["linear"]), np.array(r["saturated"])
        print(f"   {r['current']:5.0f} A: peak Bz linear {lin.max():8.3f} mT, saturated {sat.max():8.3f} mT "
              f"({sat.max() / lin.max():.3f}), {r['solves']} solves, {r['seconds']:.1f} s, lowest chi {min(r['chi']):.0f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(runs, open(OUT, "w"))
