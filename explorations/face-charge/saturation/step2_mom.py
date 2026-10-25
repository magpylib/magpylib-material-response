"""Step 2: material-response's MoM with each cell's chi on the B-H curve.

mesh_uniform at 2, 4 and 6 cells across the 20 mm core (10, 5 and 3.3 mm
cells), each meshed once, at the currents of step 1. Writes
results/step2.json; skips what is saved.
"""

import json
import os

import numpy as np
from common import HERE, mom

OUT = os.path.join(HERE, "results", "step2.json")
CURRENTS = (25.0, 100.0, 400.0)
runs = json.load(open(OUT)) if os.path.exists(OUT) else {}
for k in (2, 4, 6):
    name = f"MoM, {k} cells across"
    if name in runs:
        continue
    runs[name] = mom(CURRENTS, k)
    print(f"{name}: {runs[name][0]['cells']} cells", flush=True)
    for r in runs[name]:
        lin, sat = np.array(r["linear"]), np.array(r["saturated"])
        print(f"   {r['current']:5.0f} A: peak Bz linear {lin.max():8.3f} mT, saturated {sat.max():8.3f} mT "
              f"({sat.max() / lin.max():.3f}), {r['solves']} solves, {r['seconds']:.1f} s, lowest chi {r['chi_min']:.0f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(runs, open(OUT, "w"))
