"""Step 3: bisection where the size-weighted jump is large.

From a uniform level: mark the panels whose size-weighted jump is above
`frac` times the largest, bisect each marked panel along both of its in-plane
axes (in the tensor mesh, the whole strips through it), solve again; repeat
until the unknowns pass a cap. Writes results/step3.json; skips runs saved.
"""

import json
import os

import numpy as np
from common import HERE, NAMES, NEAR, PlateBody, errors, indicator, solve

OUT = os.path.join(HERE, "results", "step3.json")
CAP = 11000  # unknowns: about uniform level 11
runs = json.load(open(OUT)) if os.path.exists(OUT) else {}
ref = np.array(json.load(open(os.path.join(HERE, "results", "step1.json")))["uniform m=11"]["B"])

for m0, frac in ((3, 0.5), (3, 0.25), (5, 0.5)):
    name = f"from m={m0}, mark > {frac} of the largest"
    if name in runs:
        continue
    body, rounds = PlateBody(m0), []
    while True:
        c, B, secs = solve(body)
        eta = indicator(body, c)
        e = errors(B, ref)
        rounds.append({"unknowns": int(body.Nd), "seconds": secs, "indicator": float(eta.max()),
                       "near": float(e[NEAR].max()), "far": float(e[5:].max()), "B": B.tolist()})
        print(f"{name}: {body.Nd:6d} unknowns, {secs:5.1f} s, indicator {eta.max():.2%}, "
              f"worst near {e[NEAR].max():.2%}, far {e[5:].max():.3%}", flush=True)
        p = body.pan
        marked = np.where(eta > frac * eta.max())[0]
        extra = [list(x) for x in body.extra]
        for n in marked:
            extra[p.i[n]].append(0.5 * (p.p1[n] + p.p2[n]))
            extra[p.j[n]].append(0.5 * (p.q1[n] + p.q2[n]))
        nxt = PlateBody(m0, extra=extra)
        if nxt.unknowns() > CAP:
            break
        body = nxt
    runs[name] = rounds
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(runs, open(OUT, "w"), indent=1)

uni = json.load(open(os.path.join(HERE, "results", "step1.json")))
print("\nworst error 0.25 mm above the plate, against uniform level 11:")
for name, r in uni.items():
    if name.startswith("uniform") and name != "uniform m=11":
        print(f"  {name:38s} {r['unknowns']:6d} unknowns: {errors(np.array(r['B']), ref)[NEAR].max():.2%}")
for name, rounds in runs.items():
    print(f"  {name:38s} " + ", ".join(f"{r['unknowns']} -> {r['near']:.2%}" for r in rounds))
print("points:", ", ".join(NAMES[NEAR]))

# The two finest bisections agree under the magnet better than uniform level 11
# does with either: compare everything against the finest one instead.
best = np.array(runs["from m=5, mark > 0.5 of the largest"][-1]["B"])
rows = {"uniform m=9": uni["uniform m=9"], "uniform m=11": uni["uniform m=11"], "outline m=5": uni["outline m=5"]}
for name, rounds in runs.items():
    for r in rounds[2:]:
        rows[f"{name}, {r['unknowns']}"] = r
print("\nerror against the finest bisection (from m=5, 8,064 unknowns):")
print(f"{'':52s}{'unknowns':>9s}" + "".join(f"{n:>17s}" for n in NAMES))
for name, r in rows.items():
    print(f"{name:52s}{r['unknowns']:9d}" + "".join(f"{x:17.3%}" for x in errors(np.array(r["B"]), best)))
