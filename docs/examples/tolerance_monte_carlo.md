---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Assembly tolerances, by Monte Carlo

A magnet on a steel back plate and a Hall sensor between them, as in a position
switch. In production neither the magnet nor the sensor sits exactly where the
drawing puts it. Assembly tolerances with σ of 0.1 mm in position and 1° in
tilt, for each of the two, are eight degrees of freedom, as in Ergun et al.
{cite}`ergun2019`.

Four questions: how far does the sensor's reading spread, which tolerance drives
the spread, does the steel plate change the answer, and how long do 10,000
samples take?

## The model

A 5 mm cube magnet, 1 T along z, 1 mm over a 10 × 10 × 2 mm steel plate (χ =
1000). The sensor reads Bz in the gap, 0.5 mm outside the magnet's edge.

```{literalinclude} tolerance_study.py
:pyobject: assembly
```

```{code-cell} ipython3
import json
import time
from pathlib import Path

import magpylib as magpy
import numpy as np
from tolerance_study import DOF, assembly, draw, pose

from magpylib_material_response import solve_face_charge

model, magnet, sensor = assembly()
magnet_at, sensor_at = magnet.position.copy(), sensor.position.copy()

bz, solved, t0 = [], True, time.perf_counter()
for deviation in draw(200):
    pose(magnet, magnet_at, deviation, "magnet")
    pose(sensor, sensor_at, deviation, "sensor")
    solved = solve_face_charge(model, level=2, reuse=solved)
    bz.append(magpy.getB(solved, sensor)[2])
bz = np.array(bz) * 1e3
print(f"200 samples in {time.perf_counter() - t0:.1f} s, factorization kept: {solved.face_charge_info['reused']}")
print(f"Bz: {bz.mean():.1f} mT, spread (sigma) {bz.std():.1f} mT")
```

```{code-cell} ipython3
:tags: [hide-input]

magpy.show(solve_face_charge(assembly()[0], level=2), assembly()[2], backend="widget")
```

## How far does the reading spread?

The 200 samples above show the loop. The 10,000 below come from
[`tolerance_study.py`](tolerance_study.py), stored.

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go

stored = json.loads(Path("tolerance.json").read_text())
with_plate = np.array(stored["bz"]) * 1e3
alone = np.array(stored["alone"]) * 1e3
low, high = np.percentile(with_plate, [0.135, 99.865])
print(f"nominal {stored['nominal'] * 1e3:.1f} mT; over {stored['samples']:,} samples: "
      f"mean {with_plate.mean():.1f} mT, sigma {with_plate.std():.1f} mT, "
      f"99.73 % between {low:.0f} and {high:.0f} mT")

fig = go.Figure()
fig.add_trace(go.Histogram(x=alone, name="Magpylib alone, without the plate", nbinsx=80,
                           marker={"color": "#8a8984"}, opacity=0.6))
fig.add_trace(go.Histogram(x=with_plate, name="with the steel plate", nbinsx=80,
                           marker={"color": "#eb6834"}, opacity=0.75))
fig.update_layout(barmode="overlay", height=360, margin={"l": 60, "r": 20, "t": 20, "b": 50},
                  legend={"x": 0.6, "y": 0.95})
fig.update_xaxes(title_text="Bz at the sensor (mT)")
fig.update_yaxes(title_text="samples")
fig
```

Tolerances of a tenth of a millimetre and a degree spread the reading by a
third: σ of 34 mT on a nominal 105 mT, and the spread is lopsided, with a long
tail upward. The sensor sits just outside the magnet's edge, where Bz changes
fastest with position.

## Which tolerance drives it?

Each tolerance alone, moved by one σ either way, gives its share of the spread:

```{code-cell} ipython3
:tags: [hide-input]

share = np.array(stored["share"]) * 1e3
for name, s in sorted(zip(stored["dof"], share, strict=True), key=lambda p: -abs(p[1])):
    print(f"{name:<22} {s if abs(s) > 0.05 else 0:+7.1f} mT per sigma")
print(f"together, as independent shares: sigma {np.sqrt((share**2).sum()):.1f} mT")
```

The magnet's and the sensor's positions along x weigh the same and pull opposite
ways: what counts is where the sensor sits relative to the magnet. Then comes
the magnet's tilt about y. Along y and about x the shares vanish, as the model
is symmetric there. Summed as independent shares they give a σ of 33.4 mT,
within 2 % of the Monte Carlo's: a linear tolerance stack would have told the
size of the spread, but not its lopsided tail.

## Does the steel plate change the answer?

```{code-cell} ipython3
:tags: [hide-input]

print(f"            nominal    sigma    sigma / nominal")
print(f"alone      {stored['nominal_alone'] * 1e3:6.1f} mT  {alone.std():5.1f} mT  {alone.std() / (stored['nominal_alone'] * 1e3):8.0%}")
print(f"with plate {stored['nominal'] * 1e3:6.1f} mT  {with_plate.std():5.1f} mT  {with_plate.std() / (stored['nominal'] * 1e3):8.0%}")
```

It doubles the reading and widens the spread from 24 to 34 mT. A tolerance study
in Magpylib alone, without the plate's response, puts the switch's threshold in
the wrong place and underestimates its spread by 30 %.

## How long do 10,000 samples take?

Only the magnet and the sensor move, and the magnet is not a solved part, so
every sample keeps the factorization of the first: 10,000 samples in 2 minutes
here, 12 ms each. A fresh solve per sample takes 44 ms, 7 minutes for the run.
The sensor's tolerances need no solve at all: its pose only changes where the
solved model is read.

## The answers

- **Spread:** σ 34 mT on 105 mT, a third, with a long upward tail.
- **Drivers:** the sensor's position relative to the magnet along x, then the
  magnet's tilt; the linear stack gets σ within 2 %.
- **The plate:** it doubles the reading and widens the spread by 44 %; without
  it the study puts the switch's threshold at half the field.
- **Time:** 10,000 samples in 2 minutes, with the factorization kept.

This model on the <a href="../inside/map.html#mc">solver map</a>: the path it
takes through the solver, box by box.
