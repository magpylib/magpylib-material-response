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

# A gear-tooth sensor

A back-biased Hall sensor counts the teeth of a steel gear. A magnet sits behind
the sensor, facing the gear. A tooth in front of it draws the magnet's flux
through the sensor, a gap between two teeth lets it spread, and the swing
between the two is the signal. Magpylib alone sees no gear: its field at the
sensor is a flat line.

Two questions: how large is the signal, and how does the air gap set it? And one
for the engineer running the study: how fast is a sweep of solves?

## The model

A rack of seven teeth, 2 mm wide at a 4 mm pitch, on a 2 mm base, as on the rim
of a large wheel. A 5 × 4 mm NdFeB disc stands 0.5 mm behind the sensor, which
reads Bz with an air gap of 1 mm to the teeth.

```{code-cell} ipython3
import time

import magpylib as magpy
import numpy as np

from magpylib_material_response import solve_face_charge

MM = 1e-3  # lengths are in metres; this writes them in millimetres

def make_rack():
    base = magpy.magnet.Cuboid(dimension=(28 * MM, 6 * MM, 2 * MM), position=(0, 0, -1 * MM), polarization=(0, 0, 0))
    teeth = [
        magpy.magnet.Cuboid(dimension=(2 * MM, 6 * MM, 3 * MM), position=(x * MM, 0, 1.5 * MM), polarization=(0, 0, 0))
        for x in (-12, -8, -4, 0, 4, 8, 12)
    ]
    rack = magpy.Collection(base, *teeth)
    rack.susceptibility = 1000  # soft steel, set on the collection for every part
    return rack


def head(air_gap):
    """The magnet, and the sensor's height in m."""
    sensor_z = 3 + air_gap  # mm: the teeth end at 3 mm
    magnet = magpy.magnet.Cylinder(
        dimension=(5 * MM, 4 * MM), position=(0, 0, (sensor_z + 2.5) * MM), polarization=(0, 0, 1.2)
    )
    return magnet, sensor_z * MM
```

```{code-cell} ipython3
:tags: [hide-input]

magnet, sensor_z = head(1.0)
sensor = magpy.Sensor(position=(0, 0, sensor_z))
magpy.show(solve_face_charge(magpy.Collection(make_rack(), magnet), level=2), sensor, backend="widget")
```

## How large is the signal?

The rack stays where it is and the head moves along it, over two tooth pitches,
at air gaps of 1 and 2 mm: one face-charge solve per position.

```{code-cell} ipython3
positions = np.linspace(-4, 4, 33)  # mm


def sweep(air_gap):
    magnet, sensor_z = head(air_gap)
    model = magpy.Collection(make_rack(), magnet)
    solved, bz, seconds = True, [], []
    for x in positions:
        magnet.position = (x * MM, 0, magnet.position[2])
        t0 = time.perf_counter()
        solved = solve_face_charge(model, level=2, reuse=solved)
        seconds.append(time.perf_counter() - t0)
        bz.append(magpy.getB(solved, (x * MM, 0, sensor_z))[2])
    info = solved.face_charge_info
    print(f"air gap {air_gap} mm: {info['unknowns']:,} unknowns, first solve {seconds[0]:.2f} s, "
          f"then {np.median(seconds[1:]) * 1e3:.0f} ms each, reused: {info['reused']}")
    return np.array(bz) * 1e3  # mT


signals = {gap: sweep(gap) for gap in (1.0, 2.0)}
magnet, sensor_z = head(1.0)
alone = magpy.getB(magnet, (0, 0, sensor_z))[2] * 1e3
```

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=positions, y=[alone] * len(positions), name="Magpylib alone, either gap",
                         line={"color": "#8a8984", "width": 2, "dash": "dot"}))
for gap, colour in ((1.0, "#eb6834"), (2.0, "#2a78d6")):
    fig.add_trace(go.Scatter(x=positions, y=signals[gap], name=f"air gap {gap:.0f} mm",
                             line={"color": colour, "width": 3}))
for x in (-4, 0, 4):
    fig.add_vrect(x0=x - 1, x1=x + 1, fillcolor="#8a8984", opacity=0.12, line_width=0)
fig.update_xaxes(title_text="sensor position along the rack (mm); shaded: a tooth under it")
fig.update_yaxes(title_text="Bz at the sensor (mT)")
fig.update_layout(height=400, margin={"l": 60, "r": 20, "t": 20, "b": 50}, legend={"x": 0.01, "y": 0.99})
fig
```

```{code-cell} ipython3
:tags: [hide-input]

for gap, bz in signals.items():
    print(f"air gap {gap:.0f} mm: {bz.min():.0f} to {bz.max():.0f} mT, a swing of {np.ptp(bz):.1f} mT")
print(f"Magpylib alone: {alone:.0f} mT at either position")
```

The steel adds a constant part, the base and the teeth together, and the swing
on top of it is the signal. Doubling the air gap from 1 to 2 mm cuts the swing
more than sevenfold, which sets how large an air gap the sensor's electronics
must still resolve.

Level 2 is enough here. At level 4, with 15,984 unknowns, the field over a tooth
and over a gap changes the swing at 1 mm by 0.6 % (from a run not repeated on
this page).

## How fast is a sweep?

Only the magnet moves, and the magnet is not a solved part, so every solve has
the same matrix: `reuse=` keeps the factorization of the first solve, and each
later one is a new load and a back-substitution, as the timings above show. Two
changes would lose it. A cuboid magnet close to a part regrades the part's
panels toward its footprint. A magnet with its own susceptibility is a solved
part moving against the rack. Each step then solves afresh.
[Many solves](../guide/loops.md) lists what keeps the matrix and what doesn't.

## The answers

- **Signal:** a swing of 62 mT on top of 478 mT at a 1 mm air gap; Magpylib
  alone sees none.
- **Air gap:** at 2 mm the swing is 8.5 mT, seven times less.
- **Sweep:** 0.8 s for the first solve, then about 50 ms each, the factorization
  kept.
