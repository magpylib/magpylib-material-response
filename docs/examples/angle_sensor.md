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

# An angle sensor on a steel shaft

A magnetic angle sensor, as on a motor's shaft or a steering column: a diametric
magnet on the end of the shaft, and a sensor chip over it whose four Hall plates
on a 1 mm circle read the magnet's angle. The chip reads it from the differences
between opposite plates, so a uniform stray field, from a cable or a motor
nearby, cancels. The shaft is steel, and a stray field magnetizes it. The setup
follows Infineon's patent US 12,237,730 B2 {cite}`binder2025`.

Three questions: how large an angle error does the steel shaft cause in a 5 mT
stray field, how does the air gap change it, and does a soft holder ring around
the chip help?

## The model

A steel shaft, 20 mm across (μr 4000), with a 6 × 3 mm magnet, 0.5 T diametric,
on its end. The sensor sits 1.5 or 2.5 mm above the magnet. The magnet and the
holder ring are stand-ins for the patent's parts.

```{literalinclude} angle_sensor_study.py
:pyobject: angle_sensor
```

```{code-cell} ipython3
import json
import time
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np
from angle_sensor_study import ANGLES, angle_error, angle_sensor, hall_plates

from magpylib_material_response import solve_face_charge

errors, solved, t0 = [], True, time.perf_counter()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for angle in ANGLES:
        solved = solve_face_charge(angle_sensor(angle, gap=1.5, stray=5e-3), level=2, reuse=solved)
        bz = magpy.getB(solved, hall_plates(1.5))[:, 2]
        errors.append(angle_error(bz, angle))
errors = np.array(errors)
print(caught[0].message)
print(f"{len(ANGLES)} angles in {time.perf_counter() - t0:.1f} s, "
      f"{solved.face_charge_info['unknowns']:,} unknowns, factorization kept: {solved.face_charge_info['reused']}")
print(f"angle error: {np.abs(errors - errors.mean()).max():.2f} degrees at most")
```

```{code-cell} ipython3
:tags: [hide-input]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # the warning printed above
    view = solve_face_charge(angle_sensor(30, gap=1.5, stray=5e-3, holder=True), level=2)
chip = [magpy.Sensor(position=p) for p in hall_plates(1.5)]
magpy.show(view, *chip, backend="widget")
```

## How large is the error in a stray field?

The loop above turns the shaft through 12 angles at a 1.5 mm air gap, in 5 mT
along x. The shaft is the only part the solver solves, and it turns as a whole
with its magnet, so every angle keeps the factorization of the first.

At most 1.26° of error at level 2. The solver warns, as printed above, that the
magnet touches the round shaft, where face charge converges slowly: level 4
gives 1.29°, 2 % more, and the patent gives 1.3°. Without a stray field the
error stays under 0.01°: the chip's differences cancel a uniform field exactly,
and only the shaft, magnetized by it, turns part of it into a field that differs
from one plate to the next.

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
from plotly.subplots import make_subplots

stored = json.loads(Path("angle_sensor.json").read_text())
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                    subplot_titles=("the shaft alone", "with a holder ring around the chip"))
style = {(1.5, 0.0): ("#8a8984", "dot"), (1.5, 5e-3): ("#eb6834", "solid"), (2.5, 5e-3): ("#2a78d6", "solid")}
for run in stored["runs"]:
    key = (run["gap"], run["stray"])
    if key not in style:
        continue
    colour, dash = style[key]
    errors = np.array(run["errors"])
    label = f"{run['gap']} mm gap, {run['stray'] * 1e3:.0f} mT"
    fig.add_trace(go.Scatter(x=stored["angles"] + [360], y=list(errors - errors.mean()) + [errors[0] - errors.mean()],
                             name=label, legendgroup=label, showlegend=not run["holder"],
                             line={"color": colour, "dash": dash, "width": 2}), 1, 2 if run["holder"] else 1)
fig.update_xaxes(title_text="shaft angle (degrees)", tickvals=[0, 90, 180, 270, 360])
fig.update_yaxes(title_text="angle error (degrees)", col=1)
fig.update_layout(height=380, margin={"l": 60, "r": 20, "t": 40, "b": 50}, legend={"orientation": "h", "y": -0.25})
fig
```

## How does the air gap change it?

```{code-cell} ipython3
:tags: [hide-input]

print("holder   gap      stray   largest error   12 angles in")
for run in stored["runs"]:
    errors = np.array(run["errors"])
    print(f"{'yes' if run['holder'] else 'no':<6} {run['gap']:>4} mm  {run['stray'] * 1e3:>3.0f} mT  "
          f"{np.abs(errors - errors.mean()).max():10.3f} deg  {run['seconds']:8.1f} s")
```

At 2.5 mm, where the magnet's field at the plates is weaker, the error grows to
1.80°.

## Does a holder ring help?

A soft ring around the chip, 6 to 16 mm across and 4 mm tall, cuts the error
twenty- to fortyfold: to 0.06° at 1.5 mm and 0.04° at 2.5 mm.

It costs solve time. With the ring the shaft turns against a second solved part,
which changes the matrix at every angle: 40 s for the 12 angles, where the shaft
alone, keeping one factorization, takes under a second.

## The answers

- **Stray field:** the steel shaft turns a 5 mT stray field into 1.3° of angle
  error at a 1.5 mm air gap, which the chip's differences cannot cancel.
- **Air gap:** at 2.5 mm the error grows to 1.8°.
- **Holder ring:** a soft ring around the chip brings it down to 0.06° and
  0.04°, at seventy times the solve time.

This model on the <a href="../inside/map.html#shaft">solver map</a>: the path it
takes through the solver, box by box.
