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

+++ {"user_expressions": []}

# Iron pole pieces for a uniform field

A simple permanent magnet for NMR puts two disc magnets face to face and an iron
disc on the inner face of each
([doi:10.1016/j.jmr.2016.12.010](https://doi.org/10.1016/j.jmr.2016.12.010)):
the iron makes the field in the gap between them more uniform. The model here,
asked for in
[#33](https://github.com/magpylib/magpylib-material-response/issues/33), has
made-up dimensions: ferrite discs (0.39 T, as a Y30BH) and iron discs, all 120
mm across and 20 mm thick, with a 40 mm gap between the iron discs. The
uniformity is the peak-to-peak variation of Bz over ±5 mm from the centre,
relative to Bz there, in parts per million (ppm).

Four questions:

- How much more uniform do the iron discs make the field, and what do they cost
  in field strength?
- Which solve can judge a variation of 100 ppm, and which is fastest?
- Do the iron discs forgive a pair of magnets that differ: a polarization off
  the axis, or one magnet weaker than the other?
- Does the iron's material matter: its susceptibility, its saturation?

Times on this page are from an Apple M4 Pro, except those the cells print as
they run, which come from the machine that built the page.

## The model

```{literalinclude} pole_pieces_study.py
:pyobject: pole_pieces
```

```{code-cell} ipython3
:tags: [hide-input]

import json
from pathlib import Path

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pole_pieces_study import LINES, MAP_X, MAP_Z, MM, by_line, lines, pole_pieces, variation

model = pole_pieces()
for disc in model.children[1::2]:
    disc.style.opacity = 0.5
magpy.show(model, backend="widget")
```

The magnets' polarization is held fixed. The ferrite's own susceptibility, about
0.05, lowers Bz at the centre by 2.9 % and changes the variation by less than
0.1 ppm (stored below), so it is left out.

## How much more uniform?

Finite elements at order 4 solve the model in a few seconds; the magnets alone
need no solve.

```{code-cell} ipython3
import time

from magpylib_material_response.fem import solve_fem

t0 = time.perf_counter()
solved = solve_fem(pole_pieces(), order=4, observers=lines())
print(f"{solved.ndof:,} unknowns, {time.perf_counter() - t0:.1f} s")

fields = {"magnets alone": by_line(magpy.getB(pole_pieces(iron=None), lines())),
          "with the iron discs": by_line(magpy.getB(solved, lines()))}
for name, B in fields.items():
    print(f"{name}: Bz {B['along the axis'][15][2] * 1e3:.2f} mT at the centre")
    print("  over ±5 mm: " + ", ".join(f"{variation(B[line], line):,.0f} ppm {line}" for line in LINES))
```

The iron discs make the field 65 to 70 times more uniform: from 6,049 to 88 ppm
along the axis, and from 3,059 to 48 ppm across the gap. They cost 12 % of the
field at the centre, 59.4 to 52.2 mT. In the plane y = 0, the field stays within
100 ppm of its centre value out to about 7 mm from the axis and 5 mm along it
with the iron; without it, to less than 1 mm either way:

```{code-cell} ipython3
:tags: [hide-input]

stored = json.loads(Path("pole_pieces.json").read_text())
X, Z = np.meshgrid(MAP_X, MAP_Z, indexing="ij")
points = np.stack([X, 0 * X, Z], axis=-1).reshape(-1, 3)
maps = {"magnets alone": magpy.getB(pole_pieces(iron=None), points)[:, 2].reshape(X.shape),
        "with the iron discs": np.array(stored["field_map"]["Bz"])}


def mirrored(a):
    """The quarter map, mirrored about x = 0 and z = 0."""
    a = np.concatenate([a[:0:-1], a], axis=0)
    return np.concatenate([a[:, :0:-1], a], axis=1)


x, z = np.r_[-MAP_X[:0:-1], MAP_X] / MM, np.r_[-MAP_Z[:0:-1], MAP_Z] / MM
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, subplot_titles=list(maps))
for col, Bz in enumerate(maps.values(), start=1):
    ppm = np.abs(Bz / Bz[0, 0] - 1) * 1e6
    fig.add_trace(go.Contour(
        x=x, y=z, z=np.log10(np.maximum(mirrored(ppm), 1)).T, zmin=0, zmax=5, colorscale="Viridis",
        contours={"start": 1, "end": 5, "size": 1}, line={"width": 1, "smoothing": 1.3},
        customdata=mirrored(ppm).T, hovertemplate="x %{x:.1f} mm, z %{y:.1f} mm: %{customdata:,.0f} ppm<extra></extra>",
        colorbar={"title": "ppm", "tickvals": [0, 1, 2, 3, 4, 5],
                  "ticktext": ["1", "10", "100", "1,000", "10,000", "100,000"]},
        showscale=col == 2), 1, col)
    fig.update_xaxes(title_text="x (mm)", row=1, col=col)
fig.update_yaxes(title_text="z (mm)", row=1, col=1)
fig.update_layout(height=300, margin={"l": 60, "r": 20, "t": 40, "b": 50})
fig
```

## Which solve can judge 100 ppm?

A variation of 100 ppm is the difference of two fields that agree to 0.01 %. A
solve whose field is off by up to 0.05 % at each point, by different amounts,
can be off by up to 1,000 ppm in the variation, so an error bar on the field
says little about it. [`pole_pieces_study.py`](pole_pieces_study.py) solves the
model three ways and stores the results; finite elements at order 5 are the
reference.

```{code-cell} ipython3
:tags: [hide-input]

fem, fc = stored["fem"], stored["face_charge"]
runs = {f"finite elements, order {o}": (n, t, B) for o, n, t, B in zip(fem["orders"], fem["ndofs"], fem["seconds"], fem["B"])}
runs |= {f"face charge, level {lvl}": (n, t, B) for lvl, n, t, B in zip(fc["levels"], fc["unknowns"], fc["seconds"], fc["B"])}
runs |= {name: (r["cells"], r["seconds"], r["B"]) for name, r in stored["cells"].items()}
reference = fem["B"][-1]
centre = reference["along the axis"][15][2]

print("Bz off: Bz at the centre, against order 5")
print("along, across: the variation over ±5 mm, in ppm\n")
print(f"{'':<24}{'unknowns':>9}{'time':>7}{'Bz off':>8}{'along':>7}{'across':>7}")
for name, (n, seconds, B) in runs.items():
    off = B["along the axis"][15][2] / centre - 1
    print(f"{name:<24}{n:>9,}{seconds:>6.1f}s{off:>+8.2%}"
          + "".join(f"{variation(B[line], line):>7,.0f}" for line in LINES))
for name in ("mesh", "ferrite"):
    label = {"mesh": "the finer mesh", "ferrite": "ferrite χ = 0.05"}[name]
    B = stored[name]["B"]
    print(f"order 4, {label}: "
          + ", ".join(f"{variation(B[line], line):.2f}" for line in LINES) + " ppm")
```

```{code-cell} ipython3
:tags: [hide-input]

shown = [("finite elements, order 5", "#555555", "solid"), ("face charge, level 3", "#eb6834", "dash"),
         ("cubes, 9,527 per disc", "#2a78d6", "solid"), ("cubes, 37,367 per disc", "#2a78d6", "dot")]
fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=list(LINES))
for col, (line, coordinate) in enumerate((("along the axis", 2), ("across the gap", 0)), start=1):
    s = LINES[line][:, coordinate] / MM
    for name, colour, dash in shown:
        Bz = np.array(runs[name][2][line])[:, 2]
        fig.add_scatter(x=s, y=(Bz / Bz[len(s) // 2] - 1) * 1e6, name=name, legendgroup=name, showlegend=col == 1,
                        mode="lines", line={"color": colour, "dash": dash, "width": 2}, row=1, col=col)
fig.update_xaxes(title_text="z (mm)", row=1, col=1)
fig.update_xaxes(title_text="x (mm)", row=1, col=2)
fig.update_yaxes(title_text="ΔBz from the centre (ppm)", row=1, col=1)
fig.update_layout(height=380, margin={"l": 70, "r": 20, "t": 40, "b": 40}, legend={"orientation": "h", "y": -0.25})
fig
```

The error bar that counts is on the variation itself: how much it changes from
one order, level or mesh to the next. Finite elements converge: orders 4 and 5
agree on it within 0.1 ppm, and order 4 on a mesh at least twice as fine near
the lines and in the discs gives the same to 0.07 ppm. Face charge, a different
method, changes it by 1 ppm from level 2 to 3 and lands within 2 ppm of the
reference. It warns that each magnet touches an iron disc and expects errors of
a few percent near that contact, which lies 20 mm from the gap: the agreement
shows the error does not reach there. The error bar `study_face_charge` puts on
the field, 0.04 % at level 3, is 400 ppm: far too wide to judge the variation
by.

Cells do not judge it. Cylinder cells put the variation at 4,959 ppm with 52
cells and 1,390 ppm with 450, in 11 s and 1.3 GB, 56 and 16 times the reference;
their dense blocks rule out the finer meshes that might close that. Cubes are
cheap per cell, but their staircase outline puts a ripple on the field near the
iron's faces and rims: with about 9,500 cubes per disc, Bz at the centre is
within 0.05 % and the variation is 30 ppm instead of 88; with 37,000, 60 ppm. It
does not converge.

For one design, finite elements are the fastest: within 0.1 ppm in 5 s, where
face charge needs 20 s at level 3 for 2 ppm. `solve_fem` keeps nothing between
solves, though, so every design is a new solve, even one that changes only the
magnets. Face charge factors its matrix once and keeps it while only the magnets
change, which is the next question.

## Do the discs forgive a pair of magnets that differ?

The top magnet's polarization turned off the axis by up to 2°, and 2 % weaker or
stronger than the bottom one's: nine variants. Face charge at level 2 keeps its
factorization from the first solve (`reuse=True`) and solves each variant from
it; the magnets alone need no solve.

```{code-cell} ipython3
import warnings

from magpylib_material_response import solve_face_charge

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*touches.*")  # discussed above
    t0 = time.perf_counter()
    first = solve_face_charge(pole_pieces(), level=2, reuse=True)
    print(f"first solve, factorization kept: {time.perf_counter() - t0:.1f} s")

    variants, t0 = {}, time.perf_counter()
    for angle in (0.0, 1.0, 2.0):
        for strength in (0.98, 1.0, 1.02):
            solved = solve_face_charge(pole_pieces(angle=angle, strength=strength), level=2, reuse=first)
            variants[angle, strength] = {
                "no iron": by_line(magpy.getB(pole_pieces(iron=None, angle=angle, strength=strength), lines())),
                "iron": by_line(magpy.getB(solved, lines()))}
    print(f"{len(variants)} variants: {(time.perf_counter() - t0) / len(variants):.2f} s each")
```

```{code-cell} ipython3
:tags: [hide-input]

print(f"{'':>17}{'Bx at the centre, µT':>22}{'ppm along the axis':>21}")
print(f"{'angle  strength':>17}{'no iron':>11}{'iron':>11}{'no iron':>11}{'iron':>10}")
for (angle, strength), by_iron in variants.items():
    Bx = [by_iron[k]["along the axis"][15][0] * 1e6 for k in ("no iron", "iron")]
    ppm = [variation(by_iron[k]["along the axis"], "along the axis") for k in ("no iron", "iron")]
    print(f"{angle:>4.0f}°{strength:>10.2f}  {Bx[0]:>11.1f}{Bx[1]:>11.1f}{ppm[0]:>11,.0f}{ppm[1]:>10,.0f}")
print("\nagainst finite elements at order 4 (stored), ppm along the axis:")
for check in stored["variants"]:
    mine = variation(variants[check["angle"], check["strength"]]["iron"]["along the axis"], "along the axis")
    print(f"  {check['angle']:.0f}°, {check['strength']:.2f}: face charge {mine:.1f}, "
          f"finite elements {variation(check['B']['along the axis'], 'along the axis'):.1f}")
```

A polarization off the axis puts a transverse field at the centre: 1° gives 259
µT of the magnets alone and 2.7 µT with the iron discs, a hundred times less,
and the variation hardly changes. A weaker or stronger magnet adds a gradient
along the axis: the point between the magnets where the field is flattest moves
toward the weaker one. 2 % adds 1,230 ppm without the iron and 230 ppm with it.
The iron forgives an angle almost wholly, an uneven strength five times.

Each variant takes under half a second, field included, after a first solve of
about 10 s, and lands within 3 ppm of finite elements. Finite elements at that
accuracy, order 3 (4.4 ppm), take 3.2 s for each variant; order 4 takes 4.8 s.
At matched accuracy face charge is ahead from the fourth variant on and about
seven times faster per variant. A sweep of the iron's shape changes the matrix,
and there finite elements stay the faster.

## Does the iron's material matter?

The iron discs with susceptibilities of 300 to 3,999, and on a measured B–H
curve: the steel of TEAM problem 13, a structural steel standing in for a pure
iron, whose datasheet curve would rise more steeply. Finite elements at order 4,
stored; on the curve they repeat the solve until the susceptibility settles, 135
s here.

```{code-cell} ipython3
:tags: [hide-input]

from pole_pieces_study import TEAM13_B, TEAM13_H

print("|B| in iron: edge, 0.1 mm from the rim's outer edge at")
print(f"(r, z) = {stored['iron']['χ = 3999']['where (r, z) in mm']} mm; bulk, 1 mm or more inside the rim\n")
print(f"{'':<14}{'Bz centre':>10}{'ppm along':>10}{'across':>7}{'edge':>7}{'bulk':>7}")
for name, r in stored["iron"].items():
    B = r["B"]
    print(f"{name:<14}{B['along the axis'][15][2] * 1e3:>7.2f} mT"
          + "".join(f"{variation(B[line], line):>{w},.0f}" for line, w in zip(LINES, (10, 7)))
          + f"{r['max |B| in iron']:>5.2f} T{r['max |B| 1 mm inside the rim']:>5.2f} T")
print("\nTEAM 13 relative permeability:")
for h, b in zip(TEAM13_H, TEAM13_B, strict=True):
    if b in (0.0025, 0.05, 0.2, 0.3, 0.4):
        print(f"  {b / (magpy.mu_0 * h):6,.0f} at {b} T")
```

```{code-cell} ipython3
:tags: [hide-input]

colours = {"χ = 300": "#a9c8ee", "χ = 1000": "#5b9be0", "χ = 3999": "#1f4f99", "TEAM 13 curve": "#eb6834"}
fig = go.Figure()
s = LINES["along the axis"][:, 2] / MM
for name, r in stored["iron"].items():
    Bz = np.array(r["B"]["along the axis"])[:, 2]
    fig.add_scatter(x=s, y=(Bz / Bz[15] - 1) * 1e6, name=name, mode="lines", line={"color": colours[name], "width": 2})
fig.update_xaxes(title_text="z (mm), along the axis")
fig.update_yaxes(title_text="ΔBz from the centre (ppm)")
fig.update_layout(height=340, margin={"l": 70, "r": 20, "t": 20, "b": 50}, legend={"orientation": "h", "y": -0.3})
fig
```

The iron carries 0.36 T at most, 0.53 T on the TEAM 13 steel, 1 mm or more
inside the rim. At the rim's outer edge beside the magnet the field rises
without bound, as at any edge, to 0.7 T at 0.1 mm from it in this mesh: a
saturation there would stay local. So a B–H curve only matters below its knee,
and there it does matter. The susceptibility moves Bz at the centre by 0.5 %
between 300 and 3,999, but the variation by 2.3 times, from 203 to 88 ppm: the
higher the susceptibility, the closer the iron's faces come to surfaces of one
magnetic potential, between which the field is uniform. The TEAM 13 steel, with
a relative permeability of 124 at the lowest fields and about 1,100 at 0.3 T,
gives 161 ppm. So enter the iron's measured curve, at least up to the field it
carries.

## The answers

- **How much more uniform:** 65 to 70 times, from 6,049 to 88 ppm along the axis
  and from 3,059 to 48 ppm across the gap over ±5 mm, for 12 % of the field.
- **Which solve:** for one design, finite elements at order 4, within 0.1 ppm in
  5 s; for many variants of the magnets, face charge with `reuse`, under half a
  second each, within 3 ppm and about seven times faster than finite elements at
  that accuracy. Cells miss the variation by 1.5 to 56 times, even where their
  centre field is within 0.05 %. Judge the variation by how it changes with the
  order, level or mesh, not by an error bar on the field.
- **Magnets that differ:** a polarization 1° off the axis gives a hundred times
  less transverse field with the iron; a 2 % weaker magnet adds 230 ppm instead
  of 1,230.
- **The material:** the iron's bulk stays under 0.6 T, far from saturation, but
  the low-field permeability sets the variation: 88 ppm at χ = 3,999, 203 ppm at
  300, 161 ppm on the TEAM 13 steel.

These numbers compare designs: one iron disc against another, a magnet's
tolerance against its effect. An NMR magnet's absolute uniformity also depends
on what no model here holds, such as the machining of the iron's faces and how
uniform the magnets' polarization is, and is usually brought to the last ppm by
shimming.
