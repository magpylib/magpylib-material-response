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

# Mesh Convergence

The method of moments gives one answer per mesh and no error bar. `mesh_study`
solves a model on meshes four to eight times apart and estimates the error of
the finest from how the results change between them. The estimate is the grid
convergence index (Roache 1994), set out as a procedure by Celik et al. (2008):
Richardson extrapolation (Richardson 1911) over the three finest meshes, the
error relative to the finest value, times a factor of safety of 1.25. It gives
one error for the whole quantity and one per sensor, and none while the results
do not yet change steadily.

Three parts from sensor and actuator design follow. Each study runs here in
seconds; the same study on meshes eight times finer, run separately, shows how
good its estimate is.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np
import plotly.graph_objects as go

from magpylib_material_response.convergence import mesh_study
from magpylib_material_response.meshing import mesh_Cuboid

MM = 1e-3  # lengths in metres, written in millimetres


def uniform_cells(collection, n):
    """A mesher for mesh_study: every part with a susceptibility cut into cubes
    of one size, n across the thinnest part. Parts of one cell size are grids of
    one spacing, which the solver multiplies by FFT, also between parts."""
    parts = collection.sources_all
    soft = [p for p in parts if getattr(p, "susceptibility", 0)]
    size = min(min(p.dimension) for p in soft) / n
    return magpy.Collection(
        *(
            mesh_Cuboid(p, tuple(max(1, round(d / size)) for d in p.dimension))
            if p in soft
            else p.copy()
            for p in parts
        )
    )


def report(study, labels, unit, scale=1.0):
    print("  cells  " + "  ".join(f"{label:>12}" for label in labels) + "   solve")
    for n, value, t in zip(study.cells, study.values, study.seconds, strict=True):
        cols = np.linalg.norm(value, axis=-1) if np.ndim(value) == 2 else value
        print(f"{n:7d}  " + "  ".join(f"{v * scale:9.3f} {unit}" for v in cols) + f" {t:6.1f} s")
    if study.order is None:
        print(f"not steady yet: no estimate; last change {study.last_change:.1%}")
    else:
        print(f"order {study.order:.2f}; error of the finest mesh {study.error:.2%}, per entry:")
        print("  " + ", ".join(f"{label} {e:.2%}" for label, e in zip(labels, study.errors, strict=True)))
```

+++ {"user_expressions": []}

## A flux concentrator

A bar of soft nickel–iron (susceptibility 4,000), 16 × 2 × 2 mm, gathers the
Earth's field into a sensor beside its tip. What matters is the gain: the field
at the sensor over the 50 µT around it.

```{code-cell} ipython3
B0 = 50e-6  # T, the Earth's field, along the bar
bar = magpy.magnet.Cuboid(dimension=(16 * MM, 2 * MM, 2 * MM), polarization=(0, 0, 0))
bar.susceptibility = 4000
bar.H_ext = (B0, 0, 0)
sensors = np.array([(8.5, 0, 0), (9, 0, 0), (10, 0, 0)]) * MM  # 0.5, 1, 2 mm from the tip


def gain(solved):
    return (magpy.getB(solved, sensors)[:, 0] + B0) / B0


concentrator = mesh_study(magpy.Collection(bar), gain, (4, 8, 16), mesher=uniform_cells)
report(concentrator, ["0.5 mm", "1 mm", "2 mm"], "", 1)
```

+++ {"user_expressions": []}

The gain 0.5 mm from the tip falls from 4.34 to 4.27 as the cells shrink, while
the gain 2 mm out rises. The same study on 4,096 to 262,144 cells puts the gains
at 4.259, 3.346 and 2.168, to within 0.14 %. Against these, the finest mesh here
is 0.30 %, 0.07 % and 0.11 % off; `mesh_study` said 0.44 %, 0.13 % and 0.17 %.

## A Halbach array on back iron

Five 6 mm NdFeB cubes, their polarizations turning by 90° from one to the next
(a Halbach array; Halbach 1980), sit on a steel plate. Such arrays drive linear
motors and magnetic encoders, and the field profile above them is what the
design is judged by.

```{code-cell} ipython3
turns = [(1.3, 0, 0), (0, 0, 1.3), (-1.3, 0, 0), (0, 0, -1.3), (1.3, 0, 0)]
magnets = [
    magpy.magnet.Cuboid(dimension=(6 * MM,) * 3, position=(x * MM, 0, 0), polarization=p)
    for x, p in zip((-12, -6, 0, 6, 12), turns, strict=True)
]
for magnet in magnets:
    magnet.susceptibility = 0.05  # NdFeB answers a little
plate = magpy.magnet.Cuboid(
    dimension=(30 * MM, 6 * MM, 3 * MM), position=(0, 0, -4.5 * MM), polarization=(0, 0, 0)
)
plate.susceptibility = 1000
array = magpy.Collection(*magnets, plate)
x = np.linspace(-15, 15, 21)
profile = np.column_stack([x, 0 * x, np.full(21, 5.0)]) * MM  # 2 mm above the magnets

halbach = mesh_study(array, profile, (2, 4, 8), mesher=uniform_cells)
print(f"cells {halbach.cells.tolist()}, order {halbach.order:.2f}, solve of the finest {halbach.seconds[-1]:.1f} s")
print(f"error of the whole profile {halbach.error:.3%}, of the worst point {np.nanmax(halbach.errors):.3%}")
magpy.show(array, backend="widget")
```

```{code-cell} ipython3
:tags: [hide-input]

finest = halbach.values[-1] * 1e3
band = halbach.errors * np.linalg.norm(finest, axis=-1)
fig = go.Figure()
for k, name in ((0, "Bx"), (2, "Bz")):
    fig.add_trace(go.Scatter(x=x, y=finest[:, k], name=name, mode="lines+markers",
                             error_y={"type": "data", "array": band, "visible": True}))
fig.update_layout(xaxis_title="x (mm)", yaxis_title="B 2 mm above the array (mT)", height=360,
                  margin={"l": 60, "r": 20, "t": 20, "b": 40})
fig
```

+++ {"user_expressions": []}

The bars are the estimated error at each point, too small to see on the scale of
the field: 0.03 % for the whole profile, 0.1 % at the worst point, at the end of
the array. Against the same study on 3,840 to 245,760 cells, itself good to 0.03
%, the finest mesh here is 0.02 % off overall and 0.08 % at that point, under
the estimate at all 21 points.

## A steel sheet shielding a sensor

A 1 mm steel sheet (susceptibility 1,000), 24 × 24 mm, sits 3 mm above a 6 mm
magnet and shields three sensors 2 mm behind it. The field there is what is left
after the shield, so the coarse meshes overestimate it: the error has to be
measured against the finest value, not against the larger coarse ones.

```{code-cell} ipython3
magnet = magpy.magnet.Cuboid(dimension=(6 * MM,) * 3, polarization=(0, 0, 1.3))
magnet.susceptibility = 0  # only the sheet answers
sheet = magpy.magnet.Cuboid(
    dimension=(24 * MM, 24 * MM, 1 * MM), position=(0, 0, 6.5 * MM), polarization=(0, 0, 0)
)
sheet.susceptibility = 1000
behind = np.array([(0, 0, 9), (6, 0, 9), (12, 0, 9)]) * MM

shield = mesh_study(magpy.Collection(magnet, sheet), behind, (1, 2, 4), mesher=uniform_cells)
report(shield, ["axis", "6 mm out", "12 mm out"], "mT", 1e3)
```

+++ {"user_expressions": []}

On the axis the residual field falls from 12.4 mT on the coarsest mesh to 6.1
mT. The same study on 4,609 to 294,913 cells puts the three sensors at 5.52,
5.85 and 8.80 mT, to within 3.3 %, 1.1 % and 0.09 %. Against these, the finest
mesh here is 10.1 %, 3.2 % and 0.26 % off, and `mesh_study` said 13.6 %, 4.7 %
and 0.23 %. The last falls 0.03 % short, less than the uncertainty of its
reference: the error is an estimate, not a bound.

<details>
<summary>The reference values: the same studies on meshes eight times finer
(6 minutes and 6 GB, not run here)</summary>

```python
finer = [
    mesh_study(bar.parent, gain, (8, 16, 32), mesher=uniform_cells),
    mesh_study(array, profile, (4, 8, 16), mesher=uniform_cells),
    mesh_study(sheet.parent, behind, (2, 4, 8), mesher=uniform_cells),
]


def per_entry(value):
    return np.linalg.norm(value, axis=-1) if np.ndim(value) == 2 else np.abs(value)


for study, reference in zip((concentrator, halbach, shield), finer, strict=True):
    off = per_entry(study.values[-1] - reference.estimate) / per_entry(
        reference.estimate
    )
    print(f"reference {per_entry(reference.estimate)}, within {reference.errors}")
    print(f"finest mesh here off by {off}, estimated {study.errors}")
```

</details>

## Notes

- **Choose meshes 4 to 8 times apart.** Meshes closer than that change little
  without being converged. `mesh_study` warns when the cell count grows by less
  than 2.2 times, cells 1.3 times smaller, the least Celik et al. (2008)
  recommend. The finest solve takes most of the time.
- **No estimate** means the three finest results do not change steadily yet: the
  coarsest is too coarse for the error to follow a power law. Refine once more.
- **What it measures** is the error of the mesh. A faceted surface stays
  faceted, and a linear susceptibility does not saturate: neither shows here.
- **Tetrahedra** follow the surface they are given: on a detailed one, such as a
  64-sided cylinder, TetGen made several thousand cells whatever the target. The
  warning above then says so.
- **One physical quantity per study:** the entries of a function's result share
  one norm, so a force and a field in one array would be weighed against each
  other.

## References

1. Richardson, L. F. (1911). _The approximate arithmetical solution by finite
   differences of physical problems involving differential equations, with an
   application to the stresses in a masonry dam._ Philosophical Transactions of
   the Royal Society A, 210, 307–357.
   [doi:10.1098/rsta.1911.0009](https://doi.org/10.1098/rsta.1911.0009)

2. Roache, P. J. (1994). _Perspective: a method for uniform reporting of grid
   refinement studies._ Journal of Fluids Engineering, 116(3), 405–413.
   [doi:10.1115/1.2910291](https://doi.org/10.1115/1.2910291)

3. Celik, I. B., Ghia, U., Roache, P. J., & Freitas, C. J. (2008). _Procedure
   for estimation and reporting of uncertainty due to discretization in CFD
   applications._ Journal of Fluids Engineering, 130(7), 078001.
   [doi:10.1115/1.2960953](https://doi.org/10.1115/1.2960953)

4. Halbach, K. (1980). _Design of permanent multipole magnets with oriented rare
   earth cobalt material._ Nuclear Instruments and Methods, 169(1), 1–10.
   [doi:10.1016/0029-554X(80)90094-4](<https://doi.org/10.1016/0029-554X(80)90094-4>)
