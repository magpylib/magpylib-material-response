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

The method of moments gives no error estimate of its own: a result is only as
good as its mesh. `mesh_study` solves one model on several meshes and estimates
the error of the finest from how the results change between them. This example
uses it on a soft iron cube, first with identical cells, then with cells that
shrink toward the faces.

## A soft iron cube

A 10 mm cube of soft iron (susceptibility 1000) sits in a uniform 10 mT field.
The field of the magnetized cube is evaluated 5 mm above its top face and 3 mm
beside a side face; `getB` does not include the applied 10 mT, which `H_ext`
only feeds into the solve.

```{code-cell} ipython3
import magpylib as magpy

from magpylib_material_response.convergence import mesh_study
from magpylib_material_response.meshing import mesh_all


def soft_cube():
    """The unmeshed model, SI units (m, T)."""
    cube = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 0))
    cube.susceptibility = 1000
    cube.H_ext = (0, 0, 0.01)  # 10 mT along z
    return magpy.Collection(cube)


observers = [(0, 0, 0.01), (0.008, 0, 0)]  # above the top face, beside a side face


def report(study):
    print("  cells   Bz above   Bz beside   solve")
    for n, B, t in zip(study.cells, study.values, study.seconds, strict=True):
        print(f"{n:7d} {B[0, 2] * 1e3:8.4f} mT {B[1, 2] * 1e3:8.4f} mT {t:6.1f} s")
    if study.order is None:
        print(f"no steady convergence: no estimate; last change {study.last_change:.2%}")
    else:
        print(f"order {study.order:.2f}, estimated error of the finest mesh {study.error:.2%}")
        print(
            f"extrapolated: {study.estimate[0, 2] * 1e3:.4f} mT above,"
            f" {study.estimate[1, 2] * 1e3:.4f} mT beside"
        )


uniform = mesh_study(soft_cube(), observers, target_elems=(27, 216, 1728))
report(uniform)
```

+++ {"user_expressions": []}

Each mesh has eight times the cells of the previous one. The changes shrink
steadily, so `mesh_study` fits a power law to the three results: the error falls
as the cell count to the power of minus the order, and an order near 0.5 means
it falls with the cell size to the power 1.5.

The estimate holds up against finer meshes, solved separately with 4,096 and
8,000 cells (too slow to run on this page): the field converges to 3.559 mT
above the cube and −4.808 mT beside it. The finest mesh here is off by 0.01 %
above and 1.3 % beside, and the single estimate for both points says 1.3 %. Its
extrapolated value beside the cube, −4.825 mT, overshoots by 0.35 %: a 27-cell
mesh is barely fine enough for the error to follow a power law.

## Cells refined toward the faces

In a high-permeability body the magnetization crowds toward the surface, so
cells that shrink toward the faces resolve it with fewer cells. `grading=3`
makes the middle cells three times the size of those at the faces:

```{code-cell} ipython3
def graded_mesh(collection, target_elems):
    return mesh_all(collection, target_elems, grading=3)


graded = mesh_study(soft_cube(), observers, target_elems=(27, 216, 1728), mesher=graded_mesh)
report(graded)
```

+++ {"user_expressions": []}

At 1,728 cells the graded mesh is 0.5 % from the converged field beside the
cube, against 1.3 % with identical cells; above the cube both are within 0.01 %.
The coarse graded meshes do not change steadily, though: beside the cube 27
cells land close to the converged value by chance, 216 fall short and 1,728 come
back. `mesh_study` therefore gives no estimate, only the last change. Grading
makes each mesh better but its error less regular; to check it, run the study on
finer meshes.

## Notes

- The estimate assumes the error falls as a power of the cell count over the
  three finest meshes. Meshes that differ little, such as 640 and 768 cells,
  change little without being converged: grow the targets by a factor of 2 or
  more.
- One error figure covers all the observers, weighted by the size of their
  field: the relative error at a weak-field point can be larger than reported.
  Compare `study.estimate` with `study.values[-1]` point by point to see it.
- Grading needs three or more cells per side to take effect. On an assembled
  shape such as the [U-core electromagnet](ucore_electromagnet.md), three blocks
  with four cells across each leg, grading gains little: at 768 cells a grading
  of 2 or 3 leaves the peak field 10–12 % below its converged value, against 14
  % with identical cells. Tetrahedra do much better there.
