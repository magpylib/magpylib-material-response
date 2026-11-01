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

# Saturation

+++

With a constant susceptibility, iron can reach any polarization: at 1 T applied,
a χ = 1000 sphere comes out at 3 T, where real iron saturates near 1.6 to 2 T. A
`bh_curve` set on a part, or on a collection above it, replaces the constant:
`apply_demag` then puts every cell at the susceptibility its own field gives on
the curve, repeating the linear solve until none changes. The interaction matrix
is built once; the fixed point takes 10 to 40 solves with Anderson acceleration.

## A curve

A curve is a table of (H, B) points in A/m and T, from (0, 0), with H
increasing. A measured steel's table goes in as it is. Here a simple curve
stands in: J = χ₀ μ₀ H / (1 + χ₀ μ₀ H / Jₛ), with χ₀ = 1000 and Jₛ = 1.6 T, so B
= J + μ₀ H. Past the table's last point, J keeps its last value.

```{code-cell} ipython3
import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from magpylib_material_response.demag import apply_demag
from magpylib_material_response.meshing import mesh_Cuboid, mesh_TriangularMesh

MU0 = 4e-7 * np.pi
CHI0, JS = 1000.0, 1.6


def polarization(H):
    return CHI0 * MU0 * H / (1 + CHI0 * MU0 * H / JS)


H = np.r_[0.0, np.geomspace(1e-2, 1e6, 400)]  # A/m
curve = np.c_[H, polarization(H) + MU0 * H]  # (H, B) rows

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(MU0 * H, polarization(H), label="the curve, J(H)")
ax.plot(MU0 * H, CHI0 * MU0 * H, "--", label="χ = 1000")
ax.set(xlim=(0, 0.02), ylim=(0, 2), xlabel="μ₀ H inside the iron (T)", ylabel="J (T)")
ax.legend()
```

## A sphere against the exact answer

A sphere stays uniformly polarized while it saturates, so its answer is exact:
μ₀ H inside is B₀ − J / 3, and J follows the curve at that field. The sphere is
a triangle mesh cut into tetrahedra, placed in a uniform field `H_ext`.

```{code-cell} ipython3
def icosphere(diameter, splits=2):
    t = (1 + 5**0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    V = list(V / np.linalg.norm(V, axis=1)[:, None])
    for _ in range(splits):
        mid, new = {}, []
        for a, b, c in F:
            m = []
            for i, j in ((a, b), (b, c), (c, a)):
                key = (min(i, j), max(i, j))
                if key not in mid:
                    p = V[i] + V[j]
                    V.append(p / np.linalg.norm(p))
                    mid[key] = len(V) - 1
                m.append(mid[key])
            new += [(a, m[0], m[2]), (b, m[1], m[0]), (c, m[2], m[1]), (m[0], m[1], m[2])]
        F = new
    return magpy.magnet.TriangularMesh(
        polarization=(0, 0, 0), vertices=np.array(V) * diameter / 2, faces=np.array(F)
    )


sphere = icosphere(0.01)
sphere.bh_curve = curve
mesh = mesh_TriangularMesh(sphere, 300)
cells = mesh.sources_all
volumes = np.array([c.volume for c in cells])

B0 = np.array([0.05, 0.2, 0.5, 1.0, 1.5, 2.0])
solved, exact = [], []
for b in B0:
    mesh.H_ext = (0, 0, b)
    J = np.array([c.polarization for c in apply_demag(mesh, solver="direct").sources_all])
    solved.append((J[:, 2] * volumes).sum() / volumes.sum())
    exact.append(brentq(lambda J, b=b: polarization((b - J / 3) / MU0) - J, 0, 3 * b))

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(B0, exact, label="exact")
ax.plot(B0, solved, "o", label=f"apply_demag, {len(cells)} tetrahedra")
ax.plot(B0, 3 * B0 / (1 + 3 / CHI0), "--", label="χ = 1000")
ax.set(xlabel="applied field B₀ (T)", ylabel="polarization of the sphere (T)", ylim=(0, 3))
ax.legend()
print("largest error:", f"{np.max(np.abs(np.array(solved) / np.array(exact) - 1)):.1%}")
```

## Where a cube saturates

A 10 mm iron cube in 0.5 T along z, 512 cells. Each cell's polarization over Jₛ
says how far it is saturated: the bulk and the edges fully, the middle of the
two pole faces not yet.

```{code-cell} ipython3
cube = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
cube.bh_curve = curve
cells = mesh_Cuboid(cube, (8, 8, 8))
cells.H_ext = (0, 0, 0.5)
solved = apply_demag(cells, solver="direct")
constant = magpy.magnet.Cuboid(dimension=(0.01,) * 3, polarization=(0, 0, 0))
constant.susceptibility = CHI0
cells_constant = mesh_Cuboid(constant, (8, 8, 8))
cells_constant.H_ext = (0, 0, 0.5)
linear = apply_demag(cells_constant, solver="direct")

share = np.array([np.linalg.norm(c.polarization) / JS for c in solved.sources_all])
centres = np.array([c.position for c in solved.sources_all])
layers = np.unique(np.round(centres[:, 1], 9))
middle = np.isclose(centres[:, 1], layers[len(layers) // 2])  # the layer nearest y = 0
x, z, s = centres[middle, 0], centres[middle, 2], share[middle]
fig, ax = plt.subplots(figsize=(4, 3.5))
img = ax.scatter(x * 1e3, z * 1e3, c=s, s=180, marker="s", vmin=0, vmax=1, cmap="magma")
ax.set(xlabel="x (mm)", ylabel="z (mm)", aspect="equal", title="J / Jₛ, the layer nearest y = 0")
fig.colorbar(img)
J_max = max(np.linalg.norm(c.polarization) for c in linear.sources_all)
print(f"polarization: {share.min() * JS:.2f} to {share.max() * JS:.2f} T; with χ = 1000 up to {J_max:.2f} T")
```

## What it costs, and what it leaves out

- A part on a curve solves 10 to 40 times, the matrix built once; the sphere
  above takes about 20 solves at each field.
- `apply_face_charge` and `fem_solve` refuse a part on a curve: face charge
  needs a uniform susceptibility in each part, and finite elements here are
  linear.
- The curve is the material's initial curve: no hysteresis, so a field that
  comes back down returns along the same curve.
