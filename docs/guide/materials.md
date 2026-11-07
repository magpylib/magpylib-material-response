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

# Material values

A model needs numbers for its materials: a susceptibility χ for each soft part,
a remanence and a χ for each magnet. Datasheets give a relative permeability μr,
a remanence Br and a recoil permeability, often as ranges. Which number goes
where? What values do common materials have? How much does the exact value
matter? This page answers them in that order.

## From a datasheet to the model

| Datasheet                | Model                                                                     |
| ------------------------ | ------------------------------------------------------------------------- |
| relative permeability μr | `susceptibility = μr − 1`                                                 |
| remanence Br, in T       | `polarization` of Br along the magnet's direction of magnetization        |
| recoil permeability μrec | `susceptibility = μrec − 1` on the magnet                                 |
| a B–H curve              | `susceptibility = BHCurve(H=..., B=...)`, see [Saturation](saturation.md) |

Polarization J and magnetization M say the same: J = μ0 M, so 1 T is about
796 kA/m. Magpylib takes a magnet's remanence as `polarization` in T or as
`magnetization` in A/m; these pages use `polarization`. A magnet with a
susceptibility answers its own field and every other: its polarization is then
$\mathbf{J} = \mathbf{J}_r + \chi\mu_0\mathbf{H}$, with $\mathbf{J}_r$ the
remanence you gave.

A disc magnet from a datasheet with Br = 1.3 T and μrec = 1.05:

```{code-cell} ipython3
import magpylib as magpy

from magpylib_material_response import UniformField, solve_face_charge

MM = 1e-3  # lengths are in metres; this writes them in millimetres

magnet = magpy.magnet.Cylinder(dimension=(10 * MM, 5 * MM), polarization=(0, 0, 1.3))
magnet.susceptibility = 1.05 - 1
print(f"magnetization: {magnet.magnetization[2] / 1e3:.0f} kA/m")

solved = solve_face_charge(magpy.Collection(magnet))
above = (0, 0, 5 * MM)  # 2.5 mm above the disc
print(f"mean polarization: {solved.children[0].mean_polarization[2]:.3f} T")
print(f"Bz above it: {magpy.getB(magnet, above)[2] * 1e3:.1f} mT at Br, "
      f"{magpy.getB(solved, above)[2] * 1e3:.1f} mT solved")
```

The disc's own field opposes its polarization: it holds 1.27 T, not 1.3 T, and
the field above it is 3 % lower. Magpylib alone, without the susceptibility,
gives the field at Br.

## Soft materials

Typical relative permeabilities and saturation polarizations, rounded. They vary
with the grade, the heat treatment and the field, so take your supplier's values
where they matter. A datasheet's initial and maximum μr bracket the linear
range. Above μr of about 100, χ = μr − 1 is μr in all but name.

| Material                                | μr, typical                                     | Saturates near |
| --------------------------------------- | ----------------------------------------------- | -------------- |
| structural and low-carbon steel         | up to about 2,000                               | 2 to 2.1 T     |
| soft iron, annealed                     | a few thousand to over 10,000                   | 2.1 T          |
| electrical steel (silicon iron)         | a few thousand to 10,000 and more               | 2 T            |
| MnZn ferrite                            | 1,000 to 15,000                                 | 0.3 to 0.55 T  |
| NiZn ferrite                            | 10 to 2,000                                     | 0.2 to 0.4 T   |
| permalloy, mu-metal (about 80 % nickel) | tens of thousands to several hundred thousand ¹ | 0.7 to 0.8 T   |

¹ Annealed; bending and machining after the anneal lower it many times over.

Sources: VAC's soft magnetic materials catalogue (PHT-001) for iron, silicon
iron and nickel iron; Cleveland-Cliffs DI-MAX sheets for electrical steel;
Ferroxcube and Fair-Rite material sheets for ferrites.

A linear χ holds while the part stays below the knee of its curve. Ferrites and
nickel alloys reach it at a fraction of a steel's flux density: check them as
[Saturation](saturation.md) shows. A laminated stack carries flux along its
sheets far better than across them; a susceptibility of three numbers, along the
part's own axes ([Building a model](model.md)), describes that.

## Permanent magnets

Typical values at room temperature, rounded. The remanence falls as a magnet
warms: take Br at the working temperature.

| Material       | Br             | μrec                       | χ = μrec − 1 |
| -------------- | -------------- | -------------------------- | ------------ |
| sintered NdFeB | 1.1 to 1.5 T   | 1.03 to 1.10               | 0.03 to 0.10 |
| SmCo           | 0.85 to 1.2 T  | 1.02 to 1.10               | 0.02 to 0.10 |
| hard ferrite   | 0.2 to 0.45 T  | 1.05 to 1.2                | 0.05 to 0.2  |
| AlNiCo         | 0.55 to 1.35 T | 1.3 to about 6, by grade ² | 0.3 to 5     |

² Arnold's AlNiCo grades: 1.3 for AlNiCo 9, about 2 for 5-7 and 8, about 4 for
5, 6 for 2. The other rows follow MMPA 0100-00, VAC's VACODYM and VACOMAX sheets
and Arnold's catalogues.

The demagnetization curve is a straight line of slope μrec only above its knee.
A magnet driven past the knee, as an AlNiCo is easily, keeps a lower remanence:
it returns along a parallel line below the first until it is magnetized again. A
susceptibility does not follow that.

## How much does the exact value matter?

For a part with an air path around it, little, once χ is large: its shape sets
its response. An iron cube in a uniform field:

```{code-cell} ipython3
for chi in (100, 1000, 10000):
    cube = magpy.magnet.Cuboid(dimension=(10 * MM,) * 3, polarization=(0, 0, 0))
    cube.susceptibility = chi
    solved = solve_face_charge(magpy.Collection(cube, UniformField(B=(0, 0, 0.01))))
    ratio = solved.children[0].mean_polarization[2] / 0.01
    print(f"χ = {chi:>5}: polarization {ratio:.2f} times the applied field")
```

From χ = 1000 to 10,000 the cube changes by under 1 %. A magnetic circuit closed
through a small air gap is different: there the iron's path competes with the
gap, and χ counts. The remanence counts in full: a magnet's field scales with
its Br.

## In short

- χ = μr − 1; a magnet's `polarization` is its Br, its χ its μrec − 1.
- Soft iron and steels: χ in the thousands to tens of thousands, saturating near
  2 T; ferrites and nickel alloys saturate far lower.
- Magnets: χ from 0.02 to 0.2, AlNiCo 0.3 to 5.
- Where the part has air around it, a large χ need not be exact; a magnet's Br
  must be.
