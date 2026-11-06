# Examples

Worked models, each run when the documentation is built. Most compare against a
reference: finite elements, or an exact solution. The pictures are drawn by
magpylib-studio from each example's model (`docs/gallery_thumbnails.py`).

`````{grid} 1 2 3 3
:gutter: 2

````{grid-item-card}
:link: cuboids_demagnetization
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/cuboids_demagnetization-light.jpg
:alt: Cuboid demagnetization
:class: only-light
```

```{image} ../_static/gallery/cuboids_demagnetization-dark.jpg
:alt: Cuboid demagnetization
:class: only-dark
```

**Cuboid demagnetization**

Three hard cuboid magnets demagnetizing each other, refined against finite elements.

````

````{grid-item-card}
:link: soft_magnets
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/soft_magnets-light.jpg
:alt: Soft magnets
:class: only-light
```

```{image} ../_static/gallery/soft_magnets-dark.jpg
:alt: Soft magnets
:class: only-dark
```

**Soft magnets**

A hard magnet beside a turned soft-iron cube, against finite elements.

````

````{grid-item-card}
:link: ucore_electromagnet
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/ucore_electromagnet-light.jpg
:alt: U-core electromagnet
:class: only-light
```

```{image} ../_static/gallery/ucore_electromagnet-dark.jpg
:alt: U-core electromagnet
:class: only-dark
```

**U-core electromagnet**

A coil built with the polyline helpers drives a soft core; cuboid cells against tetrahedra.

````

````{grid-item-card}
:link: solver_performance
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/solver_performance-light.jpg
:alt: Solvers and performance
:class: only-light
```

```{image} ../_static/gallery/solver_performance-dark.jpg
:alt: Solvers and performance
:class: only-dark
```

**Solvers and performance**

The direct and the iterative solver: their agreement, time and memory, and which paths are fast.

````

````{grid-item-card}
:link: mesh_convergence
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/mesh_convergence-light.jpg
:alt: Mesh convergence
:class: only-light
```

```{image} ../_static/gallery/mesh_convergence-dark.jpg
:alt: Mesh convergence
:class: only-dark
```

**Mesh convergence**

`study_cells` error bars on a flux concentrator, a Halbach array on back iron and a steel shield.

````

````{grid-item-card}
:link: fem_check
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/fem_check-light.jpg
:alt: Checking with finite elements
:class: only-light
```

```{image} ../_static/gallery/fem_check-dark.jpg
:alt: Checking with finite elements
:class: only-dark
```

**Checking with finite elements**

`study_fem` and `compare` against `study_cells`, sensor by sensor: whose error bar is too short.

````

````{grid-item-card}
:link: shield_tube
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/shield_tube-light.jpg
:alt: A magnet in a steel tube
:class: only-light
```

```{image} ../_static/gallery/shield_tube-dark.jpg
:alt: A magnet in a steel tube
:class: only-dark
```

**A magnet in a steel tube**

Inside the tube cells converge; behind the shield it takes face charge or finite elements.

````

````{grid-item-card}
:link: tetrahedral_meshes
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/tetrahedral_meshes-light.jpg
:alt: Any shape as tetrahedra
:class: only-light
```

```{image} ../_static/gallery/tetrahedral_meshes-dark.jpg
:alt: Any shape as tetrahedra
:class: only-dark
```

**Any shape as tetrahedra**

A `TriangularMesh` cut into tetrahedra by TetGen, against the exact sphere.

````

````{grid-item-card}
:link: saturation
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/saturation-light.jpg
:alt: Saturation
:class: only-light
```

```{image} ../_static/gallery/saturation-dark.jpg
:alt: Saturation
:class: only-dark
```

**Saturation**

A B–H curve in place of a constant susceptibility: a sphere against its exact answer, and a cube.

````

`````

```{toctree}
:maxdepth: 1
:hidden:

cuboids_demagnetization
soft_magnets
ucore_electromagnet
solver_performance
mesh_convergence
fem_check
shield_tube
tetrahedral_meshes
saturation
```
