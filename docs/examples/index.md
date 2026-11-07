# Examples

Worked models from engineering, each showing what the package does. They run
when the documentation is built, or, where that takes minutes, plot results a
script next to the page stored. Most compare against a reference: finite
elements, or an exact solution. The pictures are drawn by magpylib-studio from
each example's model (`docs/gallery_thumbnails.py`).

`````{grid} 1 2 3 3
:gutter: 2

````{grid-item-card}
:link: holding_magnet
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/holding_magnet-light.jpg
:alt: A holding magnet on a steel plate
:class: only-light
```

```{image} ../_static/gallery/holding_magnet-dark.jpg
:alt: A holding magnet on a steel plate
:class: only-dark
```

**A holding magnet on a steel plate**

The pull against the gap with `getFT`, and how fine a solve a force needs.

````

````{grid-item-card}
:link: gear_tooth_sensor
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/gear_tooth_sensor-light.jpg
:alt: A gear-tooth sensor
:class: only-light
```

```{image} ../_static/gallery/gear_tooth_sensor-dark.jpg
:alt: A gear-tooth sensor
:class: only-dark
```

**A gear-tooth sensor**

A Hall sensor swept past steel teeth, one solve per step, the factorization kept.

````

````{grid-item-card}
:link: angle_sensor
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/angle_sensor-light.jpg
:alt: An angle sensor on a steel shaft
:class: only-light
```

```{image} ../_static/gallery/angle_sensor-dark.jpg
:alt: An angle sensor on a steel shaft
:class: only-dark
```

**An angle sensor on a steel shaft**

How much angle error the shaft makes of a stray field, and how a holder ring takes it away.

````

````{grid-item-card}
:link: hall_current_sensor
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/hall_current_sensor-light.jpg
:alt: A Hall current sensor
:class: only-light
```

```{image} ../_static/gallery/hall_current_sensor-dark.jpg
:alt: A Hall current sensor
:class: only-dark
```

**A Hall current sensor**

A slotted ferrite ring from CAD around a busbar: sensitivity, slot width, position error, where it saturates.

````

````{grid-item-card}
:link: tolerance_monte_carlo
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/tolerance_monte_carlo-light.jpg
:alt: Assembly tolerances, by Monte Carlo
:class: only-light
```

```{image} ../_static/gallery/tolerance_monte_carlo-dark.jpg
:alt: Assembly tolerances, by Monte Carlo
:class: only-dark
```

**Assembly tolerances, by Monte Carlo**

10,000 samples of a magnet, a steel plate and a sensor, two minutes with the factorization kept.

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

What a shield lets through: face charge with `refine=1` within 1 %, where cells are far off.

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

`````

```{toctree}
:maxdepth: 1
:hidden:

holding_magnet
gear_tooth_sensor
angle_sensor
hall_current_sensor
tolerance_monte_carlo
shield_tube
ucore_electromagnet
saturation
mesh_convergence
fem_check
```
