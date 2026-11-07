# Examples

Each example takes a design question from sensor or actuator engineering and
answers it with a solved model. Most run when the docs are built; the slow ones
plot results stored next to the page. Most check their answer against finite
elements or an exact solution. The settings behind them are explained in the
[user guide](../guide/index.md).

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

How hard a disc magnet holds on steel, how fast the pull falls with a gap, and
how fine a solve a force needs.

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

How large a back-biased Hall sensor's signal is over steel teeth, how the air
gap sets it, and how fast a sweep of solves runs.

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

How much angle error a steel shaft makes of a 5 mT stray field, how the air
gap changes it, and whether a soft holder ring helps.

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

A slotted ferrite ring from CAD around a busbar: mT per ampere, the slot's
width, a busbar off-centre, and the current at which the ferrite saturates.

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

How far a Hall sensor's reading spreads with assembly tolerances, which one
drives the spread, and what the steel plate changes: 10,000 samples in two
minutes.

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

How much field a steel tube around a magnet lets through, and which solve to
trust there: face charge with `refine=1` is within 1 %, cells are far off.

````

````{grid-item-card}
:link: ucore_electromagnet
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/ucore_electromagnet-light.jpg
:alt: A U-core electromagnet
:class: only-light
```

```{image} ../_static/gallery/ucore_electromagnet-dark.jpg
:alt: A U-core electromagnet
:class: only-dark
```

**A U-core electromagnet**

A coil wound on a soft-iron U-core: the field above its poles, how much of it is
the core, how fine a mesh it needs, and how hard the iron is driven.

````

````{grid-item-card}
:link: saturation
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/saturation-light.jpg
:alt: A U-core electromagnet at high current
:class: only-light
```

```{image} ../_static/gallery/saturation-dark.jpg
:alt: A U-core electromagnet at high current
:class: only-dark
```

**A U-core electromagnet at high current**

The U-core on a B–H curve from 25 to 400 A per winding: whether the solver
follows the curve, where the core saturates, what that costs in field and in
solve time.

````

````{grid-item-card}
:link: mesh_convergence
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/mesh_convergence-light.jpg
:alt: Error bars for cells: three parts
:class: only-light
```

```{image} ../_static/gallery/mesh_convergence-dark.jpg
:alt: Error bars for cells: three parts
:class: only-dark
```

**Error bars for cells: three parts**

How fine a mesh a flux concentrator, a Halbach array and a steel shield each
need, and whether the error bar covers the true error.

````

````{grid-item-card}
:link: fem_check
:link-type: doc
:class-card: magpy-gallery-card

```{image} ../_static/gallery/fem_check-light.jpg
:alt: Checking a result with finite elements
:class: only-light
```

```{image} ../_static/gallery/fem_check-dark.jpg
:alt: Checking a result with finite elements
:class: only-dark
```

**Checking a result with finite elements**

An independent finite-element solve of the same model, to check another
solver's answer and its error bar, sensor by sensor.

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
