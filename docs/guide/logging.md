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

# Logging

A solve makes choices you don't see: which solver `"auto"` took and why, how
many unknowns face charge set up, how large its jump indicator is, and how long
each step took. Turn on logging to see them. This page answers how to turn it
on, how to set what it shows and where, and how to turn it off again.

## Turn it on

`configure_logging()` turns logging on. The package logs with
[Loguru](https://loguru.readthedocs.io/), and Loguru has a sink of its own that
would print every record a second time: the first line removes it.

```{code-cell} ipython3
import sys

import magpylib as magpy
from loguru import logger
from magpylib_material_response import configure_logging, disable_logging, solve_cells, solve_face_charge
from magpylib_material_response.meshing import mesh_uniform

logger.remove(0)
configure_logging(sink=sys.stdout, show_time=False, enable_colors=False)

MM = 1e-3  # lengths are in metres; this writes them in millimetres
magnet = magpy.magnet.Cuboid(
    dimension=(10 * MM, 10 * MM, 5 * MM),
    position=(0, 0, 5.5 * MM),
    polarization=(0, 0, 1.2),
)
plate = magpy.magnet.Cuboid(dimension=(40 * MM, 40 * MM, 6 * MM), polarization=(0, 0, 0))
plate.susceptibility = 1000
model = magpy.Collection(magnet, plate)

cells = solve_cells(mesh_uniform(model, 4))
sheets = solve_face_charge(model)
```

Each solve first names the magnets it keeps fixed. `solve_cells` then counts the
clusters of cells and says which solver `"auto"` took: here the iterative one,
as the plate's cells form one grid. `solve_face_charge` gives its unknowns, its
largest jump indicator ([How accurate is a result?](accuracy.md)) and the
largest mean polarization of a solved part: compare it with the knee of the
part's B–H curve to see whether saturation matters.
[Inside cells](../inside/cells.md#solving-direct-or-iterative) shows the
`"auto"` choice on a model of mixed cells.

## Silent by default

Importing the package logs nothing and leaves every Loguru sink as it was,
following Loguru's
[recipe for libraries](https://loguru.readthedocs.io/en/stable/resources/recipes.html#configuring-loguru-to-be-used-by-a-library-or-an-application).
`configure_logging()` adds one sink that receives only the package's records.
Calling it again replaces that sink and no other.

## What it shows, and where

`configure_logging` takes these arguments; each has an environment variable that
sets its default.

| Argument        | Default      | Environment variable    | Sets                                         |
| --------------- | ------------ | ----------------------- | -------------------------------------------- |
| `level`         | `"INFO"`     | `MAGPYLIB_LOG_LEVEL`    | the least severe level shown                 |
| `sink`          | `sys.stderr` |                         | a stream, a file path or a function          |
| `enable_colors` | `True`       | `MAGPYLIB_LOG_COLORS`   | coloured output; turn off for a file         |
| `show_time`     | `True`       | `MAGPYLIB_LOG_TIME`     | a time stamp on each line                    |
| `min_log_time`  | `1.0`        | `MAGPYLIB_LOG_MIN_TIME` | the shortest step, in seconds, that is timed |

```python
configure_logging(level="DEBUG")  # internal steps too
configure_logging(sink="solve.log", enable_colors=False)  # to a file
configure_logging(min_log_time=0)  # time every step
```

A step that takes longer than `min_log_time` logs "Completed: …" with its
duration. `solve_cells`, `solve_face_charge`, `solve_fem`, `study_fem` and
`cells.demag_tensor` take a `min_log_time` of their own; left at `None`, it is
the one set here.

## Turn it off

```{code-cell} ipython3
disable_logging()
```

This removes the sink `configure_logging` added and silences the package again.

## With your own Loguru setup

An application that configures Loguru itself needs no `configure_logging`:
enabling the package sends its records to the application's sinks.

```python
from loguru import logger

logger.enable("magpylib_material_response")
```

## In short

- `configure_logging()` shows what a solve chose and how long it took;
  `disable_logging()` silences it again.
- Remove Loguru's own sink first, `logger.remove(0)`, or each record prints
  twice.
- `level`, `sink`, `min_log_time` and the environment variables set what is
  shown and where.
