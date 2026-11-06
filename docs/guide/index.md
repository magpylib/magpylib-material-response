# User guide

How to set up a model, choose a solver, run it, and know how far to trust the
answer.

- [Choosing a solver](choosing.md): which of the three fits your model, and what
  each costs.
- [Building a model](model.md): susceptibilities, fixed magnets, coils, applied
  fields, and what a solve returns.
- [Face charge](face_charge.md), [Cells](cells.md) and
  [Finite elements](fem.md): each solver's settings and limits.
- [How accurate is a result?](accuracy.md): each solver's error estimate, and
  how to check one against another.
- [Many solves](loops.md): moving magnets, sweeps and Monte Carlo, fast.
- [Saving and loading](saving.md) and [Logging](logging.md).

```{toctree}
:maxdepth: 1
:hidden:

choosing
model
face_charge
cells
fem
accuracy
loops
saving
logging
```
