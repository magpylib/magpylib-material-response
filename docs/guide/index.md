# User guide

How to set up a model, choose a solver, run it, read the result, and know how
far to trust it.

- [Building a model](model.md): susceptibilities, fixed magnets, coils, applied
  fields, and what a solver refuses.
- [Material values](materials.md): typical susceptibilities of steels, ferrites
  and magnets, and how to read them off a datasheet.
- [Choosing a solver](choosing.md): which of the three fits your model, and what
  each costs.
- [Face charge](face_charge.md), [Cells](cells.md) and
  [Finite elements](fem.md): each solver's settings and limits.
- [Reading a solved model](reading.md): fields outside and inside the parts,
  forces and torques, what a result holds, and how to draw it.
- [Saturation](saturation.md): a datasheet's B–H curve, which solver follows it,
  and how to check the result.
- [How accurate is a result?](accuracy.md): each solver's error estimate, and
  how to check one against another.
- [Many solves](loops.md): moving magnets, sweeps and Monte Carlo, fast.
- [Saving and loading](saving.md) and [Logging](logging.md).

```{toctree}
:maxdepth: 1
:hidden:

model
materials
choosing
face_charge
cells
fem
reading
saturation
accuracy
loops
saving
logging
```
