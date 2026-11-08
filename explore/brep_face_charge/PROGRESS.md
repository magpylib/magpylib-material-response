# B-rep parts for face charge: study

**Question.** A STEP file holds a part's exact geometry (B-rep); an STL export
cuts its curved faces into facets of the exporter's choosing. Would a source
that keeps the B-rep, and lets the solver facet curved faces itself, as it does
a `Cylinder` today, beat solving the export? In unknowns and time for the same
accuracy, against a reference on the exact geometry.

**Parts** (`geometry.py`), soft iron χ = 1000 in a 10 mT field along
(0.8, 0.36, 0.48), sensors near the curves and 60 mm away:

- `plate`: the validation's CAD plate, 20 × 12 × 4 mm, 3 mm fillets on its
  vertical edges, a 3 mm hole: curved faces are a small part of its surface.
- `pole`: a Ø20 × 10 mm bar with its top rim filleted 3 mm: curved faces are
  most of it.

**Arms** (`arms.py`): an export at 12, 24, 48, 96 facets per turn, fixed,
solved at levels 1 to 3; the solver's own faceting, `12 (level + 1)` per turn,
widened to keep the volume or not.

**Reference** (`reference.py`): finite elements on the exact OpenCASCADE solid,
curved elements, orders 3 to 5, as `validation/references.py` makes one.

## Steps

| Step | What | Status | Result |
| --- | --- | --- | --- |
| 1 | Geometry: exact solids, facetings, volumes | done | widened facetings keep the exact volume; unwidened lose 0.4 % (plate) to 5 % (pole) at 12/turn |
| 2 | Exact references, plate and pole | done | plate 0.012 % of B near the curves (35 s, 4.6 GB); pole 0.004 % (29 s, 11.4 GB); the solver checked to mesh the exact solid |
| 3 | Arms, plate and pole | done | `results/arms_*.jsonl`, 36 solves under the perf lock, best of two |
| 4 | Figures and conclusion | done | below |

## Result

![Error near the curves against unknowns](results/error_vs_unknowns.png)

1. **An export's facets set a floor that the level does not lower.** Raising
   the level converges to the faceted part, not to the real one: at 30° the
   plate stays near 2.8 % and the pole near 4 %; at 15° about 0.6 % and 1.4 %;
   at 7.5° the pole stays near 0.42 %.
2. **Facets cut by the solver from the exact curve, widened to keep the volume,
   converge with the level:** plate 0.76, 0.16, 0.044 %; pole 0.46, 0.10,
   0.042 % at levels 1, 2, 3.
3. **For the same error, far fewer unknowns.** Pole: 0.10 % with 3,678 unknowns
   in 0.37 s, where the finest export (3.75°) reaches 0.115 % with 19,350 in
   6.9 s: 5 times fewer unknowns, 19 times faster. Plate: 0.044 % with 11,256
   in 2.1 s against the 3.75° export's 0.043 % with 17,952 in 5.7 s: 1.6 and
   2.7 times.
4. **Most of the gain is the widening, which needs the exact curve.** With the
   same facets, widened against left on the curve: pole 0.042 against 0.42 %
   at level 3, plate 0.044 against 0.16 %. An export cannot be widened: it
   does not say which vertices lie on which curve, or where its centre is.
5. **Far away it holds too:** 60 mm off, as a share of the part's own field,
   widened facets at level 3 are 0.013 % (plate) and 0.012 % (pole) off,
   against 0.032 % and 0.11 % for the 3.75° export.
6. **Level 1 is too coarse to judge:** there the plate's unwidened facets come
   out better (0.21 against 0.76 %), by errors that cancel; from level 2 on
   the widened ones win on both parts.

**Conclusion.** Worth doing for parts whose curved faces matter: keeping the
B-rep lets the solver cut curved faces by level and widen them, which a
tessellated export cannot do. On the pole piece that is a 5 times smaller
system, 28 times less memory (it goes with the square of the unknowns), for the same accuracy. On a part that is
mostly flat, the gain is smaller (plate: 1.6 times).

**Limits.** Two parts, one field direction, a uniform drive with no magnet
close to a curve; the widening of fillet arcs is this study's (the solver
widens whole turns today); times on one M4 Pro, under the perf lock.

**If pursued:** a part that keeps its B-rep (read from STEP with the
OpenCASCADE the `fem` extra already ships), with planes, cylinders, cones,
spheres and tori faceted by level and widened, other surfaces tessellated
finely as a fallback; finite elements would mesh the exact solid as the
reference here does. Next checks: a magnet close to a curved face, and a
part with free-form (NURBS) faces.
