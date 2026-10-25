# Refinement where the charge jumps

Started 2026-10-25 (Alex: "ok for all 3").

**Question.** The solved charge jumps between panels. On a flat face of a
uniform body the true charge is continuous, so the jumps are discretization
error, free to read off a solve. Can they tell a user how accurate fields near
the surface are, and say where to refine?

**Case** (`common.py`). A soft plate, 10 × 10 × 4 mm, χ 1000, with a magnet, 4 ×
4 × 3 mm, 1.2 T, χ 0, 1 mm above it. B is read 0.25 mm above the plate (over its
centre, over (1, 1) mm, under the magnet's edge, outside it, over the rim) and
far away (8 mm above, 3 mm off the side). The plate's mesh is a tensor product,
one list of edges per axis, graded toward the body's edges.

Each step's script writes `results/stepN.json` and skips what is saved.

## Step 1: uniform levels, the reference, and which jump to read (done)

`step1_reference.py`, 45 s, 4.1 GB peak. Errors against uniform level 11:

| Mesh       | Unknowns | Largest jump | Size-weighted | Worst 0.25 mm above |    Far |
| ---------- | -------: | -----------: | ------------: | ------------------: | -----: |
| uniform 3  |      798 |         39 % |         5.2 % |              13.7 % | 0.82 % |
| uniform 5  |    2,304 |         21 % |         1.9 % |               2.1 % | 0.12 % |
| uniform 7  |    4,224 |         16 % |        0.88 % |               2.5 % | 0.02 % |
| uniform 9  |    6,720 |         16 % |        0.35 % |              0.44 % | 0.01 % |
| uniform 11 |   10,350 |         16 % |        0.14 % |                   — |      — |
| outline 3  |    3,240 |         10 % |        0.33 % |               2.2 % | 0.02 % |
| outline 5  |    9,000 |         16 % |        0.12 % |              0.59 % | 0.01 % |

"Outline" grades toward the magnet's outline the way a touching body's footprint
is graded. The worst point 0.25 mm above the plate is mostly the one just
outside the magnet's footprint, where B is weak.

- **The plain largest jump stalls at 16 % from level 7 on.** It sits in the
  first panels at the plate's top corners: the charge is singular there, and the
  graded panels repeat the same pattern at every scale, so the relative jump
  does not shrink. Under the magnet the jump keeps falling: 39, 21, 13 and 6.5 %
  at levels 3, 5, 7 and 9.
- **Weighting each panel's jump by its size** (jump × √area / √total area)
  leaves the edge zones out. The largest then always sits under the magnet's
  corners, falls from 5.2 % at level 3 to 0.14 % at level 11, and the worst
  error 0.25 mm above the plate is 1.1 to 2.8 times it on uniform levels.
- **Far fields** are 0.01 to 0.8 % off already at level 3.
- **Grading toward the outline like a contact** puts layers 4 mm deep on both
  sides of each line: 4 times the unknowns at level 3.
- **The reference is good to a few 0.1 % near the plate, not better.** Level
  11's own indicator is 0.14 %, the outline meshes differ from it by 0.1 to 0.6
  % there, and levels 5 to 9 sit 0.23 % off it at the gap centre. On those
  outline meshes the error-to-indicator ratio reads 5 to 7, which may be the
  reference's own error. Steps 2 and 3 need a better one.

## Step 2 (next): outline lines graded over the gap, and a better reference

Layers toward the magnet's outline only as deep as the gap (1 mm), not 4 mm;
levels 3 to 9. Its finest level, if it agrees with uniform level 11 within the
latter's indicator, becomes the reference.

## Step 3: bisection where the size-weighted jump is large

From level 3: bisect the strips of the panels whose size-weighted jump is above
half the largest, solve again, repeat. Error against unknowns, next to uniform
levels and step 2.
