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

## Step 2: outline lines graded over the gap only (dropped)

Graded over 1 mm (the gap) instead of 4 mm, lines toward the magnet's outline
still multiply the unknowns by 4.4 at level 3 (3,534 against 798) and 4.5 at
level 9. The mesh is a tensor product: every line runs across the whole face and
down the side faces, and its layers grade toward the line as toward a singular
contact, while a source 1 mm away leaves the plate's charge smooth. Not run;
step 3 refines only where the indicator points.

## Step 3: bisection where the size-weighted jump is large (done)

`step3_bisect.py`, under a minute, 3.5 GB peak. From a uniform level, mark the
panels whose size-weighted jump is above half the largest (or a quarter), bisect
each along both in-plane axes (the whole strips through it, in the tensor mesh),
solve again; stop before 11,000 unknowns.

Under the magnet, the two finest bisections (from levels 3 and 5) agree with
each other to 0.04 %, closer than uniform level 11 comes to either. So the
errors below are against the finest bisection (from level 5, 8,064 unknowns):

| Mesh                        | Unknowns |  Time | 0.25 mm above, under the magnet | Just outside it | Over the rim |    Far |
| --------------------------- | -------: | ----: | ------------------------------: | --------------: | -----------: | -----: |
| uniform 9                   |    6,720 | 8.5 s |                     0.19–0.30 % |          0.07 % |       0.08 % | 0.01 % |
| uniform 11                  |   10,350 |  20 s |                     0.03–0.13 % |          0.38 % |       0.03 % | 0.01 % |
| bisection, from 3, 2 rounds |    2,688 | 1.2 s |                    0.002–0.04 % |          0.25 % |       0.19 % | 0.03 % |
| bisection, from 3, 4 rounds |    7,728 |  10 s |                    0.002–0.02 % |          0.26 % |       0.05 % | 0.01 % |

- **Bisection where the indicator points reaches under the magnet in 2,688
  unknowns what uniform levels do not reach in 10,350**, in 1.2 s against 20 s.
  Each round cuts the largest indicator 3 to 4 times: 5.2, 1.4, 0.47, 0.26, 0.09
  %.
- **Just outside the magnet's footprint**, where B is weak (47 mT, against 550
  mT under the magnet's centre), every mesh family differs by about 0.3 %,
  bisections included. The indicator is relative to the peak charge, so it does
  not single out a region of weak field.
- **The plate's edges stay at the starting level**, which caps the error over
  the rim and far away (0.05 to 0.2 % and 0.01 to 0.03 % after two to four
  rounds from level 3). Starting from level 5 lowers both.
- Marking above a quarter of the largest instead of half adds unknowns for the
  same error.

## Next

Bisection on the indicator as an option of the solve, e.g.
`apply_face_charge(coll, m=3, refine=2)`: a few rounds from a coarse level. It
needs the panel split outside the tensor mesh to stay cheap on large bodies,
where whole strips cost more than here.
