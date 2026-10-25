# The U-core in saturation: face charge on slabs against the MoM

Started 2026-10-25 (Alex: "test the ucore").

**Question.** Face charge needs a uniform χ inside each body; saturation makes χ
depend on the field. Cut each soft part into touching slabs, each with its own
uniform χ taken from the B–H curve at its average field, and iterate: does that
track saturation, and how does it compare with the MoM iterated the same way
(issue #81)?

**Case** (`common.py`). The documented U-core (case F: three cuboids, χ 999, a
16-winding coil), with every winding at 25, 100 and 400 A. The B–H curve of #81:
J = χ0 μ0 H / (1 + χ0 μ0 H / Js), χ0 999, Js 1.6 T, so a cell or slab takes χ(H)
= χ0 / (1 + χ0 μ0 H / Js). A slab's average field is its moment / volume / χ,
exact for a uniform χ; a MoM cell's is the field at its centre. Both iterate on
log χ with Anderson acceleration (depth 5) to 1e-4. Compared: the peak of Bz
along the documented line 30 mm above the poles. The linear answer at 25 A,
2.151 mT, matches case F's FEM reference, 2.155 mT ± 0.2 %.

## Results (done)

`step1_face_charge.py` (3 min, 4.9 GB peak) and `step2_mom.py` (2.5 min, 5.0 GB
peak). Peak Bz in mT, and the time per current after the build:

| Method                              | Unknowns |  25 A | 100 A | 400 A | Per current |
| ----------------------------------- | -------: | ----: | ----: | ----: | ----------: |
| linear, χ 999                       |          | 2.151 | 8.605 | 34.42 |             |
| face charge, one χ per cuboid       |    1,536 | 2.137 | 8.299 | 18.56 |       0.5 s |
| face charge, 20 mm slabs            |    3,168 | 2.135 | 8.158 | 14.36 |   1.4–2.9 s |
| face charge, 10 mm slabs            |    6,720 | 2.135 | 8.148 | 14.26 |      7–21 s |
| face charge, 20 mm slabs, level 3   |    7,128 | 2.136 | 8.163 | 14.36 |      8–24 s |
| face charge, 20 mm slabs, cut 2 × 2 |   13,440 | 2.136 | 8.156 | 14.34 |    40–116 s |
| MoM, 88 cells                       |          | 1.373 | 5.409 | 12.69 |   0.3–0.9 s |
| MoM, 704 cells                      |          | 1.822 | 7.054 | 13.64 |   2.5–9.7 s |
| MoM, 2,376 cells                    |          | 1.971 | 7.576 | 13.91 |     20–74 s |

Builds: 0.5 s (cuboids), 2.1 s (20 mm slabs), 8.7 s (10 mm slabs), 36 s (cut 2 ×
2). Solves per current: 6 at 25 A, 7 to 11 at 100 A, 14 to 26 at 400 A.

- **Saturation matters here.** At 400 A the field above the poles is 0.41 times
  the linear one; the yoke's centre is down to χ 8. Even at 25 A this smooth
  curve lowers χ by 15 % there.
- **One χ per cuboid is not enough**: 30 % high at 400 A (18.6 against 14.3 mT).
  The yoke saturates in its middle and much less toward the legs.
- **Slabs converge in their length, and nothing else matters.** 20 to 10 mm
  slabs move the peak by 0.1 % at 100 A and 0.7 % at 400 A. Level 3 panels and a
  2 × 2 cut of the cross-section each move it by 0.1 % or less.
- **The MoM approaches from below and is still short.** At 2,376 cells it is 2.5
  % under the 10 mm slabs at 400 A and 7 % at 100 A. Its linear answer on the
  same mesh is 8 % under FEM, so the gap is its own discretization, shrinking as
  saturation lowers χ. The saturated-to-linear ratios agree: 0.993, 0.954, 0.438
  (MoM, 2,376 cells) against 0.992, 0.947, 0.414 (10 mm slabs).
- **Cost:** 20 mm slabs solve each current in 1.4 to 2.9 s after a 2 s build;
  the MoM's finest mesh takes 20 to 74 s, as it rebuilds its matrix every solve
  (the matrix does not depend on χ; #81's real version would build it once).

**Caveats.** No saturated reference: the evidence is the two methods converging
toward each other. The slabs were cut by hand, along each cuboid's length; a
part that saturates across its section would need the cut chosen from the field
(like `refine`, on χ instead of the charge jumps).

## What it means

Saturation does not rule face charge out: piecewise-uniform χ keeps every
region's charge on its surfaces, and the touching-body coupling is exact. For
bar-like soft parts, slabs along the length track saturation at a fraction of
the MoM's cost. The next pieces would be the split chosen automatically from the
field, and a first-pass check after a linear solve: the largest average field
per body against the curve's knee, to say whether saturation matters at all.
