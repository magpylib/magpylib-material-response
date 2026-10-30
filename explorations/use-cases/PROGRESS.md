# Use cases against today's API

Each script times a real use case with the package as it is on main, and notes what had to be written by hand. Results are in `results/`; a script skips a result that exists. Run from this folder with the package's environment.

## 1. Angle sensor on a steel shaft (`s1_shaft_sensor.py`)

Face charge level 2, 12 angles x 2 air gaps x 2 stray fields,
with and without a soft holder: 96 solves in 169 s, peak
5.31 GB. Unknowns [2232, 7008].

| holder | gap (mm) | stray (mT) | max angle error (deg) |
| --- | --- | --- | --- |
| no | 1.5 | 0 | 0.005 |
| no | 1.5 | 5 | 1.258 |
| no | 2.5 | 0 | 0.003 |
| no | 2.5 | 5 | 1.802 |
| yes | 1.5 | 0 | 0.006 |
| yes | 1.5 | 5 | 0.060 |
| yes | 2.5 | 0 | 0.003 |
| yes | 2.5 | 5 | 0.042 |

- Without a holder the shaft is the only solved part, so the whole grid has one
  matrix: 7.3 s today, about
  0.7 s with a kept factorization (estimate:
  the first solve, then each configuration's source field and a back-substitution).
- With the holder the rotor turns against it: 158 s today.
- A holder 2 mm thick, as a real one would be, takes
  16,044 unknowns alone at level 2
  (4.4 s); level 3 with the shaft is refused.
- By hand: the stray field is added back to getB (#71).

## 2. Tolerance Monte Carlo (`s2_tolerance_mc.py`)

8 degrees of freedom, 40 samples timed, extrapolated to 10,000.

| | unknowns | per sample | 10,000 samples today | kept factorization |
| --- | --- | --- | --- | --- |
| (a) magnet fixed source | 1,518 | 0.075 s | 13 min | 77 s |
| (c) magnet with recoil 0.05 | 2,166 | 0.139 s | 23 min | matrix changes |

- (b) Sensor tolerances alone need no new solve: one solve (0.07 s)
  and one getB along a 10,000-pose sensor path (0.19 s).
  This works today.
- Bz spread: 30.6 mT with all tolerances, 22.5 mT with the
  sensor's alone. The sensor sits just outside the magnet's edge, where the field
  is steep; the timing is the point here.
- The kept-factorization figure is conservative: it evaluates the magnet's field
  at the panels once per sample, where one call along a magnet path would batch it.
- A magnet with a recoil susceptibility is a solved part, so moving it changes the
  matrix. Each part's own block stays, so a solve that keeps them and recouples
  through the magnet's 648 unknowns is the lever to try.

## 3. Coils on a U-core (`s3_coils_sum.py`)

Face charge level 2, 1,320 unknowns, 20 sets of currents.

- Today, a solve per set: 2.9 s (0.14 s each).
- Kept factorization (estimate): 0.3 s.
- Sums of one solve per coil at 1 A, written by hand: 0.3 s, and
  any further set costs nothing. Largest difference from the direct solves:
  2.0e-14 of the peak.

## 4. NMR pole pieces, comparing designs (`s4_nmr_pole_pieces.py`)

Iron discs 20 mm (A) against 24 mm (B), variation of Bz over +-5 mm on the axis.
Peak 6.72 GB.

| level | unknowns A / B | Bz A (mT) | A (ppm) | B (ppm) | B - A (ppm) | s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5,232 / 4,416 | 50.968 | 80.8 | 90.4 | +9.5 | 4.6 |
| 2 | 11,448 / 10,032 | 50.887 | 90.9 | 88.4 | -2.5 | 15.7 |
| 3 | 20,640 / 18,600 | 50.871 | 89.9 | 84.3 | -5.6 | 35.7 |

- The verdict flips: B is less uniform at level 1, more uniform at levels 2 and 3,
  and the margin is still moving. Bz at the centre settled to 0.03 % by level 2.
- Level 3 of design A needs 20,640 unknowns, over the default `max_unknowns`
  (20,000); it ran with 21,000.
- The two designs get different panel layouts, so their difference carries the
  noise of both. `mesh_study` takes one model and one quantity, so the study of
  a difference is written by hand.
- #85 reports 1,138 ppm with iron, from the method of moments on 3,640 cells; here
  the variation along the axis is about 90 ppm. The issue does not say where its
  +-5 mm lies, so the two may not measure the same thing.

