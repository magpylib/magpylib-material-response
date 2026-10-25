# NGSolve probe: a soft sphere with an exact answer

A first look (2026-10-25) at NGSolve as the FEM yardstick for the plan in issue
#82: material-response must beat FEM by an order of magnitude or more at equal
accuracy, the getB calls included.

`ngs_sphere.py`: a sphere of radius a, χ 1000, in a uniform field; total scalar
potential, air out to R = 20a with φ = −H0 z there, H1 elements of order p,
BDDC-preconditioned CG. The exact field outside is the sphere's dipole. Errors
of H at five points 1.5a to 3a from the centre:

| Sphere mesh | Order | Unknowns |  Mesh |  Solve | Worst error |
| ----------- | ----: | -------: | ----: | -----: | ----------: |
| maxh 0.5a   |     2 |    8,052 | 0.2 s | 0.07 s |      6.9e-2 |
| maxh 0.5a   |     3 |   26,159 | 0.2 s | 0.22 s |      1.2e-2 |
| maxh 0.5a   |     4 |   60,875 | 0.2 s | 0.55 s |      1.9e-3 |

- ngsolve 6.2.2608 installs from pip on Python 3.13 (macOS arm64) in seconds,
  273 MB, with netgen/OCC meshing of the parts and the air.
- Refining the air near the sensors (a shell to 4a at maxh 0.5a) left order 2 at
  6.1e-2; at maxh 0.25a, order 2 went past 6 GB and was stopped.
- A naive FEM is slow and inaccurate here. An honest yardstick needs tuning:
  order, the mesh near sensors, the open boundary (a Kelvin transform rather
  than a truncated air sphere) and the solver.

Run: `uv venv --python 3.13 && uv pip install ngsolve numpy`, then
`python ngs_sphere.py` (the script as left has the refined air shell).
