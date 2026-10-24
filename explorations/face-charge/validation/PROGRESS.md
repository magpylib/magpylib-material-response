# Validation against the documented examples

Each documented example of material-response, solved by the face-charge method
and by the MoM as its page runs it, both against the page's reference. This file
is rewritten by `report.py` after every case; `run.py` appends the log below.

**0 of 6 cases done.** Next: A.

| Case | Model | Status | Reference | Face charge, finest level | MoM, docs settings, finest |
| --- | --- | --- | --- | --- | --- |
| A | Three magnets, χ 0.3 to 1, two rotated (cuboids_demagnetization) | todo | ANSYS (FEMdata_test_cuboids), 301 points | | |
| B | Hard magnet (χ 0.5) and a turned soft cube (χ 3999) (soft_magnets) | todo | ANSYS (FEMdata_test_softmag), Bx and Bz, 3 lines of 1001 points | | |
| C | Flux concentrator, χ 4000 bar in the Earth's field (mesh_convergence) | todo | docs: MoM study on 4,096 to 262,144 cells, within 0.14 % | | |
| D | Halbach array of 5 magnets (χ 0.05) on back iron (χ 1000) (mesh_convergence) | todo | MoM study (2, 4, 8) extrapolated; docs: good to 0.03 % against 245,760 cells | | |
| E | Shielding sheet (24×24×1 mm, χ 1000) behind a magnet (mesh_convergence) | todo | docs: MoM study on 4,608 to 294,912 cells, within 3.3 %, 1.1 %, 0.09 % | | |
| F | U-core electromagnet (3 cuboids, χ 999) driven by a 16-winding coil (ucore_electromagnet) | todo | FEM (scikit-fem P2, explore/cells-and-fem), h = [3.0, 2.0, 1.5, 1.2, 1.0] mm, extrapolated over the three finest (order 1.17) | | |

Errors: for fields, |B − B_ref| at each observer over the largest |B_ref| on the
set; for gains and single sensors, relative per point. Times: face charge is
mesh, assembly and solve; the MoM is `apply_demag` alone, without meshing. The
face-charge mesh level m is set by hand, not converged by a study.

## How to resume

On branch `explore/face-charge`, from this folder, with material-response's
environment and `PYTHONPATH` at the branch's `src`:

```sh
PYTHONPATH=../../../src ./guard.sh 13 python run.py        # every case without a result
PYTHONPATH=../../../src ./guard.sh 13 python run.py C      # one case
PYTHONPATH=../../../src python run.py C --redo             # run it again
```

A case writes `results/<case>.json` (summary) and `results/<case>.npz` (the
compared values of every level and the reference) when it finishes, so stopping
loses at most the case that was running. `guard.sh 13` stops a run above 13 GB.

## Cases

(none finished yet)

## Log

- 2026-10-24 17:07 set up: six cases (A–F), harness, U-core FEM references copied from explore/cells-and-fem
