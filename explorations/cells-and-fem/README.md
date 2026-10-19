# Cells and FEM: the 2026-10-19/20 study

Scripts, raw results and the adversarial review behind the tracking issue
**magpylib/magpylib-material-response#76** (read that first: it holds the
findings, the numbers and what to do next). This folder is a study, not package
code: ruff is told to skip it, nothing here is tested in CI, and nothing here is
imported by `magpylib_material_response`.

What was studied, on three cases with an independent FEM reference each:

1. which cell type (uniform cuboid, uniform tetrahedron) gets a given accuracy
   cheapest;
2. adaptive refinement with a cell budget;
3. cells whose polarization varies **linearly** inside them (12 unknowns per
   cell), with closed-form kernels for the cuboid and the tetrahedron;
4. what FEM costs for the same cases, and how the references were made;
5. what JAX would buy through the Array API core functions.

## Findings, in one line each (the issue has the numbers and the review)

1. **References first.** The three FEM references are known to 0.1–0.2 % (U-core
   peak 2.152 mT ± 0.2 %, cube moment 0.02888 A m² ± 0.1 %, plate moment 0.5633,
   range 0.562–0.5639); nothing below that level can be ranked. FEM moments are
   rigorous upper bounds.
2. **Cell type decides.** One-surface tetrahedra are within 1.7 % of the U-core
   reference from 89 cells; uniform cuboids are still −6 % at 3,773. Flat
   cuboids are 8× cheaper than the fewest tetrahedra TetGen makes on a 1 mm
   plate, 35× with the FFT.
3. `mesh_TriangularMesh`'s `target_elems` is a lower bound; `minratio` sets the
   count on thin bodies.
4. #52's iterative solver has no FFT path on multi-body cuboid models (one
   `loose` cluster) and is slower than direct above ~1,500 cells there.
5. Adaptive refinement with a cell budget works (1.2–2.5× fewer cuboid cells),
   does not make cuboids accurate, and none of the cheap error estimates is one
   except Richardson on uniform cuboid series.
6. The closed-form kernels for linearly polarized cuboids and tetrahedra are
   validated independently (1e-14 outside, 1e-9..1e-13 inside); they cost 1.1×
   (cuboid) and 1.5× (tet) magpylib's uniform kernels per pair.
7. Linear cuboids cut the error ~5× at equal cell count (U-core −0.8 % at 88
   cells with converged Galerkin); the linear-tetrahedron collocation scheme has
   a free parameter and is **not validated** (its best numbers are error
   cancellation).
8. JAX: ~2× per core; the 8–11× figures are nine cores. On cuboid grids #52's
   assembly is 92 % `np.unique`, not kernel.
9. pymagba's Rust tetrahedron kernel dropped into #52's assembly: exact,
   2.8–3.6× faster end to end on one core (989 tets 2.35 → 0.83 s, 4,016 tets
   43.5 → 12.9 s); its calls hold the GIL, so threads over sources add nothing;
   memory unchanged. magpylib_jax's tetrahedron kernel is slower per core than
   magpylib's.

## Layout

| Path                                          | What                                                                                                                                                             |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mom/adapt.py`                                | U-core geometry (`BLOCKS`, `COIL`, `LINE`, `CHI`), uniform cuboid grids, the adaptive loop (residual and adjoint indicators), `apply` timing of both #52 solvers |
| `mom/adapt_tet.py`                            | the U-core as one closed `TriangularMesh`, TetGen meshes, midpoint 1→8 tetrahedron split, adaptive loop for tetrahedra                                           |
| `mom/boxes_tet.py`                            | each U-core cuboid tetrahedralized on its own (non-conforming joints)                                                                                            |
| `mom/plate.py`                                | thin plate 40×40×1 mm: cuboid grids vs TetGen tetrahedra; `SOLVER=direct\|iterative`                                                                             |
| `mom/split.py`                                | where a dense solve spends its time: kernel assembly vs LU                                                                                                       |
| `mom/lincell.py`                              | **closed-form field of a cuboid with linear polarization** (3×12 per pair), NumPy or JAX                                                                         |
| `mom/lintet.py`                               | **closed-form field of a tetrahedron with linear polarization** (3×12 per pair), NumPy or JAX                                                                    |
| `mom/lincell_check.py`, `mom/lintet_check.py` | their validation against magpylib and against each other                                                                                                         |
| `mom/p1emul.py`                               | the first emulation of linear cells by S³ uniform sub-cuboids (superseded by the real kernels; keeps `P`, `COLLOC`, `grid`)                                      |
| `mom/p1real.py`                               | material response with linear cuboids; `COLLOC=lsq8\|galerkin`                                                                                                   |
| `mom/p1tet.py`                                | material response with linear tetrahedra; `COLLOC=child8\|corner4\|corner4c`, `FRAC=<f>`; `bench` times the kernels                                              |
| `mom/bench_lincell.py`                        | cost per pair of the cuboid kernels, NumPy and `jax.jit`                                                                                                         |
| `mom/kern_dropin.py`                          | pymagba's or magpylib_jax's tetrahedron kernel dropped into #52's assembly (`stock\|pymagba\|jax`, `POOL=<threads>`, `RAYON_NUM_THREADS`)                        |
| `mom/bench_pr854.py`                          | magpylib main vs PR #854 (Array API) cuboid core function, NumPy / JAX eager / `jax.jit`                                                                         |
| `mom/figures.py`                              | the two figures in `results/`, from the numbers in the result files                                                                                              |
| `fem/ngs_cube.py`, `fem/ngs_plate.py`         | NGSolve scalar-potential solves: tier-0 cube vs magpylib, iron cube and plate moments                                                                            |
| `fem/fem.py`, `fem/ucore_setup.py`            | scikit-fem two-potential P2 solve of the U-core and the iron cube (written in a sibling session; `ucore_setup.py` was `setup.py`)                                |
| `results/kern_dropin.txt`                     | the kernel drop-in runs and the raw kernel timings                                                                                                               |
| `results/*.txt`                               | one JSON line per solve, as printed by the scripts                                                                                                               |
| `results/fem-*.json`                          | the scikit-fem runs: mesh sizes, moments, the 161-point Bz line                                                                                                  |
| `results/mesh-*.npz`                          | the final adaptive meshes                                                                                                                                        |
| `results/fig_*.png`                           | the figures used in issues #73 and #74                                                                                                                           |
| `review/`                                     | the adversarial review: `BRIEF.md` (every claim, with its source) and one report per reviewer                                                                    |
| `guard.sh`                                    | `guard.sh <GB> cmd…` kills `cmd` above that RSS and prints the peak; use it for every run                                                                        |

## Provenance

- material-response: **#52 at 4901d1d** (`feat/demag-solver`; the iterative FFT
  solver, `demag_tensor` with the point rule) for every MoM result. The scripts
  also run on main 0.5.0 except where `solver=` is passed.
- magpylib **6.0.0rc0**; `magpy.func.cuboid_field` / `tetrahedron_field` are the
  uniform-cell references.
- Machine: Apple-silicon laptop, 24 GB, one job at a time under `guard.sh`. Two
  FEM jobs from two sessions running together crashed it once; that is why the
  guard exists.
- The U-core FEM reference (`results/fem-ucore-*.json`) and the cube series
  (`results/fem-cube-*.json`) came from a sibling session running `fem/fem.py`
  at h = 4 … 1 mm (up to 743k dofs, 89 s, ~10 GB).

## Environments

Three, because the solvers do not share wheels:

```sh
# 1. material-response + tetgen (every mom/*.py except the JAX benchmarks)
uv venv .venv && uv pip install -e ".[tetgen]" scipy matplotlib
#    to reproduce the #52 numbers exactly:  git worktree add ../mr52 feat/demag-solver  and  PYTHONPATH=../mr52/src

# 2. JAX (bench_lincell.py, p1tet.py bench, bench_pr854.py)
uv venv jax-venv && uv pip install "jax[cpu]" array-api-compat array-api-extra numpy scipy magpylib
#    bench_pr854.py needs an `overlay/` folder: magpylib's src/magpylib with PR #854's
#    array_api_utils.py and field_BH_cuboid.py copied over it, plus the alias
#    `_BHJM_magnet_cuboid = BHJM_magnet_cuboid` at the end of the latter (v6 renamed it).

# 3. NGSolve (fem/ngs_*.py)
uv venv ngs-venv && uv pip install ngsolve magpylib

# scikit-fem (fem/fem.py): uv run --with gmsh --with scikit-fem --with pyamg --with magpylib python fem/fem.py ucore 4
```

## Running

All from `mom/` (the scripts import each other by bare name). Every command
below ran under the guard; the GB figure is the cap that was enough.

```sh
G=../guard.sh; P=python  # the venv's python; add PYTHONPATH=<#52 src> for #52

# cell type (issue #73)
$G 6 $P adapt.py uniform 10 6.67 5 4 3.33 2.86        # uniform cuboids, 88 … 3,773 cells (last one 5 GB)
$G 8 $P adapt_tet.py uniform 50 200 500 1000 2000     # one-surface tetrahedra, 89 … 4,016 (last one 7.3 GB, 44 s)
$G 4 $P boxes_tet.py 50 200 500 1000                  # per-cuboid tetrahedra
$G 6 $P plate.py cub 20 3;  SOLVER=iterative $G 2 $P plate.py cub 80 12;  $G 8 $P plate.py tet 50 relaxed
$G 8 $P adapt.py apply direct 3.33;  $G 8 $P adapt.py apply iterative 3.33   # the two #52 solvers, same mesh

# adaptive refinement (issue #74)
$G 8 $P adapt.py adaptive residual 4000               # writes mesh-residual.npz
$G 8 $P adapt.py adaptive targeted 4000
$G 8 $P adapt_tet.py adaptive 50 1500                 # writes mesh-tet.npz

# linear cells (issue #70)
$G 2 $P lincell_check.py;  $G 2 $P lintet_check.py    # validation, seconds
$G 8 $P p1real.py cube 4 p1;  COLLOC=galerkin $G 8 $P p1real.py ucore 6.67 p1;  $G 8 $P p1real.py ucore 5 p1   # 3.7 GB
COLLOC=corner4 FRAC=0.4472136 $G 8 $P p1tet.py ucore 200 p1;  COLLOC=corner4 FRAC=0.5 $G 8 $P p1tet.py cube 50 p1
$G 8 $P split.py cub 4;  $G 8 $P split.py tet 500     # assembly vs LU

# kernel drop-in (a venv with pymagba, magpylib-jax, jax[cpu], tetgen, loguru, magpylib)
RAYON_NUM_THREADS=1 $G 8 python kern_dropin.py pymagba 1000;  $G 8 python kern_dropin.py stock 1000;  JAX_ENABLE_X64=1 $G 8 python kern_dropin.py jax 500

# kernels and JAX (jax-venv)
$G 8 python bench_lincell.py;  $G 8 python p1tet.py bench;  $G 6 python bench_pr854.py

# FEM references (ngs-venv):  args are  h_mm grading order [mur]
$G 3 python ../fem/ngs_cube.py 1 0.3 4 1        # tier 0, 1.4 GB
$G 3 python ../fem/ngs_cube.py 1 0.3 4 1001     # iron cube moment
$G 5 python ../fem/ngs_plate.py 2 0.3 4         # plate moment
```

Each script prints one JSON line per solve; the `results/*.txt` files are those
lines with the guard's peak-RSS line appended.
