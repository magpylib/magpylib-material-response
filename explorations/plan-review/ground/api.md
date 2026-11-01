# Today's API and what each solver returns, read from the code

Date: 31 October 2026. Source: `magpylib-material-response` on `main` at 57c7860, read in full
(`demag.py`, `face_charge/_apply.py`, `face_charge/_sheet.py`, `fem/_solve.py`, `fem/_study.py`,
`fem/_geometry.py`, `convergence.py`, `meshing.py`, `meshing_utils.py`, `utils.py`,
`validation/methods.py`, the five docs pages). Line numbers below are in those files. Nothing was
taken from the plan page on trust.

Probes (correctness and API behaviour only, no timing; the package's own `seconds` are masked or
not quoted), scripts and printed outputs in
`/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/d931a5f5-3e14-496c-bad9-7cbff61b8689/scratchpad/plan-review/ground/probes/`:

- `probe_a_face_charge.py` → `probe_a_output.txt` (face charge)
- `probe_b_mom.py` → `probe_b_output.txt` (cells, `mesh_uniform` + `apply_demag`)
- `probe_c_fem_studies_chaining.py` → `probe_c_output.txt` (FEM on a tiny setting, the studies, one solver's output fed to another)

The model of every probe is the brief's: a plate 10 × 10 × 2 mm with χ = 1000, a 5 mm cube magnet
of 1.0 T with a 1 mm gap above the plate, a sensor in the middle of the gap, `magpylib.defaults.threads = 1`,
magpylib 6.1.0, scipy 1.18.0, numpy 2.5.1 (the venv's version string says `0.7.1.dev2` from a
stale `_version.py`; the imported source is the root checkout on `main`).

## The conclusion in short

1. **Three functions, three return types.** `apply_demag` gives back a copy of the collection with
   every cell's `polarization` overwritten and attaches no information. `apply_face_charge` gives
   back a copy in which each solved part is replaced by a `ChargeSheet`, plus a
   `face_charge_info` dict on the collection. `fem_solve` gives back a `FemSolution`, which is not
   a magpylib object. (Section 1.)
2. **What they share is B and H at observers.** The validation harness already reduces all three
   to `Solved(B, seconds, counts)`: B at the case's sensors, three phase times, a count of unknowns
   (`validation/methods.py:36-43`). (Section 7.)
3. **B inside a part:** cells yes, FEM yes, face charge no. A sheet gives μ0 H without the
   part's magnetization and does not warn (`_sheet.py:59-61, 314-343`). On the plate, 0.3 mm below
   its top face: cells 269.4 mT, face charge −1.72 mT. μ0 (1 + χ) H by hand at the plate's
   centre is 206.6 mT where FEM reads 252.8 mT, 18 % off (probes A, B, C). (Section 2.)
4. **A moment and a saturation check exist only in pieces.** FEM: `.moment` of the whole model,
   magnet included (0.1098 A m² here, of which the magnet alone is 0.0995). A sheet:
   `mean_polarization` in T, no volume and no moment. Cells: a sum by hand. Nothing gives the
   soft part's moment directly. (Section 2.)
5. **Serialization:** the two collection results round-trip through `to_json`/`from_json` with
   their extras (`face_charge_info`, the sheets' colour level, a mesh's `sealed` and χ). A
   `FemSolution` cannot be saved. (Section 2.)
6. **Paths:** all three refuse a path on any source. A sensor path, inside the model or passed to
   `getB`, works for all three; FEM reads sensors through a temporary `CustomSource`. (Section 2.)
7. **Nothing records how a model was meshed, and no solver checks.** `apply_demag` accepts an
   unmeshed model and solves each part as one cell (390 mT at the sensor against 552–557 mT
   converged). `apply_face_charge` accepts a cell mesh and solves 25 cells as 25 bodies, 1,800
   unknowns where the part alone takes 270. (Sections 5, 8.)
8. **A result is not an input.** Solving a face-charge result again keeps the old charges: after
   moving the magnet, 488 mT where a fresh solve gives 424 mT, with a warning that says only that
   nothing responds. Solving a cell result again takes the solved polarization as remanence and
   returns a silently different field. The two solvers' outputs cannot be fed to each other. So
   every loop written today rebuilds the unsolved model each step. (Sections 6, 8.)
9. **`H_ext` is one triple per object**, checked to be exactly three finite numbers in all three
   solvers; the field of every result excludes it, so users add it back by hand. A per-step value
   would be refused today. (Sections 2, 8.)

## 1. Inputs, outputs, side effects and information, per solver

| | `apply_demag` (cells) | `apply_face_charge` (sheets) | `fem_solve` (finite elements) |
|---|---|---|---|
| Signature | `(collection, susceptibility=None, inplace=False, *, min_log_time=None, style=None, solver="auto", solver_tol=1e-6, max_iter=1000, interaction="point")` — `demag.py:1539-1550` | `(collection, *, level=None, refine=0, max_unknowns=20_000, min_log_time=None)` — `_apply.py:52-54` | `(collection, order=3, *, maxh=None, edge_layers=4, observers=None, observer_maxh=None, solver_tol=1e-10, max_iter=5000, min_log_time=None)` — `fem/_solve.py:885-896` |
| What it takes | A Collection whose magnets are the cells, any shape, meshed or not (no check, `demag.py:1671`); currents drive the cells at their centroids (`1728-1735`); sensors pass through; any other source is a `TypeError` (`1674-1688`) | A Collection (`TypeError` otherwise, `130-132`). Solved: magnets with a nonzero isotropic χ, on themselves or a parent, of a shape with a volume (`328-361`). Fixed: everything else, currents, sheets, magnets without χ (`338-345`) | Magnets with a volume, currents on closed paths, sensors pass through; anything else `TypeError` (`fem/_geometry.py:70-104`) |
| The mesh | Yours: `mesh_uniform`, `mesh_all`, `mesh_Cuboid`, `voxelize`, … (section 5) | Made inside every call: panels graded toward edges and contacts, one `level` for every part (`163-173`); `refine` splits where the solved charge jumps and solves again (`255-275`) | Made inside every call by NGSolve, refined toward edges and toward `observers` (`Model.__init__`, `422-467`) |
| What it returns | A new Collection, a copy (`1657-1659`), the cells' `polarization` overwritten in their own frame (`1756-1757`); `None` with `inplace=True`, the input changed (`1759-1761`; probe B) | A new Collection (`144`); each solved part replaced, at its place among its parent's children, by a `ChargeSheet` carrying the part's label, colour and opacity (`289-312`); the rest copied as it is | A `FemSolution` (`882`), not a magpylib source |
| Information attached | None: no attribute on the result (probe B). Only log lines: the cell structure, which solver `"auto"` took and why (`464-469`, `1465-1470`) | `collection.face_charge_info = {"unknowns", "level", "rounds", "seconds"}` (`145`, `250`, `319`); saved by `to_json` (`utils.py:163-166`). A log line gives the largest mean polarization and the jump indicator (`260-262`, `277-280`) | Attributes `order`, `ndof`, `elements`, `seconds` (`{"mesh", "assemble", "solve"}`), `moment` (`fem/_solve.py:324-339`) |
| Side effects | None on the input unless `inplace=True`; `style=` is set on the copy (`1660-1661`). `demag_tensor` leaves a unit polarization on cells that had none (`1421-1427`) | Every collection above a sheet gets `sealed = False` (`313-318`); the sheets of one solve share one colour level, `style.charge.level` (`281-287`) | None on the model; NGSolve's message level toggled during the work (`140-148`) |
| Settings that shape the mesh | `cells_across` or `target_elems`, on the mesher | `level` (and `max_unknowns`, which can lower it) | `maxh`, `edge_layers`, `observers`, `observer_maxh` |
| Settings that need a solve | `solver`, `interaction`, `solver_tol`, `max_iter`, `susceptibility` (overrides the objects') | `refine` | `order`, `solver_tol`, `max_iter` |
| What is factored and kept | `"direct"`: the dense matrix in place through `scipy.linalg.solve`, nothing kept (`504-509`). `"auto"` on this plate takes GMRES, FFT covers 99 % (probe B log; rule `487-501`, `143`) | The dense system, LU in single precision from `MIXED` unknowns, refined to double; nothing kept (`_bodies.py:613-637`) | Conjugate gradients with a BDDC preconditioner; nothing kept (`813-875`) |

## 2. What each result can and cannot give

| Need | Cells (`apply_demag`) | Face charge (`ChargeSheet`s) | FEM (`FemSolution`) |
|---|---|---|---|
| B outside, at points | `magpy.getB(result, points)`: 542.45 mT at the sensor, 200 cells (probe B) | Same: 557.33 mT at level 2 (probe A) | `solution.getB(points)`: 557.10 mT at order 2 (probe C). `magpy.getB(solution, …)` is refused: not a source (probe C) |
| B on sensors, pixels, sensor paths | Yes: a 5-step path gives (5, 3) (probe B) | Yes: (5, 3); a sensor with 10 poses inside the model gives (10, 3) (probes A, C) | Yes, through a temporary `CustomSource` (`fem/_solve.py:362-374`); (5, 3) in probe C |
| `result.getB(sensor)` | magpylib refuses it when the result holds its own sensor ("Collections with sensors and sources do not allow collection.getB() inputs"); `magpy.getB(result, sensor)` or `result.getB()` work (probe A) | Same | Not applicable |
| H | `magpy.getH` | `magpy.getH`; the sheet's H is right everywhere, inside the part too (`_sheet.py:59-61`) | `.getH` |
| B inside a part | Yes, B = μ0 H + J of the cell: 269.4 mT 0.3 mm below the top face (probe B) | No: B = μ0 H without M (`_apply.py:112-115`); −1.72 mT at the same point (probe A); no warning when read inside (`_sheet.py:314-343`). By hand μ0 (1 + χ) H at the centre: 1001 × 0.2064 mT = 206.6 mT where FEM reads 252.8 mT (probes A, C) | Yes: 252.8 mT at the plate's centre (probe C; `flux_density`, `623-639`) |
| J and M | Per cell: `cell.polarization`; `magpy.getJ` | `getJ`, `getM` give 0 (`_sheet.py:317-318`; probe A) | Not exposed; no `getJ` (probe C) |
| A moment | By hand: Σ Vᵢ Rᵢ Jᵢ / μ0 over the cells (`how_it_works.md:304-307`); 0.0104 A m² for the plate (probe B) | `sheet.mean_polarization`, T, in the sheet's own frame (`205-228`); no `.volume`, no `.moment` (probe A); the volume can be rebuilt from `sheet.triangles` (2.0000e-7 m³, matches), giving 0.01035 A m² by hand | `.moment`, A m², of all magnetization in the model (`332-339`): 0.1098 A m² here, of which the magnet is 0.0995; the plate's share only by subtraction (probe C) |
| A saturation check | Max |J| over the cells by hand: 0.399 T here (probe B) | `mean_polarization` (0.065 T here) and the log line "average polarization up to … above the material's knee, saturation matters" (`277-280`); `jump_indicator` measures discretization, not saturation (`230-245`) | Nothing built in; B at points inside |
| `magpy.show` | Yes: 2 plotly traces for 201 cells and the sensor (probe B) | Yes: faces coloured by charge, panel edges on request (`get_trace`, `346-420`); plotly 3 traces, matplotlib a Figure (probe A) | No traces. `magpy.show(solution, backend="plotly", return_fig=True)` did not raise (probe C); what it drew was not checked |
| Studio | A mesh is `sealed = True`, one object in studio (`meshing.py:53-54`; `quickstart.md:122-123`; studio `importer.py:117-223`) | Studio 0.10.0 handles `ChargeSheet` (studio `recording.py:809, 846`, `fieldlines.py:432`, `tests/test_charge_sheet.py`); the result is un-sealed so studio lists each sheet (`_apply.py:110-111`). Studio does not re-solve on a drag (studio #96, open) | No |
| `to_json` / `from_json` | Yes, with `sealed`, the cells' χ and any `H_ext` (`utils.py:157-162, 214-215`); 211,922 characters here, B equal after the round trip (probe B) | Yes: vertices, faces, charges, panels, `mean_polarization`, style with the colour level, `face_charge_info` (`utils.py:163-166, 197-204, 288-298`); 54,866 characters, B equal (probe A). `to_json(sheet)` alone warns that its parent is left out (`149-152`) | No: `TypeError`, unsupported type (probe C) |
| A path on a source | Refused: `ValueError` (`demag.py:1663-1670`; probe B) | Refused: `NotImplementedError` (`_apply.py:330-332`; probe A) | Refused: `ValueError` (`fem/_geometry.py:63-69`; probe C) |
| The result as the input of a new solve | Silently different: the solved polarization is read as remanence (`demag.py:1703-1711`); 542.88 mT against 542.45 mT, no warning (probe B) | Stale: the sheets are fixed sources, nothing responds (`_apply.py:338-345`); after moving the magnet 2 mm, Bz 488.5 mT and Bx 215.6 mT where a fresh solve gives 424.4 and 109.8 mT, with the warning "No magnet has a susceptibility … nothing responds" (probe A). `apply_demag` on it: `TypeError`; `mesh_uniform` on it: `ValueError` (probe C) | Not applicable |
| `H_ext` per part | One value per cell or inherited (`demag.py:311-342`) | One value per solved part (`161`, `164-166`); parts with different values are not solved as one (`629-638`) | One value for all responding parts, else `ValueError` (`fem/_geometry.py:123-131`; probe C) |
| `H_ext` in the result's field | Left out (`_apply.py:111-112`, issue #71): 5 mT along z moves B at the sensor by +0.78 mT, the plate's answer only (probe B) | Left out: +0.77 mT (probe A) | Left out (`fem/_solve.py:346-349`) |

## 3. The checks each solver runs on the model

`apply_demag` (`demag.py`):

- `solver` in auto/direct/iterative and `interaction` in point/newell, else `ValueError` (`1649-1654`).
- Any source with a path: `ValueError` (`1663-1670`).
- A source that is not a magnet, a current or a sensor: `TypeError` (`1674-1688`). No magnet: `ValueError` (`1689-1692`).
- χ from the argument, else from each object or its parents; none anywhere: a warning that nothing responds, and the copy comes back unsolved (`383-394`, `1740-1742`). Shape checks on the argument (`179-227`).
- `H_ext`: three finite numbers per owner, else `ValueError` (`363-380`); above 10 T a warning (`95`, `330-340`).
- `"newell"` on cells that are not parallel cuboids: `ValueError` (`425-450`).
- GMRES that does not converge: under `"auto"` a fallback to the direct solver when its matrix fits 2 GB, else `RuntimeError` (`1509-1535`); under `"iterative"` `RuntimeError` (`1362-1371`).

`apply_face_charge` (`face_charge/_apply.py`):

- A Collection; `level` an integer ≥ 1, `refine` an integer ≥ 0 (`130-141`).
- A path on the collection or any source: `NotImplementedError` (`330-332`).
- No magnet with χ: a warning (`333-335`), an unsolved copy with `unknowns` 0 (`146-149`).
- A χ part without a volume: `NotImplementedError` (`346-352`); an anisotropic χ: `NotImplementedError` (`353-359`).
- A triangle mesh not closed or not consistently wound: `ValueError` (`377-392`).
- More unknowns than `max_unknowns` at the chosen level: `ValueError` before assembly (`229-238`); with no `level` given, lower levels are tried, with a warning (`176-188`); a refinement round over the cap is skipped with a warning and the last round kept (`221-228`).
- A magnet touching a solved part that is not a cuboid aligned with it: a warning (`791-831`).
- `H_ext` as in `apply_demag` (`161`).

`fem_solve` (`fem/_geometry.py`, `fem/_solve.py`):

- Paths: `ValueError` (`63-69`). `TriangleSheet` currents: `TypeError` (`70-76`). Open current paths: `ValueError` (`77-86`). Other sources: `TypeError` (`87-96`). No magnet: `ValueError` (`97-100`). Flat magnets: `TypeError` (`101-104`).
- Different `H_ext` on responding magnets: `ValueError` (`123-131`).
- `maxh` positive and finite (`469-479`); `edge_layers` an integer ≥ 0 (`444-446`); `observer_maxh` needs `observers` and must be positive (`447-453`); `order` an integer ≥ 1 (`814-817`).
- Overlapping magnets: `ValueError` (`564-580`). A current through a magnet: `ValueError` (`672-681`). A current through a hole in a magnet: `ValueError` (`683-741`).
- A mesh that does not conform at any grading: `RuntimeError` (`523-538`). Conjugate gradients not converged: `RuntimeError` (`867-875`). Observers outside the mesh: `RuntimeError`; on the sphere: a warning (`589-613`).

## 4. The studies

- `mesh_study(collection, quantity, levels, mesher=None, demag_kwargs=None) → MeshStudy`
  (`convergence.py:230-347`). For each level: `mesher(collection, level)`, by default `mesh_uniform`
  with the level as `cells_across` (`319-320`, `331`); then `apply_demag` (`342`); then `getB` at the
  quantity, or `quantity(solved)` (`321-327`). The estimate is the grid convergence index over the
  three finest meshes (`194-227`). The result holds `cells, values, seconds, order, estimate, error,
  errors, last_change` and no solved collection (`53-98`; probe C: order 2.0, error 0.74 %, estimate
  555.49 mT from 25, 200 and 1,600 cells). `inplace` and `susceptibility` are refused (`309-316`).
- `fem_study(collection, quantity, orders=(2, 3, 4), …) → FemStudy` (`fem/_study.py:69-186`): one mesh
  (`155-162`), one solve per order (`164-168`), the same estimate logic (`189-206`); it keeps the
  highest order's `solution` (`49-53`).
- `compare(a, b) → Comparison` (`406-457`): last values, error bars per entry, whether they agree;
  takes any objects with `values` and `errors` (`411-415`). The docs chain
  `mesh_study` + `fem_study` + `compare` (`docs/examples/fem_check.md:62-64`).
- **No study drives `apply_face_charge`:** `mesh_study`'s mesher must return cells for `apply_demag`
  (`342`). `MeshStudy.from_values(cells, values, seconds=None)` takes any series (`100-126`), so a
  face-charge ladder can be fed to it by hand, but its order and estimate assume a volume mesh whose
  cell size is (volume / cells)^(1/3) (`30-36`, `104-110`); how good that estimate is for panels is
  unverified here.

## 5. The mesh step today

- A mesh is a Collection of cells with `sealed = True` (`meshing.py:53-54`). magpylib itself has no
  `sealed` (no match in `magpylib/src` at 30f6a014); studio reads it and shows the collection as one
  object (studio `importer.py:117-223`; `docs/quickstart.md:122-123`). Each cell gets the part's χ
  copied (`26-29`), or the mesh collection gets the one inherited from a parent (`31-34`); an inherited
  `H_ext` goes on the mesh collection (`50-52`). The cells stay ordinary magnets.
- `mesh_uniform(obj, cells_across, inplace=False, max_cells=1_000_000)` (`883-984`): cells of one size
  across the thinnest part that responds; parts without χ and currents stay (probe B: the magnet and
  the sensor untouched). It refuses when nothing responds (`943-945`; probe B), a shape it cannot mesh
  (`946-954`), or more than `max_cells` (`957-965`). It works on a detached copy that is given the
  inherited χ and `H_ext` (`meshing_utils.py:386-396`), and puts the meshed part back **at the end** of
  its parent's children (`399-407`; probe B: plate, magnet, sensor became magnet, sensor, plate),
  where `apply_face_charge` keeps a sheet at its part's place (probe A).
- `mesh_all(obj, target_elems, …)` (`772-857`): per-shape meshers with a volume-weighted target;
  refuses children that are neither magnets nor currents (`831-841`). Also `mesh_Cuboid`,
  `mesh_Cylinder`, `mesh_TriangularMesh` (TetGen), `voxelize`, `mesh_thin_CylinderSegment_with_cuboids`,
  `slice_Cuboid`.
- Nothing records which mesher made a mesh or at what setting; `apply_demag` never reads `sealed`
  (no reference in `demag.py`). The face-charge and FEM meshes exist only inside their call; the
  face-charge one can be seen afterwards on the sheets (`panels`, `style.mesh.grid`), the FEM one not
  at all.
- Names: neither magpylib nor the package has a public `mesh`, `solve` or `study` (grep of both;
  magpylib has `TriangularMesh.mesh`, a property on that class only,
  `class_magnet_TriangularMesh.py:228`; the package's `solve_system` is internal, `_bodies.py:537`).

## 6. What a user writes by hand today, per use case

Sources: the scripts on `origin/explore/use-cases` (e9e88b6), `explorations/use-cases/`, and the docs.

- **Coil sums** (`s3_coils_sum.py`): `model(i_left, i_right)` builds the core and both coils with
  the currents. Today: one `apply_face_charge` and one `getB` per set of currents. By hand: two solves
  of `model(1, 0)` and `model(0, 1)`, `unit = [getB(apply_face_charge(model(*one), level=LEVEL), LINE)
  for one in ((1, 0), (0, 1))]`, then `a * unit[0] + b * unit[1]` for each set. The script's model has
  no remanence and no `H_ext`; with either, a third solve of the zero-current state would be needed
  (my reading of linearity, not in the script). The docs' U-core composes the same way, B of core
  plus coil against the bare coil (`docs/examples/ucore_electromagnet.md:155-170, 208, 230`).
- **Stray field** (`s1_shaft_sensor.py`): the field is set as `coll.H_ext = (stray, 0, 0)` inside
  `model(angle, gap, stray, holder)`; a fresh model and solve per (holder, gap, stray, angle); then
  `B = magpy.getB(solved, plates(gap)) + np.array([stray, 0, 0])`, with the comment "getB leaves out
  H_ext (#71)". The docs do the same for a concentrator's gain,
  `(magpy.getB(solved, sensors)[:, 0] + B0) / B0` (`docs/examples/mesh_convergence.md:69-74`).
- **A difference study** (`s4_nmr_pole_pieces.py`): a loop over levels; at each, `model(iron)` and
  `apply_face_charge(…, level=level, max_unknowns=…)` for design A and for B, Bz read on the axis,
  then `row["B_minus_A_ppm"] = B["variation_ppm"] - A["variation_ppm"]`. The script's docstring:
  "mesh_study takes one model and one quantity, so a study of a difference is written by hand."
  Each design gets its own panel layout (PROGRESS.md §4).
- **A loop** (`s2_tolerance_mc.py`): per sample the model is rebuilt, `model(dxy, tilt, magnet_chi)`,
  solved with `apply_face_charge(coll)`, read with `magpy.getB(solved, sensor(…))`. The sensor's own
  tolerances need no new solve: one solve, then one `getB` along a sensor path holding every pose
  (section (b) of the script; "This works today", PROGRESS.md §2). The docs loop the same way over
  mesh sizes (`docs/examples/soft_magnets.md:87-94`; `shield_tube.md:86-90`).
- **Checking a result** (`docs/examples/fem_check.md:62-64`): `mesh_study` + `fem_study` + `compare`.

In every loop the unsolved model is rebuilt each step, because a result cannot be solved again
(section 2, row "The result as the input of a new solve").

## 7. The validation harness as an existing unification

`validation/methods.py` has one adapter class per method, each with `name`, `settings` (its own
ladder), `can_solve(case)` and `solve(collection, sensors, setting, **options) → Solved(B, seconds,
counts)` (`36-43`):

- `Magpylib`: `getB` alone, the floor every method must beat (`46-70`).
- `MoM`: `mesh_uniform(collection, setting, max_cells=20_000)` + `apply_demag(**options)` + `getB`;
  `counts = {cells, unknowns = 3 cells}`; `TooLarge` from `max_cells` (`73-115`).
- `FaceCharge`: `apply_face_charge(collection, level=setting, **options)` + `getB`; unknowns counted
  from the sheets' panels; `TooLarge` from `max_unknowns` or a skipped refinement round (`166-231`);
  `FaceChargeRefined` adds `refine=1` (`234-246`).
- `FEM`: `fem_solve(collection, order=setting, observers=sensors, **options)` + `solution.getB(sensors)`;
  `counts = {unknowns = ndof}` (`118-163`).

The common ground it found: B at the given sensors, a count of unknowns, three phase times
(prepare, solve, getB). The setting is each method's own ladder (cells across, level, order), and
nothing is reused between settings or calls.

## 8. Where the code differs from the brief or the page

1. Page, node "model": "apply_demag wants cells". It takes any magnet as one cell and never checks
   whether the model was meshed (`demag.py:1671`; probe B: the unmeshed plate as one cell gives
   390 mT at the sensor, against 552–557 mT converged). The reverse holds too: `apply_face_charge`
   accepts a 25-cell mesh and solves the cells as 25 touching bodies, 1,800 unknowns against 270
   for the part itself, and un-seals the mesh (probe C).
2. Page, node "read": "none set gives a copy and a warning". True for the two solvers
   (`demag.py:383-394`, `_apply.py:146-149`), but `mesh_uniform` raises instead (`meshing.py:943-945`;
   probe B) and `fem_solve` solves without a word (`fem/_geometry.py:124-125`).
3. Page, node "auto": an observer inside a part is "known only when the model holds sensors". Today
   no solver reads the sensors inside the model: `apply_demag` passes them through (`demag.py:1674-1677`),
   `apply_face_charge` iterates sources only (`_apply.py:337`), `fem_solve` skips them
   (`fem/_geometry.py:87-89`) and takes `observers=` explicitly, to refine the mesh there.
4. Page, node "split": "H_ext, one value per step, shape (steps, 3)". Today a value that is not
   exactly three numbers raises in all three solvers, which share `get_H_ext`
   (`demag.py:363-380`; `_apply.py:161`; `fem/_geometry.py:123`).
5. Brief and page: "all three refuse paths". On sources, yes. A sensor path inside the model is
   accepted by all three and read along the path (probe C), and FEM reads sensor paths through a
   `CustomSource` (`fem/_solve.py:362-374`).
6. Brief §1: `face_charge_info` holds `reused` "since #132". Not on `main`: `unknowns, level, rounds,
   seconds` (`_apply.py:145, 319`; probe A). #132 is an open draft (gh, 31 October).
7. `docs/face_charge.md:30-32`: `getB` works "as for any other". With a magpylib rule to know:
   `solved.getB(sensor)` is refused when the solved collection holds its own sensor;
   `magpy.getB(solved, sensor)` and `solved.getB()` work (probe A).
8. Page, node "solved": a `ChargeSheet` "with its charges per step, as TriangleSheet carries current
   densities per step". A sheet's `charge_densities` is one (m, 3) array (`_sheet.py:164-182`); only
   its pose can have a path (probe A: a 3-step pose path gives (3, 3)). The page's own point
   "1 · What a path returns" already drops per-step charges.
9. Brief §1: "`apply_demag`'s default on uniform grids is GMRES … only `solver="direct"` factors".
   Confirmed on the plate (probe B log: auto chose iterative, FFT covers 99 %; rule `demag.py:487-501`),
   with one addition: even `"direct"` keeps nothing, `scipy.linalg.solve` overwrites the matrix and
   returns only the solution (`504-509`).
10. Brief §1 table: `FemSolution` has `.getB, .getH, .moment, .ndof, .seconds`; also `.order` and
    `.elements` (probe C). Its `.moment` is the whole model's, magnet included (probe C:
    0.1098 A m² = 0.0995 for the magnet + 0.0104 for the plate), not the soft part's.
11. Page, "Two steps" table: "Face charge | level, one for the model today" and "refine is a solve
    setting": both confirmed (`_apply.py:163-173`, `255-275`); and "Method of moments | mesh made by
    you": confirmed, with the two mesh steps disagreeing on where a meshed part lands in the
    hierarchy (section 5).
12. The refusal over `max_unknowns` prints "about 0.0 GB" for 1,518 unknowns (probe A;
    `_apply.py:234-235` rounds to one decimal). Cosmetic.
13. Studio #96 says a `ChargeSheet` is "a CustomSource the studio cannot hold". Studio 0.10.0 handles
    it (`recording.py:809, 846`, `fieldlines.py:432`, `tests/test_charge_sheet.py`); what still holds
    from #96 is that studio does not re-solve on a drag.

## 9. The probes' numbers (Bz at the sensor in the middle of the 1 mm gap)

| What | Setting | Size | Bz (mT) | Probe |
|---|---|---|---|---|
| magpylib alone: the magnet without the plate | — | — | 356.66 | A |
| face charge | level 1 / 2 / 3 | 270 / 768 / 1,518 unknowns | 554.60 / 557.33 / 556.36 | A |
| cells, `apply_demag` (auto → GMRES) | cells across 1 / 2 / 4 | 25 / 200 / 1,600 cells | 503.33 / 542.45 / 552.23 | B |
| `mesh_study` on those three meshes | — | — | estimate 555.49, error 0.74 %, order 2.0 | C |
| FEM | order 2, one edge layer | 12,407 unknowns, 5,411 elements | 557.10 | C |
| `apply_demag` on the unmeshed model | — | 2 cells | 390.19 | B |
| `apply_face_charge` on the 25-cell mesh | level 1 | 1,800 unknowns | 556.58 | C |
| face charge, the solved copy solved again after the magnet moved 2 mm | level 2 | 0 unknowns | 488.49 (a fresh solve: 424.39) | A |
| cells, the solved result solved again | cells across 2 | 200 cells | 542.88 (once: 542.45) | B |
| `H_ext` = 5 mT along z on the collection | face charge level 2 / cells across 2 | | 558.10 / 543.23 (+0.77 / +0.78) | A, B |
| the magnet given χ = 0.05 | face charge level 2 | 1,056 unknowns, two sheets | — | A |

Other numbers: the plate's mean polarization 0.0650 T (face charge) and 0.0653 T (cells); its moment
by hand 0.01035 and 0.0104 A m²; FEM's moment of the whole model 0.1098 A m². Jump indicator
3.34 / 1.86 / 0.79 % at levels 1 / 2 / 3. Inside the plate at its centre: face charge μ0 H
0.2064 mT (and B reported equal to it), FEM μ0 H 0.2525 mT and B 252.8 mT. 0.3 mm below the top
face: face charge −1.72 mT, cells 269.4 mT. Serialized size: 54,866 characters for the face-charge
result, 211,922 for the 200-cell result.

## 10. Open questions this document could not settle

- What `magpy.show(FemSolution)` draws: it did not raise in probe C; nothing says what appeared.
- Whether `MeshStudy.from_values` gives a sound error estimate for a face-charge ladder, whose
  unknowns grow with the square of the level rather than as a volume refinement.
- What studio does with a solved face-charge collection or a sealed mesh when a magnet is dragged:
  read from its source and #96 only, not run.
- The cells' B at the plate's exact centre (0, 0, 0) was not quoted: that point sits on a corner shared
  by 8 cells and magpylib summed their J there (2.18 T printed in probe B). Whether magpylib means to
  count boundary points as inside every cell that touches them is unverified; the 0.3 mm point
  avoids it.
- Whether the 1.5 mT spread between face charge (556.4–557.3 mT), cells (552.2 mT at 1,600 cells,
  555.5 extrapolated) and FEM (557.1 mT at order 2) is the cells' or FEM's error: FEM was run only
  at the tiny setting; the brief's finer FEM runs were not repeated.
- Thread behaviour: every probe ran with one thread; the face-charge thread pool (`_parallel.py`)
  and magpylib's threads were not probed.
