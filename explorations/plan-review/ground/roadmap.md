# How one entry point interacts with everything else planned or promised

31 October 2026. Ground work for the review of the "apply_response dispatch"
plan (`page_text.md`, `page_map.txt`, `BRIEF.md`). Read-only: nothing in any
repo was changed; one tiny FEM probe ran (section 8). Every claim names its
source. Line numbers are on `main` at 57c7860 unless a branch is named.

## The conclusion, first

- **The unit of reuse must be the assembled operator, not the LU.** Three
  planned things need the matrix kept with χ changing or with no factorization
  at all: saturation (#81, χ per cell or slab changes every iteration),
  hierarchical matrices (#103, solved iteratively, no dense LU), and the method
  of moments' own default (GMRES with FFT products, no LU). In both solvers χ
  is separable from the assembled blocks: MoM forms `Q = I − S·T` from a χ-free
  `T` (`demag.py:504-510`) and applies χ inside its matvec (`:1255-1273`); face
  charge adds χ only on the diagonal, `1/χ + 1/2` (`_bodies.py:567`), and in the
  remanence load (`:577`). #132's cache keys on χ (`_cache.py:108` on the
  branch), so a saturation loop would miss on every iteration and re-assemble
  (43 of 74 ms at 1,518 unknowns, `BRIEF.md` §2) to redo a 13 ms factorization.
  Keep the χ-free blocks on the mesh object; derive the LU, or a preconditioner,
  from them.
- **Decide #71 (applied field as a source) before the path API and the return
  type.** The plan's "H_ext per step, shape (steps, 3)" special-cases an
  attribute #71 removes; under #71 the same thing is a `UniformField` source
  with a path, going through the ordinary load (`_apply.py:199-217`). #71 also
  decides whether B from a solve includes the applied field; today all three
  solvers leave it out (`_apply.py:112`, `docs/examples/fem_check.md:115`).
- **Studio 0.10 reads results in two shapes only**, both by name: a
  `face_charge.ChargeSheet` imported from
  `magpylib_material_response.face_charge` (`session.py:2320-2352`,
  `fieldlines.py:428-432`) and a sealed `Collection` of cells
  (`widget.py:1461-1467`). Studio's own tests call `apply_face_charge`
  (`tests/test_charge_sheet.py:31,112`). Whatever `solve()` returns must still
  be, or hand out, those two shapes, or studio changes in step.
- **Studio #96 is the plan's best customer**: it asks for exactly a body that
  "names how it was solved", a fingerprint of what the solve depended on
  (studio #89) and a re-solve callable per drag frame. The open point is
  whether the mesh object sees the dragged magnet: `apply_face_charge` works on
  a copy of the model (`_apply.py:144`).
- **Breaking changes for the documented examples: none if the three functions
  stay.** If they are deprecated: 25 `apply_demag` calls on 11 docs pages plus
  the README, 2 `apply_face_charge` calls plus the README, 0 executed
  `fem_solve` calls (docs reach FEM only through `fem_study`), 4 `mesh_study`,
  4 `fem_study` and 2 `compare` calls on 2 pages. If the old functions return a
  new type: 28 reads of `.sources_all`/`.children` on 8 pages, 2 `inplace=True`,
  17 `susceptibility=` overrides on 5 pages break (section 5).
- **Postponing #132 breaks no promise** (draft, unmerged, no release notes).
  Its χ-free split `assemble_system` / `load_system` / `split_unknowns`
  (branch `_bodies.py`), the factor-time guards (`_cache.py:126-141`) and the
  pivot copy (`:189`, #131) are the core every later piece needs and can merge
  on their own, without the store.

## 1. The map at a glance

Effect of the entry point on each issue, and what the issue needs from the
solve core. "Easier" means the entry point gives the issue a home or removes a
hand-written step; "harder" means a plan element as written would get in the
way.

| Issue | Effect of the entry point | What it needs from the core |
| --- | --- | --- |
| #130 kept factorization / PR #132 | Easier: the mesh object is its home | χ-free assembly split, fingerprint, LU with guards, pivot copy |
| #131 scipy pivots | Unrelated to the API; a rule for any kept LU | `piv.copy()` per `lu_solve`, a lock on the kept object |
| #81 saturation | Easier if the operator is kept; harder if only the LU is | χ-free operator kept across iterations; χ per cell or slab; a `bh_curve` attribute; B or mean field inside; refuse paths on B–H models |
| #103 hierarchical matrices | Unrelated to the work; constrains reuse | reuse = kept compressed operator + preconditioner + warm start, no LU; lifts `max_unknowns` |
| #71 UniformField | Harder as written (per-step `H_ext`, `read` checks on an attribute); easier once decided | applied field enters as a fixed source through `external()`; B includes it |
| #118 anisotropic face charge | Easier: one `auto` rule to change | a second inside layer on the sheet; χ tensor in the fingerprint |
| #120 thin sheets | Easier: one fallback with a reason | the mesh record says which panel model made it |
| #84 progress hooks | Easier: one `progress=` keyword | phase boundaries in assemble/load/factor, GMRES callback, refinement rounds, path chunks |
| #85 NMR pole pieces | Easier only if `study()` takes a difference of two solves | same layout across rungs; stretched layout for two shapes (later) |
| #92 thin rings in MoM | Unrelated; a rule for `auto` | warn when `auto` routes a thin curved high-χ ring to cells |
| #79 meshing per body | Easier: per-part mesh settings get a home | cell type per body, conforming meshes, a cell budget |
| #76 consolidated findings | Unrelated now; constrains the return type | a compressed solved body (FMM getB); 12 unknowns per linear cell; FMM solve has no LU |
| #60 tensor χ | Unrelated | `auto`'s anisotropy test and the fingerprint accept a (3, 3) |
| #12 large tests | Unrelated; an adapter in `validation/methods.py` | time a re-solve as its own phase |
| #11 docs in magpylib | Unrelated; names matter if the docs merge | none |
| studio #96 (+ #89) | Easier: method record + fingerprint + re-solve callable | a solve per frame as load + back-substitution; results stay ChargeSheet or sealed cells; live model, not a copy |

## 2. Issue by issue

### #130 and PR #132: the kept factorization

What #132 does (branch `face-charge-kept-factorization`, b43d600): a global
store keyed on what the matrix is assembled from, read off the meshed bodies:
panels in their own frame, χ, poses relative to the first body, joined faces,
absolute size (`_cache.py:94-124`); a matrix is factored only the second time
its key comes (`SEEN`, `solve_kept`, `:32,144-187`); 1 GB cap with eviction
(`:34,74-83`); a fork resets the store (`:48-49`); each back-substitution copies
the pivots (`:189`). It splits `solve_system` into `assemble_system`,
`load_system`, `split_unknowns` (branch `_bodies.py`, diff in the PR) and sets
`face_charge_info["reused"]` (`_apply.py` diff). Its docs section "Many solves
in a loop" promises "Nothing needs asking for" and public `set_cache_limit` /
`clear_cache` (branch `docs/face_charge.md`, section added at line 220). PR
state: open, draft, no reviews (`gh pr view 132`). Its CI timing comment shows
no case slower than 1.05× (the PR's github-actions comment).

Interaction with the plan: the 31 October reversal (`BRIEF.md` §1) makes the
mesh object hold the factorization instead. Then:

- The `assemble_system` / `load_system` split is the private core the plan
  names ("factor once, back-substitute every step"); it is χ-free by
  construction and can merge alone.
- The key becomes the fingerprint that spots an edited mesh. #132's key is
  complete for what it covers (14 model sequences checked, PR body), but it
  digests χ as one scalar per body (`_cache.py:108`): for #81 and #118 it must
  cover a χ that changes per slab or is a tensor.
- The "second sighting" rule, the cap, eviction and fork handling go away with
  the store; the guards at factor time stay (`_factor`, `:126-141`: a singular
  matrix raises, an ill-conditioned one warns and is not kept).
- `validation/methods.py` on the branch clears the cache before every solve
  (diff, `FaceCharge.solve`), so the PR timing never sees a reuse; with a kept
  object the runner should time a re-solve as a phase of its own, or the reuse
  path is unguarded against regressions (section on #12).

Nothing of #132 is promised publicly yet: the CHANGELOG entry and the docs
section exist only on the branch (`git diff main origin/face-charge-kept-factorization -- CHANGELOG.md docs/face_charge.md`).

### #131: `lu_solve` is not thread-safe on a shared LU

A scipy bug (f2py wrapper of `getrs` shifts the caller's pivots in place with
the GIL released; fixed on scipy main by scipy/scipy#25992, not in any release
the pin `scipy>=1.17.1` allows): 395–399 wrong solutions of 400 from 4 threads
on 1.15–1.18 (#131's table). Rule for any kept LU: `lu_solve((lu, piv.copy()),
b)` per call and a lock around what is kept (#131, `_cache.py:189`). The mesh
object has the same hazard as the store: a notebook's drag callback and a
kernel thread sharing one object. Unrelated to the API; a constraint on the
core, with a test (#132 adds
`test_a_kept_factorization_is_safe_from_threads`, branch `tests/test_face_charge.py`).

### #81: saturation, non-linear B–H

What #81 plans: a `bh_curve` set like `susceptibility`; inside `apply_demag`,
repeat the linear solve with each cell's χ read off the curve at its field,
Anderson-accelerated, "interaction matrix built once"; the same curves in
`fem_solve`; face charge "refuses saturation today", with a slab prototype to
decide on. Measured in the prototype: 11–35 linear solves per field level on a
512-cell cube; exact on a sphere to 0.02 %.

What the core must give it:

1. **The operator kept across iterations, with χ changing.** MoM:
   `_solve_direct(T, chi, rhs)` forms `Q` from a χ-free dense `T`
   (`demag.py:504-510`), and the iterative path applies χ inside
   `_build_matvec` (`:1255-1273`), so `T` or the FFT/dense `ops` are the thing to
   keep. Face charge: χ enters as `add_mass(…, 1/χ + 0.5)` on the diagonal
   (`_bodies.py:567`) and as `load_remanence()/χ` (`:577`); the blocks `G` are
   χ-free. A kept LU is useless here; a kept operator makes each iteration a
   refactorization (13 of 74 ms) instead of a full assembly (43 of 74 ms) at
   1,518 unknowns (`BRIEF.md` §2).
2. **χ per cell or per slab.** `apply_demag` already takes one value per cell
   (`susceptibility=` on the call, `demag.py:1539-1550`). Face charge takes one
   χ per part; the saturation exploration cut each cuboid into touching slabs
   by hand and gave each its own χ: 20 mm slabs reach 14.36 mT at 400 A where
   the finest MoM (2,376 cells) reaches 13.91 mT and is still rising, in
   1.4–2.9 s per current against 20–74 s (`explore/face-charge`,
   `explorations/face-charge/saturation/PROGRESS.md`, results table). That is a
   mesh edit ("cut this part into slabs along its length") on the mesh object,
   plus χ per slab.
3. **A field inside** to read χ from: B at cell centres (MoM), or a slab's mean
   field from its moment (`ChargeSheet.mean_polarization`, `_sheet.py`).
4. **Paths.** With a B–H curve the matrix depends on the field, so it changes
   at every step: the narrow form ("one matrix for every step", page map
   `rigid` node) must refuse paths on B–H models or iterate per step.

Where the plan contradicts the evidence: the `auto` node and the "B inside"
tile say "non-linear B–H stays with the method of moments (#81)"
(`page_map.txt:23-24, 91-96`); the exploration shows face charge on slabs
tracks saturation on bar-like parts at a fraction of MoM's cost. Leave the
rule open.

### #103: hierarchical matrices for large face-charge systems

#103 keeps far cluster pairs as interpolated blocks and solves iteratively past
`max_unknowns` instead of refusing; measured on triangles 6–8× faster assembly
at 10,008–18,216 unknowns, answers within 1.4e-5. Two consequences for the
plan: (a) there is no dense LU to keep, so reuse along a path or between frames
is a kept compressed operator, a preconditioner and a warm start, and each step
is a GMRES solve, not a 1 ms back-substitution (`BRIEF.md` §2: GMRES per step is
"a full iterative solve"); (b) the `auto` rule "more than `max_unknowns` at
level 1 → cells" (`page_map.txt:24`) changes once large models run. The mesh
object must record which assembly made it (dense or hierarchical).

### #71: replace `H_ext` with a `UniformField` source

#71's status (2026-10-29): #119 fixed the checks; the redesign "stands, for
after 0.7.0". Today the applied field is an attribute read per object or parent
(`get_H_ext`, `demag.py:311-343`) and enters each solver separately: MoM adds
`S·H_ext` to the right-hand side (`demag.py:1737`); face charge gives each body
an `applied` vector (`_apply.py:161,364-394`) added inside the load
(`_bodies.py:537-577`); FEM holds a uniform `B_ext` exactly as a lift
(`fem/_solve.py:5`). Under #71 the field becomes a fixed source in the
collection and reaches face charge through `external(points)`, which is
`magpy.getB(fixed, points) / mu0` (`_apply.py:199-217`); the per-body `applied`
and the applied-field equality test in `_same_material` (`_apply.py:629`) go
away, and FEM must still recognise the source as its lift.

What it does to the plan:

- The `read` node checks "H_ext: three finite numbers; above 10 T a warning"
  (`page_map.txt:5-7`) on an attribute #71 removes.
- "H_ext per step, shape (steps, 3)" (`split` node, `page_map.txt:17-20`;
  page "The narrow form") is a new per-step argument for one attribute. Under
  #71 it is a `UniformField` on a path, like any other source, if magpylib lets
  a custom source's parameter vary along a path (#71: "In core the field could
  vary along a path"; not verified here).
- #71 is breaking by design: `getB` of a result includes the applied field; a
  field on a sub-collection no longer acts on its cells only. The plan's
  "return B at observers" (page "The narrow form") must say which convention
  it follows; today every solver leaves `H_ext` out (`_apply.py:112`;
  `docs/examples/fem_check.md:115`; the use-cases scripts add it back by hand,
  `explore/use-cases` `PROGRESS.md:30`).
- Studio draws `H_ext` as the corner arrow (`threejs.py:1745-1780`, decision
  0044, studio 0.10.0 release `#104`) and keeps it in the document as `applied`
  (`docs/architecture.md:125-128`); `to_json` writes an `H_ext` key
  (`utils.py:160-162, 311-313`). A `UniformField` needs both to follow.

Decide #71 first; it is cheaper to decide than to build around.

### #118: anisotropic face charge, a second inside layer

#118 adds, on anisotropic bodies only, an inside single layer with the kernel
of `∇·(A∇·)`: two unknowns per panel corner there, one elsewhere; isotropic
parts unchanged; a lone anisotropic body doubles its unknowns (dense ~8×);
"fields inside need `S_A[τ]`: the sheet alone gives a wrong H there". For the
plan: the `auto` rule "anisotropic χ → cells" (`page_map.txt:24`) flips to face
charge later, so the rule belongs in one function that returns the method and
its reason; the `ChargeSheet` return type needs room for an inside layer; the
fingerprint must digest a tensor (today a scalar, `_cache.py:108`). Unrelated
to paths: the inside layer is part of the matrix.

### #120: thin sheets cost (size / thickness)² unknowns

A 40 × 40 × 0.5 mm sheet needs 40,320 unknowns at level 1 (#120's table), so it
is refused today (`max_unknowns` 20,000, `_apply.py:184`) and `apply_demag`
with flat cuboids of ≥ 3 layers is the advice (#120; #76 §2: 20×20×3 cells
−0.84 %). The plan's fallback "level 1 already needs more than `max_unknowns`
→ cells" (`page_map.txt:24`) is where #120 bites; it should name the reason
(thin part) and the recipe. A thin-shell model later is another panel model
inside face charge, which the mesh record and the fingerprint must tell apart.

### #84: progress hooks for studies and inside long solves

#84 asks for `mesh_study(..., progress=)` after each level, a GMRES-iteration
hook in `apply_demag`, progress inside `apply_face_charge` (assembly,
factorization, refinement rounds) and `fem_solve`, and a reason when no
estimate is given. One entry point gives one `progress=` keyword and one event
vocabulary. What the core must expose: the phase boundaries #132's split
already draws (mesh, assemble, factor, load, back-substitute, sheets;
`_apply.py:257-313` runs the refinement rounds), scipy's GMRES `callback`, FEM's
phases (`solution.seconds["mesh"|"assemble"|"solve"]`, read in
`validation/methods.py:148-163`) and, along a path, the chunk of steps (plan:
~100 steps a chunk, `page_map.txt:47`). Easier with the entry point; nothing
blocks it today.

### #85: NMR pole pieces, a docs example

#85 is a documentation issue: a page comparing a magnet with and without iron
discs, "comparing designs … not absolute ppm homogeneity", with `mesh_study`'s
error bar, a saturation check (#81) and, since #113, the discs by face charge.
The use-case survey found the verdict between two disc thicknesses flips with
level (+9.5, −2.5, −5.6 ppm at levels 1–3, Bz at the centre settled to 0.03 %
by level 2) and that level 3 needs 20,640 unknowns and 6.7 GB
(`explore/use-cases` `PROGRESS.md:70-84`; `page_map.txt:79-84`). What it needs:
a study of a *difference* between two models, which `mesh_study` cannot take
("takes one model and one quantity, so the study of a difference is written by
hand", `PROGRESS.md:80`); the plan's `study()` says "the quantity may compare
configurations … solved on the same layout at every rung" (`page_map.txt:50-52`)
but the two designs are two shapes, and one layout stretched across shapes is
"later, each its own design" (`page_map.txt:16`). So: easier only if `study()`
takes a function of several solves; the layout question stays open. The page
itself can be written today with `apply_face_charge` (#85's last checkbox).

### #92: MoM on thin cylinder segments

A known MoM failure: the field inside a thin ring is 300–550 % off even after
#100's averaging; face charge solves the ring (5.0 % inside at level 2, 0.25 %
on the far side); the issue stays open for saturating or anisotropic rings
(#92, update 2026-10-29). For `auto`: a linear ring goes to face charge, which
is right; a saturating or anisotropic ring goes to cells, where the answer
inside is known wrong, so `auto` should warn on thin curved high-χ cells. The
docs example `shield_tube.md` meshes the tube with `mesh_Cylinder(tube,
(around, 1, along))` and solves it with `apply_demag`
(`docs/examples/shield_tube.md:38-59`); under one entry the same model would
route to face charge by default, so the example would name `method="mom"`.
Unrelated to reuse.

### #79: meshing for `apply_demag`, cell type per body

#79 gathers the MoM meshing work: tetrahedra for compact parts (89 one-surface
tetrahedra within 1.7 % where 3,773 cuboids are −6 %), flat cuboids with ≥ 3
layers for thin parts, a cell budget by volume, conforming meshes across
touching bodies, `mesh_TriangularMesh`'s `minratio` setting the count, TetGen's
AGPL licence against Netgen, CAD meshes through `meshio`. The plan's
`mesh(model, method="mom", cells_across=…)` would wrap `mesh_uniform(obj,
cells_across, inplace=False, max_cells=1_000_000)` (`meshing.py:883`), which
gives every part one cell size. #79's per-body choices are exactly the "mesh
settings per part … inherited from parent collections" the plan leaves open
(`page_text.md`, "Two steps"). Easier: one home; but `cells_across` and
`target_elems` must not be frozen as the only MoM mesh settings. #79 lists
"objects on paths" among what cells must cover; with the narrow form on face
charge too, that reason weakens.

### #76: consolidated findings (linear cells, FMM, compute backends)

Still open in #76 (status 2026-10-29): linear cuboid cells (12 unknowns per
cell, ~5× lower error at equal cells), FMM evaluation of a solved body as a
magpylib source (13 s → 0.7 s on a 13,041-point plane, §5b), an FMM solve above
~10⁴ cells, a compute-backend protocol, JAX for gradients. Constraints on the
plan: a MoM result may become a compressed source wrapping the cells rather
than the cells themselves, so the return type should allow an opaque solved
body; "cells" must not imply 3 unknowns each (`validation/methods.py:107`
counts `3 * cells`); an FMM solve has no LU, as #103. A mesh-free polarizability
dipole for compact parts (§4e) is a candidate fourth method, so `method=` is
open-ended. Unrelated to the entry point now.

### #60: a full 3 × 3 susceptibility tensor per cell

Unrelated to the entry point. Two touch points: `auto`'s anisotropy test must
treat any non-scalar χ, including a (3, 3), as "cells" (today face charge
refuses when the 3 values differ, `_apply.py:351-357`), and the fingerprint
must digest a tensor. #81's Newton step would need #60.

### #12: large tests

Answered in substance by the validation suite: 20 cases, 17 with a stored FEM
reference, every PR checks MoM against them and reports time and memory, every
night all methods climb finer settings (CHANGELOG 0.7.0, "Validation in the
repository"). Each method is one adapter class with
`solve(collection, sensors, setting, **options) → Solved(B, seconds, counts)`
(`validation/methods.py:37-43, 73-116, 118-163, 166-232, 255`): a one-call
unification over the three solvers that returns B at sensors, the shape the
plan proposes for paths. The entry point needs its own adapter there, and the
runner should time a re-solve as a phase, since #132's version clears the cache
before each solve.

### #11: docs integration into magpylib

Unrelated. If the docs merge, names collide more visibly: magpylib has no
`mesh` or `solve` function (grep of `magpylib/src/magpylib` for `def mesh(` /
`def solve(`: none), but `magpy.getFT(..., meshing=)` and this package's
`meshing` module would sit beside a `mesh()` function.

### studio #96 (re-solve on drag) and studio #89 (say when a solve is stale)

What studio has: an editable view reports a drag's end (`revision`,
`last_edit`); a look-only view can be re-pointed after a notebook re-solves
(0.14 s + 0.02 s); a solve per frame is 46/76/141/377 ms at levels 2–5 today
and 12/19/30/50 ms with a kept LU (#96; `explore/use-cases` `PROGRESS.md:133-140`).
What it asks for (#96): (1) swap a sealed body's parts in an editable view,
(2) "a solver hook for the studio's own panel: a body names how it was solved,
the kernel registers that solve, and the studio calls it when an edit changes
what the body was solved for (the fingerprint of #89)", (3) solving during a
drag paced by cost. #89 asks where the fingerprint lives: "material-response
writes it on the body when it solves (`collection.solved_for = …`), or studio
takes it when the body is imported".

The plan's mesh object records its method and spots edits by a fingerprint
(`BRIEF.md` §1; `meshed` node, `page_map.txt:31-32`): that is #96's item 2 and
#89's fingerprint, if the fingerprint is written on the result and the
re-solve is one call that only loads and back-substitutes. Two things the plan
must settle for studio:

- **The result's shape.** Studio holds a solved sheet as the constructor
  arguments of `face_charge.ChargeSheet` (`vertices`, `faces`,
  `charge_densities`, `panels`, `mean_polarization`), imported from
  `magpylib_material_response.face_charge` the first time a document names it
  (`session.py:2320-2352`, decision 0041 `docs/decisions.md:2169`,
  `docs/architecture.md:129-137`); field lines stop inside a sheet by testing
  the class's module path (`fieldlines.py:428-432`); a meshed MoM result is a
  sealed collection drawn as one object (`widget.py:1461-1467`; `threejs.py:1714`;
  CHANGELOG 0.7.0 line 111; `docs/quickstart.md:122`). Studio's tests build
  sheets with `face_charge.apply_face_charge(…, level=1)`
  (`tests/test_charge_sheet.py:18,31,112`), and its example calls `mesh_all` +
  `apply_demag` (`examples/ucore_demo.py:126-128`). Renaming the module, the
  class or `apply_face_charge`, or returning something that is neither a
  `ChargeSheet` nor a sealed collection of cells, breaks studio 0.10 as
  shipped.
- **Live model or copy.** `apply_face_charge` solves a copy
  (`_apply.py:144`) and unseals the parents it puts sheets into
  (`:313-319`). A drag moves a magnet in studio's scene; for `solve(meshed)` to
  see it, the mesh object must reference the live sources or take them as an
  argument each frame. The brief's question 3 ("what happens when a user moves
  a magnet on that object") is the same question.

Also from studio's side: its roadmap step R7-3 names `fem_study` and
`compare` as what studio will call (`docs/roadmap.md:172-176`), and its FEM plan
quotes `apply_demag(collection, susceptibility=…)` (`docs/plans/fem.md:32`).

## 3. What the docs and release notes promise, and the plan must keep

- "A solved model is a magpylib source like any other" (`README.md:29-32`,
  the paragraph under the warning). The plan's path form returns "B at
  the observers passed to the solve, not charges per step" (page "The narrow
  form"): a result that is not a source. Say so, or wrap it as one.
- "hands back ordinary magpylib magnets that you use like any other"
  (`docs/how_it_works.md:23`).
- A `ChargeSheet` is "an ordinary magpylib source: `getB`, observer paths,
  `show()` and `to_json` work as for any other, and magpylib-studio after 0.9
  draws it in a `SceneWidget`" (`docs/face_charge.md:30-33`; CHANGELOG 0.7.0
  line 42); `to_json` keeps `face_charge_info` (`utils.py:164-166, 314-315`) and
  `sealed` (`:214-215, 319-320`).
- "A mesh is a `Collection` with `sealed = True`: magpylib-studio from 0.8
  shows it as one object" (`docs/quickstart.md:122`; CHANGELOG 0.7.0 line 111).
- Fields leave `H_ext` out, in all three solvers (`_apply.py:112`;
  `docs/examples/fem_check.md:115`); #71 would change this deliberately.
- "Objects moving along paths are refused by both" (`docs/face_charge.md:79`);
  verified for all three: `demag.py:1663-1670`, `_apply.py:331`,
  `fem/_geometry.py:63-66` and the probe in section 8.
- Not yet promised: everything in #132's branch docs ("Nothing needs asking
  for", `set_cache_limit`, `clear_cache`, `face_charge_info["reused"]`).

## 4. Breaking changes for documented examples

Counts by `grep -c` on `docs/*.md`, `docs/examples/*.md` and `README.md`
(`method.md`'s `apply_demag(` is a signature in prose; `how_it_works.md`'s
`fem_study(` and `fem_check.md`'s `fem_solve(` are prose).

| Page | `apply_demag(` | `apply_face_charge(` | `fem_solve(` | `mesh_study(` | `fem_study(` | `compare(` | result reads `.sources_all`/`.children` | `susceptibility=` | `inplace=True` | `H_ext` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| README.md | 1 | 1 | | | | | | | | |
| quickstart.md | 5 | | | | | | 1 | 3 | | 2 |
| how_it_works.md | 4 | | | | 1 (prose) | | 9 | | | 4 |
| face_charge.md | | 2 | | | | | | | | 2 |
| method.md | 1 (prose) | | | | | | | | | 3 |
| logging.md | 1 | | | | | | | | 1 | |
| serialization.md | 1 | | | | | | | | 1 | |
| api.md | | | | | | | | | | 1 |
| examples/cuboids_demagnetization.md | 1 | | | | | | 2 | 3 | | |
| examples/soft_magnets.md | 1 | | | | | | 2 | 2 | | |
| examples/ucore_electromagnet.md | 1 | | | | | | 5 | | | |
| examples/solver_performance.md | 8 | | | | | | 6 | 8 | | |
| examples/mesh_convergence.md | | | | 3 | 3 | 1 | | | | 1 |
| examples/fem_check.md | | | 1 (prose) | 1 | 1 | 1 | | | | 1 |
| examples/shield_tube.md | 1 | | | | | | 1 | | | |
| examples/tetrahedral_meshes.md | 1 | | | | | | 2 | 1 | | |
| **Total, docs** | **25 on 11 pages** | **2** | **0 executed** | **4 on 2 pages** | **4 executed on 2 pages** | **2 on 2 pages** | **28 on 8 pages** | **17 on 5 pages** | **2** | **14 on 7 pages** |

Beyond the docs: tests call `apply_demag` 52 times, `apply_face_charge` 37,
`fem_solve` 8, `mesh_study` 5, `fem_study` 6 (`grep -c` over `tests/*.py`);
`validation/methods.py` wraps all three and `validation/references.py:112-116`
calls `fem_study` and `fem_solve`; studio calls `apply_face_charge` twice in
its tests and `mesh_all` + `apply_demag` once in an example (section 2, studio).

What each plan element breaks:

| Plan element | Documented examples broken |
| --- | --- |
| Add `mesh` / `solve` / `study` beside the three functions, unchanged | none |
| Old functions as thin wrappers that return what they return today | none; two result shapes to keep |
| Deprecate `apply_demag`, `apply_face_charge`, `fem_solve` | 25 + 2 calls on 12 pages and the README; 97 test calls; studio's 2 test calls and 1 example; the validation adapters |
| `solve()`'s new return type also returned by the old functions | 28 result reads on 8 pages, 2 `inplace=True`, 17 `susceptibility=` overrides (no such argument on `solve` in the plan), `to_json` of a solved collection (`serialization.md:60-80`), studio's two consumers |
| `study()` replaces `mesh_study` / `fem_study` / `compare` | 10 calls on 2 pages (+ 2 prose mentions), 11 test calls, `validation/references.py`, studio roadmap R7-3 names `fem_study` and `compare` |
| `H_ext` per step, or #71's `UniformField` | 14 `H_ext` mentions on 7 pages; studio's corner arrow (0044) and document key; `to_json`'s `H_ext` key |
| `auto` as the default method | `shield_tube.md` (MoM on a tube by design) and every `apply_demag` example routed to face charge unless `method="mom"` is written; `how_it_works.md` is a MoM page |

## 5. What the dependencies say about order

Not a decision; what follows from sections 2 to 4.

1. Merge #132's χ-free split (`assemble_system`, `load_system`,
   `split_unknowns`), the factor-time guards and the pivot copy as a refactor
   without the store: the core for #130, #81, #84 and paths; bit-identical
   first solves are already checked on the branch (PR body).
2. Decide #71 before the path API and the return type (section 2, #71).
3. Define the kept object as the χ-free operator with a derived LU or
   preconditioner, so #81, #103, the MoM direct solver and the MoM GMRES default
   all fit; a per-call pivot copy and a lock (#131).
4. Write the fingerprint on the result (studio #89's open question) and give
   the object a one-call re-solve that takes the live sources (studio #96).
   Keep `ChargeSheet` and sealed cells as the shapes a result hands out.
5. `progress=` on the one entry once the core has phases (#84).
6. `auto` as one function returning the method and its reason; #92, #118,
   #120 and #103 each change a line there.
7. Per-part mesh settings on the mesh object, where #79's cell types land.
8. `study()` over a function of several solves (#85); a layout stretched across
   shapes later.

## 6. Open questions this ground work could not settle

- Can a magpylib custom source's parameter (a `UniformField`'s B) vary along a
  path, so that #71 covers "H_ext per step" without a special argument? Not
  probed.
- Live model or copy: does the mesh object reference the sources it was solved
  with, or copy them as `apply_face_charge` does (`_apply.py:144`)? Studio's
  drag and the brief's question 3 both hang on it.
- Where the fingerprint lives (studio #89): on the result, written by
  material-response, or in studio's document.
- Will saturation run through face charge on slabs or stay with MoM (#81's last
  checkbox)? It decides `auto`'s rule and whether the narrow form refuses B–H
  models.
- Does `max_unknowns` keep its role in `auto` once #103 solves past it?
- Will studio learn a new result type, or does material-response keep
  `ChargeSheet` and sealed cells as the only shapes it hands out?
- Do `fem_study` and `compare` keep their names? Studio's roadmap R7-3 names
  them.
- Should the validation runner time a re-solve as its own phase?

## 7. Not in this note

Question 0 of the brief ("do the rewrite at all") and the return-type sketch
(question 2) are the design's; this note gives them the constraints.

## 8. Probe

`ground/probe_fem_path.py`: a 1 mm soft cube (χ 1000, 10 mT) on a two-step
path, `fem_solve(order=1, edge_layers=1)`, run as
`nice -n 19 .venv/bin/python -I probe_fem_path.py` with
`magpylib.defaults.threads = 1`. Output:

```
path length: 2
fem_solve REFUSED: ValueError 1 objects with paths found; fem solves one position of the model, like apply_demag.
```

So all three solvers refuse paths (`fem/_geometry.py:63-66`), as the brief
says.

## Sources

- Issues: `gh issue view N -R magpylib/magpylib-material-response` for 130,
  131, 120, 118, 103, 92, 85, 84, 81, 79, 76, 71, 60, 12, 11 (all open, none
  with comments); `gh issue view 96 -R magpylib/magpylib-studio` and `89`;
  `gh pr view 132 -R magpylib/magpylib-material-response` (open, draft, no
  reviews, one github-actions timing comment).
- Branches, read with `git show`: `origin/face-charge-kept-factorization`
  (`_cache.py`, `_bodies.py`, `_apply.py`, `docs/face_charge.md`,
  `tests/test_face_charge.py`, `validation/methods.py`);
  `origin/explore/use-cases` (`explorations/use-cases/PROGRESS.md`, `kept.py`);
  `origin/explore/face-charge` (`explorations/face-charge/saturation/PROGRESS.md`).
- Code on `main` (57c7860): `src/magpylib_material_response/demag.py`,
  `face_charge/_apply.py`, `face_charge/_bodies.py`, `face_charge/_sheet.py`,
  `fem/_solve.py`, `fem/_geometry.py`, `fem/_study.py`, `convergence.py`,
  `meshing.py`, `utils.py`; `validation/methods.py`, `validation/references.py`.
- Docs: `README.md`, `CHANGELOG.md`, `docs/index.md`, `docs/how_it_works.md`,
  `docs/quickstart.md`, `docs/face_charge.md`, `docs/api.md`,
  `docs/examples/*.md`.
- Studio (read-only, 0b92f69): `magpylib_studio/session.py`, `fieldlines.py`,
  `widget.py`, `threejs.py`, `recording.py`, `tests/test_charge_sheet.py`,
  `examples/ucore_demo.py`, `docs/architecture.md`, `docs/decisions.md`
  (0037, 0041, 0044), `docs/roadmap.md`, `docs/plans/fem.md`,
  `vscode-extension/CHANGELOG.md`.
