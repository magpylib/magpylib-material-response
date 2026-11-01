# What the code would need for each piece of the plan

Ground file for the plan review, 31 October 2026. Read from main at 57c7860
(`#133` far pairs and `#134` far field in), `origin/face-charge-kept-factorization`
at b43d600 (PR #132, based on 2911060 = 0.7.0), `origin/explore/use-cases` at
e9e88b6, magpylib 6.1.0, magpylib-studio main. Line numbers are of those
revisions. Sizes and risks are estimates, not measurements; no timing was run.
One probe (`ground/probe_seams.py`, output `ground/probe_seams.out`) checked
API behaviour on tiny models.

## Conclusion first

1. **The pieces are not equally far from the code.** FEM already has a mesh
   object (`fem._solve.Model`, private) that is solved at any order; MoM's mesh
   is already an object (a `Collection` with `sealed = True`); face charge has
   no mesh object at all: its bodies live inside one 320-line function
   (`_apply.py:142-320`) and die with it. The face-charge side is where the
   plan costs lines.
2. **Four facts in the code change the plan's shape:**
   - `Collection.copy()` is a deep copy (`class_BaseGeo.py:570-607`; probe 1b):
     a factorization stored as an attribute of the model would be duplicated by
     every `copy()`, including the one `apply_face_charge` makes of its input
     (`_apply.py:144`) and the one `apply_demag` makes (`demag.py:1659`). The
     holder of the LU needs its own copy rule, or the LU must sit beside the
     model, not on it.
   - studio and the JSON serializer recognise objects by exact class
     (`utils.py:131`; studio `session.py:2395-2410`, `importer.py:61-69`), so a
     `Collection` subclass would be refused by `to_json` and unnamed in a studio
     document. An attribute on a plain `Collection`, as `sealed` is
     (`meshing.py:55-56`), is what both already read.
   - In `apply_demag` the fixed (chi = 0) magnets are cells of the matrix
     (`demag.py:1671, 1716, 1462-1479`): moving a magnet changes the matrix.
     The narrow form "sources vary freely" does not hold for MoM as it stands;
     it needs the fixed magnets taken out of the unknowns first (`demag.py:
     1728-1737` is the seam, with a Newell caveat).
   - PR #132 is based on 0.7.0, before #133 and #134 (merge base 2911060). Its
     78-line change to `_bodies.py` splits a `solve_system` that main has since
     rewritten (`cross_block` with `FAR_RULES`, `_bodies.py:357-465`); the split
     must be redone on main, not rebased.
3. **Rough size of the whole plan, as estimated below: about 1,000 to 1,500 new
   lines, 300 to 450 touched, 40 to 60 new test functions, four docs pages
   touched.** Paths in the narrow form is the largest and riskiest piece (350
   to 500 new lines); "auto" and the study are the smallest. The kernels, the
   meshers, #133's far pairs, #134's far field and the FEM internals stay
   untouched.

## Today's seams, in one table

| Where | Lines | What it does today | Why it matters to the plan |
|---|---|---|---|
| `face_charge/_apply.py` `apply_face_charge` | 52-320 | mesh, warn, solve, refine, build sheets, replace parts, in one function | the only place a face-charge "mesh" exists |
| `_apply.py` `_split` | 328-361 | solved parts vs fixed sources; refuses any path (330-332) and anisotropic chi (353-359) | the "read" node and the auto checks |
| `_apply.py` `meshed(level)` | 163-173 | builds `Body`/`TriBody`, contact lines, counts unknowns before any assembly | the mesh step of a two-step API |
| `_apply.py` `external(points)` | 199-217 | one `magpy.getB(fixed, points, sumup=True)` for the load | the loads-per-step seam |
| `_apply.py` refine loop | 219-275 | `_interfaces` + `solve_system` (247-249), sheets (251-254), bisect + re-mesh (267-275) | refine changes the bodies, so the matrix |
| `_apply.py` result | 289-320 | parts swapped for sheets in the copy, parents unsealed (313-318), `face_charge_info` (319) | what `solve()` returns today |
| `_bodies.py` `solve_system` (main) | 537-610 | assembles A and rhs together, `_solve` in place | #132 split it into assemble / load / split |
| `_bodies.py` `_solve` | 613-637 | from 2,000 unknowns: float32 LU + refinement against A in double (619-633); else `scipy.linalg.solve` overwriting A (637) | the factorization is thrown away; which one to keep is a choice |
| `_bodies.py` `near_panels`, `NEAR`, `add_contact_lines` | 265-275, 56, 293-331 | finer load rule within one panel size of a magnet; grading within 1 % of the part's size | the two thresholds of the "same matrix" test |
| #132 `_bodies.py` | 525-624 | `assemble_system`, `load_system`, `_shared_dofs`, `split_unknowns` | the private core the plan builds on |
| #132 `_cache.py` | 94-123, 126-141, 186-189 | fingerprint `_key`, `_factor` with singular/ill-conditioned guards, `lu_solve` with copied pivots (#131) | the parts worth keeping |
| #132 `_cache.py` | 31-91, 144-183 | 1 GB LRU, "seen twice" rule, fork reset, `set_cache_limit`, `clear_cache` | the parts the reversal drops |
| `demag.py` `apply_demag` | 1539-1761 | copy (1655-1659), refuse paths (1663-1670), rhs from J0 + currents at centroids + H_ext (1703-1737), solve (1741-1752), write J into cells (1756-1757) | MoM's loads and result |
| `demag.py` `_solve_cells` | 1449-1536 | `_auto_solver` (1463-1470), direct `_assemble_T_dense` + `_solve_direct` (1472-1483), GMRES with fallback (1484-1535) | where a kept operator or LU would go |
| `demag.py` `_solve_direct` | 504-509 | Q = I - S T in T's buffer, `scipy.linalg.solve` overwriting | the only MoM path with a factorization |
| `demag.py` `get_H_ext` | 311-380 | reads `H_ext` up the parents; refuses anything but three finite numbers (363-380) | per-step `H_ext` is refused today (probe 4a) |
| `demag_fft.py` | 298-334, 548-637 | cluster analysis, FFT kernel, FFT matvec | untouched |
| `fem/_solve.py` `Model` | 419-882 | geometry + netgen mesh (422-467), `solve(order)` (813-882) | already a mesh object |
| `fem/_solve.py` `FemSolution` | 304-416 | `getB/getH` (341-360), a `CustomSource` wrapper so sensors and paths work (362-374), `moment` (339) | the FEM result type |
| `fem/_solve.py` `_positions` | 118-137 | sensors (pixels, paths) to global points through a recording `CustomSource` | how observers on paths can be read |
| `fem/_study.py` `fem_study` | 69-186 | one `Model` (155-162), a solve per order (164-168), its own `_estimate` (189-206) | one of the three ladders |
| `convergence.py` `mesh_study` | 230-347 | `mesher(collection, level)` then `apply_demag` hard-wired (329-346); `_study`/`_gci` solver-agnostic (129-227) | the other ladder; face charge has none |
| `meshing.py` `_collection_from_obj_and_cells` | 25-57 | cells get chi and `H_ext` from the part or its parents; `coll.sealed = True` (55-56) | the MoM mesh object and its one attribute |
| `utils.py` to/from JSON | 131-138, 157-166, 210-215, 309-320, 343-353 | exact-class registry; keeps `susceptibility`, `H_ext`, `face_charge_info`, `sealed` | a new attribute costs 4 lines; a new class costs a registration |
| `validation/methods.py` | 36-43, 73-231, 255-258 | `Solved(B, seconds, counts)` and one adapter per method with `settings_for`, `can_solve`, `solve` | an existing one-interface over the three solvers, 258 lines |

## (a) A mesh object that records its method

**What exists.**

- MoM: `mesh_uniform` and the per-shape meshers return a plain
  `magpy.Collection` whose meshed parts are sub-collections with `sealed = True`
  (`meshing.py:55-56`; probe 7a: `[None, True]` for a magnet beside a meshed
  plate). `sealed` is not a magpylib attribute (no hit for "sealed" under
  `magpylib/src/magpylib`); studio reads it with `threejs.is_sealed`
  (`threejs.py:1714-1719`: `isinstance(obj, Collection) and getattr(obj,
  "sealed", False) is True`) and lists the parts (`session.py:2266-2267`); a
  studio document keeps `sealed` and the parts on the create event
  (`session.py:1554-1556, 3034-3039`); `to_json` keeps it (`utils.py:214-215`).
  `apply_demag` takes that collection as it is (`demag.py:1539`), and keeps
  `sealed` on its copy (probe 7c).
- Face charge: no object. `meshed(level)` (`_apply.py:163-173`) builds the
  `Body`/`TriBody` list, adds contact lines (168-169), and `body.mesh(limit)`
  counts unknowns without cutting a triangle mesh that cannot fit
  (`_tribody.py:109-132`). The default-level walk (176-189), `near_panels`
  (192) and `_warn_unaligned` (193) complete the mesh step. Everything from
  line 195 on is the solve. The `Outline`s (151-159) and `boxes` (160) of the
  fixed magnets are part of the mesh: they grade the panels and pick the fine
  rule, which is why a magnet within `NEAR` = 1 % of a part's size
  (`_bodies.py:56`) or within one panel size (`near_panels`, 265) is in the
  matrix or the load rule.
- FEM: `Model` (`fem/_solve.py:419-467`) holds the netgen mesh, the parts,
  `B_ext`, the currents, the observers' refinement; `Model.solve(order)`
  (813-882) is the solve step; `fem_study` already uses them as two steps
  (`fem/_study.py:155-168`). `fem_solve` is `Model(...).solve(order)`
  (946-954). Only the name is private.

**What would be new.**

- Face charge: a holder (working name `FaceChargeMesh`) with the bodies,
  outlines, boxes, `joined` (`_interfaces`, `_apply.py:542-594`, which depends
  on the meshed panels and so belongs to the mesh), the level, whether the
  level was chosen, and the fixed sources' identities. `mesh()` runs
  `_apply.py:142-193`; `solve()` runs 195-320 on the holder. Estimate 150 to
  250 new lines; of the function's 180 lines, about 80 move or change.
  Refinement (`refine`, 219-275) re-meshes the bodies (`bisect` + `mesh` +
  `near_panels`, 273-275), so a refined solve produces a new mesh; the holder
  must say so (the plan's "refine on the first step only" follows from this).
- "Records its method": an attribute on the plain `Collection` next to
  `sealed`, written by the meshers (`meshing.py:56`, one line each) and by the
  face-charge holder, and carried by `to_json` (`utils.py:164-166` pattern, 4
  lines) and by studio's create event (studio side, not this repo). Studio #96
  asks for exactly this ("a body names how it was solved, the kernel registers
  that solve", #96 item 2). A `Collection` subclass instead: `copy()` and
  `getB` accept it (probe 1d-1f), but `to_json` raises `TypeError: Unsupported
  object type` (`utils.py:131-138`) and studio's `type_name` returns None for
  anything but `magpy.Collection` itself (`session.py:2401-2402`); a document
  could not name it. So: attribute, not subclass.
- What `show` needs: a solved face-charge model draws its `ChargeSheet`s
  (`_sheet.py:346-420`, panel grid at 396-419); an unsolved mesh has no magpylib
  object for the panels. A preview sheet with zero charge from `_sheet_of`
  (`_apply.py:738-755`) per body gives the panel grid for free: about 20 lines.
  Studio draws `ChargeSheet` through `_EXTERNAL_TYPES` (`session.py:2345-2356`:
  module, package, and the constructor arguments it reads off an object); a
  field-line pass recognises sheets by module name (`fieldlines.py:429-432`).
- Per-part settings "inherited like susceptibility": the inheritance walk
  exists (`_get_susceptibility_from_hierarchy`, `demag.py:299-308`;
  `meshing_utils.inherited`, used at `meshing.py:32, 50`); a `level` per part
  would reuse it (about 20 lines) but the face-charge mesher takes one `m` per
  body already (`Part.__init__`, `_bodies.py:97-103`), so the solver side is
  ready.

**Risk: medium.** Two things describe one mesh: the user's `Collection` (which
they edit, with no change hook: `position`/`orientation` are plain setters,
`class_BaseGeo.py:380-446`) and the solver's bodies. An edit on the collection
after `mesh()` is invisible to the holder unless the holder fingerprints the
bodies again at `solve()` (#132's `_key`, below) or re-reads the parts. The
plan's "look at it, edit it" means editing the bodies, which have no public
API; the natural editable thing is `level`/`refine`, not panels.

## (b) A common return type for the three solvers

**What each returns today.**

- `apply_demag`: the collection copy with each cell's `polarization` set
  (`demag.py:1756-1760`), or `None` with `inplace=True` (1759-1761; probe 7b).
  `inplace` writes into the caller's cells; `mesh_study` forbids it
  (`convergence.py:309-316`). The copy has no parent, so inherited
  `susceptibility` and `H_ext` are read from the originals (1655-1657, 1672,
  1715, 1719).
- `apply_face_charge`: the collection copy with solved parts replaced by
  `ChargeSheet`s in place among their parents' children (`_apply.py:289-312`),
  parents unsealed up the tree (313-318), `face_charge_info = {unknowns, level,
  rounds, seconds}` (probe 6b: `{'unknowns': 270, 'level': 1, 'rounds': 0,
  'seconds': ...}`), no `sealed` (probe 6c). #132 adds `reused` (its diff).
- `fem_solve`: `FemSolution` with `getB/getH` at positions or sensors
  (`fem/_solve.py:341-360`), `moment` (339), `ndof`, `elements`, `seconds`
  dict; sensors are read through a throw-away `CustomSource` whose
  `field_func` is the solution (362-374), so paths and pixels work. It is not a
  magpylib object: `show` cannot draw it, `to_json` has no tag for it
  (`utils.py:113-130` lists no `CustomSource`), studio cannot hold it.

**What each can and cannot fill.**

| Need | cells (MoM) | sheets (face charge) | FEM |
|---|---|---|---|
| `getB/getH` outside | yes, magpylib | yes, magpylib (`_sheet.py:314-343`) | yes, own kernel (376-402) |
| B inside a part | yes, J is on the cell | no: mu0 H without M (`_sheet.py:57-61`; probe 6d gives Bz = -0.17 T at the plate's centre, 6e `getM` = 0) | yes (`flux_density`, 623-639) |
| `show` / studio | yes (sealed) | yes (`ChargeSheet` registered in studio) | no |
| moment | sum of J x V over cells (not written anywhere today) | `mean_polarization` x volume (`_sheet.py:205-228`; `_joined_means` for joined parts, `_apply.py:597-626`) | `moment` (339) |
| saturation check | J per cell | `mean_polarization` only (one value per part) | B inside |
| `to_json` | yes | yes (`utils.py:197, 288`) | no |
| paths in the result | polarization is a path property (`class_BaseExcitations.py:328`), so cells with one J per step are legal magpylib | `charge_densities` could become a path property through `_field_func_kwargs_ndim` (`field_BH.py:217-231`; `TriangleSheet` does it for `current_densities`, `class_current_TriangleSheet.py:103`); magpylib tiles per observer (198-205), the memory the page notes | `getB(observers)` with the observers on a path, from one solution |

**What would be new.** A record (dataclass) with `sources` (the magpylib
collection: cells or sheets; for FEM the `CustomSource` wrapper, 362-374),
`method`, `info` (unknowns/level/rounds/order/ndof/seconds), `moment`,
`getB/getH` delegating to `magpy.getB(self.sources, ...)`, and `mesh` (the
holder of (a)). Estimate 150 to 250 new lines plus about 30 touched at the
three return sites. If the record is not itself a `Collection`, every example
and test that does `magpy.getB(solved, ...)` or `solved.children` changes; if
it is a `Collection` (subclass), (a)'s registry problem returns. Test work: 6
to 10 functions (one per field, one per method); docs: `quickstart.md`,
`face_charge.md`, `examples/fem_check.md`, `api.md`. `validation/methods.py`
(`Solved`, 36-43) shows the minimum that served the validation suite: B at
sensors, seconds per phase, counts.

**Risk: medium-high (API).** This is the breaking change of the plan.
`apply_demag`'s `inplace=True` has no place in a record; `face_charge_info` on
the collection would move or be duplicated; studio's `_EXTERNAL_TYPES` and
`to_json` keep working only if `sources` stays a plain collection of known
classes.

## (c) Reuse through the object

**Where the factorization is made today.** `_bodies._solve`
(`_bodies.py:613-637`): below 2,000 unknowns, `scipy.linalg.solve(A, b,
overwrite_a=True)` factors A in place and keeps nothing (637); from 2,000 on,
`lu_factor` of a float32 copy (627) refined against A in double (629-633).
Keeping *that* LU means keeping A (8 n^2 bytes) and the float32 LU (4 n^2)
for the refinement loop, 12 n^2 bytes; #132 instead factors in double once
(`_cache.py:126-141`, with `getrf` + `gecon` guards) at 8 n^2 bytes, which is
why its factoring call is "up to 26 % slower" (PR #132 body) and a hit
"differs from a fresh solve by rounding, up to 7e-14" above 2,000 unknowns
(issue #130). Which to keep is a decision: 1.5x memory and bit-identity, or
one double LU.

**Where it would live.** On the mesh holder of (a) (`lu`, `piv`, and the
fingerprint it was made for), made at the first `solve()`; `load_system` +
`lu_solve` on later solves (#132 `_bodies.py:583-601`, `_cache.py:186-189`).
Lifetime: the holder's. Memory: 8 n^2 bytes (18 MB at 1,518 unknowns, 1.7 GB
at 14,454, `_cache.py:61-66`); a cap is then the user's business (one object,
one matrix), which removes #132's LRU, limit, "seen twice" rule, fork reset
and the two public functions (`_cache.py:31-91, 144-183`, about 120 lines).

**Copies.** `Collection.copy()` deep-copies attributes (probe 1b: a plain
array attribute is a different object on the copy; `class_BaseGeo.py:596-607`
calls `deepcopy`). `apply_face_charge` copies its input at `_apply.py:144`
and `apply_demag` at `demag.py:1659`. So wherever the LU hangs off a magpylib
object, each copy doubles it unless the holder defines `__deepcopy__` to share
or drop it (about 15 lines). If the LU is on the *result* and the result is
solved again, it doubles again. Decision needed: the LU on the mesh holder
only, never on the returned sources.

**Edit detection.** #132's `_key` (`_cache.py:94-123`) hashes, per body: type,
chi, `noise`, the panels in the body's own frame (`panels.corners()` or `T` +
`shell`), the pose relative to the first body rounded to 1e-12 of the parts'
size (`ROUND`, 31), the joined interfaces, and the absolute size (99, the x2
scaling miss review found). It reads the *meshed bodies*, so a magnet that
came close enough to regrade the panels misses by itself (`_cache.py:5-7`).
On the holder the same key, computed at `mesh()` and again at `solve()`,
tells an edited mesh from an unedited one; the "parts swap places in the
list" miss (page: "Misses") disappears because the holder fixes the order.
What the key cannot see: an edit of the user's `Collection` after `mesh()`
(a moved part), unless `solve()` re-reads the parts and rebuilds the bodies to
compare; that re-mesh is the 0.7 ms of a 20 ms call at 1,518 unknowns (page,
"What a call still pays"). Estimate: `_key` moves as is (30 lines) plus a
"rebuild and compare" of about 40 lines.

**A plain `solve(model)` loop.** Without a store, each call meshes, assembles
and factors as today (`_apply.py:142-320`). A middle ground that needs no
global state: `solve(model, reuse=previous)` where `previous` is a result
holding its mesh holder; `solve` fingerprints the new bodies against
`previous.mesh` and takes its LU on a hit. That is #132's hit path
(`_cache.py:167-170`) with the user holding the reference: about 40 lines, and
it is what studio #96 needs between drag frames (one object, re-solved).

**Thread safety.** `scipy.linalg.lu_solve` shifts the pivots in place during
the call on scipy 1.15 to 1.18 (#131, with the f2py wrapper quoted); the fix
is `lu_solve((lu, piv.copy()), b)` per call (`_cache.py:186-189`), and the LU
array itself is read-only during `getrs` (#131: "`lu` itself stays
unchanged"). #132 also takes a lock around the store (`_cache.py:152-166,
177-182`); on a single holder the lock is needed only where the kept state is
set or dropped, not around the back-substitution. The face-charge thread pool
is module-global (`_parallel.py:22, 72-76`) and follows
`magpylib.defaults.threads` (37-39; `defaults_classes.py:166-177`), and the
tests already cover forked children (`test_face_charge.py:1163`).

**Estimate.** Re-do #132's split on main's `solve_system` (`_bodies.py:
537-610`; #132's version predates `FAR_RULES` and `flux_by_distance`, so its
78-line diff does not apply): about 60 touched lines. Move `_factor`,
`_back_substitute`, `_key`: about 100 lines new in the holder. `__deepcopy__`:
15. `reuse=`: 40. Tests: #132's four functions (`test_a_kept_factorization_
serves_only_its_own_matrix` over its 14 model sequences, `..._is_safe_from_
threads`, `..._stays_within_its_limit` (drops), `..._needs_a_sound_matrix`)
re-targeted, plus copy semantics and `reuse=`: 5 to 7 functions. Docs: the
"Many solves in a loop" section #132 wrote for `face_charge.md` (28 lines)
rewritten for the object.

**Risk: medium.** The fingerprint's completeness is the known failure mode
(the x2 model read 0.21 T for -0.11 T before the size went in, page: "Wrong
answers"); the copy rule is a new one; mixed precision vs double LU is a
choice with a speed cost either way.

## (d) Paths in the narrow form

**What magpylib's paths impose** (read-only, `magpylib/src/magpylib`).

- Every class declares its path properties: `position`, `orientation`
  (`class_BaseGeo.py:134`), `dimension` for Cuboid/Cylinder/CylinderSegment
  (`class_magnet_Cuboid.py:106`, `..._Cylinder.py:105`, `..._CylinderSegment.
  py:118`), `diameter` (Sphere 95), `vertices` (Tetrahedron 107,
  TriangularMesh 147), `polarization`/`magnetization` (`class_BaseExcitations.
  py:328`), `current` (415), `moment` (Dipole 82). `_get_path_len` is the
  longest of them (`class_BaseGeo.py:207-214`; probe 2d: a 4-row `dimension`
  counts 4). MR's `_has_path` is `_get_path_len() > 1` (`demag.py:152-155`).
- Paths of different lengths are edge-padded to the longest (`_sync_path_
  lengths` docstring, `class_BaseGeo.py:222-232`; probe 2e: a 5-step source
  against an 11-step sensor equals the source padded with its last pose).
  `getB` returns `(sources, path, observers, ..., 3)` squeezed
  (`field_BH.py:598` docstring; probe 2a: 5 steps x 7 points -> (5, 7, 3)).
  A `CustomSource` on a path gets all steps x observers at once, flattened in
  its own frame (probe 3a: (35, 3) for 5 x 7; `field_BH.py:195-205, 266-271`),
  with per-step keyword arrays tiled per pixel when the class declares them in
  `_field_func_kwargs_ndim` (217-231).
- So a step is a joint index over every object's path, and a (steps, points,
  3) field of the fixed sources at the load points is one `magpy.getB` call
  with `fixed` on their paths: the `external(points)` closure at `_apply.py:
  199-217` already makes that call (with `sumup=True`), today for one step.

**The "same matrix" test.** `_split` refuses all paths (`_apply.py:330-332`;
probe 5a). The narrow test reads, per step: the solved parts' shapes (a
`dimension`/`vertices`/`diameter` path on a solved part: refuse), their poses
relative to each other (`_relative`, `_bodies.py:278-281`, over the steps:
refuse if any changes beyond `ROUND`), and the fixed cuboids' distance to the
solved parts' faces at every step against the two thresholds the code has:
`near_tol` (`NEAR` = 1 % of the smaller part, `_bodies.py:56, 69-72`, which
decides the contact lines in `add_contact_lines`, 293-331) and the fine load
rule within one panel size (`near_panels`, 265-275, `reach=1.0`). The page's
"1 % of the parts' size" is `NEAR`; the load rule is the second threshold,
and the frozen-layout probe shows what each costs when ignored: 2.0 % of B
at 0.1 mm off (load rule), 5.4 % at 0.01 mm off (panels) (brief section 2;
`PROGRESS.md` table under `c7_frozen_layout.py`). A magnet's own `polarization`
path, `H_ext` per step and sensor paths pass. Estimate: 80 to 120 lines, with
the refusal message naming the part and the count of distinct matrices (the
page's refuse node) another 30.

**Loads per step.** `load_system` (#132 `_bodies.py:583-601`) calls
`a.load(field)` once per body; `Galerkin._load` (`_panels.py:291-305`) reshapes
`field(pts)` to `(nq^2, N, 3)` and contracts with the weights; `TriBody._load`
(`_tribody.py:281-285`) likewise. With `field` returning `(steps, points, 3)`
both gain one leading axis: about 10 lines each; the fine rule (`fine`, 16
points a side on cuboid panels, 64 x 7 on triangles, `_panels.py:279-289`,
`_tribody.py:269-279`) stays per panel, fixed at step 0 (the page's
"load rule, finer near a magnet" caveat). Chunking: `kept.py:28-30, 117-138`
takes 100 steps a call (`STEPS_PER_CALL`), because the loads of 10,000 steps
at once took 4.5 GB and magpylib peaked at 3.5 GB on one 200-step call
(brief; `kept.py:28-30`). The remanence load (`load_remanence() / chi`,
#132 `_bodies.py:596-597`; `Body:178-180`, `TriBody:287-290`) is per body and
constant; a `polarization` path on a solved part makes it `(steps, n_dofs)`:
about 15 lines. `H_ext` per step: `get_H_ext` refuses `(steps, 3)` today
(`demag.py:363-380`; probe 4a) and `Part.applied` is one 3-vector
(`_bodies.py:98`) added inside the per-body `field` closure (#132 `_bodies.
py:592-593`); relaxing `_checked_H_ext` to `(steps, 3)` and carrying
`applied` as `(steps, 3)` is about 30 lines, with the ownership walk
(`demag.py:320-342`) unchanged. `_same_material` compares `applied` between
joined parts (`_apply.py:629-638`); per step that comparison must hold at
every step.

**The solve.** `lu_solve` takes a right-hand side `(n, k)`: `kept.py:140-142`
solves all steps of a chunk in one call. Three lines.

**B at observers.** The prototype returns B, not sheets: `kept.py:144-161`
reads each observer with its own step's unknowns through `charge_field_rows`
(164-209), a per-row copy of `_triangle.charge_field`, for cuboid bodies
only, without #134's far rule (`_sheet.py:324-331`, `charge_field_far`) and
without `TriBody` reads. The alternative the page rejected (sheets with a
`charge_densities` path) is legal magpylib through `_field_func_kwargs_ndim`
but tiles the charges per observer (`field_BH.py:198-205`): 243 MB at 10,000
steps and 24 GB for a 100-point map (page, "What a path returns"). Observers
on paths: `fem/_solve.py:118-137` shows how sensors with pixels and paths
become global points (a recording `CustomSource`); it drops the step axis
(returns one concatenated array), which a per-step read needs back: about 20
lines. Estimate for the read: 120 to 180 lines (rows kernel for rectangles and
triangles, far rule per row, sensor frames back through magpylib's rotation
as `fem/_solve.py:351-353` does by handing a `CustomSource` to `magpy.getB`).

**MoM.** The loads are `pol_magnets` (`demag.py:1703-1711`), the currents'
field at the cell centroids (`magpy.getB(currents_list, pos, sumup=True)`,
1733; currents on a path give `(steps, n, 3)` natively) and `H_ext`
(1719-1724, refused per step). `_solve_direct` (504-509) would take a
`(3n, steps)` right-hand side in one `scipy.linalg.solve` call but keeps no
LU; keeping one is `lu_factor`/`lu_solve` in its place (10 lines) for
`solver="direct"` only. GMRES (`_solve_iterative`, 1329-1372) is one right-hand
side at a time with a warm start at `rhs` (1360); the operator `ops`
(`_build_operator`, 1137-1245) can be kept across steps, each step still a
full iterative solve (the page's node "core", "Not for"). The result could be
the cells with a `polarization` path (`class_BaseExcitations.py:328`), written
at 1756-1757 as a `(steps, 3)` per cell. But: the fixed magnets are cells in
the matrix (`magnets_list` at 1671 includes chi = 0 cells; T over all of them,
1462, 1473-1479; their rows of Q are identity rows since S = 0), so a moving
magnet changes T. To make MoM's narrow form useful the chi = 0 cells must
leave the unknowns and enter the load as the currents do (1728-1735): for
`interaction="point"` that is the same field (`_point_block_T` is `magpy.getH`
of the cells, 617-641); for `"newell"` the load would need the cell-averaged
field, which `magpy.getB` at a centroid is not. About 40 to 60 lines in
`apply_demag` and `_solve_cells`, and a behaviour change worth its own PR.

**Estimate for (d).** Face charge: narrow test 80-120, refusal 30, loads
per step 40, `H_ext`/`J0` per step 45, multi-RHS 3, read 120-180, chunking
and the path/observer bookkeeping 60: about 350 to 500 new lines, 60 touched.
MoM direct: 60 to 100 more, after the fixed-magnet restructure. Tests: 8 to 12
functions (accept/refuse per rule, equality to a loop of single solves to the
0.7 nT the prototype reached, `H_ext` and polarization per step, a chunk
boundary, sensors with pixels on paths, `refine` refused past step 0). Docs:
a page section with the rules and the refusal.

**Risk: high.** Memory per chunk; the test's thresholds are the solver's
constants, so a change to `NEAR` or `reach` silently changes what is accepted;
the read kernel duplicates `charge_field` (a second copy of a validated
kernel); `refine` and the fine rule are frozen at step 0 by construction.

## (e) One `study()` over the three ladders

**What exists.** `mesh_study` (`convergence.py:230-347`) takes `mesher(
collection, level)` (default `mesh_uniform`, 319-320) and hard-wires
`apply_demag(meshed, **demag_kwargs)` (342) and `evaluate(solved)` as
`magpy.getB(solved, quantity)` (321-327); it counts the cells that respond
(332-334, `_responds`, `meshing.py:860-870`). The estimate is solver-agnostic:
`_study(cells, values, seconds)` (129-160) and `_gci` (194-227, Richardson
over the three finest with h = (volume / cells)^(1/3), `PLAUSIBLE_ORDERS`
0.3 to 3 in h, `SAFETY_FACTOR` 1.25), exposed as `MeshStudy.from_values`
(100-126). `fem_study` (`fem/_study.py:69-186`) makes one `Model` and solves
per order (155-168), with its own `_estimate` (189-206: the last change as the
error when the changes shrink; no extrapolation). `compare` (406-457) puts two
studies side by side by their last values and per-entry errors. Face charge
has no study; the only face-charge ladder in the repo is the validation
adapter `FaceCharge.settings = (1, 2, 3, 4, 5)` solving `level=setting`
(`validation/methods.py:166-231`).

**What would be new.** A dispatcher `study(model, quantity, rungs, method=
...)` that maps a rung to a solve (cells_across -> `mesh_uniform` +
`apply_demag`; level -> `apply_face_charge(level=)`; order -> `Model.solve`),
reads the count axis (cells / unknowns / ndof), and picks the estimator:
`_gci` for h-refinement, `_estimate` for p-refinement. Face charge's unknowns
grow with the square of the level (`_apply.py:97-98`) on a surface, so the
cell-size exponent of `_gci` (volume / cells)^(1/3) is not its h; a surface
h = (area / panels)^(1/2) or the level itself needs choosing and checking
against the validation ladders (`validation/`, cases with FEM references).
Estimate 150 to 250 new lines reusing `_study`, `_gci`, `_estimate`, and the
`METHODS` adapter shape of `validation/methods.py:255-258`; `mesh_study` and
`fem_study` become thin calls or deprecations (40 touched lines). Tests: 4 to 6
functions (one ladder each, the estimator choice, the refusal of a method
that cannot solve the model). Docs: `examples/mesh_convergence.md`,
`examples/fem_check.md`, `api.md`.

**The "same layout" need.** The page's difference studies (plate length, NMR
pole pieces) need one layout at every rung; face charge's layout follows the
length (brief section 2: slope 3.7 vs 3.9 mT/mm) and MoM's `mesh_uniform`
re-meshes per level. A fixed-layout option is a mesher feature (stretch one
layout with the part), not a study feature; nothing in the code does it.

**Risk: low to medium.** Code is small; the estimator semantics for face
charge are unproven.

## (f) "auto" and its checks

What the model can tell at `mesh()`/`solve()` time, with the seam for each:

- Anisotropic chi: `_split` (`_apply.py:353-359`) raises today; auto routes to
  cells instead: 5 lines, plus the inheritance walk already used.
- An observer inside a part: only when sensors are in the model
  (`Collection.sensors_all`, `class_Collection.py:221`; probe 7d). Point-in-
  solid tests exist without the FEM extra: `meshing_utils.mask_inside_Cuboid /
  _Cylinder / _Sphere / _CylinderSegment` (`meshing_utils.py:176-240`, in the
  part's own frame), and `_tribody._winding` (`_tribody.py:383-395`) for closed
  triangle meshes and tetrahedra (`_closed_surface`, `_apply.py:398-437`, gives
  the faces). Sensor positions with pixels and paths come from the recording
  trick at `fem/_solve.py:118-137`. About 40 to 60 lines. Without sensors the
  solve cannot know; the page's fallback, a sheet that warns when read inside
  its body, is a check in `ChargeSheet._field` (`_sheet.py:314-343`) against
  the sheet's own closed surface (`_winding` again): about 20 lines and one
  winding-number pass per `getB` call on the near observers.
- Unknowns at level 1: `meshed(1)` (`_apply.py:163-173`) counts `n_dofs`
  before any assembly; triangle meshes that cannot fit are counted, not cut
  (`_tribody.py:118-132`, `fewest_triangles`); `test_too_many_unknowns_raise_
  before_assembly` (`test_face_charge.py:1096`) covers the path. The default-
  level walk (176-189) is the same logic in reverse. About 20 lines to expose
  a count.
- `max_unknowns` and the refusal text: `_apply.py:53, 229-238`.
- Name clash: `apply_demag` already has `solver="auto"` meaning direct vs
  iterative (`demag.py:422, 487-501, 1463-1470`); the log line "solver='auto'
  chose ..." (1466-1470) is the pattern for "picks a method, says why".

Estimate: 60 to 100 lines, 4 to 6 tests, one docs section. **Risk: low in
code, medium in meaning**: a silent choice between methods that disagree by
1 to 5 % near parts (brief, B inside; `_warn_unaligned` cases) is a
documentation and warning problem more than a code one.

## What stays untouched

- Kernels: `_rectangle.py` (rectangle fields, far variants), `_triangle.py`
  (`charge_field`, `charge_field_far`, `far_points`, `normal_field*`),
  `newell.py`, `demag_fft.build_fft_kernel` and `demag_fft_matvec`
  (`demag_fft.py:548-637`).
- #133's far pairs: `FAR_RULES`, `FAR_PAIRS`, `FAR_CHUNKS`, `flux_by_distance`
  (`_panels.py:36-118`), `cross_block` (`_bodies.py:357-465`),
  `Galerkin.assemble` (`_panels.py:249-277`), `TriBody.assemble`
  (`_tribody.py:183-218`).
- #134's far field: `ChargeSheet._field`'s far branch and `FAR_SIDES`
  (`_sheet.py:27-29, 327-342`). (A per-row read for paths would copy, not
  change, it.)
- Meshers: `meshing.py` (except one attribute line per mesher), `_mesh.py`
  (graded cells), `_tribody.py`'s grading and contact lines, `_apply.py`'s
  `_closed_surface`, `_turned`, `_sphere`, `_interfaces`, `_warn_unaligned`.
- MoM machinery: cluster analysis (`demag_fft.analyze_structure`), T assembly,
  the GMRES operator, preconditioner and fallback (`demag.py:548-1372`), except
  the fixed-magnet restructure if MoM paths are wanted.
- FEM internals: `Model._mesh`, `locate`, `_two_potentials`, `solve`
  (`fem/_solve.py:499-882`); only a public name and the study hook change.
- The validation harness (`validation/`), which keeps calling the three
  functions, is the regression net for all of the above.

## Sizes and risks at a glance (estimates)

| Piece | New lines | Touched lines | Tests (functions) | Docs | Risk |
|---|---|---|---|---|---|
| (a) mesh object, method attribute | 150-250 (face charge holder) + 30 | 80 (`_apply.py`), 10 (meshers, JSON), 20 (FEM name) | 8-12 | `face_charge.md`, `quickstart.md`, `api.md` | medium |
| (b) common return | 150-250 | 30 at three return sites; every example that reads the collection | 6-10 | 4 pages | medium-high (API) |
| (c) reuse on the object | 150 (factor, pivots, key, deepcopy, `reuse=`) | 60 (re-do #132's split on main) | 5-7 (from #132's 4) | the #132 section rewritten | medium |
| (d) paths, narrow form | 350-500 (+60-100 MoM direct after the fixed-magnet restructure) | 60 | 8-12 | a page section | high |
| (e) one `study()` | 150-250 | 40 | 4-6 | 3 pages | low-medium |
| (f) auto and checks | 60-100 | 10 | 4-6 | a section | low (code), medium (meaning) |
| Total | about 1,000-1,500 | 300-450 | 35-55 | 4-5 pages | |

For scale: `_apply.py` is 831 lines, `_bodies.py` 637, `demag.py` 1,761,
`fem/_solve.py` 954, `convergence.py` 457; `test_face_charge.py` has 44 test
functions, `test_basic.py` 19, `test_convergence.py` 10, `test_fem.py` 11
(`grep -c "def test_"`); #132 adds 4 test functions and 189 lines of
`_cache.py`.

## Facts that bear on the ordering

- The private core (#132's `assemble_system` / `load_system` /
  `split_unknowns`) is needed by (c) and (d) alike and by nothing else;
  it is the first thing to land, on main, and it is 60 lines of moving code.
- (a) for face charge is the holder that (c) hangs the LU on and (d) runs the
  steps on; (c) without (a) is #132 again (a store keyed by fingerprint).
- (d) is independent of (b) only if it returns B at observers (as the page
  decided); returning sheets with per-step charges would couple it to (b).
- MoM reuse covers `solver="direct"` only (`_solve_direct`, `demag.py:
  504-509`); `solver="auto"` picks GMRES whenever FFT covers a tenth of the
  matrix (`AUTO_MIN_FFT_SHARE`, 143, 487-501), which is every uniform grid, so
  a kept LU serves MoM only when the user names `direct`. #81's saturation
  loop needs T kept across chi changes (its plan: "interaction matrix built
  once"), which is a kept *T* (`_assemble_T_dense`, 995-1052), not a kept LU of
  Q (`_subtract_susceptibility_product` folds chi into Q, 268-287); the two
  are different objects to keep.
- #71 (`UniformField` replacing `H_ext`) would move the per-step applied field
  into a source on a path, which magpylib handles natively; (d)'s `H_ext`
  per-step code (about 30 lines) would then be thrown away.
- #103 (hierarchical matrices) replaces the dense A above `max_unknowns` with
  blocks solved iteratively: no LU to keep, so (c) and (d) apply below the
  threshold only; the holder design should not assume a dense LU exists.
- #118 (anisotropic face charge) doubles the unknowns of an anisotropic body
  and changes the "auto" rule of (f) (anisotropy would no longer force cells).
- #120 (thin shells) changes the mesh, not the seams here.
- #84 (progress hooks) wants hooks inside the solve loops that (a)/(c) touch;
  the holder is the natural place for a callback.
- studio #96 wants: a part that names how it was solved (the (a) attribute),
  a solver hook the kernel registers, and a re-solve per drag frame with the
  previous solve kept (the (c) `reuse=` form, one object).

## Open questions I could not settle from the code

1. Which factorization to keep above 2,000 unknowns: the float32 LU plus A
   for refinement (12 n^2 bytes, bit-identical to today) or a double LU
   (8 n^2 bytes, #132's choice, slower factor call, 7e-14 off). The page's
   times are for the double LU.
2. Whether the LU lives on the mesh holder only or also travels with the
   returned sources; `copy()` doubles whatever it is on (probe 1b).
3. Whether "edit it" on the face-charge mesh means anything beyond `level`
   and `refine`; the bodies have no public API and no change hook.
4. For MoM paths: whether to take the fixed magnets out of the unknowns
   (changes `apply_demag` for `interaction="newell"`), or to leave MoM out of
   the narrow form.
5. Whether `solve()` returns a record or a `Collection`; both break something
   (examples and `getB(solved, ...)` vs `to_json` and studio registries).
6. The h-exponent for face-charge levels in a Richardson estimate; nothing in
   the code or the validation results fixes it.
7. How studio would store the method attribute and a `reuse=` handle across
   drag frames (studio side; `session.py:3034-3039` is where `sealed` is read
   on a create event).
8. Not verified: whether a `Collection` subclass is drawn by studio's
   *document* path (the importer accepts it by `isinstance`, `importer.py:
   65-69`; `type_name` returns None, `session.py:2399-2400`); not probed in a
   real studio session. (The `type_name` check for `Collection` is at
   `session.py:2401-2402`.)

## Sources

- Code as cited (`path:line`), main at 57c7860; `git show
  origin/face-charge-kept-factorization:src/.../_cache.py` and `_bodies.py`;
  `git merge-base main origin/face-charge-kept-factorization` = 2911060; `git
  diff --stat 2911060 origin/face-charge-kept-factorization` (78 lines in
  `_bodies.py`, 189 in `_cache.py`, 268 in `tests/test_face_charge.py`).
- `git show origin/explore/use-cases:explorations/use-cases/kept.py` and
  `PROGRESS.md`.
- Issues #130, #131, #71, #81, #103, #84, #79, #118, #120, #92, #85, #60; PR
  #132 body and its timing comment; studio #96 (`gh issue view`, `gh pr view`).
- `ground/probe_seams.py` -> `ground/probe_seams.out` (magpylib 6.1.0, one
  thread, models of 270 unknowns and 2-cell meshes): results 1a-1f, 2a-2e,
  3a, 4a, 5a, 6a-6f, 7a-7d quoted above.
- Numbers in mT, ms, GB and unknowns come from the brief and the page, not
  from any run here.
