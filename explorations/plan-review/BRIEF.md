# Brief: review of the plan for one entry point over the solvers of magpylib-material-response

Date: 31 October 2026. The owner, Alex (maintainer of magpylib-material-response and of magpylib's display code), asked: "ok lets review the plan properly". The review's output is a design proposal he will read and decide on, point by point. He wants plain words, numbers with units, and every claim backed by a source. He is sceptical: on 31 October he asked "tolerances on magnets I can do with magpylib alone, what is the benefit of all this? how do I even know it is accurate?". His standing condition from 29 October: the plan "must be backed by real use cases before any design issue or code".

## 1. The plan under review

magpylib-material-response (MR) adds soft-magnetic response to magpylib models. It has three solvers today, each its own function with its own inputs and outputs:

| Solver | Function | Takes | Returns |
|---|---|---|---|
| Method of moments (MoM) on volume cells | `demag.apply_demag(collection, solver="auto", interaction="point", inplace=False, ...)` | A collection already meshed into cells (`meshing.mesh_uniform`, `mesh_all`, `mesh_Cuboid`, `voxelize`, ...) | A copy of the collection with each cell's polarization set (or None when `inplace=True`) |
| Face charge (surface unknowns of linear, uniform, isotropic parts) | `face_charge.apply_face_charge(collection, level=None, refine=0, max_unknowns=20_000)` | The unmeshed model; it meshes the surfaces itself into graded panels | A copy of the collection with each solved part replaced by `ChargeSheet` sources, plus `face_charge_info` (unknowns, level, rounds, seconds, and since #132 `reused`) |
| Finite elements (FEM, NGSolve, the `[fem]` extra, for verification) | `fem.fem_solve(collection, order=3, maxh=None, edge_layers=4, observers=None, ...)` | The unmeshed model | A `FemSolution` object with `.getB(observers)`, `.getH`, `.moment`, `.ndof`, `.seconds` |

Studies: `convergence.mesh_study` (MoM, a ladder of `cells_across`), `fem.fem_study` (a ladder of orders), `convergence.compare`. No study exists for face charge. All three solvers refuse magpylib paths (an object with several positions/orientations/polarizations). `H_ext`, the applied field, is an attribute on objects (issue #71 proposes replacing it by a source).

**The plan** (page "apply_response dispatch", artifact https://claude.ai/artifact/KTQDByzmN2czMmRw9XYUjY, full text in `page_text.md` and the map's node descriptions and 16 use-case tiles in `page_map.txt`, both next to this brief):

- One entry point over the three solvers, in two steps, the first optional:
  ```python
  meshed = mesh(model, method="face_charge", level=3)   # look at it, edit it
  solved = solve(meshed, refine=1)
  solved = solve(model, method="face_charge", level=3, refine=1)  # the same in one call
  solved = solve(model)             # picks a method, says why
  solved = solve(model_with_paths)  # one factorization, every step (narrow form)
  ```
- The mesh object records which method made it, so `solve` never guesses whether a model is meshed; it is the unit of reuse. Mesh settings (cells_across, level, maxh) sit on `mesh`, solve settings (solver/interaction, refine, order) on `solve`. Per-part settings could be inherited from parent collections like `susceptibility` is.
- `"auto"`: face charge unless something makes it wrong or too big: anisotropic χ → cells; an observer inside a part → cells (B inside from face charge is μ0(1+χ)H, error ×(1+χ)); more than `max_unknowns` at level 1 → cells. FEM only when named.
- A single `study()` over each method's ladder (cells_across / level / order) with an error estimate, replacing `mesh_study`/`fem_study`.
- Paths in the "narrow form": accept a path only if the matrix is the same at every step (solved parts keep their shapes, their relative poses, and no fixed cuboid within the contact distance moves along a solved part). Sources, sensors, currents, `H_ext` (per step, shape (steps, 3)) and a solved part's own polarization may vary. Refuse other paths with a message naming the part, how many matrices it would take, suggesting a loop. Return B at observers passed to the solve, not charges per step. `refine` on the first step only. Iterative MoM named in the refusal.
- Private core: factor once, back-substitute every step; loads in chunks of ~100 steps.
- Method names recommended (not confirmed by Alex): `"mom"`, `"face_charge"`, `"fem"`, `"auto"`.

**Decided on 30 October** ("ok" from Alex): a cache that keeps the factorization between calls inside the solvers first (no new API), then paths in the narrow form on the same core. The cache was built as PR #132 (branch `face-charge-kept-factorization`, closes #130).

**Reversed on 31 October:** #132 is postponed, unmerged, after this reasoning: in the rewrite the mesh object would hold the factorization itself, so the global store (an LRU of LU factorizations keyed by a fingerprint of the meshed bodies, capped at 1 GB, with eviction, fork handling and public `set_cache_limit`/`clear_cache`) is the riskiest part and may be unnecessary; a key that misses an input returns silently wrong fields (review found one: a model scaled ×2 hit the wrong matrix); reusing the object could be faster than the cache (the cache still re-meshes and rebuilds sheets, 21 ms a call at 1,518 unknowns, against 6 ms on one thread / 1.8 ms on 8 for the path prototype); the public API might need deprecation; hierarchical matrices (#103) have no dense factorization to keep. Parts of #132 worth keeping in a rewrite: the split of `solve_system` into `assemble_system`/`load_system`/`split_unknowns`, the singular/ill-conditioned guards at factor time, copying pivots per `lu_solve` call (scipy bug, MR #131), the fingerprint for detecting an edited mesh, the 14 model sequences checked against a fresh solve.

So the plan as it stands is out of date: its last review put the cache first, assumed the cache is the base for everything else, and the plan has no design issue yet.

## 2. What was measured (M4 Pro, magpylib 6.1.0, MR 0.7.0/0.7.1-dev; sources on branch `origin/explore/use-cases`, folder `explorations/use-cases`, see its `PROGRESS.md` and `results/`)

- One `apply_face_charge` call of the tolerance Monte Carlo (plate 10×10×2 mm χ=1000 under a 5 mm cube magnet, sensor in the gap, 1,518 unknowns): 74 ms = assembly 43 + factorization 13 + load 9 + mesh/copies/sheets 9.
- Kept factorization (#132) from the third call: 768 unknowns 44→13 ms; 1,518: 77→21; 2,520: 142→30; 4,104: 355→53; 5,670: 680→62; 7,488: 1,173→92; 9,558: 1,876→121 ms (15.5×). Factoring call up to 26 % slower. Shaft 48 configurations 7.5→2.5 s; shaft inside a soft holder (two solved parts moving against each other) 160→164 s (each matrix comes back once).
- 300-call loops at 1,518 unknowns, cache off → on: only the magnet varies 22.4→6.3 s (298 of 300 reused); 10 plate lengths × 30 magnet samples 23.3→7.2 s; 40 lengths shuffled 23.6→12.0 s; 100 lengths shuffled 28.1→19.5 s; a new length every sample 74.2→74.6 ms a call (0.6 % overhead).
- Path prototype (`kept.py`, cuboids that do not touch, builds no sheets, returns B at observers): 10,000 samples 6.1 ms/sample on one thread (65 s), 1.8 ms on 8 threads (20 s), loop with kept LU 8.2 ms (84 s), today 75 ms (755 s). Matches `apply_face_charge` to 0.7 nT on fields of 47–173 mT. Loads for all 10,000 steps at once: 4.5 GB, so chunks.
- Accuracy of the plate model (Bz at the sensor and its slope per mm of magnet shift): FEM maxh 0.5 mm order 3: 105.05 mT, 218.49 mT/mm (57 s); FEM maxh 1.0: 104.86, 217.72 (25 s); face charge level 3: 104.95, 218.55; level 5: 105.05, 218.32; one solve kept and the magnet moved afterwards in magpylib: 104.95, 145.71; magpylib alone, no plate: 52.23, 145.71. Linear models only; no measurement exists.
- Frozen layout (keeping the first step's panels/load rule along a path): a magnet sliding 1 mm off the plate: error 0.00002 mT; 0.1 mm off: 0.59 mT (2.0 % of B), from the load rule; 0.01 mm off: 1.4 mT (5.4 %), from the panels; a solved part's own polarization kept from step 0: wrong by 220 mT, right with one load per step.
- Face charge on a changing dimension: its panel layout moves with the length, so the slope dB/dL is wrong (3.7 vs MoM's 3.9 mT/mm, and the trend over L is missed); MoM on one stretched layout gives the same slope coarse and fine. A difference needs the same layout.
- B inside a soft part from face charge at χ=1000, 0.3 mm below the surface: 952, 91, 359 mT at levels 3, 4, 5 where cells give 442–446 mT. Face charge cannot give B inside.
- `apply_demag`'s default on uniform grids is GMRES with FFT products: no factorization. Only `solver="direct"` factors.
- Coils: the response is linear in the currents; one solve per coil then weighted sums, exact to 2e-14; written by hand today.
- NMR pole pieces (#85): the verdict between two designs flips with level (+9.5, −2.5, −5.6 ppm at levels 1–3); each design has its own layout.
- Use-case survey: 9 kinds of use from the repos and 13 references (page section "What the use cases say"): fields are read outside the soft parts in all but saturation checks; many configurations is the norm; most change only the right-hand side; differences matter more than absolute values; optimizers are gradient-free; saturation is checked more than solved (#81).
- Studio (magpylib-studio) does not re-solve on drag yet (studio #96); a solve per frame at levels 2–5 is 46/76/141/377 ms today.

## 3. The questions the review must answer

0. **Do the rewrite at all?** What do users gain over today's three functions? If mostly "tidier", does it justify breaking changes? Consider the alternatives: keep three functions and add only what the use cases need (e.g. a common return type, paths, a face-charge study, coil sums); or do nothing now and build on demand.
1. **What users gain**, concretely, per use case: name the use case, what is written by hand today, what the proposal saves. Alex's question: "what is the benefit?"
2. **What `solve()` returns.** Today three different things. One return type must serve getB/getH, show (magpylib and studio draw the result), B inside (cells only), moment, saturation checks, serialization (to_json/from_json), paths. Sketch it. Say what each solver can and cannot fill.
3. **How reuse works.** The mesh object holds the factorization and spots edits (fingerprint). What happens when a user moves a magnet on that object, edits a cell, changes χ? What does a plain `solve(model)` loop get without #132's hidden store? Is there a middle ground (e.g. `solve` accepts a previous solution)? Thread safety (scipy #131: `lu_solve` shifts pivots in place; copy per call).
4. **MoM's default solver** has no factorization. Should reuse cover MoM at all, and how (direct only; keep the operator; saturation iteration in #81 needs the matrix kept across iterations)?
5. **What `"auto"` can decide.** It cannot know at solve time whether an observer will be inside a part unless sensors are in the model. Is a method chosen silently acceptable? Alternatives: no auto; auto that explains; default to face charge with a warning when read inside.
6. **The old functions and names.** Do `apply_demag`, `apply_face_charge`, `fem_solve` stay as thin wrappers, get deprecated, or stay the real API with `solve` on top? Names `"mom"`, `"face_charge"`, `"fem"`, `"auto"`. Naming of `mesh`/`solve` themselves (collisions with magpylib? with meshing module?).
7. **Order.** Which pieces ship first; whether the narrow form of paths is still needed once the mesh object exists; where #132's reusable parts go; how it interacts with planned work: saturation (#81), hierarchical matrices (#103), meshing per body (#79), UniformField replacing H_ext (#71), anisotropic face charge (#118), thin sheets (#120), progress hooks (#84), studio re-solve on drag (studio #96).

## 4. Where things are (read-only unless said otherwise)

- Repo root: `/Users/alexandreboisselet/Documents/GitRepos.nosync/magpylib-material-response` on `main` at 57c7860 (today's #133 Gauss points for far panel pairs, #134 far-field getB of sheets merged). Package: `src/magpylib_material_response/` (`demag.py`, `demag_fft.py`, `meshing.py`, `convergence.py`, `face_charge/` with `_apply.py`, `_bodies.py`, `_mesh.py`, `_sheet.py`, `_tribody.py`, `_panels.py`, `_parallel.py`; `fem/` with `_solve.py`, `_study.py`, `_geometry.py`). Docs: `docs/` (quickstart.md, how_it_works.md, method.md, face_charge.md, api.md, examples/). Validation harness: `validation/` (`methods.py` wraps the three solvers behind one `solve(collection, sensors, setting)` interface, worth reading as an existing unification). Tests: `tests/`. Changelog: `CHANGELOG.md`.
- Python: `/Users/alexandreboisselet/Documents/GitRepos.nosync/magpylib-material-response/.venv/bin/python` (magpylib 6.1.0, scipy 1.18.0, numpy 2.5.1, ngsolve installed). The package is installed editable from the root checkout.
- Branches (read with `git show origin/<branch>:<path>` or `git ls-tree -r --name-only origin/<branch>`; do NOT check them out, the root checkout is shared with other sessions): `origin/explore/use-cases` (e9e88b6: `explorations/use-cases/PROGRESS.md`, `common.py`, `kept.py`, `cached.py`, `p_model.py`, `c1`–`c8`, `p1_tolerance_kept.py`, `results/*.json`); `origin/face-charge-kept-factorization` (b43d600, PR #132: `src/magpylib_material_response/face_charge/_cache.py`, the split `_bodies.py`, tests, `docs/face_charge.md` section "Many solves in a loop").
- magpylib source (for conventions: Collection, paths, getB, sealed, defaults.threads, CustomSource): `/Users/alexandreboisselet/Documents/GitRepos.nosync/magpylib/src/magpylib` (read-only, another project). magpylib-studio: `/Users/alexandreboisselet/Documents/GitRepos.nosync/magpylib-studio` (read-only).
- Issues: `gh issue view N -R magpylib/magpylib-material-response` for 130, 131, 120, 118, 103, 92, 85, 84, 81, 79, 76, 71, 60; `gh issue view 96 -R magpylib/magpylib-studio`; `gh pr view 132 -R magpylib/magpylib-material-response`.
- This folder (`/private/tmp/claude-501/-Users-alexandreboisselet-Documents-GitRepos-nosync-magpylib/d931a5f5-3e14-496c-bad9-7cbff61b8689/scratchpad/plan-review/`): `BRIEF.md` (this), `page_text.md`, `page_map.txt`; write your outputs into the subfolder named in your task (`ground/`, `designs/`, `judge/`, `verify/`, `final/`), creating it if needed.

## 5. Rules for every agent

- The repo and the other repos are read-only. No `git checkout`, `git stash`, `git commit`, no file edits under any repo, no installs. Work in the scratchpad folder above only.
- No timing or memory measurements of any kind, and no claims about speed beyond the numbers in this brief (other sessions share the machine; a performance lock protocol exists and you are not part of it). Probes are for correctness and API behaviour only: small models (≤ 3,000 face-charge unknowns, ≤ 5,000 cells, FEM only with tiny models if at all), each probe under about 30 s, run as `nice -n 19 <python> -I script.py` from the scratchpad, one at a time. Set `magpylib.defaults.threads = 1` at the top of a probe. If a probe would be large, skip it and say so.
- Verify before you assert. Cite the source of every claim: `path:line`, an issue number, a probe script and its printed output, a reference. Mark what you could not verify as unverified. Do not take the page's claims on trust where the code can be checked.
- Write in plain words, as to an engineer who has not read the code: short sentences, no code jargon in headlines, numbers with units, no filler. Alex reads nothing long: lead with the conclusion.
- Alternatives are real options, including "do not do this". Say what each costs and what it breaks.
- Your final text is returned to a script, not to a person: when a structured output is asked for, fill every field; put long material into the file named in your task and return its path.
