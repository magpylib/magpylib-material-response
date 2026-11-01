# The solve entry point, reviewed

> Decided 2026-11-01: yes to all, tracked in the issue. Corrections since: #132's split did rebase cleanly (step 1 went into #135 with step 2); the factoring call is up to about twice a plain solve above 2,000 unknowns, not 26 %, after #133; `reuse=` measured 72 to 18 ms at 1,518 unknowns, and a poor match costs time except in a window about 1e-12 of the parts' size around a contact threshold.

31 October 2026. For Alex. Built on five fact sheets written today from the code on `main` (57c7860), the measured scripts on `explore/use-cases`, the open issues, studio's source and the documentation of seven comparable solver libraries (`ground/api.md`, `uses.md`, `roadmap.md`, `comparables.md`, `seams.md`). The designs, judges and adversarial checks of the planned review did not run (usage budget); this is one pass by me over the sheets, so treat the recommendations as argued, not independently verified.

## The conclusion

Do not do the rewrite. Everything that was measured to help users comes from one capability, reusing a solved model's matrix when only the sources change, and that fits in today's `apply_face_charge` as one keyword, `reuse=`. The rest of the plan, `mesh()`/`solve()`, one return type, `"auto"`, one `study()`, has no measured gain, would break 27 documented calls and studio's two result shapes, and costs 1,000 to 1,500 lines against about 250. Paths come later, only if someone runs 100,000 samples. #132 is superseded: its core is kept, its store is dropped.

## Decisions for Alex

| # | Question | Recommendation | Why, in one line |
|---|---|---|---|
| 0 | Do the rewrite? | **No.** Keep the three functions. | The measured gains (below) need none of `mesh()`/`solve()`, a common return type or `auto`; those pieces have no number behind them (`uses.md`, conclusion). |
| 1 | What do users gain, then? | A tolerance study with a soft part goes from 12.7 min to about 3.3 min per 10,000 samples, a turning shaft from 7.4 to 2.5 s, a studio drag frame from 46 to 12 ms once studio #96 exists, with no change to their code but one keyword. | Measured with the cache, which does the same work (`c1`, `c2`, `c4`, `c6`). |
| 2 | One return type? | **No.** Each solver keeps what it returns; every result gets one attribute saying which method made it. | Studio reads exactly two shapes, `ChargeSheet` and sealed cells, by class name; `to_json` has a class registry; a new type or subclass breaks both (`roadmap.md`, `seams.md` (b)). |
| 3 | How does reuse work? | `apply_face_charge(model, reuse=previous)`: the call fingerprints the newly meshed parts; if they match `previous`, it takes its factorization (load and back-substitution); if not, it solves afresh. The factorization lives on the result and dies with it. No global store. | A wrong `reuse` can only cost a fresh solve, never a wrong field; memory is the user's object, not a hidden 1 GB; this is how every comparable library does it (`comparables.md`, lessons 1 to 3). |
| 4 | Reuse for the method of moments? | **Not now.** | Its default solver never factors, and moving a magnet changes its matrix because fixed magnets are cells in it (`demag.py:1671, 1462-1479`); keeping the operator pays inside #81's saturation loop, which is a separate PR. |
| 5 | An `"auto"` method choice? | **No.** The docs' table chooses; each solver refuses or warns where it is wrong. Add the one missing warning: a sheet read inside its body. | No library read today picks a discretization for the user; the solve cannot know where `getB` will read; the two methods differ by 1 to 5 % near parts and 2 to 10× inside (`comparables.md` lesson 4; `api.md` §2). |
| 6 | Old functions and names? | Unchanged. The method attribute uses the module names: `"face_charge"`, `"demag"`, `"fem"`. | Nothing to deprecate; the names already exist in the package. |
| 7 | Order? | (1) #132's private core on `main` as a refactor; (2) `reuse=`; (3) the inside warning; (4) the method attribute; (5) decide #71; (6) paths, only on demand, as their own design issue. Close #132 in favour of (1)+(2). | (1) is needed by everything after it and is 60 lines of moving code; (6) is the largest and riskiest piece, 350 to 500 lines, measured on one case only (`seams.md` (d)). |

## What users gain, with the evidence

The benefit of material-response over magpylib alone is correctness, not speed. On the plate model (a 5 mm cube magnet 1 mm over a 10 × 10 × 2 mm plate, χ = 1000, the sensor in the gap; `uses.md`, probe):

| How | Bz at the sensor | Per mm the magnet shifts | What is wrong |
|---|---:|---:|---|
| magpylib alone, no plate | 52.2 mT | 144.9 mT/mm | B is half; every sensitivity to the magnet's pose a third low |
| face charge, solved once, the magnet then moved in magpylib | 104.95 mT | 144.9 mT/mm | B right, the sensitivity still a third low: the plate's share is frozen in the sheet, with no warning |
| face charge, a solve per position | 104.95 mT | 217.8 mT/mm | right to 0.1 % of finite elements (105.05 mT, 218.5 mT/mm) |

The middle row is the trap: a solved collection is an ordinary source, so moving the magnet in it is the natural thing to do, and it silently loses the plate's part of the sensitivity. A tolerance study with a soft part therefore needs a solve per sample. That solve costs 76 ms today; with the matrix reused, 20 ms (`c2`: 1,149 hits of 1,150 calls, worst difference 4e-14). The magnet's tolerances are a third of the study's variance (30.6 mT spread with all 8 tolerances, 22.5 mT with the sensor's alone; `s2`), and they are the part magpylib alone gets wrong.

Where reuse pays, measured on the branch (`uses.md`, table 2): the Monte Carlo (12.7 → 3.3 min), the turning shaft (7.4 → 2.5 s), a drag frame (46 → 12 ms at level 2, 377 → 50 ms at level 5). Where it does not: anything that changes a solved part's shape, pose against another solved part or χ (the NMR designs, the plate length, the shaft inside its holder, a recoil magnet that moves), and one-off solves. Coils need no feature: one solve per coil and weighted sums, exact to 2e-14, four lines; a docs paragraph.

How accurate, honestly: face charge agrees with finite elements to 0.1 % on B and 0.3 % on the slope on this model, and within each reference's uncertainty on 15 validation cases (0.64 % at worst). All of that is model against model on a linear material; no measurement exists. Saturation is not modelled (#81): the plate's mean polarization is 0.065 T, but nobody has checked the plate under the magnet's edge against the 1.6 to 2 T knee. The magnet's recoil (χ 0.05) moves B by 0.4 %, more than the discretization error, and is left out of the showcase. B inside a part from face charge is wrong by 2 to 10× (952 mT where cells give 442) and today only documented, not warned. Design differences flip with the level (the NMR verdict: +9.5, −2.5, −5.6 ppm at levels 1 to 3).

## The API, sketched

```python
# One configuration: as today, bit-identical, nothing kept
solved = apply_face_charge(model, level=3)
B = magpy.getB(solved, sensor)

# A loop that reuses: one keyword. True keeps the first call's matrix;
# a previous result lends its matrix when the solved parts match it.
solved = True
for pose in poses:                       # magnet poses, polarizations, H_ext: the matrix stays
    solved = apply_face_charge(model_at(pose), reuse=solved)
    B.append(magpy.getB(solved, sensor))
solved.face_charge_info["reused"]        # True when a kept matrix served the call

# A path, the narrow form: later, if 100,000-sample studies are real
model.magnet.position = poses            # (10000, 3)
B = apply_face_charge(model, observers=sensor)   # B of shape (10000, 3); refused when the matrix would change
```

What `reuse=` does, in order: mesh the solved parts as today (0.7 ms at 1,518 unknowns); read the fingerprint off the meshed bodies (panels in each body's own frame, χ, poses relative to the first body, joined faces, absolute size: #132's `_key`, which review completed with the size); if it equals the previous result's, load the sources and back-substitute on its factorization, else assemble and factor as today. The factorization is held by a private object on the result (8 n² bytes: 18 MB at 1,518 unknowns, 1.7 GB at 14,454), shared, not doubled, by `copy()`, freed when the result is dropped. Each back-substitution copies the pivots (#131). Without `reuse=`, nothing changes: the first solve stays bit-identical to today and no memory is held. With it, the call that factors takes up to 26 % longer above 2,000 unknowns, because the kept factorization is in double, and later calls differ from a fresh solve by up to 7e-14.

The messages a user sees:

- A path on a source, as today, with one more sentence: `NotImplementedError: objects moving along paths are refused: solve each step in a loop, passing reuse= to keep the matrix between steps`.
- A result read inside its own part: `UserWarning: 3 of 10 observers lie inside the plate: a ChargeSheet gives μ0 H there, not B; use apply_demag on cells or fem_solve for fields inside a part`.
- A `reuse=` that does not match: no warning, a fresh solve, and `face_charge_info["reused"] == False`; a log line at info level says what differed (a part moved, a part's length, χ).

## Order of work, with sizes (estimates from `seams.md`)

| Step | What | Size | Keeps from #132 |
|---|---|---|---|
| 1 | The private core on `main`: `assemble_system`, `load_system`, `split_unknowns`, the singular and ill-conditioned guards, the pivot copy. Redone on `main`, since #133 rewrote the function #132 split; bit-identical first solves, checked as #132 did on 18 model and option combinations | 60 touched lines, 1 test | the split, `_factor`, `_back_substitute` |
| 2 | `reuse=` on `apply_face_charge`, the fingerprint, the private holder with its copy rule, `face_charge_info["reused"]`; measured on the Monte Carlo and the shaft | about 150 new lines, 5 to 7 tests | `_key`, the 14 model sequences, the thread test, the sound-matrix test, the docs section rewritten |
| 3 | Warn when a sheet is read inside its body (a winding-number pass on the near observers) | 20 lines, 1 test | |
| 4 | One attribute on every result naming its method, kept by `to_json`; what studio #96 asks for, with the fingerprint studio #89 asks for | 15 lines, 1 test | |
| 5 | Decide #71 (`UniformField` in place of `H_ext`) before any path design; it decides whether a field varies per step as a source or as an argument, and whether `getB` of a result includes it | a decision | |
| 6 | Paths in the narrow form, as a design issue, only when a user needs 100,000 samples: loads per step in chunks of 100, `lu_solve` with many right-hand sides, B at observers, the six refusals | 350 to 500 lines, high risk, one case measured | |

Dropped from #132: the global store, the second-sighting rule, the 1 GB cap and eviction, the fork reset, `set_cache_limit`, `clear_cache`, the validation runner clearing a cache. Not planned: `mesh()`, `solve()`, a common return type, `"auto"`, one `study()`; a study for face charge can be fed to `MeshStudy.from_values` by hand today, and its error estimate on panels is unproven anyway.

## The open issues

| Issue | Effect of this recommendation |
|---|---|
| #130, #132 | Superseded by steps 1 and 2; close #132 when step 2 opens |
| #131 scipy pivots | The rule stays: a pivot copy per `lu_solve`, with the thread test |
| #81 saturation | Unaffected; it keeps the χ-free operator inside its own iteration, which is where a kept MoM operator pays |
| #103 hierarchical matrices | Unaffected; `reuse=` applies below `max_unknowns`, where a dense factorization exists |
| #71 UniformField | Decide before paths (step 5); unaffected by steps 1 to 4 |
| #118, #120 | Unaffected: no `auto` rule to change |
| #84 progress hooks | Easier after step 1: the phases exist as functions |
| #85 NMR | Unaffected; a difference of two designs stays a hand-written loop, and its verdict still needs a layout that holds across shapes |
| #79 meshing per body | Unaffected |
| studio #96, #89 | Served by steps 2 and 4: hold the last result per body, pass it as `reuse=` each frame; read the method attribute and the fingerprint off the result |

## Unverified

- The 20 ms per call with `reuse=` is expected from the cache's measurement of the same work, not measured in this form.
- The shaft, the coils and anything above 2,000 unknowns were never run along a path; the 100,000-sample figures are linear extrapolations from 10,000.
- Whether the plate under the magnet's edge saturates in the showcase: only a bulk mean and one inside point exist.
- Whether studio can hold a result across drag frames and pass it back: read from its source, not run.
- The designs, judges and adversarial checks of the planned review did not run; the sheets' facts cite code lines, issues and probe outputs, the recommendations are mine.
