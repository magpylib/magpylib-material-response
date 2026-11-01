# What users gain, use case by use case

Ground truth for the plan review, 31 October 2026. From the measured scripts on `origin/explore/use-cases` (e9e88b6), the docs on `main` (57c7860), the issues, and one small probe run for this review (values only, no timing). Every number names its source; the tags are listed at the end.

## Conclusion

- **Measured gains exist in two use cases.** The tolerance Monte Carlo: 10,000 samples take 12.7 min today, 3.3 min with a kept factorization, 20 s along a path on 8 threads (65 s on one) [c1][c2][p1]. The turning shaft: 48 configurations 7.4 s today, 2.5 s with a kept factorization [c4]. Studio's drag would get the same per-frame gain, 46 → 12 ms at level 2, once studio #96 exists [c6]. Nothing else in the plan has a measured gain: coils are 4 hand-written lines today [s3], the NMR comparison reuses nothing [s4], the one-configuration cases have nothing to reuse, and the refused cases get a message.
- **The benefit over magpylib alone is correctness, not speed.** With the plate, the sensor's B doubles (52.23 → 104.95 mT) and its sensitivity to the magnet's position rises by half (144.9 → 217.8 mT/mm) [probe]. A user who solves once and then moves the magnet in magpylib gets the right B and the wrong sensitivity, 144.9 mT/mm, with no warning [probe]. A re-solve per sample is what makes a tolerance study right. The plan makes that re-solve cost 2–20 ms instead of 76 ms [c2][p1].
- **Accuracy is known against finite elements only**, on a linear material: 0.1 % on B, 0.3 % on the slope [brief]; the kept solver matches `apply_face_charge` to 0.7 nT [p1]; the cache matches a fresh solve to 4e-14 [c3]. Unknown: no measurement exists; saturation is not modelled (#81); the showcase ignores the magnet's recoil, worth 0.4 % of B [s2]; face charge is wrong inside a part (952 mT where cells give 442) [brief]; the NMR verdict is smaller than its own discretization error [s4].
- **Ranked by gain for the cost:** (1) a kept factorization for repeated solves of the same parts, in the existing calls; (2) paths for batch Monte Carlos; (3) write down the coil sums; (4) a warning when a sheet is read inside its body; (5) same-layout differences for design comparisons, the one accuracy fix users need, not in the plan; (6) a common return type, `auto` and new names: no measured gain, breaks three functions.
- **What this implies for "do the rewrite at all?":** the measured gains come from two capabilities, keeping a factorization and taking paths. Neither needs `mesh()`/`solve()`, a common return type or `"auto"`; both fit inside `apply_face_charge` and the direct `apply_demag`. The rewrite's own pieces have no number behind them.

## Alex's two questions

### "Tolerances on magnets I can do with magpylib alone. What is the benefit of all this?"

For a system without soft parts, nothing: magpylib's paths already do 10,000 sensor poses in 0.19 s on one solved model [s2]. With a soft part next to the sensor, the plate model of the Monte Carlo (a 5 mm cube magnet 1 mm over a 10 × 10 × 2 mm plate, χ 1000, sensor in the gap), three ways of doing it [probe]:

| How | Bz at the sensor | Slope per mm the magnet shifts | What is wrong |
| --- | ---: | ---: | --- |
| magpylib alone, no plate | 52.23 mT | 144.9 mT/mm | B is half; every sensitivity to the magnet's pose is a third low |
| face charge level 3, solved once, then the magnet moved in magpylib | 104.95 mT | 144.9 mT/mm | B right, sensitivity still a third low: the plate's 72.9 mT/mm is frozen in the sheet |
| face charge level 3, a solve per position | 104.95 mT | 217.8 mT/mm | right, to 0.1 % of FEM [brief] |

The brief's numbers for the same model, from a session probe [brief]: 52.23 / 145.71 mT/mm alone, 104.95 / 218.55 at level 3, FEM 105.05 / 218.49; the slopes differ from the probe's by the finite-difference step. The middle row is the trap: the solved collection is an ordinary magpylib source, so moving the magnet in it is the natural thing to do, and it silently loses the plate's part of the sensitivity. c7 shows the same on a solved part's own polarization: kept from the first step it is wrong by 220 mT, right to 9e-14 mT with a load per step [c7].

So the benefit is this: a tolerance study with a soft part needs a solve per sample, which costs 76 ms today, 12.7 min for 10,000 samples and 2.1 h for the 100,000 that [3] runs (linear extrapolation, unverified) [c1]. A kept factorization makes it 20 ms (3.3 min per 10,000) with no code change [c2]; a path makes it 1.8 ms on 8 threads (20 s) [p1]. The spread the study is after is 30.6 mT with all 8 tolerances, 22.5 mT with the sensor's alone [s2]: the magnet's tolerances are a third of the variance, and they are the part magpylib alone gets wrong.

### "How do I know it is accurate?"

What is checked, all on linear material:

- Against finite elements on the plate model: FEM (maxh 0.5 mm, order 3) 105.05 mT and 218.49 mT/mm; face charge level 3: 104.95 and 218.55; level 5: 105.05 and 218.32 [brief]. Level 4 in this review: 105.06 mT, 217.72 mT/mm [probe]. B agrees to 0.1 %, the slope to 0.3 %.
- The kept solver (paths) against `apply_face_charge`, 20 samples: worst 0.7 nT on fields of 66 to 173 mT [p1]. The cache against no cache: worst 4e-14 relative over 20 sequences that change one thing at a time (PROGRESS.md; 1.8e-14 in the keep2 result file) [c3]; the shaft's 48 configurations 1.5e-15 [c4].
- The whole package against FEM on 15 validation cases: face charge within each reference's uncertainty, 0.64 % at worst [docs face_charge.md:49-52]; magpylib alone is 721 % off under the soft plate [docs fem_check.md:145-147].
- Where a kept solve goes wrong: a magnet within 0.1 mm of the plate, kept from the first step, 2.0 % of B; within 0.01 mm, 5.4 % [c7]. The narrow form refuses those; the cache's key misses them by itself [c3].

What is not known:

- **No measurement.** Every check is model against model, and FEM uses the same linear χ: "FEM checks the numerics, not the material model" [docs fem_check.md:135-136].
- **Saturation is not modelled** (#81). The solved plate's mean polarization is 0.065 T [probe], but that is a bulk average that cancels between the region under the magnet and the return flux; B inside the plate 0.3 mm below its surface is 442–446 mT from cells [brief], below the 1.6–2 T knee #81 names. Nobody has checked the plate under the magnet's edge. `mean_polarization` is the only saturation check the package offers [docs face_charge.md:220-226].
- **The showcase ignores the magnet's recoil.** With χ 0.05 on the magnet, Bz's mean moves by 0.5 mT, 0.4 % (111.97 vs 111.48 mT over 40 samples) [s2], larger than the discretization error, and the path is then refused.
- **B inside a part from face charge is wrong** by 2 to 10 times at χ 1000 (952, 91, 359 mT at levels 3–5 where cells give 442–446) [brief]; today it is documented [docs face_charge.md:73-75] and not warned: `_sheet.py` has no warning [grep, `_sheet.py:60` is the only mention].
- **Differences between designs** are not settled by any of this: the NMR verdict is +9.5, −2.5, −5.6 ppm at levels 1–3, each design on its own panel layout, and level 3 needs 20,640 unknowns, over the 20,000 cap [s4]. #85 itself notes a 0.6 % absolute error against a 1,138 ppm quantity. A plate length swept along a path gives face charge a biased slope (3.7 vs 3.9 mT/mm) because its panels move with the length [page, session probe, not re-run].

## Table 1: the sixteen use cases today

Lines are counted in the branch scripts and the docs; "solves" is what the script runs; "matrix" says whether the solved parts' system changes between configurations, as the cache's hits and misses found [c3].

| # | Use case (tile) | What the user wants | Written by hand today | Solves | What changes the matrix |
| --- | --- | --- | --- | --- | --- |
| 1 | Shaft angle sensor | Angle error over 12 angles × 2 gaps × 2 stray fields: 1.26° at 1.5 mm in 5 mT, patent says 1.3° [s1] | model 24 lines (`s1:32-55`), the grid of loops with its bookkeeping ~34 lines (`s1:74-107`), +1 line adding the stray field back to `getB` (#71, `s1:83`) | 48 (the gap is a sensor move, 24 would do) | Nothing: rotor turns as one, stray field only loads; 46 hits of 48 [c4] |
| 2 | Tolerance Monte Carlo | Spread of Bz over 8 tolerances: 30.6 mT; sensor's alone 22.5 mT [s2] | model + sensor helpers 42 lines (`p_model.py`), the loop 4 lines (`c1:26-29`) | 10,000; 76.3 ms each, 12.7 min [c1] | Nothing: 1 miss in 1,150 calls [c2] |
| 3 | U-core coils | Bz on a line for 20 sets of two currents [s3] | model 22 lines (`s3:28-50`); the per-coil sums 4 lines (`s3:72-75`) | 20 today, 2 with the sums | Nothing: currents only load; sums exact to 2e-14 [s3] |
| 4 | NMR pole pieces (#85) | Is design B more uniform than A, in ppm over ±5 mm | the study of a difference ~23 lines (`s4:60-82`); `mesh_study` takes one model and one quantity [docs convergence.py:230-249] | 6 (3 levels × 2 designs), 56 s, 6.7 GB [s4] | Every solve: shape and level differ; "one length changes" 0 hits [c3] |
| 5 | Plate length tolerance | dB/dL at a sensor over 9.5–10.5 mm | a loop of re-meshed solves; a stretched cell layout by hand via `mesh_Cuboid(cuboid, (nx, ny, nz))` [docs meshing.py:66-68] | 21 per method [page] | Every step: 0 hits [c3]; face charge's panels move with L, slope biased [page] |
| 6 | B inside a part | Flux in the iron, saturation check | `apply_demag` on cells; face charge documented as wrong inside [docs face_charge.md:73-75] | 1 | n/a |
| 7 | Hall-chip concentrator | Gain at a sensor µm from the surface; 4.27 at the finest mesh vs FEM 4.259 [docs mesh_convergence.md:83-86] | `mesh_study` 1 call (`mesh_convergence.md:77`); FEM `observers=` refines there [docs _solve.py:924-926] | 1 (a ladder for the error bar) | n/a |
| 8 | Shield sheet | Field behind a 1 mm sheet; MoM finest 9.9 % off on axis [docs mesh_convergence.md:161-164], face charge 0.28 s to 1 % [docs face_charge.md:185] | `apply_face_charge(..., level=2)` 1 line; the method choice is the user's | 1 | n/a; sheets over 20,000 unknowns refused (#120) |
| 9 | A magnet's own demag | Field drop of a χ 0.3 magnet: 10 % [docs quickstart.md:40-51] | `mesh_Cuboid` + `apply_demag`, 2 lines | 1 | n/a |
| 10 | Laminated core | Anisotropic χ | `apply_demag` with a 3-vector χ [docs demag.py:1578-1582]; face charge raises [docs _apply.py:126-128] | 1 | n/a |
| 11 | FEM check | Do two methods agree, with error bars | `mesh_study` + `fem_study` + `compare`, 3 lines (`fem_check.md:62-64`) | a ladder each | n/a |
| 12 | Design optimizer | Best shape over a population, gradient-free [1] | not measured; a solve per candidate | populations × steps | Every candidate when the iron's shape varies; nothing when only magnets or currents vary (unmeasured) |
| 13 | Shaft in a holder | Angle error with a soft ring: 0.06° in 5 mT [s1] | as 1 | 48; 3.3 s each, 158 s [s1] | Every angle: 0 hits of 48, cache 4 % slower [c4]; a real 2 mm ring is 16,044 unknowns alone [s1] |
| 14 | Recoil magnet Monte Carlo | As 2 with χ 0.05 on the magnet | as 2 | 10,000; 139 ms each, 23 min [s2] | Every sample: "a recoil magnet moves" 0 hits [c3] |
| 15 | Magnet sliding, almost touching | Field while a magnet slides 0.01 mm off the plate | a loop of fresh solves, correct | one per position | Every position within 1 % of the size: panels graded to the footprint; frozen layout 5.4 % wrong [c7]; the cache misses by itself [c3] |
| 16 | Studio drag | A re-solve on each drag frame | nothing: studio does not re-solve (studio #96); a sheet is passed as Triangles, 0.2–5 % off near the part (studio #96) | one per frame: 46 / 76 / 141 / 377 ms at levels 2–5 [c6] | Nothing while only the magnet moves |

## Table 2: what each option gives

(a) the full rewrite: `mesh()`/`solve()`, the mesh object holding its factorization, paths in the narrow form, `"auto"`, one `study()`, a common return type. (b) a kept factorization alone, in the existing calls, no API change (#132 or the object holding it). (c) paths alone: `apply_face_charge` accepts a path, factors once inside the call, returns B at observers. (d) a common return type alone. (e) nothing. Time is total for the configurations in Table 1; "code" is what the user stops writing.

| # | Use case | (a) full rewrite | (b) kept factorization alone | (c) paths alone | (d) common return type alone | (e) nothing |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Shaft | ≤ 2.5 s (the cache's number; a path on the shaft was not run, estimate 0.7 s [s1], unverified); code: 4 loops → one call with a 12-step rotation and `H_ext` of shape (24, 3); the `H_ext` add-back stays unless #71 | 7.4 → 2.5 s, 2.9× [c4]; code unchanged | same matrix, so eligible; not measured; loops → one call | 0 s; 0 lines | 7.4 s, a job of seconds today |
| 2 | Tolerance MC | 20 s on 8 threads, 65 s on one [p1]; code: the 4-line loop → one call; the path model is 12 lines (`p1:38-47, 72-73`) against 12 + loop today: about equal; memory 4.5 GB for all loads at once, so chunks of 100 [kept.py:28, brief] | 12.7 → 3.3 min, 3.8× [c1][c2]; code unchanged; a loop without sheets could reach 8.1 ms, 83 s [p1 loop] | the same 20 s / 65 s [p1]; one call | 0 s; 0 lines | 12.7 min; 2.1 h at 100,000 samples (extrapolated); the sensor-only part is 0.19 s today [s2] |
| 3 | Coils | a current path could find the sums itself [page]; not measured; saves the 4 lines | 2.9 → ~0.3 s estimated [s3 `kept_factorization_s`], unmeasured; the sums already give 0.29 s | 20 loads at one factorization; not measured; cannot beat 2 solves + sums | 0 | 0.29 s with 4 lines by hand, exact to 2e-14 [s3] |
| 4 | NMR | only `study()` applies, if it takes a difference; the verdict still flips (+9.5, −2.5, −5.6 ppm) [s4]; level 3 still over the cap | 0: nothing repeats [c3] | refused: shapes differ | 0 | 56 s, 6.7 GB, verdict unsettled [s4] |
| 5 | Plate length | a refusal naming the part and suggesting a loop [page]; the stretched layout is "later" | 0 hits [c3] | refused | 0 | by hand with stretched cells today [docs meshing.py:66-68]; same slope coarse and fine, 3.98 / 3.86 mT/mm [page, unverified] |
| 6 | B inside | `auto` → cells only if a sensor inside is in the model; else a proposed warning [page] | 0 (MoM's default never factors [docs demag.py:1603-1607]) | 0 | a result that warns when read inside: the one cheap correctness win | the user must know [docs face_charge.md:73-75]; no warning in code |
| 7 | Concentrator | refine near observers for face charge, proposed, unmeasured [page] | 0 | 0 | 0 | works: 0.04 s to 1 % [docs face_charge.md:180] |
| 8 | Shield sheet | `auto` falls back to cells over the cap; same as choosing today | 0 | 0 | 0 | works; the user chooses the method |
| 9 | Magnet's demag | 0 | 0 | 0 | 0 | works, 2 lines |
| 10 | Laminated core | `auto` picks cells: one decision saved | 0 | 0 | 0 | works with `apply_demag` |
| 11 | FEM check | one `study()` over the ladders: same 3 lines | 0 | 0 | saves adapter code for whoever compares methods: `validation/methods.py` has 42 (MoM), 65 (face charge), 45 (FEM) lines of it | works, 3 lines |
| 12 | Optimizer | nothing when shapes vary; when only magnets vary, (b) applies, unmeasured | hits when only sources move [c3 "fixed source moves"]; gain unmeasured on an optimizer | shapes vary: refused | 0 | a solve per candidate |
| 13 | Shaft in holder | refused with a message; block recoupling "later" [page] | 158 → 163 s, 4 % slower [c4] | refused | 0 | 158 s today |
| 14 | Recoil MC | refused | 0 hits [c3] | refused | 0 | 23 min [s2] |
| 15 | Sliding, touching | refused: correct | correct by itself, no gain [c3] | refused | 0 | a loop of fresh solves, correct |
| 16 | Studio drag | = (b), held by the object or a cache; a path never sees frames | 46 → 12, 76 → 19, 141 → 30, 377 → 50 ms per frame [c6]; needs studio #96 | 0 | a sheet studio can draw would drop the Triangle stand-in (studio #96) | no re-solve at all today (studio #96) |

What the table says in one line: time is saved in rows 1, 2 and 16 by (b), and in row 2 again by (c); code is saved almost nowhere (rows 1 and 3, about 35 and 4 lines); the refused rows (5, 13, 14, 15) get a message, which beats a silent 2–5 % error [c7] but gives nothing to compute with.

## Ranked: what gives users the most for the least

1. **Keep the factorization for repeated solves of the same parts**, inside the existing calls, whichever object holds it. Measured: 76 → 20 ms per Monte Carlo call [c2], 7.4 → 2.5 s on the shaft [c4], 46 → 12 ms per drag frame at level 2 [c6], 4.22 s → 0.21 s at 14,454 unknowns [c5]. No user code changes. Serves what a path cannot: loops, optimizers over magnet poses, studio's drag, stray-field sweeps. Costs, all measured on the branch: memory held between calls, 0.05 GB at 2,520 unknowns and 1.67 GB at 14,454 [c5]; the factoring call 7–36 % slower [c5]; a key that misses an input returns wrong fields silently, one such miss found by review (a model scaled ×2) [brief]; threads need their own pivots or 6 % wrong answers [brief, PROGRESS.md]; a stream of new matrices gains nothing (the holder, 4 % slower) [c4].
2. **Paths in the narrow form, for batch Monte Carlos.** 3.3 min → 20 s on 8 threads, 65 s on one, for 10,000 samples [p1]. Only one case measured. The loads dominate: 16.8 of the 18 s is magpylib's field at 18,896 points [p1 t8 `loads_s`], so the gain over (1) is the threads on the loads, not the algebra. Needs chunks of 100 steps (4.5 GB otherwise) [brief], a load per step (220 mT wrong otherwise) [c7], and six kinds of refusal [page]. A loop becomes one call; the line count stays about the same.
3. **Write down the coil sums.** 2.9 → 0.29 s for 20 sets, any further set free, exact to 2e-14, 4 lines [s3]. Zero code in the package; a docs paragraph. Or let a current path detect it.
4. **Warn when a sheet is read inside its body.** Prevents a 2–10× wrong B (952 vs 442 mT) [brief]; today documented only [grep `_sheet.py`]. A few lines.
5. **Same-layout differences for comparisons and dimension tolerances.** The one item that fixes an accuracy problem users have: the NMR verdict flips with level [s4], the plate's dB/dL is biased on moving panels [page]. Not in the plan (the narrow form refuses it). Possible by hand with cells today (`mesh_Cuboid` with a fixed triple) [docs meshing.py:66-68], costly to design for face charge.
6. **A common return type, `"auto"`, one `study()`, new names.** No measured time gain in any row. Code saved: adapters like `validation/methods.py`'s 150 lines, only for whoever compares methods. Cost: three public functions become wrappers or get deprecated; `auto` cannot know where `getB` will read unless sensors are in the model [page]. Lowest gain for the cost.

## Sources

- [s1] `explorations/use-cases/s1_shaft_sensor.py` and `results/s1_shaft_level2_angles12.json` on `origin/explore/use-cases` (e9e88b6); unchanged since b193c4f (`git diff --stat` empty). Line numbers `s1:NN` refer to this file.
- [s2] `s2_tolerance_mc.py`, `results/s2_tolerance_samples40.json`: 40 samples timed, extrapolated to 10,000.
- [s3] `s3_coils_sum.py`, `results/s3_coils_level2_sets20.json`.
- [s4] `s4_nmr_pole_pieces.py`, `results/s4_nmr_levels1-3_max21000.json`.
- [c1] `c1_today.py`, `results/c1_today_200_magpylib-6.1.0.json`: 76.35 ms per sample on magpylib 6.1.0, MR 0.7.1.dev3.
- [c2] `c2_cache_loop.py`, `results/c2_cache_loop_1000_magpylib-6.1.0.json`: 19.76 ms per call, 1,149 hits, 1 miss, 50 samples checked at 0.0 error.
- [c3] `results/c3_cache_correct_keep2.json`: 20 sequences, all ok, worst 1.8e-14.
- [c4] `results/c4_shaft_cache_level2_angles12_keep2.json`.
- [c5] `results/c5_cache_size_level4.json`, `..._level10.json`.
- [c6] `results/c6_drag_levels.json`: 30 frames per level, 28 hits.
- [c7] `c7_frozen_layout.py`, `results/c7_frozen_layout.json`, table in `PROGRESS.md`.
- [p1] `p1_tolerance_kept.py`, `kept.py` (209 lines, cuboids that do not touch, no sheets), `results/p1_tolerance_kept_path_10000_magpylib-6.1.0_t8.json` (1.80 ms/sample, 19.7 s total, loads 16.8 s, peak 2.26 GB), `..._t1.json` (6.13 ms, 65.4 s), `..._loop_10000_magpylib-1059.json` (8.14 ms, 83.1 s); check 7.0e-10 T on 0.0658–0.1725 T.
- [probe] `ground/probes/plate_slope.py`, output in `ground/probes/plate_slope.out`, run for this review on `main` with one thread, no timing: the plate model of `p_model.py`, central differences of ±0.05 mm.
- [brief] `BRIEF.md` §2: the FEM and level-5 numbers, B inside, the scaled-model miss, the thread failures; from session probes not on the branch, not re-run here.
- [page] `page_text.md`: the length-tolerance chart and slopes, the refusal design, the coil-path idea; the length and B-inside probes ran in a session scratchpad and could not be re-run from a file.
- [docs] files on `main` at 57c7860, `path:lines`.
- Issues: #71, #81, #85, #103, #118, #120, #130; studio #96.
- [1], [3]: the page's references (Ortner et al. 2023; Ergun et al. 2019).

## Open questions

- A path was never run on the shaft, the coils or anything above 2,000 unknowns; the shaft's 0.7 s is an estimate that was "too hopeful" for the cache (2.5 s measured).
- Whether the plate under the magnet's edge saturates in the showcase: only a bulk mean (0.065 T) and one inside point (442 mT) exist; no cell-by-cell check against a knee.
- How much a design optimizer over magnet poses with fixed iron would gain from a kept factorization: no script exists; the page counts optimizers as "a solve each".
- The 100,000-sample figure (2.1 h today, 3.3 min on 8 threads) is a linear extrapolation from 10,000.
- The length-tolerance and B-inside numbers come from session probes that are not on the branch; they are quoted from the page and the brief, not re-run.
- Whether `H_ext` per step and a `getB` that includes it belong to the return type or to #71's `UniformField`: the shaft's add-back line disappears only with the latter.
