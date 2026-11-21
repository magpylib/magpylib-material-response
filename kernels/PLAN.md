# Reference files for the field kernels (#208)

Goal: language-neutral files that every implementation of the field kernels must
pass (NumPy, `array_api_strict`, the Rust crate natively, WebAssembly, later
ports, and in time magpylib itself). Cuboid first; face charge after; then any
magpylib source kind. Branch `feat/wasm-kernels-208`, worktree
`magpylib-material-response.worktrees/wasm-kernels-208`, own `.venv` with
magpylib main (never test against a stale magpylib).

## Committed

- 602bf11: the Rust crate `kernels/rust` (core, C ABI, pyo3 `python` feature,
  `wasm` feature), the first reference file and checkers.
- 9f38bba: the cuboid's one implementation,
  `src/magpylib_material_response/_array_api/cuboid.py` (Jack Smith's array API
  rewrite of magpylib's cuboid, with #1066's stable logarithms), run on NumPy
  and, for exact values, at 50 digits on `kernels/reference/mp_namespace.py`
  (the accuracy guardrail, generator only). `cuboid.json` (61 named points),
  `make_cuboid.py`, `reference_format.py`, `show.py`; tests in
  `tests/test_kernel_reference.py` (implementation on NumPy and
  `array_api_strict`, installed magpylib as a port).

## Uncommitted in the worktree (2026-11-21)

Face charge, to be redone in the new design below, not committed as is:

- `face_charge/_triangle.py` written against the array API for `frames`,
  `charge_field`, `far_points`, `charge_field_far` (the solver's own kernels
  stay NumPy). NumPy results bit-identical to before on contiguous input (36 of
  36 saved outputs; only a transposed-layout V differs by 1 ulp, tensordot vs
  einsum). Passes on `array_api_strict` (needed `V[:, 0, ...]` and
  `[:, None, :]` indexing). Full MR suite: 1135 passed. Keep this part.
- `_array_api/__init__.py`: shared `namespace()` and `quiet()`; `cuboid.py` uses
  them. Keep.
- `mp_namespace.py`: `matmul`, `tensordot`, `linalg.cross/vector_norm`, `sum`,
  `mean`, `sign`, `permute_dims`. Keep.
- `make_face_charge.py`, `face_charge.json` (135 points, prose-heavy),
  `reference_format.py` changes, checker rewrites (`reference.rs`, `check.mjs`,
  the test file) and `show.py` for triangles: superseded by the design below;
  reuse what fits, drop the rest.

## The agreed design (Alex, 2026-11-21)

Keep it simple. The JSON is the single source of truth, for machines; `SPEC.md`
is for humans and agents.

1. **`kernels/reference/<kind>.json`**, one file per source kind, data only, no
   prose, deterministic, one meaning per key:
   - top level: `rtol`, `sources`, `points`, `relations`;
   - a source: `kind` (the `magpylib.func` name without `_field`, e.g. `cuboid`;
     MR's own kinds get magpylib-style names) and that function's keyword
     parameters, singular (`dimension`, `polarization`, ...). Core functions
     only: no position or orientation (the pose is magpylib's layer on top);
   - a point: `id`, `source`, `observer`, one key per field (`B`, `H`, `J`,
     `M`), `atol_<field>` only where needed. No `why`, `group`, `convention` or
     `needs` in the file;
   - one field per line (vectors on one line), so a diff names the field; floats
     as shortest round-trip decimals; `"nan"`, `"inf"`, `"-inf"` as strings.
2. **Extending** = appending a point with only `id`, `source`, `observer`, then
   running **`fill.py <file>`**, the one generic tool kept in the repo. It holds
   no points; it fills missing values only (existing ones untouched unless
   asked). Rule, in its code and the SPEC: expected = the implementation at 50
   digits rounded to double, `atol_<field>` = 16 x numpy's own error where that
   passes `rtol`, rounded up to a power of ten; where the 50-digit value is
   undefined (numpy NaN or infinite) or numpy differs beyond the tolerance (a
   cuboid face: the side IEEE signed zeros pick), numpy's value, and `fill.py`
   prints those points for review. Generators that define points are used once
   to write a file, never committed (`make_cuboid.py`, `make_face_charge.py`,
   `reference_format.py` go).
3. **`SPEC.md`**: the format and every key; the encoding; the comparison rule
   (per field: `|got - expected| <= rtol |expected| + atol`, Euclidean;
   non-finite components exactly); per kind its parameters, fields, units and
   conventions (faces, edges, zero dimension or polarization, inside rule,
   `mu_0`); what each point id and relation id means; which points take numpy's
   value; how to add a point.
4. **Relations** (physics and cross-comparison), same file while cuboid only:
   `{id, terms: [{source, field, observer, weight}], target, atol}`; `weight` a
   number, 1x3 row or 3x3 matrix; a term's own `observer`, so a shifted cell
   needs no pose. The target is the law's own value (0, J_t, ...), never a
   50-digit residual of the same formula (that would check nothing); `fill.py`
   sets `atol` from numpy's error plus the discretization error of approximate
   laws. Exact identities first; stencils and Gauss later, tens of observers at
   most.
5. **Checkers** read every file, skip and report kinds or fields they lack. The
   magpylib port in the Python test skips the two near-edge points
   (`face-y+-by-edge`, `beside-edge-1e-12`) when magpylib <= 6.3.0 (#1066): that
   list lives in the test, not in the file.
6. **Later kinds** with elliptic integrals (cylinder, circle, cylinder segment)
   need `cel` in `mp_namespace` and in the array API implementation.

## Cuboid in the new design (2026-11-21)

Committed as 03e7fa6; the working tree then narrowed it (uncommitted, below).

- `cuboid_field` has `J` and `M`: bit-identical with magpylib's
  `_BHJM_magnet_cuboid` for B, H, J, M on NumPy and `array_api_strict`.
- `kernels/reference/fill.py`: the one generic tool (kind table: `cuboid`).
  Idempotent; `--redo` reproduces the file byte for byte; never takes numpy's
  value; refuses a point without a finite 50-digit value.
- `cuboid.json`: points and relations as in SPEC.md; the near-edge points equal
  magpylib's independent 90-digit literals bit for bit. Planted sign bugs (3
  mutants), regenerated: each fails relations at 50 digits; `cycle-axes` catches
  all three.
- Checkers: Python test, Rust (J, M reported as not in the crate), Node/wasm (B
  only), each printing the `why` beside a failure.
- magpylib 6.3.0 is off or NaN at `beside-edge-1e-12` and the nine `near-edge-*`
  points (#1066): the skip list in the test.

Held back from the committed file (13 points, uncommitted): faces `face-x+`,
`face-x-`, `face-y+`, `face-y+-by-edge`, `face-x+-tolerance`,
`beyond-tolerance-x+`; edges and corners `edge-x`, `edge-z`, `corner-ppp`,
`corner-mmm`; `zero-dimension` (on the zero-thickness plate B tends to J inside,
probed at 1e-6..1e-12, not 0); `issue-1058`, `observer-nan`, `observer-inf`.
Their observers and the `cell_1058` source are in 03e7fa6; `face-z+` (observer
[-0.0015, 0.002, 0.015] on `cube`) was never committed. They come back, with the
agreed rule computed by `fill.py`, once #1073 is decided. Which class each band
gives them: SPEC.md and the 2026-11-21 table (today's 1e-15 h, #1081's 100 eps,
1e-12 and 1e-8 of the size).

Verified finding for #1073 (not applied): on an x face magpylib's B is the
outside limit while H subtracts J, so H matches neither side. Cause: after the
octant mapping the zero `x - a` is `+0.0`, times a negative factor an `atan2`
numerator `-0.0` over a negative denominator gives `-pi` instead of `+pi`.
Forcing that zero to `-0.0` gave the inside limit on all four faces tried. This
answers Michael's question on how to compute a face average: the same formula
evaluated twice with the zero signed each way gives both limits exactly, no
nudging and no separate expression. Tested for the x face only (flipping y and z
faces to the outside was not).

Scope (Alex, 2026-11-21): MR lays the testing foundation and the wasm for field
lines in the browser, not every edge case. The files hold only clear, agreed
behaviour. Faces, edges, corners, the tolerance band and non-finite observers
wait for magpylib #1073 (proposal #1081; Michael agreed on NaN at edges and
corners, asked how a face average is computed, proposed a band per object, maybe
a keyword). 13 points held back, ids reserved in SPEC.md (uncommitted). If the
band becomes a keyword of `magpylib.func`, it is a source parameter here and
band-dependent points can state it.

## Face charge in the new design (2026-11-21, uncommitted)

- Kinds `triangle_linear_charge` (closed form; renamed from `triangle_charge`,
  magpylib's name for a uniform charge) and `charge_sheet` (far rule from
  `far_from` out; 0 = everywhere), field `H` in A/m, array API entry points
  `triangle_linear_charge_field` and `charge_sheet_field` at the end of
  `face_charge/_triangle.py`. A file may mix kinds: its relations tie them.
- `face_charge.json`: 14 sources, 46 points, 8 relations. Held back (ids
  reserved in SPEC.md): points in or within 1e-9 sides of a triangle's plane
  (the code lifts them 1e-12 sides, a convention), zero-area triangles,
  non-finite and overflow observers. The off-plane points from 1e-3 to 1e-12
  sides stay, flagged as band-sensitive; the cutoff is the cuboid's (#1081's
  band).
- `fill.py` has a per-kind scope check and refuses the held-back cases, the
  cuboid's too (all 14 earlier held-back cuboid points are refused).
- Relations: cuboid vs its 12 surface triangles (3), far rule vs closed form
  (2), triangle vs its four midpoint parts (the linear charge terms) (2), cycle
  of the axes. 8 planted formula bugs, regenerated: each fails relations
  (far-rule ones catch all eight; the box ones need the c0 path).
- Checkers: Python 305 passed; Rust and Node pass (Node skips cuboid H, J, M);
  `kernels/rust/www/plate.json` is the timing plate (84 ms for 2,744 points, as
  before). Old generators `make_face_charge.py`, `reference_format.py` deleted:
  `plate.json` is now the only record of the solved plate (magnet J (0, 0, 1.2)
  T, 10 x 10 x 5 mm at z = 8.5 mm over iron 60 x 30 x 4 mm, level 1); it cannot
  be regenerated. `ChargeSheet._field` and the reference share
  `beyond_far_from`, `charge_field` and `far_points`; its chunking and threads
  are not covered by the file.

Review (3 independent reviewers, 2026-11-21), all fixed: `far_points` back to
`einsum` on NumPy (bit-identical to before; `tensordot` only for other
namespaces); `fill.py` scope-checks every point, refuses non-finite residuals,
checks numpy against relations too and exits 1 on any disagreement, computes the
triangle plane distance at 50 digits per triangle (a sliver's double normal let
in-plane points through), refuses near-collinear triangles and observers on the
`far_from` sphere, enforces the `atol` rule for approximate laws (law error

- 16 x numpy's rounding; `jump-x`, `jump-y` tightened 1e-8 -> 1e-9); misleading
  sliver ids and whys fixed; SPEC states the frame (n, e1 from corner 0 to 1),
  "sides" (sqrt(area)) and the far rule's digits; wasm exposes `cuboid_h` (box
  relations now checked in the browser build); Rust and Node name the relations
  they skip; the Rust thread pass runs only with `--features parallel` and says
  so.

Foundation review (3 reviewers, 2026-11-21: design via SPEC alone, 65 Rust
mutants, branch vs main), all fixed: NumPy floor 2.1 (`unstack`; the solver
broke on 2.0.x, CI's lowest pins); the test runs without `array_api_strict` (CI
jobs without the test group); lint; negative dimensions held back (magpylib
flips B: undecided); kind renamed `triangle_linear_charge`; SPEC says J, M come
from the inside rule; `fill.py` rejects unknown keys, sources, kinds and fields,
and decides the `far_from` sphere exactly (rounding vs exact side) with a new
binary-exact point `sheet-on-far-from`; `b-h-j-inside` is now B - mu0 H = J
(law's value as target, B and H only), `m-j-inside` dropped; every checker runs
batches (cuboid rows mixing sources, tiled past 2 x 64) and fails if nothing was
checked (Rust and Node: 1,744 checks); `quiet()` restores main's warning flags
per function; the Python cuboid matches magpylib's zero-size product and warns
nowhere at 1e200; the Python binding checks shapes; FFI documents that `out`
must not alias. The two port bugs that survived (chunk index, per-row cuboid
parameters) are now caught by batches.

When #1073 is decided (the format needs no change): multi-source cases are
relations (each term its own source, observer, weight; mixed kinds allowed), as
`split-bar-*`. #1081's touching cells map to whole - sum of cells = 0 for B and
H on a shared face (exact under the average rule), J relations for the 1/2, 1/4,
1/8 shares, per-cell points with "nan" for B at edges and corners. Three things
change then: (1) the scope checks in `fill.py` (now refusing every face, edge,
corner and band observer, terms included); (2) the 50-digit truth on a face:
mp_namespace has no signed zero, so `fill.py` averages the two one-sided limits
at 50 digits (steps 1e-30 and 1e-40 of the size, inward and outward, which must
agree; the inward half was built and verified, then removed when faces were held
back); (3) the implementations follow the rule (for the cuboid, forcing the sign
of the zero in `x - a` each way gives both limits exactly, verified on x faces).
Not expressible, by design: a point rounded onto a face through a turned parent
(#1081), which is the pose layer's.

Speed, step 1 (2026-11-21, uncommitted): a `tolerance` per call (0 = exact, the
default, bit-identical to before; studio e.g. 1e-3 while dragging). A pair takes
the far rule from k longest edges of a triangle's centroid, k from `PAIR_RULE`:
bounds 6.5e-3 / k^4 of the one-sign field, twice the worst case an adversarial
review found (a needle with a linearly falling charge, seen along its axis:
3.3e-3 / k^4; my first random sampling had claimed 3x margin over 1.3e-3,
wrong). In `_triangle.charge_sheet_field`, the Rust `Sheet` (per-triangle
contiguous far charges) and the wasm `Sheet.h(x, tolerance)`. Plate, 2,744
points, wasm one thread, perf lock: 87.5 ms exact, 82.5 ms at 1e-6, 16.1 ms at
1e-3 (5.4x); far rule alone 15.2 ms, the floor of this step. Reference: every
sheet has `tolerance`; box sheets with mixed pairs at 1e-3 and 1e-6, one where
`far_from` decides first, one exact-binary triangle per table row with points at
and just inside k L (inside points dropped for k^2 2304, 4096: NumPy's own error
there hides the far rule's). `fill.py` refuses observers where rounding picks a
pair's rule. Planted Rust bugs: all caught (rows, row choice, >=, the
whole-sheet switch, centroid). Not wired in `ChargeSheet._field` (studio's
Python side). Next: treecode (decision pending), SIMD, workers in studio.

Open:

## Decided earlier

- C ABI: keep pyo3 for Python; test the C functions through ctypes on the same
  files (the extension's `.so` exports them). Not built yet.
- Subagents: cheapest model that fits (memory `subagent-models`).
- Commit only on Alex's word, cuboid and face charge separately.
- For Jack (Alex tells him, when he chooses): his `array-api` branch predates
  #1066 (+-inf at two points by an edge); `xp.where(mask, -1.0, 1.0)` with two
  scalars is not in the standard (fails `array_api_strict`).

## Open, later

- Browser speed: the solved plate's H at 2,744 points takes 84 ms in wasm (one
  thread, 47 ns per pair) against a 16 ms frame: per-pair switch to the 6-point
  rule a few sides out, web workers, or SIMD.
- Inside test kernel (MR's `_inside` convention) and points on a solved part's
  surface (#208).
- Native one thread is ~1.3x NumPy on `charge_field`, ~12x on all threads.
