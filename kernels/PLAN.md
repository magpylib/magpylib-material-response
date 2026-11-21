# Shared reference files for the field kernels (#208)

Goal: one set of language-neutral files that every implementation of MR's field
kernels must pass (numpy, Rust native, WebAssembly, the C ABI, any later port).
Cuboid and face charge first. If it works, magpylib may adopt the format later,
after agreement with Michael.

Revised after an independent review (2026-11-19); the review's points are marked
[R n].

## What exists (branch `feat/wasm-kernels-208`)

- `kernels/reference/make_cuboid.py` writes `cuboid.json` (named sources, points
  with ids and reasons) from the cuboid's one implementation,
  `_array_api/cuboid.py`, run on NumPy and on `mp_namespace` (50 digits);
  `show.py` plots it. Face charge is in progress, uncommitted.
- Checkers: `tests/test_kernel_reference.py` (the implementation on NumPy and
  `array_api_strict`, installed magpylib as a port),
  `kernels/rust/tests/reference.rs` (Rust, 1 and all threads),
  `kernels/rust/www/check.mjs` (WebAssembly in Node). All pass.
- Flaws of the first format, fixed in step 1: a case-wide `atol` let small rows
  pass with any answer [R1]; expected values were numpy's output [R2];
  `far_points` rows mixed metres and charges in one norm [R7].

## Format

1. **One family per file**: `face_charge.json`, `cuboid.json`, later
   `inside.json`. One output row per line, so a diff shows which value moved.
2. **Header**: format version, generator versions and commits, constants
   (`mu_0`), the kernel registry (inputs with shapes and units, outputs and
   units), the comparison rules, and the conventions that are part of the
   contract:
   - the lift off a triangle's plane: `w == 0` exactly gives `w = 1e-12 s`; no
     case puts an observer in a tilted plane, where the lift would hinge on
     rounding [R9];
   - the cuboid's boundary rule (`|x| - d/2 < 1e-15 d/2` is inside; on edges and
     corners the field is 0) and the inside test's own rule (step 3) [R10];
   - floats are shortest round-trip decimals: a reader must parse correctly
     rounded (serde_json `float_roundtrip`, a correct `strtod` in C).
3. **Shared arrays**: a table of named arrays (a triangulated cuboid, the solved
   plate) that cases reference by id, so relations do not repeat them [R11].
4. **Case kinds**:
   - `value`: inputs, expected rows, per-row `rtol` and `atol`. Expected = the
     50-digit value rounded to double (numpy's value only where no exact one is
     defined: on a cuboid face, non-finite rows) [R2]. Non-finite components
     must match exactly.
   - `relation`: evaluations `[{kernel, inputs, weight}]`, `weight` a scalar,
     1x3 or 3x3 matrix; per row `sum_k W_k out_k = target`. For approximate laws
     (stencils, `+-h` jumps) `target` is the 50-digit residual, so the tolerance
     is rounding only [R4].
   - Every case: `name`, `why`, `requires` (kernels it needs; a checker skips
     and reports a case whose kernels it lacks). No version strings [R8].
5. **Tolerances**: per row, `atol` and `rtol` from that row's own size, not the
   case's [R1]: 16 x numpy's error against 50 digits, floor 1e-12. The generator
   asserts (a) numpy passes every case, (b) every row has teeth: an effective
   relative tolerance under 1e-6 unless flagged with a reason, and scaled,
   perturbed or sign-flipped expected values fail [R1].
6. **Source of truth**: the generator defines the cases (random ones from a
   fixed seed) and writes the files; nobody edits values by hand. A new edge
   case = a few lines in the generator, reviewed by diff. Regeneration is a
   deliberate run, never part of a test.

## Status

- 2026-11-20: step 1 done for face charge (`face_charge.json`, 17 cases, rows
  per case); the cuboid moved to a file of points (`cuboid.json`): 8 named
  sources, 61 points, each with its own id, group, reason, B and H, one field
  per line; `show.py` plots it in 3D.
- 2026-11-20: the cuboid has one implementation, `_array_api/cuboid.py` in the
  package (no new dependency: numpy 2 arrays name their own namespace) (Jack
  Smith's array API rewrite of magpylib's cuboid, with #1066's stable logarithms
  carried in), kept here until magpylib's rewrite takes over. The generator runs
  it on NumPy and, for the exact values, on `mp_namespace` (mpmath, the accuracy
  guardrail): the mirrored 50-digit cuboid is gone, and no expected value moved.
  Tests run it on NumPy and `array_api_strict`, and installed magpylib as a
  port; 6.3.0 skips only the two `needs` points. Found on the way: `where` with
  two Python scalars is not in the standard (in Jack's branch too); the branch
  predates #1066 and gives inf at the two points by an edge. Face charge follows
  the same way.

## Steps

1. **Split, reformat, fix the flaws** (no new cases): family files, shared
   arrays, header and conventions, per-row tolerances, exact expected values,
   teeth assertions, `far_points` as two outputs; update the three checkers.
   numpy's cuboid check: run when the installed magpylib reproduces one #1066
   edge row, else skip with the reason [R8].
2. **Relation cases**, in order of strength [R5] (generic checker in each
   language, ~20 lines):
   - cross-kernel: `Cuboid` H = `charge_field(c = (J.n / mu0, 0, 0)) / 4 pi` on
     the cuboid's faces, two triangles each (exact; H inside included, the
     -J/mu0 comes from the charges); B adds J inside; observers off the surface
     and edges [R3];
   - Gauss: the flux of `charge_field` over a closed surface round a sheet is 4
     pi times its total charge [R5];
   - jumps at +-h: cuboid face, tangential B by J_t, normal B continuous, normal
     H by J.n / mu0; sheet, normal `charge_field` by 4 pi sigma at the foot,
     tangential continuous [R4];
   - exact identities: mirror symmetry, 90 degree rotation, power-of-two length
     scaling, linearity in J, superposition of coefficients, the list order of
     triangles (not their vertex order, which changes the frame and with it c1,
     c2) [R6];
   - curl H = 0, div B = 0 by stencils: optional, weakest [R5].
3. **Inside test kernel** `inside_cuboid`, with MR's `_inside` convention:
   inside iff inside, and a step of `max(1e-6 |high - low|, 1e-9)` (the box
   diagonal) to all six sides stays inside; on the surface is outside. Value
   cases: centre, on faces, edges, corners, within and beyond the step, NaN, and
   observers on the surface of a solved part, as #208 asks [R10].
4. **C ABI check**: the Python checker also calls the C functions through ctypes
   on the same cases (the extension's `.so` exports them). Depends on the C ABI
   decision.
5. **Later, own decisions**: the face-charge assembly matrix for a fixed mesh as
   value cases (a kernel output, named in #208's first comment) [R12]; norms and
   inequalities (decay, bounds), rates (convergence order); the mesh inside
   test; browser speed (per-pair switch to the 6-point rule, web workers, SIMD);
   magpylib.

Not in the files, by nature: iterative and nonlinear solves (tolerances,
pivoting, B-H Newton) [R12]; the object API (paths, objects, error and warning
wording); binding behaviour (threads, GIL, speed and memory budgets). Shapes and
broadcasting at the kernel level can be value cases.
