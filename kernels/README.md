# Field kernels (#208)

The files in `reference/` say what the field kernels compute. Every
implementation is checked against them: the Python one in the package (on NumPy
and `array_api_strict`), the Rust crate natively, and its WebAssembly build for
studio's field lines in the browser.

```
reference/   the single source of truth
  SPEC.md          format, comparison rule, per-kind conventions, every id
  *.json           points and relations, data only (one file per family)
  fill.py          fills in a new point's values (50 digits, rounded)
  mp_namespace.py  the 50-digit array namespace fill.py runs the kernels on
  show.py          plots a file's points
rust/        one crate, built for the browser (wasm feature)
  src/             the kernels (cuboid.rs, triangle.rs), the browser binding (wasm.rs)
  tests/           the Rust checker
  js/              the WebAssembly checker, and plate.json for its timing
```

The Python kernels live in the package:
`src/magpylib_material_response/_array_api/cuboid.py` and
`src/magpylib_material_response/face_charge/_triangle.py`.

## Run the checkers

```sh
uv run pytest tests/test_kernel_reference.py      # Python

cd kernels/rust
cargo test --release                              # Rust, one thread
cargo test --release --features parallel          # Rust, threads too
wasm-pack build --release --target web --out-dir pkg -- --features wasm
node js/check.mjs                                 # WebAssembly, then the plate's timing
```

CI runs all of them. The wasm build needs SIMD, which `rust/.cargo/config.toml`
switches on.

## The browser API

`rust/src/wasm.rs`, built by the commands above. Arrays are flat `Float64Array`s
in C order; every source is in its own frame (a caller moves observers into it
and turns the field back); a wrong argument throws.

```js
api_version(); // 2: a caller refuses any other
cuboid_field(field, observers, dimensions, polarizations); // magpylib.func.cuboid_field: "B" or "H"; one cuboid (3,) or one per observer (M, 3)
const sheet = new ChargeSheet(triangles, charge_densities); // ChargeSheet.from_triangles' arguments
sheet.h(observers, tolerance); // H in A/m; B = mu_0 H outside the body only
```

`ChargeSheet.from_coefficients` and `charge_field` exist for `js/check.mjs`
only. Raise `api_version` with every change a caller must follow.

## Add a case

A case starts where a kernel is used or goes wrong: a use case met while
developing (a solve, a studio page), or a bug found, in MR or in magpylib (many
magpylib issues were found while developing MR: #1066's cases are in
`cuboid.json`, #1058's id is reserved until #1073). Before it becomes an entry:

1. **Is it in scope?** SPEC.md's Scope: a value the mathematics defines, off
   every face, edge, corner and triangle plane. If its value waits for a rule
   still under discussion (#1073), reserve its id in SPEC.md and stop there.
2. **Is it covered?** Search the file for the same source and region: a point
   that already checks it, or a source to reuse. A new source is listed in
   SPEC.md's section of its kind.
3. **Point or relation?** Where a law ties fields (B − μ₀H = J, a body against
   its parts, a symmetry), write a relation: a wrong formula cannot regenerate
   its way past it. Otherwise a point.
4. **Write it**: `id` (a new one, its group first: `near-edge-*`; never reused),
   `why` (one line, citing the SPEC rule it relies on), `source`, `observer`. No
   values.
5. **Fill it**:
   `uv run --group reference python kernels/reference/fill.py <file>`. It
   refuses a point out of scope, with the reason; it exits 1 where NumPy
   disagrees with the 50-digit value: a bug in the implementation, to fix before
   the entry goes in.
6. **Check every implementation** (the commands above). A failure in one is a
   bug in that port. In magpylib, it is filed there, and the Python test skips
   the point for the releases before its fix, as for #1066.
7. **Commit**: the file's diff shows only the new entry; `fill.py --redo`
   changes nothing.

A new kind needs more: its rules in a SPEC.md section, its Python implementation
on the array API (it must run on `mp_namespace` for `fill.py`), and its line in
`fill.py`'s `KINDS`, the Python test's `IMPLS`, the Rust checker and
`js/check.mjs` (`charge_sheet_corners` is an example).

## The timing plate

`rust/js/plate.json` is studio's solved plate, kept only to time the browser
build: a magnet with J (0, 0, 1.2) T, 10 x 10 x 5 mm at z = 8.5 mm over an iron
plate of 60 x 30 x 4 mm, solved by face charge at level 1 (664 triangles). It
has no expected values.

## When magpylib #1073 is decided

Faces, edges, corners, the tolerance band and non-finite observers wait for it
(their ids are reserved in SPEC.md). The format needs no change:

- Multi-source cases are relations, each term with its own source, observer and
  weight, as `split-bar-*`. #1081's touching cells become "whole minus the sum
  of its cells = 0" on a shared face, J relations for the 1/2, 1/4 and 1/8
  shares, and points with `"nan"` for B at edges and corners.
- `fill.py`'s scope checks (`cuboid_scope`, `triangles_scope`) stop refusing
  what the rule then defines.
- The 50-digit value on a face: `mp_namespace` has no signed zero, so `fill.py`
  averages the two one-sided limits (steps of 1e-30 and 1e-40 of the size, which
  must agree).
- The implementations follow the rule. For the cuboid, forcing the sign of the
  zero in `x - a` each way gives both one-sided limits exactly (checked on x
  faces): today on an x face B is the outside limit while H subtracts J, so H
  matches neither side.

## Open

- A treecode for large meshes: #214.
- Kinds with elliptic integrals (cylinder, circle, cylinder segment) need `cel`
  in `mp_namespace` and an array API implementation.
