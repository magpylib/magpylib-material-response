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

## Add a case

Append a point with only its `id`, `source`, `observer` and `why`, then run
`fill.py` on the file (needs the `reference` dependency group). SPEC.md says how
values and tolerances are set and what `fill.py` refuses.

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
