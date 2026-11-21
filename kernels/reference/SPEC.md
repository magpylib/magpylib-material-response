# Reference files for the field kernels

Every implementation of a field kernel (NumPy, `array_api_strict`, the Rust
crate, WebAssembly, any later port, magpylib itself) must reproduce
`<kind>.json` in this folder (#208). The files hold data only; this page says
what they mean.

## Format

One file per source kind. Top level:

| key         | value                                                        |
| ----------- | ------------------------------------------------------------ |
| `rtol`      | the relative tolerance of every comparison                   |
| `sources`   | name → source: `kind` and the kind's parameters              |
| `points`    | list of points                                               |
| `relations` | list of relations: laws that tie fields at several observers |

A **source** is `kind` (the `magpylib.func` name without `_field`, e.g.
`cuboid`), `why`, and that function's keyword parameters, singular (`dimension`,
`polarization`, ...). Core functions only: no position or orientation; the
source sits in its own frame.

A **point**: `id`, `why`, `source`, `observer` (3 numbers, m), one key per field
of the kind (`B` in T, `H` in A/m, `J` in T, `M` in A/m; 3 numbers each), and
`atol_<field>` where that field needs one (0 when absent).

A **relation**: `id`, `why`, `terms`, `target`, `atol`. A term is `source`,
`field`, `observer` and `weight`: a number, or a matrix of rows of 3
(`[[...], ...]`). The relation's value is the sum over the terms of
`weight @ field(observer)`; it must equal `target`, the law's own value, never a
computed residual. Each term has its own observer, so a shifted part needs no
pose.

An **id** names a point or relation as a test name does: a lowercase slug
(`a-z`, `0-9`, `.`, `+`, `-`), unique in its file, group first
(`near-edge-z-1e-6`), so that its prefix is its group. Once committed it is
never renamed or reused; a point whose meaning changes gets a new id. Elsewhere
(an issue, a commit, a skip list) a point is `<kind>/<id>`, e.g.
`cuboid/near-edge-z-1e-6`.

A **`why`** is one line saying what the entry checks, as a test's docstring
does. Checkers ignore it, except to print it beside a failure. A rule shared by
many points is written once, here, and a `why` cites it by name ("SPEC: faces").

## Encoding

- Numbers are JSON numbers, the shortest decimal that reads back to the same
  double; read them correctly rounded (serde_json: feature `float_roundtrip`; C:
  `strtod`).
- Non-finite values are the strings `"nan"`, `"inf"`, `"-inf"`.
- One key per line and every vector on one line, so that a diff names the value
  that moved.

## Comparison

Per field at a point, and per relation:
`|got - expected| <= rtol |expected| + atol`, Euclidean norm over the
components. An expected value with a non-finite component is matched exactly,
component by component (NaN by NaN, an infinity by the same infinity).

A checker reads every file, checks every point and relation whose kinds and
fields it has, and reports the ones it skips.

## Where the values come from

`fill.py <file>` fills what is missing; it holds no points. For each missing
field of a point:

- expected = the one implementation
  (`src/magpylib_material_response/_array_api/`) run at 50 digits on
  `mp_namespace.py`, rounded to the nearest double;
- `atol_<field>` = 16 times numpy's own error, rounded up to a power of ten,
  where that exceeds `rtol |expected|`;
- numpy's value instead, with no `atol`, where the 50-digit value is not finite,
  numpy's is not, or numpy is off by more than 1e-6 relative. `fill.py` prints
  these points; each must be one listed below.

A relation without `atol` must hold at 50 digits (to 1e-30 of its terms' sizes):
an exact law. It gets `atol` = 16 times numpy's residual, at least `rtol` times
the sum of its terms' sizes, rounded up to a power of ten. A relation with
`atol` is an approximate law, or one filled before: `fill.py` checks it at 50
digits and with numpy, and never changes it.

**Adding a case**: append a point with only `id`, `why`, `source` and `observer`
(or a relation with its terms and target; `atol` too if the law is approximate),
then run `python kernels/reference/fill.py kernels/reference/<kind>.json`, which
refuses an entry without `why`. `--redo` recomputes the points' values when the
implementation changes.

## cuboid

`magpylib.func.cuboid_field`, in the cuboid's frame (centered, sides along the
axes). Parameters: `dimension` (3, m), `polarization` (J, 3, T). Fields `B`,
`H`, `J`, `M`; mu_0 = 1.25663706127e-6 (CODATA 2022, as magpylib).

Conventions, by the names the `why`s cite, with half-dimensions
`h = |dimension| / 2`:

- **inside**: `|x_k| - h_k < 1e-15 h_k` on every axis, faces included. There
  `J = polarization`, else 0; `M = J / mu_0`; `H = (B - J) / mu_0`.
- **edges**: on an edge or a corner (within 1e-15 h of two faces' planes and
  inside along the third axis), `B = H = 0`; J and M as above.
- **degenerate**: for a zero dimension or polarization, `B = H = 0`.
- **faces**: on a face, B jumps by J. The side taken is the one IEEE signed
  zeros pick in magpylib's formulas (`atan2`), so these points take numpy's
  value. Outside, in a face's plane, the field is continuous and exact.

A magpylib checker skips `face-y+-by-edge`, `beside-edge-1e-12`, `issue-1058`
and `near-edge-*` for magpylib <= 6.3.0, which gives NaN or is off there (fixed
by #1066); that list is in the test, not in the file.

Not taken from magpylib's tests: `test_vs_mag2` (magpylib 2's outputs; these
values are exact), `test_Cuboid_basics` (the pose). Waiting for other kinds: the
cuboid against charged triangles, tetrahedra, polylines, triangle strips,
dipoles and triangle meshes.
