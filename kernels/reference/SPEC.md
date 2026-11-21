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

## Scope

A file holds only behaviour that is clear and agreed: values the mathematics
defines, at points where no convention under discussion changes them. Points on
a face, an edge or a corner (within today's 1e-15 h, or within the band proposed
in #1081, 100 eps max(size, |p|)), on a zero-thickness cuboid, or at a
non-finite observer wait for the surface rules (magpylib #1073; proposal #1081)
and are added once those are decided. Their ids are reserved: listed per kind
below, to be used with the same meaning.

## Where the values come from

`fill.py <file>` fills what is missing; it holds no points. For each missing
field of a point:

- expected = the one implementation
  (`src/magpylib_material_response/_array_api/`) run at 50 digits on
  `mp_namespace.py`, rounded to the nearest double. A point without a finite
  value there is refused: it is out of scope.
- `atol_<field>` = 16 times numpy's own error, rounded up to a power of ten,
  where that exceeds `rtol |expected|`.

NumPy never sets a value. Where it is off by more than 1e-6 relative, `fill.py`
prints the point and writes no `atol`, and the checkers fail on it.

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

- **inside**: `J = polarization` inside, 0 outside; `M = J / mu_0`;
  `H = (B - J) / mu_0`.
- **degenerate**: for a zero polarization, or a zero dimension with the observer
  off the cuboid's plane, `B = H = 0`.

**Band-sensitive** (in the file, regular today and under #1081's band of 100 eps
max(size, |p|); a wider band would make them face or edge points): `near-edge-*`
and `beside-edge-1e-12` (magpylib's #1066 tests), and the relations
`split-bar-*` and `jump-*`.

**Reserved, waiting for #1073**: `face-x+`, `face-x-`, `face-y+`,
`face-y+-by-edge`, `face-x+-tolerance`, `beyond-tolerance-x+`, `edge-x`,
`edge-z`, `corner-ppp`, `corner-mmm`, `zero-dimension` (on the zero-thickness
plate: B tends to J inside, not 0), `face-z+`, `issue-1058` (magpylib's #1058
test: on a face today, on an edge under #1081's band), `observer-nan`,
`observer-inf`.

A magpylib checker skips `beside-edge-1e-12` and `near-edge-*` for magpylib <=
6.3.0, which gives NaN or is off there (fixed by #1066); that list is in the
test, not in the file.

Not taken from magpylib's tests: `test_vs_mag2` (magpylib 2's outputs; these
values are exact), `test_Cuboid_basics` (the pose). Waiting for other kinds: the
cuboid against charged triangles, tetrahedra, polylines, triangle strips,
dipoles and triangle meshes.
