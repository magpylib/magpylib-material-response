# Reference files for the field kernels

Every implementation of a field kernel (NumPy, `array_api_strict`, the Rust
crate, WebAssembly, any later port, magpylib itself) must reproduce
`<family>.json` in this folder (#208). The files hold data only; this page says
what they mean.

## Format

One file per family (`cuboid`, `face_charge`): its kinds and the relations that
tie them. Top level:

| key         | value                                                        |
| ----------- | ------------------------------------------------------------ |
| `rtol`      | the relative tolerance of every comparison                   |
| `sources`   | name → source: `kind` and the kind's parameters              |
| `points`    | list of points                                               |
| `relations` | list of relations: laws that tie fields at several observers |

A **source** is `kind` (the `magpylib.func` name without `_field`, e.g.
`cuboid`; a kind magpylib lacks gets a name in its style), `why`, and that
function's keyword parameters, singular (`dimension`, `polarization`, ...). Core
functions only: no position or orientation; the source sits in its own frame. A
file may hold sources of several kinds: its relations tie them (a cuboid to the
charge on its surface).

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
(an issue, a commit, a skip list) a point is `<family>/<id>`, e.g.
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
in #1081, 100 eps max(size, |p|)), on a zero-thickness cuboid, in a charged
triangle's plane or within the same band of it, on a triangle without area (corners in a
line to within 100 eps of the product of two sides), within 1e-12 of `far_from`
from a sheet's center (where rounding picks the branch), or at a non-finite
observer wait for the surface rules (magpylib #1073; proposal #1081)
and are added once those are decided. Their ids are reserved: listed per kind
below, to be used with the same meaning.

## Where the values come from

`fill.py <file>` fills what is missing; it holds no points, and refuses a point
or a relation's observer outside the scope above. For each missing field of a
point:

- expected = the one implementation
  (`src/magpylib_material_response/_array_api/`) run at 50 digits on
  `mp_namespace.py`, rounded to the nearest double. A point without a finite
  value there is refused: it is out of scope.
- `atol_<field>` = 16 times numpy's own error, rounded up to a power of ten,
  where that exceeds `rtol |expected|`.

NumPy never sets a value. Where it is off by more than 1e-6 relative, `fill.py`
prints the point and writes no `atol`, and the checkers fail on it.

A relation that holds at 50 digits (to 1e-30 of its terms' sizes) is an exact
law: its `atol` is 16 times numpy's residual, at least `rtol` times the sum of
its terms' sizes. Any other is an approximate law, and its `atol`, written in
the file by hand so that it is declared, must be the law's own 50-digit
residual plus 16 times numpy's difference from the 50-digit value. Both are
rounded up to a power of ten; `fill.py` writes the first, checks both, and
refuses a non-finite residual or numpy off by more than 1e-6 of the terms'
size.

**Adding a case**: append a point with only `id`, `why`, `source` and `observer`
(or a relation with its terms and target; `atol` too if the law is approximate),
then run `python kernels/reference/fill.py kernels/reference/<family>.json`, which
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

## triangle_charge

Charge on triangles, summed: MR's face-charge kernel
(`face_charge/_triangle.py`, `charge_field`). Parameters: `vertices` (N, 3, 3,
m), `charge` (N, 3, A/m): per triangle the coefficients `c0, c1, c2` of the
density `c0 + (c1 (u - u_c) + c2 (v - v_c)) / s`. For corners `a, b, c` (in that
order): `n = (b - a) x (c - a) / |(b - a) x (c - a)|`, `e1 = (b - a) / |b - a|`,
`e2 = n x e1`; `u`, `v` are the coordinates along `e1`, `e2`, `u_c`, `v_c` the
centroid's, and `s = sqrt(area)`. The corners' order thus fixes the sign of
`e2`, and with it of `c2`'s term. Positive charge fields away from itself.
Field `H` in A/m. The parameters are shared by all observers, unlike the
cuboid's per-observer rows. In the `why`s a side is `s`, the triangle's
`sqrt(area)`.

## charge_sheet

`triangle_charge`, with the far rule: at or beyond `far_from` (m) from `center`
(m), `H` is that of six point charges per triangle, at the barycentric points
and weights of Dunavant's degree-4 rule, each carrying the density there times
its weight times the area; closer, the closed form. The points are the three
permutations of `(A, B, B)` and of `(C, D, D)`, with `A = 0.816847572980459`,
`B = 0.091576213509771`, `C = 0.108103018168070`, `D = 0.445948490915965`;
their weights `0.109951743655322` and `0.223381589678011`, each divided by the
sum of all six in double precision (`_triangle.py`: `_FAR_POINTS`,
`_FAR_WEIGHTS`). `far_from` 0 is the far rule everywhere. Parameters: those of
`triangle_charge`, `center`, `far_from`. The solver's `center` (the mean of the
centroids) and `far_from` (the reach plus a number of sides) are parameters
here, not derived.

Sources: `tilted` three triangles, 1 mm across, acute, obtuse and right-angled,
at generic orientations off the origin, each with a charge of every coefficient
nonzero; `plain` one in the plane z = 0; `sliver` 1e6 times longer than wide;
`moved` plain 4 m from the origin (rounding in the frame); `nm`, `km` the first
tilted triangle at 1 nm and 1 km (the field of a charge density does not depend
on the length unit); `uncharged` no charge; `box_surface` a cuboid's 12 surface
triangles; `tri0`, `tri0_split` one triangle and its four midpoint parts;
`tilted_cycled`, `tilted_far`, `tilted_sheet` tilted turned, as a sheet by the
far rule everywhere, and as a sheet switching at 20 mm.

**Reserved, waiting for #1073**: points in a triangle's plane (the production
code lifts them 1e-12 sides off it, a convention: `plane-*`, `near-edge-*`,
`near-corner-*`, `plain-200-sides-x`); `zero-area-*`; `observer-nan`,
`observer-inf`, `observer-1e200`, `observer-1e-300`.

**Band-sensitive** (in the file, regular today and under #1081's band of
100 eps max(size, |p|), about 2e-17 m for `plain`; a wider band would make them
sheet points): `off-plane-*`, 1e-3 to 1e-12 sides off the plane.

**Relations**: `box-vs-charge-*` a cuboid's H is the field of its surface charge
`J.n / mu_0` (approximate: `c0` is rounded to a double; two points outside and
one inside); `far-rule-vs-closed-*` the far rule against the closed form 70 and
200 sides away (approximate: the rule's own error); `split-triangle-*` (approximate: the parts'
coefficients are rounded doubles) a
triangle's H is the sum of its four midpoint parts' with the same linear density
in each part's frame (the linear terms); `cycle-axes` a cyclic permutation of
the axes turns H the same way (the frames follow the triangle).

Not in these files: the solved plate of studio (664 triangles) is in
`kernels/rust/www/plate.json`, a timing input without expected values; the far
rule's points and charges are not separate fields.
