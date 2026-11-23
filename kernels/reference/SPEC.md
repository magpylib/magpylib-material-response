# Reference files for the field kernels

Every implementation of a field kernel (NumPy, `array_api_strict`, the Rust
crate, WebAssembly, any later port, magpylib itself) must reproduce
`<family>.json` in this folder (#208). The files hold data only; this page says
what they mean.

## Format

One file per family (`cuboid`, `face_charge`, `current`, `sphere`): its kinds
and the relations that tie them. Top level:

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
triangle's plane or within the same band of it, on a triangle without area
(`charge_sheet_corners` drops it instead; corners in a line to within 100 eps of
the product of two sides), where a sheet's double and exact distance to its
center fall on different sides of `far_from` or a pair's side of `k L` (rounding
picks the branch), on a circular loop's wire (within today's 1e-15 r0 in its
plane, or the same band of it), of a cuboid with a negative dimension
(magpylib's formula then flips B) or a loop with a negative diameter, or at a
non-finite observer wait for the surface rules (magpylib #1073; proposal #1081)
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
  where that exceeds `rtol |expected|`. The tolerances are thus set by NumPy's
  rounding: a port must be about as accurate as the NumPy implementation, and a
  more accurate one is not held to more.

NumPy never sets a value. Where it is off by more than 1e-6 relative, `fill.py`
prints the point, exits 1 and leaves the file unchanged: a bug in the NumPy
implementation, to fix first.

A relation that holds at 50 digits (to 1e-30 of its terms' sizes) is an exact
law: its `atol` is 16 times numpy's residual, at least `rtol` times the sum of
its terms' sizes. Any other is an approximate law, and its `atol`, written in
the file by hand so that it is declared, must be the law's own 50-digit residual
plus 16 times numpy's difference from the 50-digit value. Both are rounded up to
a power of ten; `fill.py` writes the first, checks both (the power of ten above
is accepted too, as another platform's rounding can move numpy's error across
one), and refuses a non-finite residual or numpy off by more than 1e-6 of the
terms' size.

**Adding a case**: append a point with only `id`, `why`, `source` and `observer`
(or a relation with its terms and target; `atol` too if the law is approximate),
then run `python kernels/reference/fill.py kernels/reference/<family>.json`,
which refuses an entry without `why`. `--redo` recomputes the points' values
when the implementation changes.

## cuboid

`magpylib.func.cuboid_field`, in the cuboid's frame (centered, sides along the
axes). Parameters: `dimension` (3, m, none negative), `polarization` (J, 3, T).
Fields `B` and `H` (as `magpylib.func.cuboid_field`), `J` and `M` (by the inside
rule below; magpylib has them only internally); mu_0 = 1.25663706127e-6 (CODATA
2022, as magpylib).

- **inside**: `J = polarization` inside, 0 outside; `M = J / mu_0`;
  `H = (B - J) / mu_0`.
- **degenerate**: for a zero polarization, or a zero dimension with the observer
  not on the zero-thickness plate, `B = H = 0`.

**Band-sensitive** (in the file, regular today and under #1081's band of 100 eps
max(size, |p|); a wider band would make them face or edge points): `near-edge-*`
and `beside-edge-1e-12` (magpylib's #1066 tests), and the relations
`split-bar-*` and `jump-*`.

**Reserved, waiting for #1073**: `face-x+`, `face-x-`, `face-y+`,
`face-y+-by-edge`, `face-x+-tolerance`, `beyond-tolerance-x+`, `edge-x`,
`edge-z`, `corner-ppp`, `corner-mmm`, `zero-dimension` (on the zero-thickness
plate: B tends to J inside, not 0), `face-z+`, `issue-1058` (magpylib's #1058
test: on a face today, on an edge under #1081's band), `observer-nan`,
`observer-inf`, `negative-inside`, `negative-outside`, `table-negative` (a
negative dimension).

A magpylib checker skips `beside-edge-1e-12` and `near-edge-*` for magpylib <=
6.3.0, which gives NaN or is off there (fixed by #1066); that list is in the
test, not in the file.

The relation `far-is-dipole` ties a cuboid to the dipole `J V / mu_0` 100 sides
off, in the direction of magpylib's `test_dipole_approximation` (approximate;
its dipole in this file). Its own 1300 sides are not taken: there the cuboid's
formula in doubles is off by 3e-6 (1.5e-6 at 1000 sides, 4e-3 at 10000; magpylib
#1098), more than this file's rule allows.

Not taken from magpylib's tests: `test_vs_mag2` (magpylib 2's outputs; these
values are exact), `test_Cuboid_basics` (the pose). Waiting for other kinds: the
cuboid against charged triangles, tetrahedra, triangle strips and triangle
meshes.

## sphere, dipole

`magpylib.func.sphere_field` and `magpylib.func.dipole_field`, at the origin. A
dipole: `moment` (3, A m^2); fields `B` and `H`,
`H = (3 (m.r) r / r^5 - m / r^3) / (4 pi)`, `B = mu_0 H`; NaN at the dipole, as
magpylib's `getB` and `func.dipole_field`. A sphere: `diameter` (m),
`polarization` (J, 3, T); fields `B`, `H`, `J`, `M`. Outside (`r > d/2`), the
dipole of moment `J V / mu_0`, `V` its volume, and `J = 0`; inside, the surface
counted inside as magpylib does today (its points are reserved), `B = 2 J / 3`
and `J` the polarization; `H = (B - J) / mu_0`, `M = J / mu_0`.

Sources: `ball` 12 mm across, every component of `J` nonzero; `ball2` 2 cm
across with another `J`, so that a batch's rows differ; `ball_nm`, `ball_km` a
million times smaller and larger (B depends on `x / R` only); `dip` a generic
moment; `dip_ball` the moment `J`; `dip_a`, `dip_b`, `dip_ab` exact binary
moments and their sum. Points inside, at the center, at 0.999 and 1.001 radii,
along and across `J`, 100 radii out, at both scales; a dipole's along, across
and generic, 1e-9 m and 1 km out, and of `dip_a` and `dip_b`. From magpylib's
tests: `test_BHJM_magnet_sphere`'s four rows (`bhjm_ball_*`),
`test_BHJM_dipole`'s three (`bhjm_dip_*`), `test_func_dipole` and
`test_Dipole_basicB/H` (`dip_123`); `test_magnet_sphere_Bfield` is `center`,
`test_core_physics_dipole_sphere` the relation `sphere-outside-is-dipole`.
**Relations**: `sphere-outside-is-dipole` (approximate: the weight `V / mu_0` is
a rounded double), `sphere-b-h-j-inside`, `dipole-linear` (exact binary
moments), `dipole-b-mu0-h`.

**Reserved, waiting for #1073**: on a sphere's surface (`surface-*`); a diameter
not positive (`zero-diameter`, `negative-diameter`); at a dipole
(`dipole-origin`: NaN, magpylib's `test_Dipole_zero_position`); `observer-nan`,
`observer-inf`.

Not taken from magpylib's tests: `test_Sphere_basics`, `test_Dipole_basic*`'s
poses and `test_func_sphere*` (the pose and the interface, not the kernel).

## polyline

`magpylib.func.polyline_field` for one straight segment, in its own frame (no
position or orientation). Parameters: `segment_start`, `segment_end` (3, m),
`current` (A), flowing from start to end. Fields `B` in T and `H` in A/m,
`B = mu_0 H`. With `e` the unit direction, `L` the length, `r1 = x - start`,
`r2 = x - end`, their lengths `R1`, `R2` and `t1 = e.r1`, `t2 = e.r2`:
`H = I / (4 pi) k (e x r1)`, `k = (t1/R1 - t2/R2) / |e x r1|^2`; where
`t1 t2 > 0` (beyond an end) rationalized,
`k = L (t1 + t2) / (R1 R2 (t1 R2 + t2 R1))`. Magpylib subtracts the sines there
instead: they cancel beyond an end, more the farther along the axis and the
nearer to it (`axis-far-1e3` off by 3e-6, `axis-far-1e3-before` by 9e-6:
magpylib's check skips them; magpylib #1097).

- **on the line**: within 1e-15 L of the segment's line, its extensions
  included, `B = H = 0`, as magpylib.
- **without length or current**: `B = H = 0` everywhere (magpylib's
  `test_field_line_special_cases`, `test_Polyline_specials`).

Sources: `seg` 6 mm at a generic orientation off the origin, 2.5 A; `seg_km` a
million times larger, -1.5 A; `bin` in exact binary coordinates, its halves
`bin_first`, `bin_second` and `bin_reversed`. Points beside the middle, the
start and the end (square to it: the branches' boundary), beyond either end,
1000 lengths along the axis on either side (the rationalized branch), 1e-6 and
1e-9 lengths from the middle, 100 lengths away, and at the km scale. From
magpylib's tests: `test_BHJM_current_polyline`'s three rows (`bhjm_*`),
`test_field_line_special_cases`' normal case (`diag`), a segment without length
and one without current (`test_Polyline_specials`), `test_func_polyline4/6`
without the turn (`vertical`). **Relations**: `split-halves` (superposition,
exact midpoint), `reverse-current`, `b-mu0-h`; `long-wire`, a 2000 km segment
against the infinite wire (`test_Polyline_vs_Infinite`; approximate);
`square-loop-far-is-dipole`, four segments against the dipole `I A`
(`test_core_phys_moment_of_current_square`; approximate, its dipole in this
file).

**Reserved, waiting for #1073**: on the segment's line, its extensions included
(the distance there is rounding: `axis-beyond-end`; magpylib's
`test_field_line_special_cases` and `test_Polyline_specials` set 0 there, at an
end too); `observer-nan`, `observer-inf`.

Not taken from magpylib's tests: `test_vs_mag2_line` and `test_Polyline_basic*`
(magpylib 2's outputs, with poses); `test_core_physics_cube_current_replacement`
(4000 segments: it tests the discretization); `test_line_position_bug`,
`test_discontinous_line` (the interface). Waiting for other kinds: triangle
strips and current sheets. The polyline against `Circle` is not taken (see
circle).

## circle

`magpylib.func.circle_field` for one loop in its own frame: centered on the
origin in the `z = 0` plane, its current counter-clockwise about `+z`.
Parameters: `diameter` (m), `current` (A). Fields `B` in T and `H` in A/m,
`B = mu_0 H`. In cylindrical coordinates scaled by the radius `r0`, `H_r` and
`H_z` are each one complete elliptic integral `cel` (Ortner et al., Magnetism
2023), by Bulirsch's iteration from its first step until a step changes it by
less than `errtol` relative. The iteration converges quadratically, so `errtol`
is about the square root of the precision: magpylib's 1e-8 in doubles, 5e-26 at
50 digits (1e-8 there would stop near double precision). `H_x = H_r x / r`,
`H_y = H_r y / r` off the axis; on it `H_r` is exactly 0, and `H_x = H_y = 0`
(NaN for a NaN observer, as magpylib).

- **on the wire**: within 1e-15 r0 of the wire in the loop's plane, `B = H = 0`,
  as magpylib; also where the distance from the wire squares below the smallest
  normal double (about 1e-154 radii off it): Bulirsch's iteration never stops at
  0 (magpylib's hangs there, magpylib #1100), and divides by that square.
- **without diameter or current**: `B = H = 0` everywhere (magpylib's
  `test_field_loop_specials`).

Sources: `loop` 6 mm across, 2.5 A; `loop_reversed` its current reversed;
`loop_km` a million times larger, -1.5 A; `loop_axis_exact` with exact inputs.
Points at the center, on the axis above and below, 1e-9 radii off it, in the
plane inside and outside, generic and mirrored below, 1e-3, 1e-6 and 1e-9 radii
from the wire, 100 and 1e4 radii away, and at the km scale. From magpylib's
tests: `test_BHJM_circle`'s three rows (`loop_bhjm_*`), the on-axis values of
`test_core_physics_circle_VS_webpage_numbers` (`loop_webpage`) and of
`test_obj_Circle`'s `test_BHJM_circle` (`loop_d1`, `loop_d2`, the latter also
`test_func_circle`), `test_Circle_basic_B/H` (`loop_basic`),
`test_field_loop_specials` off the wire (`loop_d2_2a`, `loop_no_diameter`,
`loop_no_current`), and `test_loop_field`'s point in the loop's frame
(`loop_stability`). **Relations**: `circle-b-mu0-h`; `circle-on-axis`,
`H_z = I r0^2 / (2 (r0^2 + z^2)^(3/2))`, exactly 1 A/m for its inputs
(`test_current_circle_on_axis`); `circle-mirror` in the loop's plane;
`circle-turn-about-axis`; `circle-reverse-current`; `circle-scaling`, 1000 times
the size, observer and current (`test_scaling_loop`); `circle-far-is-dipole`,
against the dipole `I A` (`test_core_phys_moment_of_current_circle`;
approximate, its dipole in this file).

**Reserved, waiting for #1073**: on the wire (`circle-on-wire`: magpylib's
`test_field_loop_specials` sets 0 there); a negative diameter
(`circle-negative-diameter`: magpylib takes its absolute value);
`circle-observer-nan`, `circle-observer-inf`.

Not taken from magpylib's tests: `test_Circle_basic*`'s sensor, `test_repr`,
`test_Circle_centroid`, `test_func_squeeze` and `test_circle_dipole_moment` (the
interface); `test_loop_field`'s 1000 turns (the pose: an observer off the axis
by rounding is `circle-near-axis-1e-9`); `test_core_phys_circle_polyline`,
`test_Polyline_vs_Circle`, `test_core_physics_long_solenoid` and
`test_core_physics_current_replacement` (299 to 9999 segments, or 1000 to 5000
loops: they test the discretization); `test_dipole_approximation`'s loop, whose
moment is not the loop's `I A` (magpylib #1099). Waiting for other kinds:
`test_Circle_vs_Cylinder_field` and the cylinder's current replacement.

## triangle_linear_charge

A linear charge density on triangles, summed: MR's face-charge kernel. Not
magpylib's `triangle_charge`, a uniform charge with fields B and H
(`face_charge/_triangle.py`, `charge_field`). Parameters: `vertices` (N, 3, 3,
m), `charge` (N, 3, A/m): per triangle the coefficients `c0, c1, c2` of the
density `c0 + (c1 (u - u_c) + c2 (v - v_c)) / s`. For corners `a, b, c` (in that
order): `n = (b - a) x (c - a) / |(b - a) x (c - a)|`, `e1 = (b - a) / |b - a|`,
`e2 = n x e1`; `u`, `v` are the coordinates along `e1`, `e2`, `u_c`, `v_c` the
centroid's, and `s = sqrt(area)`. The corners' order thus fixes the sign of
`e2`, and with it of `c2`'s term. Positive charge fields away from itself. Field
`H` in A/m. In the `why`s a side is `s`, the triangle's `sqrt(area)`, except in
`off-plane-*`, `plain-*-sides-oblique` and `far-rule-vs-closed-*`, which count
the nominal 1 mm.

## charge_sheet

`triangle_linear_charge`, with the far rule: at or beyond `far_from` (m) from
`center` (m), `H` is that of six point charges per triangle, at the barycentric
points and weights of Dunavant's degree-4 rule, each carrying the density there
times its weight times the area; closer, the closed form. The points are the
three permutations of `(A, B, B)` and of `(C, D, D)`, with
`A = 0.816847572980459`, `B = 0.091576213509771`, `C = 0.108103018168070`,
`D = 0.445948490915965`; their weights `0.109951743655322` and
`0.223381589678011`, each divided by the sum of all six in double precision.
`far_from` 0 is the far rule everywhere.

`tolerance` sets the per-pair rule. At 0 (exact) every pair closer than
`far_from` takes the closed form, as MR's solver does. Above 0, a pair of
observer `x` and triangle takes the far rule where
`|x - centroid|^2 >= k^2 L^2`, `L` the triangle's longest edge: the centroid is
the mean of the corners, `((a + b) + c) / 3`; each square is summed x, y, z in
that order; `L^2` is the largest of `|b - a|^2`, `|c - b|^2`, `|a - c|^2`; and
`k^2` is from the first row of this table whose bound the tolerance reaches
(none below the last: exact):

| bound  | k^2  | bound  | k^2 | bound   | k^2  |
| ------ | ---- | ------ | --- | ------- | ---- |
| 1.3e-3 | 2.25 | 5.1e-6 | 36  | 2.0e-8  | 576  |
| 4.1e-4 | 4    | 1.6e-6 | 64  | 6.2e-9  | 1024 |
| 8.1e-5 | 9    | 3.2e-7 | 144 | 1.3e-9  | 2304 |
| 2.6e-5 | 16   | 1.0e-7 | 256 | 3.9e-10 | 4096 |

A pair taken by the far rule is then off by at most `bound` times the field its
charge would make all of one sign (area times the largest corner density, over
the distance squared). The bounds are `6.5e-3 / k^4`, rounded up: twice the
worst case an adversarial search found, a needle whose charge falls linearly
along it, seen along its axis at `k L` (`3.3e-3 / k^4` at k = 1.5,
`2.5e-3 / k^4` far out); measured at 50 digits, not proven.

Closer than that, a pair takes the middle rule where
`|x - centroid|^2 >= m^2 L^2` (the same sums), `m^2` from this second table,
chosen the same way: sixteen point charges per triangle, Dunavant's degree-8
rule, at the centroid (weight `0.144315607677787`), the permutations of
`(a, b, b)` for `(a, b) = (0.081414823414554, 0.459292588292723)` (weight
`0.095091634267285`), `(0.658861384496480, 0.170569307751760)`
(`0.103217370534718`) and `(0.898905543365938, 0.050547228317031)`
(`0.032458497623198`), and of
`(0.008394777409958, 0.263112829634638, 0.728492392955404)`
(`0.027230314174435`), the weights divided by their sum in double precision. Its
bounds are `1.7e-4 / m^8`, rounded up: twice the worst case found (the same
needle at 1.5, `8.3e-5 / m^8`; `5.6e-5 / m^8` far out).

| bound  | m^2  | bound  | m^2 | bound   | m^2 |
| ------ | ---- | ------ | --- | ------- | --- |
| 6.7e-6 | 2.25 | 2.6e-8 | 9   | 1.1e-10 | 36  |
| 6.7e-7 | 4    | 2.6e-9 | 16  | 1.1e-11 | 64  |

Closer still, the closed form. On the solved plate's grid the error of H
relative to |H| is about 3e-6 at a tolerance of 1e-3 and 3e-9 at 1e-6. A
negative or NaN tolerance, or one below both tables' last bounds, is exact.

Parameters: those of `triangle_linear_charge`, `center`, `far_from`,
`tolerance`. `center` and `far_from` are parameters here, so that a case can
switch where it chooses; `charge_sheet_corners` derives them as the solver does.

Sources: `tilted` three triangles, 1 mm across, acute, obtuse and right-angled,
at generic orientations off the origin, each with a charge of every coefficient
nonzero; `plain` one in the plane z = 0; `sliver` 1e6 times longer than wide;
`moved` plain 4 m from the origin (rounding in the frame); `nm`, `km` the first
tilted triangle at 1 nm and 1 km (the field of a charge density does not depend
on the length unit); `uncharged` no charge; `box_surface` a cuboid's 12 surface
triangles; `tri0`, `tri0_split` one triangle and its four midpoint parts;
`tilted_cycled`, `tilted_far`, `tilted_sheet`, `tilted_sheet_exact` tilted
turned, as a sheet by the far rule everywhere, as a sheet switching at 20 mm,
and as a sheet centered on the origin switching at 2^-6 m, where a point at
exactly `far_from` (an exact binary distance) tells "at or beyond" from
"beyond". `box_sheet_1e-3`, `box_sheet_1e-6` the cuboid's surface charge as
sheets at those tolerances, with points where the 12 pairs take different rules;
`box_sheet_switch` the same sheet switching whole at 5 cm, closer than its pairs
would; `pair_row_<k^2>` and `pair_middle_<m^2>` one triangle in exact binary
coordinates per row of each table, at that row's bound, with a point exactly at
`k L` (or `m L`) from its centroid and one just inside, each taking the rule its
`why` names. Points where the rules differ by less than twice the files'
tolerance are left out (`pair-row-64-inside`, `pair-middle-36-inside`,
`pair-middle-64-*`): there the closed form is only as accurate as NumPy, which
cannot tell the rules apart, and an error in those rows changes H by less than
that.

**Reserved, waiting for #1073**: points in a triangle's plane (the production
code lifts them 1e-12 sides off it, a convention: `plane-*`, `near-edge-*`,
`near-corner-*`, `plain-200-sides-x`); `zero-area-*`; `observer-nan`,
`observer-inf`, `observer-1e200`, `observer-1e-300`.

**Band-sensitive** (in the file, regular today and under #1081's band of 100 eps
max(size, |p|), about 2e-17 m for `plain`; a wider band would make them sheet
points): `off-plane-*`, 1e-3 to 1e-12 sides off the plane.

**Relations**: `box-vs-charge-*` a cuboid's H is the field of its surface charge
`J.n / mu_0` (approximate: `c0` is rounded to a double; two points outside and
one inside); `far-rule-vs-closed-*` the far rule against the closed form 70 and
200 sides away (approximate: the rule's own error); `split-triangle-*`
(approximate: the parts' coefficients are rounded doubles) a triangle's H is the
sum of its four midpoint parts' with the same linear density in each part's
frame (the linear terms); `cycle-axes` a cyclic permutation of the axes turns H
the same way (the frames follow the triangle).

Not in these files: the solved plate of studio (664 triangles) is in
`kernels/rust/js/plate.json`, as `charge_sheet_corners` takes it, a timing input
without expected values; the far rule's points and charges are not separate
fields.

## charge_sheet_corners

`charge_sheet` of a sheet as `ChargeSheet` holds it, which is what a saved page
sends the browser's kernel: no coefficients, center or far distance to carry, so
none that can disagree with the kernel reading them. Parameters: `vertices` (N,
3, 3, m), `charge_densities` (N, 3, A/m), the density at each triangle's corners
in their order, and `tolerance`. A triangle without area (corners in a line to
within 100 eps of the product of two sides, `_triangle.has_area`) is dropped
first: it carries no charge. Derived from the rest, as
`_triangle.sheet_parameters`:

- **coefficients**: the plane through the corners' densities. With each corner's
  `u`, `v` (`triangle_linear_charge`'s frame, from the centroid, over `s`) and
  differences from corner 0, `du_k = u_k - u_0` (likewise `dv_k`, `dq_k`):
  `det = du_1 dv_2 - du_2 dv_1`, `c1 = (dq_1 dv_2 - dq_2 dv_1) / det`,
  `c2 = (du_1 dq_2 - du_2 dq_1) / det`, `c0 = q_0 - c1 u_0 - c2 v_0`.
- **center**: the mean of the centroids.
- **far_from**: the largest distance of a corner from the center, plus 64
  (`FAR_SIDES`) times the longest edge of any triangle. For no triangle the
  center is 0 and `far_from` infinite.

Sources: `tilted_corners` tilted with a different density at each corner;
`tilted_corners_no_area` the same with a fourth triangle, its corners in a line;
`spread_corners` two triangles 1 m apart, whose `far_from` the reach sets;
`box_corners_1e-6`, `box_corners_1e-3` the cuboid's surface charge varying
linearly over it, at those tolerances. Points by the closed form, by per-pair
rules, and at 1.5 and 1.001 times the derived `far_from` (the far rule for the
whole sheet). At 1.001 a port with a larger `far_from` takes the closed form,
off by 1e-9; a center or reach computed wrong (the sum over N + 1, no reach)
fails there, in the opposite direction, or at `spread_corners`. Just inside
`far_from` the two rules differ by less than numpy's error, so a port's smaller
`far_from` is not seen: it is as accurate. The relation `corners-drop-no-area`
checks the dropped triangle. A checker's derived `far_from` is a rounded double:
a point within 1e-9 times `far_from` of it is out of scope (`fill.py` refuses
it).
