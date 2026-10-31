"""Write PROGRESS.md from the result files."""

import json

from common import HERE, RESULTS


def load(prefix):
    files = sorted(RESULTS.glob(f"{prefix}*.json"))
    return json.loads(files[-1].read_text()) if files else None


def s1(d):
    rows = "\n".join(
        f"| {'yes' if r['holder'] else 'no'} | {r['gap_mm']:g} | {r['stray_mT']:g} "
        f"| {r['error_max_deg']:.3f} |"
        for r in d["rows"]
    )
    return f"""## 1. Angle sensor on a steel shaft (`s1_shaft_sensor.py`)

Face charge level {d['level']}, {d['angles']} angles x 2 air gaps x 2 stray fields,
with and without a soft holder: {d['solves']} solves in {d['wall_s']:.0f} s, peak
{d['peak_rss_gb']} GB. Unknowns {d['unknowns']}.

| holder | gap (mm) | stray (mT) | max angle error (deg) |
| --- | --- | --- | --- |
{rows}

- Without a holder the shaft is the only solved part, so the whole grid has one
  matrix: {d['no_holder_today_s']:.1f} s today, about
  {d['no_holder_kept_factorization_s']:.1f} s with a kept factorization (estimate:
  the first solve, then each configuration's source field and a back-substitution).
- With the holder the rotor turns against it: {d['with_holder_today_s']:.0f} s today.
- A holder 2 mm thick, as a real one would be, takes
  {d['thin_ring_level2']['unknowns']:,} unknowns alone at level 2
  ({d['thin_ring_level2']['seconds']:.1f} s); level 3 with the shaft is refused.
- By hand: the stray field is added back to getB (#71).
"""


def s2(d):
    a, b, c = d["a_magnet_fixed"], d["b_sensor_only"], d["c_magnet_recoil"]
    return f"""## 2. Tolerance Monte Carlo (`s2_tolerance_mc.py`)

8 degrees of freedom, {d['samples']} samples timed, extrapolated to {d['target']:,}.

| | unknowns | per sample | {d['target']:,} samples today | kept factorization |
| --- | --- | --- | --- | --- |
| (a) magnet fixed source | {a['unknowns']:,} | {a['per_sample_s']:.3f} s | {a['today_10k_s'] / 60:.0f} min | {a['kept_factorization_10k_s']:.0f} s |
| (c) magnet with recoil 0.05 | {c['unknowns']:,} | {c['per_sample_s']:.3f} s | {c['today_10k_s'] / 60:.0f} min | matrix changes |

- (b) Sensor tolerances alone need no new solve: one solve ({b['solve_s']:.2f} s)
  and one getB along a {d['target']:,}-pose sensor path ({b['getB_10k_poses_s']:.2f} s).
  This works today.
- Bz spread: {a['Bz_std_mT']:.1f} mT with all tolerances, {b['Bz_std_mT']:.1f} mT with the
  sensor's alone. The sensor sits just outside the magnet's edge, where the field
  is steep; the timing is the point here.
- The kept-factorization figure is conservative: it evaluates the magnet's field
  at the panels once per sample, where one call along a magnet path would batch it.
- A magnet with a recoil susceptibility is a solved part, so moving it changes the
  matrix. Each part's own block stays, so a solve that keeps them and recouples
  through the magnet's {c['unknowns'] - a['unknowns']:,} unknowns is the lever to try.
"""


def s3(d):
    return f"""## 3. Coils on a U-core (`s3_coils_sum.py`)

Face charge level {d['level']}, {d['unknowns']:,} unknowns, {d['sets']} sets of currents.

- Today, a solve per set: {d['today_s']:.1f} s ({d['per_set_s']:.2f} s each).
- Kept factorization (estimate): {d['kept_factorization_s']:.1f} s.
- Sums of one solve per coil at 1 A, written by hand: {d['by_sums_s']:.1f} s, and
  any further set costs nothing. Largest difference from the direct solves:
  {d['sums_vs_solves_worst_rel']:.1e} of the peak.
"""


def s4(d):
    def row(r):
        a, b = r["A"], r["B"]
        if "refused" in a or "refused" in b:
            return f"| {r['level']} | refused: {(a.get('refused') or b.get('refused'))} | | | | | |"
        return (
            f"| {r['level']} | {a['unknowns']:,} / {b['unknowns']:,} "
            f"| {a['Bz_centre_mT']:.3f} | {a['variation_ppm']:,.1f} "
            f"| {b['variation_ppm']:,.1f} | {r['B_minus_A_ppm']:+,.1f} "
            f"| {a['seconds'] + b['seconds']:.1f} |"
        )

    rows = "\n".join(row(r) for r in d["rows"])
    return f"""## 4. NMR pole pieces, comparing designs (`s4_nmr_pole_pieces.py`)

Iron discs 20 mm (A) against 24 mm (B), variation of Bz over +-5 mm on the axis.
Peak {d['peak_rss_gb']} GB.

| level | unknowns A / B | Bz A (mT) | A (ppm) | B (ppm) | B - A (ppm) | s |
| --- | --- | --- | --- | --- | --- | --- |
{rows}

- The verdict flips: B is less uniform at level 1, more uniform at levels 2 and 3,
  and the margin is still moving. Bz at the centre settled to 0.03 % by level 2.
- Level 3 of design A needs 20,640 unknowns, over the default `max_unknowns`
  (20,000); it ran with 21,000.
- The two designs get different panel layouts, so their difference carries the
  noise of both. `mesh_study` takes one model and one quantity, so the study of
  a difference is written by hand.
- #85 reports 1,138 ppm with iron, from the method of moments on 3,640 cells; here
  the variation along the axis is about 90 ppm. The issue does not say where its
  +-5 mm lies, so the two may not measure the same thing.

"""


def p1(runs):
    today = load("s2_")["a_magnet_fixed"]
    rows = "\n".join(
        f"| {r['mode']} | {r['magpylib']} | {r['samples']:,} | {r['per_sample_ms']:.2f} "
        f"| {r['total_s']:.1f} |"
        for r in runs
    )
    r = runs[0]
    return f"""## P1. Tolerance Monte Carlo with a kept factorization (`p1_tolerance_kept.py`, `kept.py`)

s2 (a) again: the plate solved once ({r['first_solve_s']:.2f} s, {r['unknowns']:,}
unknowns), its LU kept ({r['factor_s'] * 1e3:.0f} ms), then each sample a new load
(the magnet at {r['load_points']:,} points), a back-substitution and a read at its
own sensor pose. Today each sample is a solve: {today['per_sample_s'] * 1e3:.0f} ms.

| mode | magpylib | samples | ms per sample | total s |
| --- | --- | --- | --- | --- |
{rows}

- "loop" takes one sample a call, as a loop over a kept solver would; "path" takes
  all at once along magnet and sensor paths, as a solver that accepts paths could.
  "1059" is magpylib with #1059's threads, its source first on PYTHONPATH.
- Against apply_face_charge on {r['checked_samples']} samples the kept solver is off
  by at most {max(x['check_max_abs_error_T'] for x in runs) * 1e9:.1f} nT, on Bz of
  {min(x['check_Bz_range_T'][0] for x in runs) * 1e3:.0f} to {max(x['check_Bz_range_T'][1] for x in runs) * 1e3:.0f} mT.
"""


def caching():
    """The cache's sections, from the c* results."""
    today = load("c1_")
    loop = load("c2_")
    plate = sorted(load_all("c5_"), key=lambda r: r["level"])
    shaft = {r["holder"]: r for r in load("c4_shaft_cache_level2_angles12_keep2")["rows"]}
    drag = load("c6_")["rows"]
    wrong = [load(f"c3_cache_correct_keep{k}") for k in (1, 2)]
    frozen = load("c7_")
    plate_rows = "\n".join(
        f"| {r['level']} | {r['unknowns']:,} | {r['plain_call_s'] * 1e3:,.0f} | "
        f"{r['first_cached_call_s'] * 1e3:,.0f} | {r['hit_call_s'] * 1e3:,.0f} | "
        f"{r['plain_call_s'] / r['hit_call_s']:.1f} |"
        for r in plate
    )
    drag_rows = "\n".join(
        f"| {r['level']} | {r['unknowns']:,} | {r['plain_ms']:.0f} | {r['hit_ms']:.0f} |"
        for r in drag
    )
    frozen_rows = "\n".join(
        f"| {r['case']} | {r['x_mm']} | {r['gap_mm']} | {r['unknowns_here']:,} | "
        f"{r['error_mT']:.5f} | {r['error_share_of_B'] * 100:.3f} | "
        f"{r['move_changes_B_mT']:.2f} |"
        for r in frozen["rows"]
    )
    worst = max(r["relative_error"] for d in wrong for r in d["rows"])
    polar = frozen["solved_part_polarization"]
    return f"""## C. A cache behind apply_face_charge (`cached.py`, `c1`-`c7`)

On magpylib {today['magpylib']} and material-response 0.7.0. The cache swaps
`solve_system` inside `apply_face_charge`, so a user's call stays as it is.

- Monte Carlo, per call: {today['per_sample_ms']:.0f} ms today, {loop['per_sample_ms']:.0f} ms with the cache.
- Wrong answers: none in {len(wrong[0]['rows'])} sequences, worst difference {worst:.0e},
  keeping an LU at the first sighting and at the second.
- Turning shaft, 48 configurations: {shaft[False]['plain_s']:.1f} s today, {shaft[False]['cached_s']:.1f} s.
- Shaft inside its holder: {shaft[True]['plain_s']:.0f} s today, {shaft[True]['cached_s']:.0f} s. Each matrix comes
  back once, so factoring it is wasted.

The plate at rising levels, one call in ms:

| level | unknowns | today | the call that factors | a hit | gain |
| --- | --- | --- | --- | --- | --- |
{plate_rows}

A solve per frame of a magnet moving over the plate, ms (studio #96):

| level | unknowns | today | a hit |
| --- | --- | --- | --- |
{drag_rows}

### Why "rigid" is the wrong test for a path (`c7_frozen_layout.py`)

Everything kept from a path's first step, against a full solve at each position.
The plate never moves.

| case | x (mm) | gap (mm) | unknowns of a full solve | error (mT) | of B (%) | the move changed B by (mT) |
| --- | --- | --- | --- | --- | --- | --- |
{frozen_rows}

A solved part's polarization kept from the first step is off by
{polar[-1]['kept_from_first_step_mT']:.0f} mT at {polar[-1]['Jz_T']} T; with one load per step, by
{polar[-1]['per_step_mT']:.0e} mT.
"""


def load_all(prefix):
    return [json.loads(f.read_text()) for f in sorted(RESULTS.glob(f"{prefix}*.json"))]


parts = ["# Use cases against today's API\n",
         "Each script times a real use case with the package as it is on main, and notes "
         "what had to be written by hand. Results are in `results/`; a script skips a "
         "result that exists. Run from this folder with the package's environment.\n"]
for prefix, fmt in (("s1_", s1), ("s2_", s2), ("s3_", s3), ("s4_", s4)):
    d = load(prefix)
    parts.append(fmt(d) if d else f"## {prefix} not run yet\n")
runs = [r for r in load_all("p1_") if r["samples"] >= 1000]
if runs:
    parts.append(p1(sorted(runs, key=lambda r: (r["samples"], r["magpylib"], r["mode"]))))
parts.append(caching())
(HERE / "PROGRESS.md").write_text("\n".join(parts))
print("wrote PROGRESS.md")
