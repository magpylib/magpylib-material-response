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


parts = ["# Use cases against today's API\n",
         "Each script times a real use case with the package as it is on main, and notes "
         "what had to be written by hand. Results are in `results/`; a script skips a "
         "result that exists. Run from this folder with the package's environment.\n"]
for prefix, fmt in (("s1_", s1), ("s2_", s2), ("s3_", s3), ("s4_", s4)):
    d = load(prefix)
    parts.append(fmt(d) if d else f"## {prefix} not run yet\n")
(HERE / "PROGRESS.md").write_text("\n".join(parts))
print("wrote PROGRESS.md")
