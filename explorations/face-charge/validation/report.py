"""Rewrite PROGRESS.md from results/*.json and log.txt: python report.py"""

import json
import os

from cases import CASES

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(x):
    if x is None:
        return "–"
    return f"{100 * x:.2g} %" if x >= 1e-4 else f"{100 * x:.1g} %"


def _secs(s):
    return f"{s:.1f} s" if s >= 0.1 else f"{1e3 * s:.0f} ms"


def write_progress():
    rows, details, done = [], [], []
    for cid, make in CASES.items():
        path = os.path.join(HERE, "results", f"{cid}.json")
        if not os.path.exists(path):
            c = make()
            rows.append(f"| {cid} | {c['title']} | todo | {c['ref_source']} | | |")
            continue
        r = json.load(open(path))
        done.append(cid)
        f = r["fc"][-1]
        mom = r["mom"][-1]
        fc_cell = (f"{f['unknowns']:,} unknowns, {_secs(f['build_s'] + f['solve_s'])}: "
                   f"**max {pct(f['error']['max'])}**, rms {pct(f['error']['rms'])}")
        if "fc_last_change" in r:
            fc_cell += f" (last refinement changed it by {pct(r['fc_last_change']['max'])})"
        mom_cell = f"{mom['label']}, {_secs(mom['solve_s'])}: **max {pct(mom['error']['max'])}**, rms {pct(mom['error']['rms'])}"
        rows.append(f"| {cid} | {r['title']} | done | {r['ref_source']} | {fc_cell} | {mom_cell} |")
        lines = [f"### {cid}. {r['title']}", "", f"Reference: {r['ref_source']}. Compared: {r['unit']} at {r['observers']} observers.", "",
                 "| Method | Size | Time | Max error | RMS error |", "| --- | ---: | ---: | ---: | ---: |"]
        for x in r["fc"]:
            lines.append(f"| face charge, m = {x['m']} | {x['unknowns']:,} unknowns | {_secs(x['build_s'] + x['solve_s'])} "
                         f"| {pct(x['error']['max'])} | {pct(x['error']['rms'])} |")
        for x in r["mom"]:
            lines.append(f"| MoM, {x['label']} | {x['cells']:,} cells | {_secs(x['solve_s'])} "
                         f"| {pct(x['error']['max'])} | {pct(x['error']['rms'])} |")
        if "mom_estimate_error" in r:
            e = r["mom_estimate_error"]
            lines.append(f"| MoM study's extrapolation | | | {pct(e['max'])} | {pct(e['rms'])} |")
        details += lines + [""]
    todo = [c for c in CASES if c not in done]
    status = f"**{len(done)} of {len(CASES)} cases done.** " + (f"Next: {todo[0]}." if todo else "All done.")
    log = open(os.path.join(HERE, "log.txt")).read() if os.path.exists(os.path.join(HERE, "log.txt")) else ""
    text = f"""# Validation against the documented examples

Each documented example of material-response, solved by the face-charge method
and by the MoM as its page runs it, both against the page's reference. This file
is rewritten by `report.py` after every case; `run.py` appends the log below.

{status}

| Case | Model | Status | Reference | Face charge, finest level | MoM, docs settings, finest |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

Errors: for fields, |B − B_ref| at each observer over the largest |B_ref| on the
set; for gains and single sensors, relative per point. Times: face charge is
mesh, assembly and solve; the MoM is `apply_demag` alone, without meshing. The
face-charge mesh level m is set by hand, not converged by a study.

## How to resume

On branch `explore/face-charge`, from this folder, with material-response's
environment and `PYTHONPATH` at the branch's `src`:

```sh
PYTHONPATH=../../../src ./guard.sh 13 python run.py        # every case without a result
PYTHONPATH=../../../src ./guard.sh 13 python run.py C      # one case
PYTHONPATH=../../../src python run.py C --redo             # run it again
```

A case writes `results/<case>.json` (summary) and `results/<case>.npz` (the
compared values of every level and the reference) when it finishes, so stopping
loses at most the case that was running. `guard.sh 13` stops a run above 13 GB.

## Cases

{chr(10).join(details) if details else "(none finished yet)"}

## Log

{log}"""
    with open(os.path.join(HERE, "PROGRESS.md"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    write_progress()
