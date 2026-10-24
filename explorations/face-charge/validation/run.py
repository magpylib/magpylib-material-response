"""Run validation cases: python run.py [A B ...] [--redo]. A case with a result
file is skipped, so the run can stop at any time and resume where it was."""

import datetime
import json
import os
import sys

import numpy as np

import fc
from cases import CASES
from report import write_progress

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def log(msg):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(HERE, "log.txt"), "a") as f:
        f.write(f"- {stamp} {msg}\n")
    print(msg, flush=True)


def err(q, ref):
    """Max and RMS error over the observers. Fields (n, k): |difference| per
    point over the largest |reference|; values (n,): relative per point."""
    q, ref = np.asarray(q, float), np.asarray(ref, float)
    if q.ndim == 2:
        d = np.sqrt(np.nansum((q - ref) ** 2, axis=1))
        scale = np.nanmax(np.sqrt(np.nansum(ref**2, axis=1)))
        e = d / scale
    else:
        e = np.abs(q - ref) / np.abs(ref)
    return {"max": float(np.nanmax(e)), "rms": float(np.sqrt(np.nanmean(e**2))), "per_point": e.tolist() if e.size <= 30 else None}


def run_case(cid):
    c = CASES[cid]()
    obs = c["observers"]
    if "quantity" in c:
        qfun = c["quantity"]
    else:
        comps = c["comps"]
        qfun = lambda B: np.asarray(B)[:, comps]  # noqa: E731
    out = {"id": cid, "title": c["title"], "ref_source": c["ref_source"], "unit": c.get("unit", "B T"),
           "observers": len(obs), "fc": [], "mom": []}
    arrays = {}

    # face charge, coarse to fine
    for m in c["fc_levels"]:
        B, info = fc.solve(c["build"](), m, obs)
        q = qfun(B)
        arrays[f"fc_m{m}"] = q
        out["fc"].append(info)
        log(f"{cid}: face charge m={m}: {info['unknowns']} unknowns, build {info['build_s']:.1f} s, "
            f"couple+solve {info['solve_s']:.1f} s, field {info['eval_s']:.1f} s")

    # the MoM as its docs page runs it
    est = errs = None
    if "mom" in c:
        for r in c["mom"]():
            q = qfun(r["B"])
            arrays[f"mom_{r['cells']}"] = q
            out["mom"].append({"label": r["label"], "cells": r["cells"], "solve_s": r["solve_s"]})
            log(f"{cid}: MoM {r['label']}: {r['cells']} cells, {r['solve_s']:.1f} s")
    else:
        levels, est, errs = c["mom_study"]()
        for r in levels:
            Q = r["Q"]
            q = Q if c.get("study_gives") == "quantity" else qfun(Q)
            arrays[f"mom_{r['cells']}"] = q
            out["mom"].append({"label": r["label"], "cells": r["cells"], "solve_s": r["solve_s"]})
            log(f"{cid}: MoM study level {r['cells']} cells, {r['solve_s']:.1f} s")
        if est is not None:
            arrays["mom_estimate"] = est if c.get("study_gives") == "quantity" else qfun(est)
            out["mom_study_errors"] = None if errs is None else np.atleast_1d(errs).tolist()

    ref = c["reference"]
    if isinstance(ref, str):
        ref = arrays["mom_estimate"]
    else:
        ref = qfun(ref) if ("quantity" not in c and np.ndim(ref) == 2) else np.asarray(ref)
    arrays["reference"] = ref

    for k, info in zip([f"fc_m{m}" for m in c["fc_levels"]], out["fc"], strict=True):
        info["error"] = err(arrays[k], ref)
    if len(c["fc_levels"]) >= 2:
        a, b = (f"fc_m{m}" for m in c["fc_levels"][-2:])
        out["fc_last_change"] = err(arrays[a], arrays[b])
    for r in out["mom"]:
        r["error"] = err(arrays[f"mom_{r['cells']}"], ref)
    if "mom_estimate" in arrays and not isinstance(c["reference"], str):
        out["mom_estimate_error"] = err(arrays["mom_estimate"], ref)

    tmp = os.path.join(RES, f".{cid}.tmp")
    np.savez(tmp + ".npz", **{k: np.asarray(v) for k, v in arrays.items()})
    with open(tmp + ".json", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp + ".npz", os.path.join(RES, f"{cid}.npz"))
    os.replace(tmp + ".json", os.path.join(RES, f"{cid}.json"))
    best = out["fc"][-1]
    log(f"{cid}: done. Face charge m={best['m']}: max error {best['error']['max']:.1e} vs reference; "
        f"MoM finest: {out['mom'][-1]['error']['max']:.1e}")


if __name__ == "__main__":
    redo = "--redo" in sys.argv
    ids = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CASES)
    os.makedirs(RES, exist_ok=True)
    for cid in ids:
        if os.path.exists(os.path.join(RES, f"{cid}.json")) and not redo:
            print(f"{cid}: has a result, skipped (--redo to run it again)")
            continue
        log(f"{cid}: start")
        run_case(cid)
        write_progress()
