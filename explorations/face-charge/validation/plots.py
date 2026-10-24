"""One figure per finished case, results/<case>.png: python plots.py [A ...]

Sensor lines: the compared field along each line from the reference, the
finest face-charge level and the finest MoM, and below it both errors.
Single sensors: the error of every level of both methods.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from examples import CASES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FC, MOM = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, validated
NAMES = ["Bx", "By", "Bz"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def _lines(obs):
    """Observer index groups along x: one per distinct (y, z)."""
    keys = np.round(obs[:, 1:] * 1e9).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(inv.max() + 1)]
    return sorted(groups, key=lambda g: -obs[g[0], 2])  # the line nearest above first


def plot(cid):
    r = json.load(open(os.path.join(HERE, "results", f"{cid}.json")))
    d = np.load(os.path.join(HERE, "results", f"{cid}.npz"))
    c = CASES[cid]()
    obs = np.asarray(c["observers"])
    fc_key = f"fc_m{r['fc'][-1]['m']}"
    mom_key = f"mom_{r['mom'][-1]['cells']}"
    ref, fcq, mq = d["reference"], d[fc_key], d[mom_key]
    fc_label = f"face charge, m = {r['fc'][-1]['m']} ({r['fc'][-1]['unknowns']:,} unknowns)"
    mom_label = f"MoM, {r['mom'][-1]['label']} ({r['mom'][-1]['cells']:,} cells)"

    if ref.ndim == 2 or (ref.ndim == 1 and len(ref) > 10):  # fields along sensor lines
        comps = c.get("comps", [None])
        q2 = lambda a: a if a.ndim == 2 else a[:, None]  # noqa: E731
        ref2, fc2, m2 = q2(ref), q2(fcq), q2(mq)
        names = [NAMES[k] for k in comps] if comps[0] is not None else [r["unit"]]
        groups = _lines(obs)
        ncol = ref2.shape[1] * len(groups)
        fig, axes = plt.subplots(2, ncol, figsize=(3.6 * ncol + 0.6, 5.2), dpi=170, sharex="col",
                                 gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
        fig.patch.set_facecolor(SURF)
        scale = np.nanmax(np.sqrt(np.nansum(ref2**2, axis=1)))
        unit = 1e3 if "T" == r["unit"].split()[-1] else 1.0
        col = 0
        for g in groups:
            x = obs[g, 0] * 1e3
            for k, name in enumerate(names):
                a, b = axes[0, col], axes[1, col]
                _style(a), _style(b)
                a.plot(x, ref2[g, k] * unit, color=INK2, lw=2.4, alpha=0.45, label="reference")
                a.plot(x, m2[g, k] * unit, color=MOM, lw=1.2, ls=(0, (4, 2)), label=mom_label)
                a.plot(x, fc2[g, k] * unit, color=FC, lw=1.2, label=fc_label)
                lbl = f"{name} ({'mT' if unit == 1e3 else r['unit'].split()[-1]})"
                if len(groups) > 1:
                    lbl += f", z = {obs[g[0], 2] * 1e3:g} mm"
                a.set_title(lbl, loc="left", fontsize=9, color=INK)
                b.plot(x, 100 * np.abs(m2[g, k] - ref2[g, k]) / scale, color=MOM, lw=1.1, ls=(0, (4, 2)))
                b.plot(x, 100 * np.abs(fc2[g, k] - ref2[g, k]) / scale, color=FC, lw=1.1)
                b.set_xlabel("x (mm)")
                if col == 0:
                    b.set_ylabel("|error| (% of max |B|)")
                col += 1
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(f"{cid}. {r['title']}", x=0.01, ha="left", fontsize=11, color=INK)
        fig.text(0.01, 0.93, f"reference: {r['ref_source']}", fontsize=8.5, color=INK2)
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.915), ncol=3, frameon=False, fontsize=8.5)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
    else:  # single sensors: error of every level
        fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=170)
        fig.patch.set_facecolor(SURF)
        _style(ax)
        fc_err = [x["error"]["max"] * 100 for x in r["fc"]]
        fc_n = [x["unknowns"] for x in r["fc"]]
        m_err = [x["error"]["max"] * 100 for x in r["mom"]]
        m_n = [3 * x["cells"] for x in r["mom"]]
        ax.loglog(m_n, m_err, "o--", color=MOM, ms=6, label="MoM (unknowns = 3 × cells)")
        ax.loglog(fc_n, fc_err, "o-", color=FC, ms=6, label="face charge")
        ax.set_xlabel("unknowns")
        ax.set_ylabel("max error vs reference (%)")
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_title(f"{cid}. {r['title']}", loc="left", fontsize=10, color=INK)
        fig.text(0.01, 0.01, f"reference: {r['ref_source']}", fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(HERE, "results", f"{cid}.png")
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for cid in sys.argv[1:] or [k for k in CASES if os.path.exists(os.path.join(HERE, "results", f"{k}.json"))]:
        print(plot(cid))
