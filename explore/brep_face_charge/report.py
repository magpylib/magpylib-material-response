"""Figure of the arms: error near the curves against unknowns, one panel per
part.  python report.py -> results/error_vs_unknowns.png, and a summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREYS = {12: "#c3c2b7", 24: "#9a988f", 48: "#6f6d66", 96: "#3a3936"}
TITLES = {
    "plate": "Plate with fillets and a hole: curves a small part of it",
    "pole": "Pole piece with a filleted rim: curves most of it",
}


def rows(part):
    out = [json.loads(line) for line in (HERE / "results" / f"arms_{part}.jsonl").open()]
    return [r for r in out if "refused" not in r]


def series(part):
    data = rows(part)
    s = {}
    for n in (12, 24, 48, 96):
        s[f"export, {360 / n:g}° facets"] = (
            GREYS[n],
            [r for r in data if r["arm"] == "export" and r["facets_per_turn"] == n],
        )
    s["solver facets the B-rep, widened"] = (BLUE, [r for r in data if r["arm"] == "brep" and r["widened"]])
    s["solver facets the B-rep, on the curve"] = (
        ORANGE,
        [r for r in data if r["arm"] == "brep" and not r["widened"]],
    )
    return s


def err(r):
    return 100 * r["errors"]["near the curves"]["of_B"]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, part in zip(axes, ("plate", "pole")):
        for label, (color, rs) in series(part).items():
            rs = sorted(rs, key=lambda r: r["level"])
            x = [r["unknowns"] for r in rs]
            y = [err(r) for r in rs]
            brep = not label.startswith("export")
            ax.plot(x, y, "-o", color=color, lw=2, ms=7, mec="white", mew=1.5,
                    zorder=4 if brep else 3, label=label if brep else None)
            if not brep:  # the facet angle, at the line's end
                ax.annotate(label.replace("export, ", "").replace(" facets", ""),
                            (x[-1], y[-1]), xytext=(7, -1), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK2)
        ref = json.loads((HERE / "results" / f"reference_{part}.json").read_text())
        ax.axhline(100 * ref["groups"]["near the curves"]["error"], color=INK2, lw=1, ls=":")
        ax.text(5.8e4, 100 * ref["groups"]["near the curves"]["error"] * 1.15,
                "the reference's own error", fontsize=8, color=INK2, ha="right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e3, 6e4)
        ax.set_ylim(2e-3, 8)
        ax.set_title(TITLES[part], fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("unknowns (levels 1, 2, 3 along each line)", color=INK2)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("error near the curves, % of B", color=INK2)
    axes[0].plot([], [], "-o", color=GREYS[48], lw=2, ms=7, mec="white", mew=1.5,
                 label="a CAD export, its facets fixed at 30°, 15°, 7.5°, 3.75°")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=9, frameon=False, labelcolor=INK2)
    fig.suptitle(
        "Face charge on CAD parts: facets cut by the solver from the exact geometry, widened to keep the volume, "
        "reach a given error with fewer unknowns",
        fontsize=11, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = HERE / "results" / "error_vs_unknowns.png"
    fig.savefig(out, dpi=130)
    print("saved", out)
    for part in ("plate", "pole"):
        print(part)
        for r in rows(part):
            e, f = r["errors"]["near the curves"], r["errors"]["60 mm away"]
            print(f"  {r['arm']:6s} {r['facets_per_turn']:3d}/turn w={r['widened']!s:5} L{r['level']}: "
                  f"{r['unknowns']:6d} unk {r['seconds']:6.2f} s  near {100*e['of_B']:.3f} %  far {100*f['of_response']:.3f} % of response")


if __name__ == "__main__":
    main()
