"""Figure of the adversarial run: worst error over the four scenarios (three
drives, a magnet 1 mm off a curve), near the curves, as a share of the part's
own field, against unknowns.  -> results/adversarial.png"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {  # arm -> colour, label
    "export": ("#9a988f", "a CAD export, its facets fixed"),
    "B-rep, widened": ("#2a78d6", "solver facets the B-rep, widened"),
    "export, widened from the mesh": ("#1baf7a", "the export, widened from the mesh alone (no B-rep)"),
}
TITLES = {"plate": "Plate with fillets and a hole", "pole": "Pole piece with a filleted rim"}


def worst(part):
    rows = [json.loads(line) for line in (HERE / "results" / f"adv_arms_{part}.jsonl").open()]
    acc = defaultdict(list)
    for r in rows:
        acc[(r["arm"], r["facets_per_turn"], r["level"])].append(r)
    return {
        k: (v[0]["unknowns"], 100 * max(x["error_of_own_field"]["near the curves"] for x in v))
        for k, v in acc.items()
    }


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    for ax, part in zip(axes, ("plate", "pole")):
        w = worst(part)
        for arm, (color, label) in STYLE.items():
            pts = sorted((k, v) for k, v in w.items() if k[0] == arm)
            if arm == "export":  # one line per faceting, levels 2 -> 3
                for n in sorted({k[1] for k, _ in pts}):
                    xy = [v for k, v in pts if k[1] == n]
                    ax.plot(*zip(*xy), "-o", color=color, lw=2, ms=7, mec="white", mew=1.5, zorder=3)
                    ax.annotate(f"{360 / n:g}°", xy[-1], xytext=(7, 7), textcoords="offset points",
                                va="center", fontsize=8.5, color=INK2)
            elif arm == "B-rep, widened":
                xy = [v for k, v in sorted(pts, key=lambda p: p[0][2])]
                ax.plot(*zip(*xy), "-o", color=color, lw=2, ms=7, mec="white", mew=1.5, zorder=4,
                        label=label)
            else:
                for n in sorted({k[1] for k, _ in pts}):
                    xy = [v for k, v in pts if k[1] == n]
                    ax.plot(*zip(*xy), "-o", color=color, lw=2, ms=7, mec="white", mew=1.5, zorder=4,
                            label=label if n == 48 else None)
                    ax.annotate(f"{360 / n:g}°", xy[-1], xytext=(7, -9), textcoords="offset points",
                                va="center", fontsize=8.5, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1.5e3, 4e4)
        ax.set_ylim(0.08, 4)
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([2e3, 5e3, 1e4, 2e4]))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_major_locator(mpl.ticker.FixedLocator([0.1, 0.2, 0.5, 1, 2]))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g} %"))
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_title(TITLES[part], fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("unknowns (levels along each line)", color=INK2)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("worst error near the curves over 4 scenarios,\n% of the part's own field", color=INK2)
    axes[0].plot([], [], "-o", color=STYLE["export"][0], lw=2, ms=7, mec="white", mew=1.5,
                 label=STYLE["export"][1])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9,
               frameon=False, labelcolor=INK2)
    fig.suptitle("Widening an export from its mesh alone does as well as the B-rep, "
                 "except a coarse (15°) export of a doubly curved face", fontsize=11, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = HERE / "results" / "adversarial.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
