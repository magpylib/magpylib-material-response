"""PR figure: memory and time of the blocks between two bodies whose grids differ."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BEFORE, AFTER = "#eb6834", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

# (label, plate cells, magnet cells, kept fraction of the forced CSR); both directions
cases = [("0.6 mm", 8712, 700, 1.0, False), ("0.45 mm", 21296, 1521, 1.0, True), ("0.4 mm", 30000, 2250, 0.991, True)]
dense = [2 * 72 * a * b / 1e9 for _, a, b, _, _ in cases]           # 8 B per entry, 9 entries per pair
csr = [2 * 108 * a * b * k / 1e9 if forced else d for (_, a, b, k, forced), d in zip(cases, dense)]  # 12 B per kept entry

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200, gridspec_kw={"width_ratios": [1.15, 1]})
fig.patch.set_facecolor(SURF)
for ax in (ax1, ax2):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)

# 1. memory
x = range(len(cases)); w = 0.34
b1 = ax1.bar([i - w / 2 - 0.01 for i in x], csr, w, color=BEFORE, label="main: stored sparse above 2 GB")
b2 = ax1.bar([i + w / 2 + 0.01 for i in x], dense, w, color=AFTER, label="this PR: dense")
for bars in (b1, b2):
    for r in bars:
        ax1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.25, f"{r.get_height():.1f}", ha="center", va="bottom", fontsize=9, color=INK)
ax1.set_xticks(list(x), [f"{c[0]} cells\n{c[1] + c[2]:,} in all" for c in cases])
ax1.set_ylabel("GB, both directions")
ax1.set_ylim(0, 19.5)
ax1.yaxis.grid(True, color=GRID, lw=0.8); ax1.set_axisbelow(True)
ax1.set_title("Memory of the blocks between plate and magnet", loc="left", fontsize=11, color=INK, pad=22)
ax1.text(0, 1.02, "χ = 1000: the row budget keeps 99–100 % of the entries", transform=ax1.transAxes, fontsize=9, color=INK2)
# while a forced block is built: 24 B per entry of index triplets, then a concatenated copy
for i, (_, a, b, k, forced) in enumerate(cases):
    if forced:
        ax1.text(i - w / 2 - 0.01, csr[i] + 1.25, f"≈ {2 * 24 * 9 * a * b * k / 1e9:.0f} GB\nwhile built", ha="center", va="bottom", fontsize=8, color=INK2)
ax1.text(-0.05, dense[0] + 1.6, "below 2 GB:\nalready dense", fontsize=8.5, color=INK2, ha="center")
ax1.legend(loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(0, 0.93))

# 2. time
labels = ["0.5 mm, one grid\n(17,152 cells)", "0.4 mm, main\n(32,250 cells)", "0.4 mm, this PR\n(32,250 cells)"]
vals = [7.0 / 60, 12.0, 122.2 / 60]
cols = [INK2, BEFORE, AFTER]
y = [2, 1, 0]
ax2.barh(y, vals, height=0.42, color=cols)
ax2.set_yticks(y, labels)
ax2.set_xlabel("minutes")
ax2.set_xlim(0, 15)
ax2.xaxis.grid(True, color=GRID, lw=0.8); ax2.set_axisbelow(True)
ax2.text(vals[0] + 0.2, 2, "7 s (FFT between the grids)", va="center", fontsize=9, color=INK)
ax2.text(11.9, 0.55, "stopped at 12 min, 10.8 GB,\nstill building", va="center", ha="right", fontsize=9, color=INK)
ax2.text(vals[2] + 0.2, 0, "2 min, peak 10 GB", va="center", fontsize=9, color=INK)
ax2.annotate("", xy=(14.6, 1), xytext=(12.0, 1), arrowprops=dict(arrowstyle="->", color=BEFORE, lw=1.5, ls=(0, (2, 2))))
ax2.set_title("apply_demag, magnet on iron plate", loc="left", fontsize=11, color=INK, pad=22)
ax2.text(0, 1.02, "0.4 mm cells: 5 mm plate → 12 cells of 0.4167, 4 mm magnet → 10 of 0.4", transform=ax2.transAxes, fontsize=9, color=INK2)

fig.tight_layout()
fig.savefig("pr86_dense_cross.png", facecolor=SURF)
print([round(v, 2) for v in csr], [round(v, 2) for v in dense])
