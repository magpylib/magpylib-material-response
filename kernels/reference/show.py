"""Plot a reference file of points for inspection: each source, its points
colored by group, every point's id, reason and expected values on hover.

    python kernels/reference/show.py [kernels/reference/cuboid.json] [--out page.html]

One view per source (a dropdown switches): the sources span 3 nm to 3000 km.
Far points start hidden, as do the B arrows (direction only); click their
legend entries to show them. Reads the file alone, not the generator: what
the plot shows is what every implementation is checked against.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.colors
import plotly.graph_objects as go

HIDDEN = {"far"}  # groups shown only on demand


def _num(v):
    return float({"nan": "nan", "inf": "inf", "-inf": "-inf"}.get(v, v))


def _box(dimension, name):
    """A cuboid's outline as a translucent mesh, centered, sides along the axes."""
    h = np.abs(np.array(dimension, float)) / 2
    corners = (
        np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        * h
    )
    i = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
    j = [1, 3, 5, 7, 1, 5, 3, 7, 2, 6, 3, 7]
    k = [3, 2, 7, 6, 5, 4, 7, 6, 6, 4, 7, 5]
    x, y, z = corners.T
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.15, color="grey",
                     name=name, hoverinfo="name", showlegend=False)  # fmt: skip


def _hover(p):
    fields = "<br>".join(f"{f} = {p[f]}" for f in ("B", "H") if f in p)
    flags = " (convention)" if p.get("convention") else ""
    needs = f"<br>needs: {p['needs']}" if "needs" in p else ""
    return f"<b>{p['id']}</b>{flags}<br>{p['why']}<br>{fields}{needs}"


def figure(doc, title):
    fig = go.Figure()
    groups = sorted({p["group"] for p in doc["points"]})
    colors = plotly.colors.qualitative.Dark24
    palette = {g: colors[n % len(colors)] for n, g in enumerate(groups)}
    owner = []  # the source of each trace, for the dropdown
    for name, src in doc["sources"].items():
        pts = [p for p in doc["points"] if p["source"] == name]
        if not pts:
            continue
        fig.add_trace(_box(src["dimension"], name))
        owner.append(name)
        for g in groups:
            sel = [p for p in pts if p["group"] == g]
            if not sel:
                continue
            X = np.array([[_num(v) for v in p["observer"]] for p in sel])
            ok = np.isfinite(X).all(axis=1)
            fig.add_trace(go.Scatter3d(
                x=X[ok, 0], y=X[ok, 1], z=X[ok, 2], mode="markers",
                marker={"size": 5, "color": palette[g],
                        "symbol": ["diamond" if p.get("convention") else "circle"
                                   for p, k in zip(sel, ok, strict=True) if k]},
                name=g, legendgroup=g, hovertext=[_hover(p) for p, k in zip(sel, ok, strict=True) if k],
                hoverinfo="text", visible="legendonly" if g in HIDDEN else True,
            ))  # fmt: skip
            owner.append(name)
        X = np.array([[_num(v) for v in p["observer"]] for p in pts])
        B = np.array([[_num(v) for v in p["B"]] for p in pts])
        keep = np.isfinite(X).all(axis=1) & np.isfinite(B).all(axis=1)
        keep &= np.linalg.norm(B, axis=1) > 0
        keep &= np.array([p["group"] not in HIDDEN for p in pts])
        U = B[keep] / np.linalg.norm(B[keep], axis=1)[:, None]
        fig.add_trace(go.Cone(
            x=X[keep, 0], y=X[keep, 1], z=X[keep, 2], u=U[:, 0], v=U[:, 1], w=U[:, 2],
            sizemode="absolute", sizeref=0.25 * float(np.max(np.abs(src["dimension"]))),
            anchor="tail", showscale=False, colorscale=[[0, "black"], [1, "black"]],
            name="B direction", visible="legendonly", showlegend=True, hoverinfo="skip",
        ))  # fmt: skip
        owner.append(name)
    names = list(dict.fromkeys(owner))
    buttons = [
        {"label": f"{n}: {doc['sources'][n].get('why', '')[:60]}", "method": "update",
         "args": [{"visible": [_initial(fig.data[i]) if o == n else False
                               for i, o in enumerate(owner)]}]}
        for n in names
    ]  # fmt: skip
    for i, o in enumerate(owner):
        fig.data[i].visible = _initial(fig.data[i]) if o == names[0] else False
    fig.update_layout(
        title=title,
        scene={"aspectmode": "data", "xaxis_title": "x (m)", "yaxis_title": "y (m)",
               "zaxis_title": "z (m)"},
        margin={"t": 110},
        updatemenus=[
            {"buttons": buttons, "x": 0, "y": 1.0, "xanchor": "left", "yanchor": "bottom"}
        ],
        legend={"title": "group (diamond: by convention)"},
    )  # fmt: skip
    return fig


def _initial(trace):
    """A trace's visibility when its source is shown."""
    if trace.name == "B direction" or trace.name in HIDDEN:
        return "legendonly"
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "file", nargs="?", default=Path(__file__).with_name("cuboid.json")
    )
    parser.add_argument("--out", help="write the page here instead of opening it")
    args = parser.parse_args()
    path = Path(args.file)
    doc = json.loads(path.read_text())
    if "points" not in doc:
        parser.error(f"{path.name} has no points: only files of points can be shown")
    fig = figure(doc, f"{path.name}: {len(doc['points'])} points")
    if args.out:
        fig.write_html(args.out, include_plotlyjs="cdn")
        print(f"wrote {args.out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
