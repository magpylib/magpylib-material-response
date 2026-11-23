"""Plot a reference file for inspection: each source, its points colored by
group (the id up to its first '-'), the observers of the relations' terms,
every id and expected value on hover; what the ids mean is in ``SPEC.md``.

    python kernels/reference/show.py [kernels/reference/cuboid.json] [--out page.html]

One view per source (a dropdown switches): the sources span 3 nm to 3000 km.
Far points start hidden, as do the field arrows (direction only); click their
legend entries to show them. Reads the file alone: what the plot shows is
what every implementation is checked against.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.colors
import plotly.graph_objects as go

# groups shown only on demand: far points would shrink the source to a dot
HIDDEN = {"far"}
FIELDS = ("B", "H", "J", "M")


def _group(p):
    return p["id"].split("-")[0]


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


def _triangles(V, name):
    """Triangles (N, 3, 3) as a translucent mesh."""
    V = np.array(V, float).reshape(-1, 3)
    x, y, z = V.T
    i = np.arange(0, len(V), 3)
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=i + 1, k=i + 2, opacity=0.3, color="grey",
                     name=name, hoverinfo="name", showlegend=False)  # fmt: skip


def _ball(diameter, name):
    """A sphere's outline as a translucent mesh, centered."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 12))
    x, y, z = (np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
               * abs(diameter) / 2).reshape(3, -1)  # fmt: skip
    return go.Mesh3d(x=x, y=y, z=z, alphahull=0, opacity=0.15, color="grey",
                     name=name, hoverinfo="name", showlegend=False)  # fmt: skip


def _shape(src, name, X):
    """A source's outline and its size; a dipole, a point, takes the size of
    its observers' distances ``X``."""
    kind = src["kind"]
    if kind == "cuboid":
        return _box(src["dimension"], name), float(np.max(np.abs(src["dimension"])))
    if kind == "sphere":
        return _ball(src["diameter"], name), abs(float(src["diameter"]))
    if kind == "polyline":
        S = np.array([src["segment_start"], src["segment_end"]], float)
        line = go.Scatter3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], mode="lines",
                            line={"color": "grey", "width": 6}, name=name,
                            hoverinfo="name", showlegend=False)  # fmt: skip
        return line, float(np.linalg.norm(S[1] - S[0])) or 1.0
    if kind == "circle":
        t = np.linspace(0, 2 * np.pi, 73)
        r = abs(float(src["diameter"])) / 2
        ring = go.Scatter3d(x=r * np.cos(t), y=r * np.sin(t), z=0 * t, mode="lines",
                            line={"color": "grey", "width": 6}, name=name,
                            hoverinfo="name", showlegend=False)  # fmt: skip
        return ring, 2 * r or 1.0
    if kind == "dipole":
        dot = go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                           marker={"size": 6, "color": "grey", "symbol": "x"},
                           name=name, hoverinfo="name", showlegend=False)  # fmt: skip
        r = np.linalg.norm(X[np.isfinite(X).all(axis=1)], axis=1)
        return dot, float(np.median(r[r > 0])) if (r > 0).any() else 1.0
    V = np.array(src["vertices"], float).reshape(-1, 3)
    return _triangles(src["vertices"], name), float(np.max(np.ptp(V, axis=0)))


def _hover(p):
    if "term" in p:
        return f"<b>{p['relation']}</b><br>{p['why']}<br>{p['term']}<br>target {p['target']}"
    fields = "<br>".join(f"{f} = {p[f]}" for f in FIELDS if f in p)
    return f"<b>{p['id']}</b><br>{p.get('why', '')}<br>{fields}"


def _relation_points(doc):
    """Each relation's terms as points, grouped as 'relations'."""
    return [
        {"id": "relations", "relation": r["id"], "why": r.get("why", ""), "source": t["source"],
         "observer": t["observer"], "term": f"{t['field']} x {t['weight']}",
         "target": r["target"]}
        for r in doc.get("relations", []) for t in r["terms"]
    ]  # fmt: skip


def figure(doc, title):
    fig = go.Figure()
    relation_points = _relation_points(doc)
    groups = sorted({_group(p) for p in doc["points"] + relation_points})
    colors = plotly.colors.qualitative.Dark24
    palette = {g: colors[n % len(colors)] for n, g in enumerate(groups)}
    owner = []  # the source of each trace, for the dropdown
    for name, src in doc["sources"].items():
        pts = [p for p in doc["points"] if p["source"] == name]
        rel = [p for p in relation_points if p["source"] == name]
        if not pts + rel:
            continue
        observers = [[_num(v) for v in p["observer"]] for p in pts + rel]
        outline, size = _shape(src, name, np.array(observers))
        fig.add_trace(outline)
        owner.append(name)
        for g in groups:
            sel = [p for p in pts + rel if _group(p) == g]
            if not sel:
                continue
            X = np.array([[_num(v) for v in p["observer"]] for p in sel])
            ok = np.isfinite(X).all(axis=1)
            fig.add_trace(go.Scatter3d(
                x=X[ok, 0], y=X[ok, 1], z=X[ok, 2], mode="markers",
                marker={"size": 5, "color": palette[g],
                        "symbol": "diamond" if g == "relations" else "circle"},
                name=g, legendgroup=g, hovertext=[_hover(p) for p, k in zip(sel, ok, strict=True) if k],
                hoverinfo="text", visible="legendonly" if g in HIDDEN else True,
            ))  # fmt: skip
            owner.append(name)
        if not pts:
            continue
        X = np.array([[_num(v) for v in p["observer"]] for p in pts])
        # each point's field, whichever it carries
        B = np.array(
            [[_num(v) for v in next(p[f] for f in FIELDS if f in p)] for p in pts]
        )
        keep = np.isfinite(X).all(axis=1) & np.isfinite(B).all(axis=1)
        keep &= np.linalg.norm(B, axis=1) > 0
        keep &= np.array([_group(p) not in HIDDEN for p in pts])
        U = B[keep] / np.linalg.norm(B[keep], axis=1)[:, None]
        fig.add_trace(go.Cone(
            x=X[keep, 0], y=X[keep, 1], z=X[keep, 2], u=U[:, 0], v=U[:, 1], w=U[:, 2],
            sizemode="absolute", sizeref=0.25 * size,
            anchor="tail", showscale=False, colorscale=[[0, "black"], [1, "black"]],
            name="field direction", visible="legendonly", showlegend=True, hoverinfo="skip",
        ))  # fmt: skip
        owner.append(name)
    names = list(dict.fromkeys(owner))
    buttons = [
        {"label": n, "method": "update",
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
        legend={"title": "group (diamond: a relation's term)"},
    )  # fmt: skip
    return fig


def _initial(trace):
    """A trace's visibility when its source is shown."""
    if trace.name == "field direction" or trace.name in HIDDEN:
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
