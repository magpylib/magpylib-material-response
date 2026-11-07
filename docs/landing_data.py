"""The data behind the scrolling story on the landing page, `index.md`, stored in
`_static/landing/data.json`: field lines of the magnet on the plate of
`getting_started.md` while the plate's susceptibility rises and while the magnet
lifts off it, the holding force, how each solver cuts the plate, and the field on
the axis by each solver. It takes about two minutes here.

    uv run --group docs python docs/landing_data.py

Run it after a change to the solvers, and commit the JSON.
"""

import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull

from magpylib_material_response import solve_cells, solve_face_charge, solve_fem
from magpylib_material_response.meshing import mesh_uniform

MM = 1e-3
#: the window of the drawing, in mm, in the plane y = 0
X0, X1, Z0, Z1 = -32.0, 32.0, -14.0, 22.0
STEP = 0.125  # mm, the grid the field lines are traced on
LINES = 16  # field lines through the magnet
POINTS = 200  # points per drawn line
CHI = (0, 0.3, 1, 3, 10, 30, 100, 300, 1000)
GAPS = (0, 0.25, 0.5, 1, 2, 3, 5, 8)  # mm between the magnet and the plate


def model(chi=1000.0, gap=0.0):
    """The magnet on the plate of the getting-started page, `gap` mm above it."""
    magnet = magpy.magnet.Cuboid(
        dimension=(10 * MM, 10 * MM, 5 * MM),
        position=(0, 0, (5.5 + gap) * MM),
        polarization=(0, 0, 1.2),
    )
    plate = magpy.magnet.Cuboid(
        dimension=(40 * MM, 40 * MM, 6 * MM), polarization=(0, 0, 0)
    )
    plate.susceptibility = chi
    return magpy.Collection(magnet, plate)


def field_grid(solved, inside_plate):
    """Bx and Bz on the grid, in T: from face charge outside the plate, and
    inside it from `inside_plate`, a finite-element solution. Face charge reads
    mu0 H inside, too small to follow next to the field of the faces close by."""
    x = np.arange(X0, X1 + STEP / 2, STEP)
    z = np.arange(Z0, Z1 + STEP / 2, STEP)
    X, Z = np.meshgrid(x, z, indexing="ij")
    points = np.column_stack([X.ravel(), 0 * X.ravel(), Z.ravel()]) * MM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the warning that it reads mu0 H inside
        B = magpy.getB(solved, points).reshape(*X.shape, 3)
    inside = (np.abs(X) < 20) & (np.abs(Z) < 3)
    if inside_plate is not None:
        B[inside] = magpy.getB(inside_plate, points.reshape(*X.shape, 3)[inside])
    return x, z, B[..., 0], B[..., 2]


def seeds(solved, z_mid):
    """Points across the magnet's middle, spaced by equal flux, so that the
    density of the lines is the density of the flux."""
    x = np.linspace(-5, 5, 801)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bz = magpy.getB(solved, np.column_stack([x, 0 * x, 0 * x + z_mid]) * MM)[:, 2]
    flux = np.concatenate([[0], np.cumsum((bz[1:] + bz[:-1]) / 2)])
    return np.interp((np.arange(LINES) + 0.5) / LINES * flux[-1], flux, x)


def trace(x, z, bx, bz, starts, z_mid, sign):
    """Follow the direction of B from each start, with fourth-order Runge-Kutta
    steps of 0.1 mm, until the line leaves the window or comes back to the
    magnet's middle. Returns one array of points per line and whether it
    closed."""
    ix = RegularGridInterpolator((x, z), bx, bounds_error=False, fill_value=0)
    iz = RegularGridInterpolator((x, z), bz, bounds_error=False, fill_value=0)

    def direction(p):
        d = np.column_stack([ix(p), iz(p)])
        return sign * d / np.maximum(np.linalg.norm(d, axis=1), 1e-30)[:, None]

    h = 0.1
    p = np.column_stack([starts, np.full(len(starts), z_mid)])
    paths = [[q.copy()] for q in p]
    alive = np.ones(len(p), bool)
    closed = np.zeros(len(p), bool)
    length = 0.0
    for _ in range(6000):
        q = p[alive]
        k1 = direction(q)
        k2 = direction(q + h / 2 * k1)
        k3 = direction(q + h / 2 * k2)
        k4 = direction(q + h * k3)
        new = q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        length += h
        idx = np.flatnonzero(alive)
        for j, i in enumerate(idx):
            paths[i].append(new[j].copy())
            out = not (X0 <= new[j, 0] <= X1 and Z0 <= new[j, 1] <= Z1)
            back = (
                length > 3
                and abs(new[j, 0]) < 5
                and (q[j, 1] - z_mid) * (new[j, 1] - z_mid) <= 0
                and sign * (new[j, 1] - q[j, 1]) > 0
            )
            if back:
                paths[i][-1] = np.array([starts[i], z_mid])
                closed[i] = True
            if out or back:
                alive[i] = False
        p[alive] = new[alive[idx]]
        if not alive.any():
            break
    return [np.array(path) for path in paths], closed


def resample(path, n=POINTS):
    """`n` points equally spaced along the path."""
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])])


def field_lines(chi, gap):
    """The lines of one frame, as integers in 0.01 mm, x and z interleaved, and
    the strength of B along each, 0 to 99 on a log scale from 1 mT to 3 T."""
    solved = solve_face_charge(model(chi, gap)) if chi else model(chi, gap)
    fem = solve_fem(model(chi, gap), order=3) if chi else None
    x, z, bx, bz = field_grid(solved, fem)
    z_mid = 5.5 + gap
    # the model is symmetric in x: trace the right half, mirror the left
    starts = seeds(solved, z_mid)[LINES // 2 :]
    forward, closed = trace(x, z, bx, bz, starts, z_mid, 1)
    backward, _ = trace(x, z, bx, bz, starts, z_mid, -1)
    forward += [f * (-1, 1) for f in forward]
    backward += [b * (-1, 1) for b in backward]
    closed = np.concatenate([closed, closed])
    strength = RegularGridInterpolator(
        (x, z), np.hypot(bx, bz), bounds_error=False, fill_value=1e-3
    )
    lines = []
    for f, b, c in zip(forward, backward, closed, strict=True):
        path = f if c else np.vstack([b[::-1], f[1:]])
        # where the line passes through the magnet's middle, which the page
        # grows it from: its start if it closes, else where the halves meet
        back = 0.0 if c else np.linalg.norm(np.diff(b, axis=0), axis=1).sum()
        total = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        path = resample(path)
        level = np.log10(np.clip(strength(path), 1e-3, 3)) + 3
        lines.append(
            {
                "xz": np.round(path * 100).astype(int).ravel().tolist(),
                "b": np.round(level / np.log10(3000) * 99).astype(int).tolist(),
                "closed": bool(c),
                "seed": round(back / total * (POINTS - 1)),
            }
        )
    return solved, lines


def above(solved, gap):
    """Bz 0.5 mm above the magnet's top, in mT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(magpy.getB(solved, (0, 0, (8.5 + gap) * MM))[2] * 1e3)


def face_charge_slice(solved):
    """Where the plate's sheet crosses y = 0: segments in mm with the charge at
    each end, in A/m, and the panel each lies on."""
    sheet = solved.children[1]
    v = np.asarray(sheet.vertices) / MM
    out = []
    for face, sigma, panel in zip(
        sheet.faces, sheet.charge_densities, sheet.panels, strict=True
    ):
        y = v[face, 1]
        cuts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (y[a] < 0) != (y[b] < 0):
                t = y[a] / (y[a] - y[b])
                p = v[face[a]] + t * (v[face[b]] - v[face[a]])
                cuts.append((p[0], p[2], sigma[a] + t * (sigma[b] - sigma[a])))
        if len(cuts) == 2:
            (xa, za, sa), (xb, zb, sb) = cuts
            out.append(
                [
                    *(round(c, 3) for c in (xa, za, xb, zb)),
                    round(sa),
                    round(sb),
                    int(panel),
                ]
            )
    return out


def cells_slice(solved):
    """The plate's cells that cross y = 0: centre, size in mm and polarization
    in T, x and z."""
    out = []
    for cell in solved.sources_all:
        if not isinstance(cell, magpy.magnet.Cuboid) or cell.dimension[0] > 9 * MM:
            continue
        y, dy = cell.position[1], cell.dimension[1]
        if abs(1e-6 - y) < dy / 2 and abs(cell.position[2]) < 3 * MM:
            pos, dim, pol = cell.position / MM, cell.dimension / MM, cell.polarization
            out.append(
                [
                    *(round(c, 3) for c in (pos[0], pos[2], dim[0], dim[2])),
                    round(pol[0], 4),
                    round(pol[2], 4),
                ]
            )
    return out


def fem_slice(solution):
    """The tetrahedra crossing y = 0 near the parts, cut into polygons in mm."""
    mesh = solution._model
    ng = mesh.mesh.ngmesh
    pts = np.array([[p[0], p[1], p[2]] for p in ng.Points()])
    pts = (pts * mesh.length + np.asarray(mesh.center)) / MM
    elements = [[v.nr - 1 for v in el.vertices] for el in ng.Elements3D()]
    near = np.linalg.norm(pts[:, [0, 2]], axis=1) < 1.5 * mesh.radius * mesh.length / MM
    out = []
    for element in elements:
        if not near[element].all():
            continue
        p = pts[element]
        y = p[:, 1] + 1e-9
        if (y > 0).all() or (y < 0).all():
            continue
        # the elements are convex: the cut is the hull of where the plane
        # crosses the segments between their corners
        a, b = np.triu_indices(len(p), 1)
        crossing = (y[a] < 0) != (y[b] < 0)
        a, b = a[crossing], b[crossing]
        t = (y[a] / (y[a] - y[b]))[:, None]
        cut = (p[a] + t * (p[b] - p[a]))[:, [0, 2]]
        if not (
            (cut[:, 0] > X0 - 5).all()
            and (cut[:, 0] < X1 + 5).all()
            and (cut[:, 1] > Z0 - 5).all()
            and (cut[:, 1] < Z1 + 5).all()
        ):
            continue
        if len(cut) > 3:
            cut = cut[ConvexHull(cut).vertices]
        out.append(np.round(cut * 100).astype(int).ravel().tolist())
    return out


def machine():
    """The processor's name, which the timings belong to."""
    if platform.system() == "Darwin":
        cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout.strip()
    return platform.processor() or platform.machine()


def main():
    data = {"window": [X0, X1, Z0, Z1], "machine": machine()}

    data["chi"] = []
    for chi in CHI:
        solved, lines = field_lines(chi, 0)
        data["chi"].append(
            {"chi": chi, "bz": round(above(solved, 0), 1), "lines": lines}
        )

    data["lift"] = []
    for gap in GAPS:
        solved, lines = field_lines(1000, gap)
        force = magpy.getFT(solved.children[1], solved.children[0], meshing=2000)[0][2]
        data["lift"].append(
            {
                "gap": gap,
                "force": round(float(-force), 2),
                "bz": round(above(solved, gap), 1),
                "lines": lines,
            }
        )

    z = np.linspace(8.5, 30, 44)
    axis = np.column_stack([0 * z, 0 * z, z]) * MM
    plain = model()
    solves, seconds = {}, {}
    for name, solve in (
        ("face charge", lambda: solve_face_charge(plain)),
        ("cells", lambda: solve_cells(mesh_uniform(plain, 4))),
        ("finite elements", lambda: solve_fem(plain, order=4, observers=axis)),
    ):
        t0 = time.perf_counter()
        solves[name] = solve()
        seconds[name] = round(time.perf_counter() - t0, 3)
    data["solvers"] = {
        "seconds": seconds,
        "face charge": face_charge_slice(solves["face charge"]),
        "cells": cells_slice(solves["cells"]),
        "finite elements": fem_slice(solves["finite elements"]),
        "panels": len(set(np.asarray(solves["face charge"].children[1].panels))),
        "cell count": len(solves["cells"].sources_all) - 1,
        "fem elements": int(solves["finite elements"].elements),
    }
    data["axis"] = {"z": np.round(z, 3).tolist()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data["axis"]["magnet alone"] = np.round(
            magpy.getB(plain.children[0], axis)[:, 2] * 1e3, 3
        ).tolist()
        for name, solved in solves.items():
            bz = magpy.getB(solved, axis)[:, 2] * 1e3
            data["axis"][name] = np.round(bz, 4).tolist()

    out = Path(__file__).parent / "_static" / "landing" / "data.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
