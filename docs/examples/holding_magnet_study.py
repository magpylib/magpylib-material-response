"""The model and the stored results of `holding_magnet.md`: the pull of a disc
magnet on a steel plate against the gap, how it converges with face charge,
cells and finite elements, and the plate's thickness. The runs take minutes,
so they are stored in `holding_magnet.json`.

    uv run --group docs python docs/examples/holding_magnet_study.py

Run it after a change to a solver or to the model, and commit the JSON.
"""

import json
import warnings
from pathlib import Path

import magpylib as magpy
import numpy as np

from magpylib_material_response import solve_cells, solve_face_charge, solve_fem
from magpylib_material_response.meshing import mesh_uniform

MM = 1e-3  # lengths in metres, written in millimetres
GAPS = (0, 0.1, 0.25, 0.5, 1, 2, 4, 8)  # mm between the magnet and the plate
THICKNESSES = (1, 2, 4, 6)  # mm


def holding_magnet(gap=0.0, thickness=6.0):
    """A 10 x 5 mm NdFeB disc, `gap` mm above a 40 x 40 mm steel plate."""
    magnet = magpy.magnet.Cylinder(
        dimension=(10 * MM, 5 * MM),
        position=(0, 0, (2.5 + gap) * MM),
        polarization=(0, 0, 1.3),
    )
    plate = magpy.magnet.Cuboid(
        dimension=(40 * MM, 40 * MM, thickness * MM),
        position=(0, 0, -thickness / 2 * MM),
        polarization=(0, 0, 0),
    )
    plate.susceptibility = 1000
    return magnet, magpy.Collection(magnet, plate)


def pull(sources, magnet):
    """The force on the magnet toward the plate, in N, from one source or the
    sum over several."""
    force = np.asarray(magpy.getFT(sources, magnet, meshing=2000)[0])
    return float(-force.reshape(-1, 3).sum(axis=0)[2])


def plate_of(fem, magnet):
    """The plate's field in a finite-element solution: the solution's, less the
    fixed magnet's own, which pulls on nothing."""

    def field(field, observers):
        b = magpy.getB(fem, observers) - magpy.getB(magnet, observers)
        return b if field == "B" else b / magpy.mu_0

    return magpy.misc.CustomSource(field_func=field)


def flux_into_plate(solved):
    """The flux through the magnet's bottom face, in Wb, at contact: it turns
    sideways into the plate under the magnet's rim, through a wall of 2 pi r t."""
    r = np.linspace(0, 5, 101)[:-1] + 0.025
    points = np.column_stack([r, 0 * r, 0 * r + 0.05]) * MM
    bz = magpy.getB(solved, points)[:, 2]
    return np.sum(bz * 2 * np.pi * r * MM * 0.05 * MM)


def main():
    out = {"gaps": list(GAPS)}
    with warnings.catch_warnings():
        # a disc on a block warns that face charge converges slowly there:
        # these runs measure how slowly
        warnings.simplefilter("ignore")
        out["pull"] = []
        for gap in GAPS:
            magnet, model = holding_magnet(gap)
            out["pull"].append(
                pull(solve_face_charge(model, refine=2).children[1], magnet)
            )

        out["convergence"] = {}
        for gap in (0, 0.5):
            rows = []
            for label, kw in (
                ("face charge, default", {}),
                ("face charge, refine=1", {"refine": 1}),
                ("face charge, refine=2", {"refine": 2}),
                ("face charge, level=5, refine=2", {"level": 5, "refine": 2}),
            ):
                magnet, model = holding_magnet(gap)
                solved = solve_face_charge(model, **kw)
                unknowns = solved.face_charge_info["unknowns"]
                rows.append([label, unknowns, pull(solved.children[1], magnet)])
            for n in (4, 6, 8):
                magnet, model = holding_magnet(gap)
                cells = [
                    c
                    for c in solve_cells(mesh_uniform(model, n)).sources_all
                    if isinstance(c, magpy.magnet.Cuboid)
                ]
                rows.append(
                    [f"cells, {n} across the plate", len(cells), pull(cells, magnet)]
                )
            for order in (3, 4, 5):
                magnet, model = holding_magnet(gap)
                fem = solve_fem(model, order=order)
                rows.append(
                    [
                        f"finite elements, order {order}",
                        fem.ndof,
                        pull(plate_of(fem, magnet), magnet),
                    ]
                )
            out["convergence"][str(gap)] = rows

        out["thickness"] = []
        for t in THICKNESSES:
            magnet, model = holding_magnet(0, t)
            solved = solve_face_charge(model, refine=2)
            flux = flux_into_plate(solved)
            out["thickness"].append(
                {
                    "thickness": t,
                    "pull": pull(solved.children[1], magnet),
                    "flux": flux,
                    "rim": flux / (2 * np.pi * 5 * MM * t * MM),
                }
            )
    Path(__file__).with_name("holding_magnet.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
