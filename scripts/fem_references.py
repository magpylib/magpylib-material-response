"""Regenerate the finite-element reference data of the docs and the tests.

    python scripts/fem_references.py            # the files that do not exist yet
    python scripts/fem_references.py cuboids    # these, even if they exist

Needs the fem extra. Each model is solved by ``fem_study`` at orders 3, 4 and 5
on one mesh, with fine elements at the sensors, and its estimated error is
stored with it. About 5 minutes and 7 GB in all. The models are those of the docs pages and the tests that read
the files; change both together.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import magpylib as magpy
import ngsolve
import numpy as np

import magpylib_material_response
from magpylib_material_response.fem import fem_study

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "src" / "magpylib_material_response" / "package_data" / "datasets"
TESTDATA = ROOT / "tests" / "testdata"
ORDERS = (3, 4, 5)


def docs_cuboids():
    """docs/examples/cuboids_demagnetization.md"""
    cube1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.001))
    cube1.move((-0.0015, 0, 0))
    cube1.susceptibility = 0.3
    cube2 = magpy.magnet.Cuboid(
        polarization=(0.9, 0, 0), dimension=(0.001, 0.001, 0.001)
    )
    cube2.rotate_from_angax(-45, "y").move((0, 0, 0.0002))
    cube2.susceptibility = 1.0
    mx, my = 0.6 * np.sin(np.deg2rad(30)), 0.6 * np.cos(np.deg2rad(30))
    cube3 = magpy.magnet.Cuboid(
        polarization=(mx, my, 0), dimension=(0.001, 0.001, 0.002)
    )
    cube3.move((0.0016, 0, 0.0005)).rotate_from_angax(30, "z")
    cube3.susceptibility = 0.5
    sensor = np.linspace((-0.004, 0, -0.001), (0.004, 0, -0.001), 301)
    return magpy.Collection(cube1, cube2, cube3), sensor


def docs_soft_magnets():
    """docs/examples/soft_magnets.md. Three sensor lines, stored point by
    point across the lines, the row order of the page's dataframe."""
    cube1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.001, 0.001, 0.002))
    cube1.move((0, 0, 0.0005))
    cube1.susceptibility = 0.5
    cube2 = magpy.magnet.Cuboid(polarization=(0, 0, 0), dimension=(0.001, 0.001, 0.001))
    cube2.rotate_from_angax(angle=45, axis="y").move((0.0015, 0, 0))
    cube2.susceptibility = 3999
    lines = [
        np.linspace((-0.004, 0, z), (0.006, 0, z), 1001)
        for z in (-0.001, -0.003, -0.005)
    ]
    return magpy.Collection(cube1, cube2), np.stack(lines, axis=1).reshape(-1, 3)


def cube_magnet(polarization, susceptibility):
    """tests/test_isotropic_anisotropic.py: a 1 mm cube at the origin."""

    def build():
        magnet = magpy.magnet.Cuboid(
            dimension=(1e-3, 1e-3, 1e-3), polarization=polarization
        )
        magnet.susceptibility = susceptibility
        return magpy.Collection(magnet), np.loadtxt(TESTDATA / "grid_points.pts")

    return build


# name: (model, file, element size in the magnets, at the sensors)
REFERENCES = {
    "cuboids": (docs_cuboids, DATASETS / "FEMdata_test_cuboids.json", 0.25e-3, 0.05e-3),
    "soft_magnets": (
        docs_soft_magnets,
        DATASETS / "FEMdata_test_softmag.json",
        0.25e-3,
        0.05e-3,
    ),
    "isotropic": (
        cube_magnet((0, 0, 1.1), 0.1),
        TESTDATA / "isotropic_results_fem.txt",
        0.25e-3,
        0.03e-3,
    ),
    "anisotropic": (
        cube_magnet((0, 0, 1.1), (0.3, 0.2, 0.1)),
        TESTDATA / "anisotropic_results_fem.txt",
        0.25e-3,
        0.03e-3,
    ),
    # mu_r = -0.1 makes the system indefinite: conjugate gradients converge only
    # without fine elements at the sensors, to about 1 mT there (the test allows 6.5)
    "negative": (
        cube_magnet((0, 0, -0.1), -1.1),
        TESTDATA / "negative_susceptibility_fem.txt",
        0.25e-3,
        0.25e-3,
    ),
}


def method(study, maxh, observer_maxh, seconds):
    return {
        "solver": "magpylib_material_response.fem.fem_study",
        "orders": [int(p) for p in study.orders],
        "maxh_m": maxh,
        "observer_maxh_m": observer_maxh,
        "estimated_error": study.error,
        "seconds": round(seconds, 1),
        "ngsolve": ngsolve.__version__,
        "magpylib_material_response": magpylib_material_response.__version__,
    }


def write(path, text):
    """Write `text` to `path` through a temporary file, so that an interrupted
    run leaves no half-written reference."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main(names):
    for name, (model, path, maxh, observer_maxh) in REFERENCES.items():
        if (names and name not in names) or (not names and path.exists()):
            continue
        collection, sensors = model()
        start = time.perf_counter()
        study = fem_study(
            collection, sensors, ORDERS, maxh=maxh, observer_maxh=observer_maxh
        )
        seconds = time.perf_counter() - start
        B = study.estimate
        info = method(study, maxh, observer_maxh, seconds)
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            result = data["results"][0]
            result["computation"] = "FEM"
            result["value"] = {
                k: B[:, i].tolist() for i, k in enumerate(("Bx", "By", "Bz"))
            }
            result["method"] = info
            write(path, json.dumps(data, indent=2) + "\n")
        else:
            header = f"B in T at grid_points.pts: x y z Bx By Bz. {json.dumps(info)}"
            rows = np.column_stack([sensors, B])
            lines = [header, *(" ".join(f"{v: .16e}" for v in row) for row in rows)]
            write(path, "\n".join(lines) + "\n")
        print(
            f"{name}: {seconds:.0f} s, estimated error {study.error:.1e} -> {path.relative_to(ROOT)}",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
