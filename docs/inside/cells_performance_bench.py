"""The measurements behind `cells_performance.md`, stored next to this script
in `cells_performance.json`. The docs are built on a machine with 2 CPUs, whose
timings would say little, so the page plots these instead.

    uv run --group docs python docs/inside/cells_performance_bench.py

Run it on a quiet machine after a change to the solvers, and commit the JSON.
"""

import datetime
import json
import platform
import subprocess
import time
import tracemalloc
from collections import Counter
from pathlib import Path

import magpylib as magpy
import numpy as np

import magpylib_material_response as mmr
from magpylib_material_response import solve_cells
from magpylib_material_response.demag_fft import analyze_collection
from magpylib_material_response.meshing import mesh_Cuboid, mesh_Cylinder

SIZES_DIRECT = [216, 512, 1000, 1728]
SIZES_ITERATIVE = [216, 512, 1000, 1728, 4096, 8000]


def cpu_name():
    """The processor's marketing name where the system gives it."""
    try:
        if platform.system() == "Darwin":
            cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True
            ).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.machine()


def max_rel_diff(coll_a, coll_b):
    """Maximum relative polarization difference between two solved collections."""
    pol_a = np.array([s.polarization for s in coll_a.sources_all])
    pol_b = np.array([s.polarization for s in coll_b.sources_all])
    return np.abs(pol_a - pol_b).max() / np.abs(pol_a).max()


def unit_cube(pos=(0, 0, 0)):
    return magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1e-3, 1e-3, 1e-3), position=pos
    )


def measure_solve(target_elems, solver):
    """Wall time, then peak memory in a second, traced run, of one meshed cube."""
    coll = magpy.Collection(mesh_Cuboid(unit_cube(), target_elems=target_elems))
    t0 = time.perf_counter()
    solve_cells(coll, susceptibility=3.0, solver=solver, solver_tol=1e-8)
    dt = time.perf_counter() - t0
    tracemalloc.start()
    solve_cells(coll, susceptibility=3.0, solver=solver, solver_tol=1e-8)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "solver": solver,
        "cells": len(coll.sources_all),
        "time": dt,
        "peak_mb": peak / 1e6,
    }


def grid_positions(k, pitch=1.8e-3):
    side = int(np.ceil(np.sqrt(k)))
    return [(i * pitch, j * pitch, 0) for i in range(side) for j in range(side)][:k]


def topologies():
    """The models of the page's topology section, by label."""
    out = {
        "single body": magpy.Collection(mesh_Cuboid(unit_cube(), 1000)),
        "4 bodies": magpy.Collection(
            *(mesh_Cuboid(unit_cube(p), 250) for p in grid_positions(4))
        ),
        "24 bodies": magpy.Collection(
            *(mesh_Cuboid(unit_cube(p), 48) for p in grid_positions(24))
        ),
    }
    meshes = []
    for i, p in enumerate(grid_positions(24, pitch=2.2e-3)):
        m = mesh_Cuboid(unit_cube(p), 48)
        m.rotate_from_angax(i * 15.0, (0, 0, 1), anchor=p)
        meshes.append(m)
    out["24 bodies, individually rotated"] = magpy.Collection(*meshes)
    cyl = magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(1e-3, 1e-3))
    out["meshed cylinder"] = magpy.Collection(mesh_Cylinder(cyl, 100))
    return out


def bench_topology(label, coll):
    n = len(coll.sources_all)
    sus = [0.5] * n
    t0 = time.perf_counter()
    coll_direct = solve_cells(coll, susceptibility=sus, solver="direct")
    t_direct = time.perf_counter() - t0
    t0 = time.perf_counter()
    coll_iter = solve_cells(
        coll, susceptibility=sus, solver="iterative", solver_tol=1e-8
    )
    t_iter = time.perf_counter() - t0
    kinds = Counter(c["kind"] for c in analyze_collection(coll.sources_all)[1])
    return {
        "topology": label,
        "cells": n,
        "structure": " + ".join(f"{v} {k}" for k, v in sorted(kinds.items())),
        "iter / direct time": round(t_iter / t_direct, 2),
        "agreement": f"{max_rel_diff(coll_direct, coll_iter):.0e}",
    }


def main():
    # warm up both solvers, so that one-time costs stay out of the first timing
    warmup = magpy.Collection(mesh_Cuboid(unit_cube(), 64))
    solve_cells(warmup, susceptibility=3.0, solver="direct")
    solve_cells(warmup, susceptibility=3.0, solver="iterative", solver_tol=1e-8)
    records = [measure_solve(s, "direct") for s in SIZES_DIRECT]
    records += [measure_solve(s, "iterative") for s in SIZES_ITERATIVE]
    out = {
        "machine": f"{cpu_name()}, {magpy.defaults.threads} threads",
        "date": datetime.datetime.now(tz=datetime.UTC).date().isoformat(),
        "versions": {
            "magpylib": magpy.__version__,
            "magpylib-material-response": mmr.__version__,
        },
        "scaling": records,
        "topology": [
            bench_topology(label, coll) for label, coll in topologies().items()
        ],
    }
    Path(__file__).with_name("cells_performance.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
