"""Can the cache give a wrong answer? Sequences of models that change one thing
at a time, each solved with the cache and without, read at points around them.
A hit is fine only if the answer stays the same; each sequence also says how
many hits it expects, so a key that is too strict shows too.

    python c3_cache_correct.py [keep_after]
"""

import sys

import cached
import magpylib as magpy
import numpy as np
from common import done, mm, write_result
from scipy.spatial.transform import Rotation as R

from magpylib_material_response.face_charge import apply_face_charge

cached.KEEP_AFTER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NAME = f"c3_cache_correct_keep{cached.KEEP_AFTER}"
if done(NAME):
    sys.exit(f"results/{NAME}.json exists")

C = magpy.magnet.Cuboid
OBS = (
    np.array(
        [
            [12, 0, 0],
            [0, 12, 1],
            [0, 0, 12],
            [7, 7, 7],
            [-9, 4, -6],
            [3, -8, 5],
            [0, 0, -9],
        ],
        float,
    )
    * mm
)


def soft(dim=(10, 10, 2), pos=(0, 0, 0), chi=1000, pol=(0, 0, 0)):
    p = C(polarization=pol, dimension=np.array(dim) * mm, position=np.array(pos) * mm)
    p.susceptibility = chi
    return p


def magnet(pos=(0, 0, 4.5), tilt=0.0, pol=(0, 0, 1), dim=(5, 5, 5)):
    return C(
        polarization=pol,
        dimension=np.array(dim) * mm,
        position=np.array(pos) * mm,
        orientation=R.from_euler("x", tilt, degrees=True),
    )


def col(*parts, h=None):
    c = magpy.Collection(*parts)
    if h is not None:
        c.H_ext = h
    return c


def rigid(k):
    c = col(soft(), soft(dim=(4, 4, 4), pos=(0, 0, 6)))
    c.rotate(R.from_euler("xyz", (20 * k, 13 * k, 7 * k), degrees=True), anchor=0)
    c.move(np.array([k, -2 * k, 0.5 * k]) * mm)
    return col(magnet(pos=(15, 0, 0)), c)


def rigid_steps(k):
    c = col(soft(), soft(dim=(4, 4, 4), pos=(0, 0, 6)))
    for _ in range(50 * k):  # the rounding of many small turns
        c.rotate(
            R.from_euler("xyz", (3.1, 1.7, 0.9), degrees=True), anchor=(1 * mm, 0, 0)
        )
    return col(magnet(pos=(15, 0, 0)), c)


def cylinder(angle):
    shaft = magpy.magnet.Cylinder(
        polarization=(0, 0, 0), dimension=(20 * mm, 40 * mm), position=(0, 0, -20 * mm)
    )
    shaft.susceptibility = 3999
    m = magpy.magnet.Cylinder(
        polarization=(0.5, 0, 0), dimension=(6 * mm, 3 * mm), position=(0, 0, 1.5 * mm)
    )
    return col(
        magpy.Collection(shaft, m).rotate(
            R.from_euler("z", angle, degrees=True), anchor=0
        )
    )


def mesh_part(shift):
    verts = (
        np.array([[x, y, z] for x in (-5, 5) for y in (-5, 5) for z in (-1, 1)], float)
        * mm
    )
    faces = [
        [0, 1, 3],
        [0, 3, 2],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [2, 3, 7],
        [2, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
        [1, 5, 7],
        [1, 7, 3],
    ]
    t = magpy.magnet.TriangularMesh(
        polarization=(0, 0, 0), vertices=verts, faces=faces, reorient_faces=True
    )
    t.susceptibility = 1000
    return col(magnet(pos=(shift, 0, 4.5)), t)


CASES = {
    "fixed source moves": (
        [
            lambda k=k: col(magnet(pos=(k, k / 2, 4.5), tilt=k), soft())
            for k in range(4)
        ],
        3,
    ),
    # from about 0.01 mm off the plate its panels grade toward the magnet's outline
    "fixed source comes very near, a new layout once": (
        [
            lambda z=z: col(magnet(pos=(0, 0, z)), soft())
            for z in (4.5, 3.51, 3.502, 3.5001)
        ],
        2,
    ),
    "H_ext changes": (
        [
            lambda h=h: col(magnet(), soft(), h=(h, -h, h / 2))
            for h in (0.0, 5e-3, -0.2, 1.0)
        ],
        3,
    ),
    "the part's remanence changes": (
        [
            lambda j=j: col(magnet(), soft(chi=0.05, pol=(0, 0, j)))
            for j in (0.2, 0.5, -1.0)
        ],
        2,
    ),
    "two parts move as one": ([lambda k=k: rigid(k) for k in range(4)], 3),
    "two parts move as one, by many small turns": (
        [lambda k=k: rigid_steps(k) for k in range(4)],
        3,
    ),
    "a turning cylinder and its magnet": (
        [lambda a=a: cylinder(a) for a in (0, 30, 77, 190)],
        3,
    ),
    "a mesh, a source moves": ([lambda s=s: mesh_part(s) for s in (0, 1, 2)], 2),
    "two parts of one material touch, a source moves": (
        [
            lambda k=k: col(
                magnet(pos=(k, 0, 10)), soft(pos=(-5, 0, 0)), soft(pos=(5, 0, 0))
            )
            for k in range(3)
        ],
        2,
    ),
    "the same model again after another": (
        [
            lambda: col(magnet(), soft()),
            lambda: col(magnet(), soft(chi=500)),
            lambda: col(magnet(), soft()),
        ],
        1,
    ),
    # a matrix that changes: every call must miss
    "parts move apart": (
        [
            lambda k=k: col(magnet(pos=(15, 0, 0)), soft(), soft(pos=(0, 0, 3 + k)))
            for k in range(3)
        ],
        0,
    ),
    "a part turns against another": (
        [
            lambda a=a: col(
                soft(),
                soft(dim=(2, 2, 2), pos=(0, 0, 4)).rotate_from_angax(a, "x"),
                magnet(pos=(15, 0, 0)),
            )
            for a in (0, 10, 25)
        ],
        0,
    ),
    "chi changes": (
        [lambda x=x: col(magnet(), soft(chi=x)) for x in (1000, 999, 10)],
        0,
    ),
    "one length changes": (
        [lambda L=L: col(magnet(), soft(dim=(L, 10, 2))) for L in (10, 10.01, 10.05)],
        0,
    ),
    "a recoil magnet moves": (
        [
            lambda k=k: col(
                soft(),
                soft(dim=(5, 5, 5), pos=(k * 0.3, 0, 4.5), chi=0.05, pol=(0, 0, 1)),
            )
            for k in range(3)
        ],
        0,
    ),
    "a magnet lands on a part, its footprint grades the panels": (
        [lambda k=k: col(magnet(pos=(k * 0.7, 0, 3.5)), soft()) for k in range(3)],
        0,
    ),
    "the parts swap places in the list": (
        [
            lambda: col(
                magnet(pos=(15, 0, 0)), soft(), soft(dim=(4, 4, 4), pos=(0, 0, 6))
            ),
            lambda: col(
                magnet(pos=(15, 0, 0)), soft(dim=(4, 4, 4), pos=(0, 0, 6)), soft()
            ),
        ],
        0,
    ),
    "touching, then one material less": (
        [
            lambda: col(
                magnet(pos=(0, 0, 10)), soft(pos=(-5, 0, 0)), soft(pos=(5, 0, 0))
            ),
            lambda: col(
                magnet(pos=(0, 0, 10)),
                soft(pos=(-5, 0, 0)),
                soft(pos=(5, 0, 0), chi=500),
            ),
        ],
        0,
    ),
}
KWARGS = {
    "level changes": [{"level": 2}, {"level": 3}, {"level": 4}],
    "refine rounds": [{"refine": 1}, {"refine": 1}, {"refine": 2}],
}


def read(model, **kwargs):
    return magpy.getB(apply_face_charge(model(), **kwargs), OBS)


rows = []


def case(name, models, expected, kwargs=None):
    kwargs = kwargs or [{}] * len(models)
    cached.uninstall()
    plain = [read(m, **k) for m, k in zip(models, kwargs, strict=True)]
    cached.install()
    cached.clear()
    with_cache = [read(m, **k) for m, k in zip(models, kwargs, strict=True)]
    error = max(
        float(np.abs(a - b).max() / np.abs(a).max())
        for a, b in zip(plain, with_cache, strict=True)
    )
    hits = cached.stats["hits"]
    # the calls before a matrix is kept are not hits
    if expected is not None:
        expected = max(0, expected - (cached.KEEP_AFTER - 1))
    ok = error < 1e-9 and (expected is None or hits == expected)
    rows.append(
        {
            "case": name,
            "calls": len(models),
            "hits": hits,
            "expected": expected,
            "relative_error": error,
            "ok": ok,
        }
    )
    print(
        f"{'ok ' if ok else 'BAD'} {name}: {hits} hits (expected {expected}), error {error:.1e}",
        flush=True,
    )


for name, (models, expected) in CASES.items():
    case(name, models, expected)
base = lambda: col(magnet(), soft())
case("level changes", [base] * 3, 0, KWARGS["level changes"])
case(
    "refine rounds",
    [lambda k=k: col(magnet(pos=(k, 0, 4.5)), soft()) for k in range(3)],
    None,
    KWARGS["refine rounds"],
)
cached.uninstall()
write_result(NAME, {"rows": rows, "all_ok": all(r["ok"] for r in rows)})
