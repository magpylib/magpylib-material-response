"""sheet_levels.png: a soft-iron plate under a magnet, solved by face charge at
mesh levels 3 and 5 and drawn by magpylib's pyvista backend, the plate as a
ChargeSheet coloured by its charge, level 3 with its panel edges. Prints B at
three sensors per level.

    python sheet_levels.py [face-charge folder]

The folder holds source.py and validation/fc.py; by default this study's. The
PR figure is drawn with the merge branch's: <face-charge checkout>/explorations/face-charge."""

import os
import sys
import warnings

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..")
sys.path[:0] = [CODE, os.path.join(CODE, "validation")]
import fc  # noqa: E402
from source import ChargeSheet  # noqa: E402

warnings.simplefilter("ignore")
soft = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.004), polarization=(0, 0, 0))
soft.susceptibility = 1000.0
magnet = magpy.magnet.Cuboid(dimension=(0.004, 0.004, 0.003), position=(0, 0, 0.0045), polarization=(0, 0, 1.2))
coll = magpy.Collection(soft, magnet)
sensors = np.array([[0.0, 0.0, 0.0035], [0.004, 0.0, 0.003], [0.0, 0.0, -0.004]])
shots = []
for m in (3, 5):
    solved = fc.apply_face_charge(coll, m)
    sheet = next(s for s in solved.sources_all if isinstance(s, ChargeSheet))
    sheet.style.mesh.grid.show = m == 3
    pl = pv.Plotter(off_screen=True, window_size=(600, 520))
    magpy.show(solved, backend="pyvista", canvas=pl)
    pl.add_text(f"m={m}: {len(sheet.faces)} faces" + (", panel edges" if m == 3 else ""), font_size=11)
    shots.append(pl.screenshot(return_img=True))
    print(f"m={m}: {len(sheet.faces)} faces; Bz at the sensors (mT): {np.round(solved.getB(sensors)[:, 2] * 1e3, 2)}")
plt.imsave(os.path.join(HERE, "sheet_levels.png"), np.hstack(shots))
