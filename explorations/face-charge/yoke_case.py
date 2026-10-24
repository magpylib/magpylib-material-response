"""Magnet glued on an iron plate: geometry and sensors shared by the face-charge and MoM runs."""
import numpy as np
YOKE = dict(dimension=(20e-3, 20e-3, 5e-3), position=(0, 0, 0), chi=1000.0)
MAG = dict(dimension=(6e-3, 6e-3, 4e-3), position=(2e-3, 1e-3, 4.5e-3), chi=0.05, J0=(0, 0, 1.2))
D = np.array([0.5, 1, 2, 5]) * 1e-3

def sensors():
    pts, tags = [], []
    for d in D:
        pts += [[2.5e-3, 1.5e-3, 6.5e-3 + d],   # above the magnet
                [7.5e-3, 1e-3, 2.5e-3 + d],     # on the plate, beside the magnet
                [0, 0, -2.5e-3 - d],            # under the plate
                [10e-3 + d, 3e-3, 0]]           # off the plate's side
        tags += [("above magnet", d), ("beside magnet", d), ("under plate", d), ("plate side", d)]
    return np.array(pts), tags
