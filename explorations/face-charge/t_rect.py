import numpy as np, magpylib as magpy
from bem import Panels
MU0 = 4e-7*np.pi
dim = np.array([0.02, 0.012, 0.007]); half = dim/2
rng = np.random.default_rng(0)
obs = rng.uniform(-1.6, 1.6, (2000, 3))*half  # inside and outside
J = np.array([0.3, -0.7, 1.1])
pan = Panels(half, 3, kind="cheb")
sigma = pan.normal @ (J/MU0)
H = pan.field(obs, sigma)
Href = magpy.getH(magpy.magnet.Cuboid(dimension=dim, polarization=J), obs)
print("max rel err", np.abs(H-Href).max()/np.abs(Href).max())
