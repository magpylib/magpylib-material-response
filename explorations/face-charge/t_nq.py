import numpy as np
from bem import geo_panels, kprime_gal, solve, moment
half = np.array([1., 1., 1.]); V = 8.
pan = geo_panels(half, 0.25, 3, 0.25)
for nq in [2, 3, 5, 8, 12]:
    K = kprime_gal(pan, nq)
    print(nq, moment(pan, solve(pan, 1e12, pan.normal[:, 2], K=K))[2]/V)
