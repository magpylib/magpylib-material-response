import numpy as np
from p1 import lin_rect
g, w = np.polynomial.legendre.leggauss(600)
p1_, p2_, q1_, q2_ = -0.3, 0.9, 0.2, 1.4
p = 0.5*(p1_+p2_) + 0.5*(p2_-p1_)*g; wp = 0.5*(p2_-p1_)*w
q = 0.5*(q1_+q2_) + 0.5*(q2_-q1_)*g; wq = 0.5*(q2_-q1_)*w
PP, QQ = np.meshgrid(p, q, indexing="ij"); WW = np.outer(wp, wq)
S = (PP - 0.5*(p1_+p2_))/(0.5*(p2_-p1_)); T = (QQ - 0.5*(q1_+q2_))/(0.5*(q2_-q1_))
worst = 0
for o in [(0.3, 0.7, 0.5), (1.5, -0.4, 0.2), (-0.7, 2.5, -0.3), (0.1, 0.3, -0.05), (2.0, 0.5, 0.0)]:
    d = np.array(o)[:, None, None] - np.stack([PP, QQ, 0*PP])
    r3 = np.sum(d**2, 0)**1.5
    num = np.array([[(dens*d[c]/r3*WW).sum() for c in range(3)] for dens in (1+0*S, S, T)])/(4*np.pi)
    ana = lin_rect(*o, p1_, p2_, q1_, q2_)
    err = np.abs(num-ana).max()/np.abs(num).max(); worst = max(worst, err)
    print(o, f"{err:.1e}")
print("worst", worst)
