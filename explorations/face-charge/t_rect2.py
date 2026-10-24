import numpy as np
from bem import rect_field
# numeric quadrature of a unit-charge rectangle [0,1]x[0,2] at a few points
g, w = np.polynomial.legendre.leggauss(400)
p = 0.5*(g+1); wp = 0.5*w
q = (g+1); wq = w
PP, QQ = np.meshgrid(p, q, indexing="ij"); WW = np.outer(wp, wq)
for o in [(0.3, 0.7, 0.5), (1.5, -0.4, 0.2), (-0.7, 2.5, -0.3)]:
    d = np.array(o)[:, None, None] - np.stack([PP, QQ, 0*PP])
    r3 = np.sum(d**2, 0)**1.5
    num = (d/r3*WW).sum((1, 2))/(4*np.pi)
    ana = np.array(rect_field(o[0], o[1], o[2], 0., 1., 0., 2.))
    print(o, num, ana)
