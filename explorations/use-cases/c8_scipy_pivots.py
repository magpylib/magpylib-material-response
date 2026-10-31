import numpy as np, scipy, scipy.linalg, sys
from concurrent.futures import ThreadPoolExecutor
rng=np.random.default_rng(0); n=800
A=rng.standard_normal((n,n))+n*np.eye(n); lu,piv=scipy.linalg.lu_factor(A,check_finite=False)
bs=rng.standard_normal((400,n)); truth=np.array([scipy.linalg.lu_solve((lu,piv.copy()),q,check_finite=False) for q in bs])
sys.setswitchinterval(1e-5); keep=piv.copy()
with ThreadPoolExecutor(4) as p: got=np.array(list(p.map(lambda i:scipy.linalg.lu_solve((lu,piv),bs[i],check_finite=False),range(400))))
print(scipy.__version__, np.__version__, scipy.__config__.show(mode="dicts")["Build Dependencies"]["blas"]["name"], 'bad',int((np.abs(got-truth).max(axis=1)>1e-9*np.abs(truth).max()).sum()),'of 400; piv changed',not np.array_equal(keep,piv))
