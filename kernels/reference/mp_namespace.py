"""A minimal array API namespace on mpmath, for the reference values (#208):
the same kernel code that runs on NumPy runs here at ``mp.mp.dps`` digits,
so that no formula is written twice.

Arrays are NumPy object arrays of ``mpf``, wrapped in :class:`MPArray`, whose
``__array_namespace__`` returns this module: kernels that take their
namespace from their arrays run here unchanged. Only what the kernels use is here. Where mpmath would raise or turn
complex, the results follow IEEE: ``x / 0`` is an infinity or NaN, ``log`` of
0 is ``-inf``, ``log`` and ``sqrt`` of a negative number are NaN. mpmath has no
negative zero: decisions made on a zero's sign are taken from the double run
by the caller, not from here.
"""

import math
import sys

import mpmath as mp
import numpy as np

FLOAT = "mpf"  # the one floating dtype
BOOL = "bool"
pi = mp.pi
newaxis = None


def _mpf(v):
    if isinstance(v, mp.mpf):
        return v
    v = float(v)
    if math.isnan(v):
        return mp.nan
    return mp.mpf(v)  # exact, infinities included


class MPArray:
    """An object array of mpf (dtype ``FLOAT``) or a boolean array (``BOOL``)."""

    __slots__ = ("_a", "dtype")

    def __init__(self, a, dtype=None):
        a = np.asarray(a)
        if dtype is None:
            dtype = BOOL if a.dtype == bool else FLOAT
        if dtype == FLOAT and a.dtype != object:
            a = (
                np.frompyfunc(_mpf, 1, 1)(a).astype(object)
                if a.ndim
                else np.array(_mpf(a), object)
            )
        self._a, self.dtype = a, dtype

    def __array_namespace__(self, api_version=None):
        return sys.modules[__name__]

    shape = property(lambda self: self._a.shape)
    ndim = property(lambda self: self._a.ndim)

    def __getitem__(self, key):
        return MPArray(self._a[key], self.dtype)

    def tolist(self):
        return self._a.tolist()

    # arithmetic, with Python numbers taken exactly
    def _bin(self, other, f):
        return MPArray(f(self._a, _raw(other)), FLOAT)

    __add__ = lambda s, o: s._bin(o, np.add)
    __radd__ = lambda s, o: s._bin(o, lambda a, b: np.add(b, a))
    __sub__ = lambda s, o: s._bin(o, np.subtract)
    __rsub__ = lambda s, o: s._bin(o, lambda a, b: np.subtract(b, a))
    __mul__ = lambda s, o: s._bin(o, np.multiply)
    __rmul__ = lambda s, o: s._bin(o, lambda a, b: np.multiply(b, a))
    __truediv__ = lambda s, o: s._bin(o, _div)
    __rtruediv__ = lambda s, o: s._bin(o, lambda a, b: _div(b, a))
    __pow__ = lambda s, o: s._bin(o, np.power)

    def __neg__(self):
        return MPArray(-self._a, FLOAT)

    # comparisons give booleans; NaN compares false, as in IEEE
    def _cmp(self, other, f):
        return MPArray(np.frompyfunc(f, 2, 1)(self._a, _raw(other)).astype(bool), BOOL)

    __lt__ = lambda s, o: s._cmp(o, lambda a, b: bool(a < b))
    __gt__ = lambda s, o: s._cmp(o, lambda a, b: bool(a > b))
    __le__ = lambda s, o: s._cmp(o, lambda a, b: bool(a <= b))
    __ge__ = lambda s, o: s._cmp(o, lambda a, b: bool(a >= b))
    __eq__ = lambda s, o: s._cmp(o, lambda a, b: bool(a == b))
    __ne__ = lambda s, o: s._cmp(o, lambda a, b: not bool(a == b))
    __hash__ = None

    def __and__(self, other):
        return MPArray(self._a & _raw(other), BOOL)

    def __or__(self, other):
        return MPArray(self._a | _raw(other), BOOL)

    def __invert__(self):
        return MPArray(~self._a, BOOL)


def _raw(v):
    if isinstance(v, MPArray):
        return v._a
    if isinstance(v, bool):
        return v
    return _mpf(v)


def _scalar_div(a, b):
    if b == 0:
        if a == 0 or mp.isnan(a):
            return mp.nan
        return mp.inf if a > 0 else -mp.inf
    return a / b


_div = np.frompyfunc(_scalar_div, 2, 1)


def _unary(f):
    def ufunc(x):
        return MPArray(np.frompyfunc(f, 1, 1)(x._a), FLOAT)

    return ufunc


def _log(v):
    if mp.isnan(v) or v < 0:
        return mp.nan
    return -mp.inf if v == 0 else mp.log(v)


def _sqrt(v):
    return mp.nan if mp.isnan(v) or v < 0 else mp.sqrt(v)


log = _unary(_log)
sqrt = _unary(_sqrt)
abs = _unary(lambda v: v if mp.isnan(v) else mp.fabs(v))


def atan2(y, x):
    def f(a, b):
        return mp.nan if mp.isnan(a) or mp.isnan(b) else mp.atan2(a, b)

    return MPArray(np.frompyfunc(f, 2, 1)(_raw(y), _raw(x)), FLOAT)


def where(cond, a, b):
    out = np.where(cond._a, _raw(a), _raw(b))
    return MPArray(out, BOOL if out.dtype == bool else FLOAT)


def asarray(v, dtype=None):  # noqa: ARG001  the standard signature; one dtype here
    return v if isinstance(v, MPArray) else MPArray(v)


def ones_like(x):
    return MPArray(np.full(x.shape, mp.mpf(1), object), FLOAT)


def stack(arrays, axis=0):
    return MPArray(np.stack([a._a for a in arrays], axis=axis), arrays[0].dtype)


def unstack(x, axis=0):
    return tuple(MPArray(a, x.dtype) for a in np.moveaxis(x._a, axis, 0))


def any(x, axis=None, keepdims=False):
    return MPArray(np.any(x._a, axis=axis, keepdims=keepdims), BOOL)


def all(x, axis=None, keepdims=False):
    return MPArray(np.all(x._a, axis=axis, keepdims=keepdims), BOOL)


def isdtype(dtype, kind):
    return dtype == FLOAT and kind in ("real floating", "numeric")


def result_type(*_dtypes):
    return FLOAT


def astype(x, dtype, copy=True):
    return MPArray(x._a.copy() if copy else x._a, dtype)


def from_doubles(a):
    """A float array as an MPArray, every double taken exactly."""
    return MPArray(np.asarray(a, float))
