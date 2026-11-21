"""Field kernels written against the Python array API standard (#208), and
what they share: finding their arrays' namespace, and NumPy's warnings off
where infinities are expected."""

from contextlib import nullcontext

import numpy as np


def namespace(*arrays):
    """The arrays' namespace, from their own ``__array_namespace__``: NumPy 2
    arrays, any array API library's, and the 50-digit reference arrays."""
    spaces = {a.__array_namespace__() for a in arrays}
    if len(spaces) > 1:
        names = sorted(xp.__name__ for xp in spaces)
        msg = f"arrays of one library expected; received {names}"
        raise TypeError(msg)
    return spaces.pop()


def quiet(xp, *warnings):
    """NumPy's ``warnings`` ('divide', 'invalid', 'over') off, for code where
    infinities and NaN are expected; other namespaces do not warn."""
    if xp is np:
        return np.errstate(**dict.fromkeys(warnings, "ignore"))
    return nullcontext()
