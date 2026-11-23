"""Field kernels written against the Python array API standard (#208), and
what they share: finding their arrays' namespace, and NumPy's warnings off
where infinities are expected."""

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


def quiet(*warnings):
    """NumPy's ``warnings`` ('divide', 'invalid', 'over') off, for code where
    infinities and NaN are expected: on NumPy and on the namespaces built on it
    (``array_api_strict``); a namespace that does not warn ignores it."""
    return np.errstate(**dict.fromkeys(warnings, "ignore"))


# Bulirsch's stop for cel in doubles, magpylib's
ERRTOL = 1e-8


def cel_errtol(eps):
    """Bulirsch's relative stop for ``cel`` at the precision ``eps``. The
    iteration converges quadratically, so a step below ``sqrt(eps)`` leaves the
    integral exact to it: magpylib's 1e-8 in doubles, ``sqrt(eps)`` above
    (float32: 1e-8 would never be met) and below (5e-26 at 50 digits)."""
    root = float(eps) ** 0.5
    return min(ERRTOL, root) if float(eps) <= 2.0**-52 else root
