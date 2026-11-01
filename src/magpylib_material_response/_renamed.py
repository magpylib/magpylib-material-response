"""What the public names were before the solvers took one form, for one release."""

from __future__ import annotations

import sys
import warnings


def old_names(module, names):
    """A module's ``__getattr__`` that gives the old names in `names`, a dict of
    old to new, with a warning, from the module named `module`."""

    def __getattr__(name):
        if name not in names:
            msg = f"module {module!r} has no attribute {name!r}"
            raise AttributeError(msg)
        warnings.warn(
            f"{name} is now {names[name]}; the old name goes in the next release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(sys.modules[module], names[name])

    return __getattr__
