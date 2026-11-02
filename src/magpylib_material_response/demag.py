"""The module :mod:`magpylib_material_response.cells` under its old name, for one
release."""

from __future__ import annotations

import warnings

from magpylib_material_response import cells

# what `import *` gave, through __getattr__ below
# pylint: disable=undefined-all-variable
__all__ = ["apply_demag", "demag_tensor", "get_H_ext", "get_susceptibilities"]  # noqa: F822


def __getattr__(name):
    if name.startswith("_"):
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    new = "solve_cells" if name == "apply_demag" else name
    value = getattr(cells, new)
    warnings.warn(
        f"magpylib_material_response.demag.{name} is now "
        f"magpylib_material_response.cells.{new}; the old module goes in the next "
        "release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return value
