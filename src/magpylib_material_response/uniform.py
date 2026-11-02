"""A uniform applied field, as a source in a model."""

from __future__ import annotations

import warnings

import magpylib as magpy
import numpy as np

MU0 = magpy.mu_0
B_WARN = 10.0  # T: stronger is most likely H in A/m given as B


class UniformField(magpy.misc.CustomSource):
    """A field that is the same everywhere: a source like a coil, in a model's
    collection, which every solver takes as driving its parts, and whose field
    ``getB`` of the solved model includes.

    Parameters
    ----------
    B, H : array-like, shape (3,)
        The flux density in T, or the field strength in A/m, in the source's own
        frame. Give one of them: in vacuum ``B = mu0 H``.
    position, orientation, style, **kwargs
        As for any magpylib source. It turns with its orientation; its position
        changes nothing.

    Examples
    --------
    >>> import magpylib as magpy
    >>> from magpylib_material_response import UniformField
    >>> soft = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 0))
    >>> soft.susceptibility = 1000
    >>> model = magpy.Collection(soft, UniformField(B=(0, 0, 0.1)))
    """

    def __init__(
        self,
        *,
        B=None,
        H=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        if (B is None) == (H is None):
            msg = "UniformField takes one of B, in T, and H, in A/m."
            raise ValueError(msg)
        value = np.array(B, float) if H is None else np.array(H, float) * MU0
        if value.shape != (3,) or not np.isfinite(value).all():
            msg = f"UniformField needs three finite numbers; got {B if H is None else H!r}."
            raise ValueError(msg)
        if H is None and np.linalg.norm(value) > B_WARN:
            warnings.warn(
                f"UniformField of {np.linalg.norm(value):.3g} T: is it H in A/m, given "
                "as B?",
                stacklevel=2,
            )
        value.flags.writeable = False
        self._B = value
        super().__init__(
            position=position, orientation=orientation, style=style, **kwargs
        )
        # a bound method, unlike a closure, follows copy() to the copy
        self.field_func = self._field

    @property
    def B(self):
        """The flux density in T, in the source's own frame."""
        return self._B

    @property
    def H(self):
        """The field strength in A/m, in the source's own frame."""
        return self._B / MU0

    def _field(self, field, observers):
        n = len(observers)
        if field in ("J", "M"):
            return np.zeros((n, 3))
        return np.tile(self._B if field == "B" else self.H, (n, 1))


def applied(sources):
    """The flux density in T, global frame, of all the UniformFields in `sources`."""
    total = np.zeros(3)
    for source in sources:
        if isinstance(source, UniformField):
            total += source.orientation.apply(source.B.copy()).ravel()
    return total


def refuse_both(sources):
    """Raise if `sources` have a UniformField and ``H_ext``: the field would count
    twice."""
    if not any(isinstance(s, UniformField) for s in sources):
        return
    for source in sources:
        owner = source
        while owner is not None:
            if getattr(owner, "H_ext", None) is not None:
                msg = (
                    f"{owner} has H_ext in a model with a UniformField, and the field "
                    "would count twice: use the UniformField alone."
                )
                raise ValueError(msg)
            owner = owner.parent
