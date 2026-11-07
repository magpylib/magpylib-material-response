"""A uniform applied field, as a source in a model."""

from __future__ import annotations

import contextvars
import warnings

import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation

MU0 = magpy.mu_0
B_WARN = 10.0  # T: stronger is most likely H in A/m given as B
#: Context variables of recorders of magpylib calls, by name: magpylib's hook
#: and studio's ``@scene``. Their default is None, recording nothing.
RECORDERS = ("magpylib_recorder", "magpylib_studio_building")
#: Set while from_json loads: a field saved as B may have been given as H.
LOADING = contextvars.ContextVar("magpylib_material_response_loading", default=False)
#: The arrow's length, in tenths of the scene (magpylib's autosize) per unit of
#: ``style.size``: a dipole's is one.
ARROW_LENGTH = 3


def unseen(function, *args):
    """Run `function` out of sight of a recorder of magpylib calls, as in studio's
    ``@scene``: what this package makes for itself is no object of the scene. The
    caller's context without its recorders, so warnings filters, numpy's error
    settings and log context still hold."""
    context = contextvars.copy_context()
    for variable in context:
        if variable.name in RECORDERS:
            context.run(variable.set, None)
    return context.run(function, *args)


def shallow_copy(source):
    """What ``copy.copy`` gives of a source whose field function is its own bound
    method: bound to the new one, or the copy would read the original's field. One
    set by hand stays."""
    new = type(source).__new__(type(source))
    new.__dict__.update(source.__dict__)
    func = source._field_func
    if getattr(func, "__self__", None) is source:
        new._field_func = getattr(new, func.__name__)
    return new


class UniformField(magpy.misc.CustomSource):
    """A field that is the same everywhere: a source like a coil, in a model's
    collection, which every solver takes as driving its parts, and whose field
    ``getB`` of the solved model includes.

    Setting a value on the field of a solved model changes the field and not the
    response, as moving a magnet of it does: solve again.

    Parameters
    ----------
    B, H : array-like, shape (3,)
        The flux density in T, or the field strength in A/m, in the source's own
        frame. Give one of them: in vacuum ``B = mu0 H``.
    position, orientation, style, **kwargs
        As for any magpylib source. It turns with its orientation; its position
        changes no field. The style is a dipole's: ``show`` draws an arrow above
        the scene, ``style.size`` times three tenths of the scene long.

    Examples
    --------
    >>> import magpylib as magpy
    >>> from magpylib_material_response import UniformField
    >>> soft = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 0))
    >>> soft.susceptibility = 1000
    >>> model = magpy.Collection(soft, UniformField(B=(0, 0, 0.1)))
    """

    _style_class = magpy.graphics.style.DipoleStyle
    _style_family = ("dipole",)
    _autosize = True  # show() passes the scene's size to get_trace
    _scene_box = True  # and, from magpylib 6.3, the box the scene fills

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
        self._B = None
        if H is None:
            self._set(B, 1.0)
            self._warn_if_strong()
        else:
            self._set(H, MU0)
        super().__init__(
            position=position, orientation=orientation, style=style, **kwargs
        )
        # a bound method, unlike a closure, follows copy() to the copy
        self.field_func = self._field

    @property
    def B(self):
        """The flux density in T, in the source's own frame. Read-only: set the
        whole array to change it."""
        value = self._B.view()
        value.flags.writeable = False
        return value

    @B.setter
    def B(self, value):
        self._set(value, 1.0)
        self._warn_if_strong()

    @property
    def H(self):
        """The field strength in A/m, in the source's own frame."""
        return self._B / MU0

    @H.setter
    def H(self, value):
        self._set(value, MU0)

    def _warn_if_strong(self):
        if np.linalg.norm(self._B) > B_WARN and not LOADING.get():
            warnings.warn(
                f"UniformField of {np.linalg.norm(self._B):.3g} T: is it H in A/m, given "
                "as B?",
                stacklevel=3,  # the user's line, from __init__ or the setter
            )

    def _set(self, value, factor):
        field = np.array(value, float)
        if field.shape != (3,) or not np.isfinite(field).all():
            msg = f"UniformField needs three finite numbers; got {value!r}."
            raise ValueError(msg)
        field = field * factor
        field.flags.writeable = False
        self._B = field

    def __copy__(self):
        return shallow_copy(self)

    @property
    def _default_style_description(self):
        return f"B={np.linalg.norm(self._B):.3g} T"

    def get_trace(self, autosize=None, scene_box=None, path_index=-1, **kwargs):
        """An arrow along B above the scene's centre, ``style.size`` times
        three tenths of the scene long, or ``style.size`` metres with
        ``style.sizemode="absolute"``. A field of zero is a dot. Without the
        scene's box, shown alone or by a magpylib before 6.3, it is drawn at the
        field's position."""
        style = self.style
        size = style.size
        if autosize is not None and style.sizemode == "scaled":
            size *= ARROW_LENGTH * autosize
        # magpylib places the trace at the pose it is drawn at: undo that pose
        # to put it above the scene, clear of it for any direction. Not past a
        # corner: the default views of plotly and pyvista look along the
        # diagonal, and put it in front of the corner
        offset = np.zeros(3)
        if scene_box is not None:
            box = np.asarray(scene_box, float)
            anchor = np.append(box[:2].mean(axis=1), box[2, 1] + 1.5 * size)
            position = self._position[path_index]
            offset = self._orientation[path_index].inv().apply(anchor - position)
        norm = np.linalg.norm(self._B)
        if norm == 0:
            trace = {"type": "scatter3d", "mode": "markers", "marker_size": 10}
            trace |= dict(zip("xyz", offset[:, None], strict=True))
            return {**trace, "marker_color": style.color, **kwargs}
        # the arrow is built along z: turn z onto B
        turn = Rotation.align_vectors(self._B[None] / norm, [[0, 0, 1]])[0]
        arrow = magpy.graphics.model3d.make_Arrow(
            "plotly-dict",
            base=10,
            diameter=0.3 * size,
            height=size,
            pivot=style.pivot,
            position=offset,
            orientation=turn,
            color=style.color,
        )
        return {**arrow, **kwargs}

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
