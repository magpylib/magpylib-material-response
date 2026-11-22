"""The field of straight current segments (magpylib's ``Polyline``), written
against the Python array API standard: one implementation for NumPy, any
other array library, and the 50-digit namespace that writes the reference
values of ``kernels/reference/current.json`` (#208). Only its test and
``kernels/reference/fill.py`` use it; the solvers keep magpylib's.

Biot-Savart for a segment from ``a`` to ``b`` with unit direction ``e`` and
length ``L``, at an observer ``x`` with ``r1 = x - a``, ``r2 = x - b``, their
lengths ``R1``, ``R2`` and their components along the segment ``t1 = e.r1``,
``t2 = e.r2 = t1 - L``:

    H = I / (4 pi) * k * (e x r1),   k = (t1/R1 - t2/R2) / d^2,

``d = |e x r1|`` the distance from the segment's line. Where ``t1`` and
``t2`` have one sign (the observer beyond an end) the two cosines nearly
cancel; rationalized, ``k = L (t1 + t2) / (R1 R2 (t1 R2 + t2 R1))``, a sum
of terms of one sign, and ``d^2`` drops out. Magpylib subtracts the sines
there.
"""

from magpylib_material_response._array_api import namespace, quiet
from magpylib_material_response._array_api.cuboid import MU0, _promote

RTOL_LINE = 1e-15  # distance from the line, in segment lengths, taken as on it


def polyline_field(field, observers, segment_start, segment_end, current):
    """B in T or H in A/m (``field`` 'B' or 'H') of straight current segments,
    one per observer: ``segment_start``, ``segment_end`` (i, 3) in m, the
    current (i,) in A flowing from start to end. As magpylib: 0 within
    ``RTOL_LINE`` lengths of the segment's line, its extensions included,
    and for a segment without length."""
    if field not in ("B", "H"):
        msg = f"field must be 'B' or 'H'; instead received {field!r}."
        raise ValueError(msg)
    xp = namespace(observers, segment_start, segment_end, current)
    observers, segment_start, segment_end, current = _promote(
        xp, observers, segment_start, segment_end, current
    )
    with quiet(xp, "divide", "invalid"):
        Lv = segment_end - segment_start
        L = xp.linalg.vector_norm(Lv, axis=-1)
        e = Lv / L[:, None]
        r1 = observers - segment_start
        r2 = observers - segment_end
        R1 = xp.linalg.vector_norm(r1, axis=-1)
        R2 = xp.linalg.vector_norm(r2, axis=-1)
        t1 = xp.sum(e * r1, axis=-1)
        t2 = xp.sum(e * r2, axis=-1)
        c = xp.linalg.cross(e, r1)
        d2 = xp.sum(c * c, axis=-1)
        beyond = t1 * t2 > 0
        k = xp.where(
            beyond,
            L * (t1 + t2) / (R1 * R2 * (t1 * R2 + t2 * R1)),
            (t1 / R1 - t2 / R2) / d2,
        )
        on_line = d2 <= (RTOL_LINE * L) ** 2
        k = xp.where(on_line | (L == 0), 0.0, k)
        H = (current * k / (4 * xp.pi))[:, None] * xp.where((L == 0)[:, None], 0.0, c)
    return H if field == "H" else MU0 * H
