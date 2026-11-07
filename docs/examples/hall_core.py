"""The slotted ferrite core of `hall_current_sensor.md`, as the triangles a CAD
export of it gives: the round walls and the flat top and bottom in one strip of
triangles per 7.5 degrees, the slot's faces in two triangles each. With a real
part, read its STL file instead, with any mesh library: its vertices and faces
are what this returns."""

import numpy as np


def core_mesh(gap=2.0, r_in=8.0, r_out=13.0, height=6.0, step=7.5):
    """Vertices in mm and faces of a ring of radii `r_in` and `r_out`, `height`
    tall, with a slot `gap` wide cut through it along the x axis."""
    a_out, a_in = np.arcsin(gap / 2 / r_out), np.arcsin(gap / 2 / r_in)
    n = int(np.ceil((2 * np.pi - 2 * a_out) / np.radians(step)))
    t = np.linspace(0, 1, n + 1)
    outline = []
    for r, a in ((r_out, a_out), (r_in, a_in)):
        phi = a + t * (2 * np.pi - 2 * a)
        outline.append(np.column_stack([r * np.cos(phi), r * np.sin(phi)]))
    ring = np.concatenate(outline)  # outer arc 0..n, inner arc n+1..2n+1
    m = len(ring)
    vertices = np.concatenate(
        [np.column_stack([ring, np.full(m, z)]) for z in (height / 2, -height / 2)]
    )

    def outer(k, top):
        return k + (0 if top else m)

    def inner(k, top):
        return n + 1 + k + (0 if top else m)

    faces = []
    for k in range(n):
        o0, o1, i0, i1 = outer(k, 1), outer(k + 1, 1), inner(k, 1), inner(k + 1, 1)
        faces += [[o0, o1, i1], [o0, i1, i0]]  # top
        o0, o1, i0, i1 = outer(k, 0), outer(k + 1, 0), inner(k, 0), inner(k + 1, 0)
        faces += [[o0, i1, o1], [o0, i0, i1]]  # bottom
        faces += [[outer(k, 0), outer(k + 1, 0), outer(k + 1, 1)]]
        faces += [[outer(k, 0), outer(k + 1, 1), outer(k, 1)]]
        faces += [[inner(k, 0), inner(k + 1, 1), inner(k + 1, 0)]]
        faces += [[inner(k, 0), inner(k, 1), inner(k + 1, 1)]]
    for k in (0, n):  # the slot's two faces
        faces += [[outer(k, 0), inner(k, 0), inner(k, 1)]]
        faces += [[outer(k, 0), inner(k, 1), outer(k, 1)]]
    return vertices, np.array(faces)
