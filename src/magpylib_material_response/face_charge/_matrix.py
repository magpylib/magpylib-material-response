"""The field of solved sheets at fixed observers as one matrix: built once,
then one product for each new charge on the same mesh."""

from __future__ import annotations

import weakref

import magpylib as magpy
import numpy as np
from scipy.sparse import csr_matrix

from magpylib_material_response.face_charge._parallel import CHUNK, run, threads
from magpylib_material_response.face_charge._rectangle import FOUR_PI
from magpylib_material_response.face_charge._sheet import ChargeSheet
from magpylib_material_response.face_charge._triangle import (
    charge_terms,
    far_terms,
)

# a panel's charge is linear across it when each of its faces' coefficients are
# its first face's, carried over, to this share of the sheet's largest
LINEAR_TOL = 1e-9
# a sheet's mesh and pose are the matrix's when they are within this share of its
# size: rounding, not a move
SAME_TOL = 1e-12
# the build holds the kernel's evaluations of a chunk and their terms, 9 numbers
# each: in chunks half getB's, it holds about as much memory at once as getB, in
# the same time (a plate at level 1, 2,601 points: 762 -> 373 MB, getB 363 MB;
# a third took 1.2 times as long)
BUILD_SHARE = 2


def _sheets_of(sources):
    """The ChargeSheets in `sources`, a sheet, a Collection or a list of them."""
    if isinstance(sources, ChargeSheet | magpy.Collection):
        sources = [sources]
    out = []
    for source in sources:
        if isinstance(source, magpy.Collection):
            out.extend(_sheets_of(source.sources_all))
        elif isinstance(source, ChargeSheet):
            out.append(source)
        else:
            msg = (
                f"field_matrix takes ChargeSheets and Collections of them; got "
                f"{source!r}. Add the field of other sources with getB."
            )
            raise TypeError(msg)
    if not out:
        msg = "field_matrix needs at least one ChargeSheet."
        raise ValueError(msg)
    return out


def _pose(sheet):
    """The sheet's position and rotation matrix; one pose only."""
    position = np.atleast_2d(np.asarray(sheet.position, float))
    if len(position) > 1:
        msg = (
            f"{sheet!r} has a path of {len(position)} poses: build a matrix for "
            "each, or one for a sheet at a single pose."
        )
        raise ValueError(msg)
    rotation = sheet.orientation.as_matrix()
    return position[0], rotation[-1] if rotation.ndim == 3 else rotation


def _basis(P, cen, e1, e2, s):
    """``1, u, v`` of each face's frame (centroid, axes, scale; (k, 3) and (k,))
    at its points ``P`` (k, 3, 3): (k, point, basis)."""
    d = P - cen[:, None]
    u = np.einsum("kpd,kd->kp", d, e1) / s[:, None]
    v = np.einsum("kpd,kd->kp", d, e2) / s[:, None]
    return np.stack([np.ones_like(u), u, v], axis=-1)


class _Unknowns:
    """A sheet's charge as the fewest numbers that give it: three per panel
    whose charge is linear across it, three per face otherwise.

    ``map`` (3 faces, unknowns) takes them to the faces' coefficients of ``1, u,
    v``, coefficient by coefficient; ``heads`` are the faces the unknowns are read from, three each; ``carry`` (faces, 3, 3) gives each face's coefficients from its
    panel's first face's."""

    def __init__(self, sheet):
        cen, _, e1, e2, s, _ = sheet._frames
        c = sheet._c
        m = len(c)
        leader = np.arange(m)
        if sheet._panels is not None:
            _, first, inverse = np.unique(
                sheet._panels, return_index=True, return_inverse=True
            )
            leader = first[inverse]
        carry = np.broadcast_to(np.eye(3), (m, 3, 3)).copy()
        others = np.flatnonzero(leader != np.arange(m))
        if len(others):
            f, V = leader[others], sheet._V[others]
            # each face's own coefficients from its corners' charges
            own = _basis(V, cen[others], e1[others], e2[others], s[others])
            lead = _basis(V, cen[f], e1[f], e2[f], s[f])
            carry[others] = np.linalg.solve(own, lead)
        scale = np.abs(c).max() if c.size else 0.0
        off = np.abs(np.einsum("fkj,fj->fk", carry, c[leader]) - c).max(
            axis=1, initial=0
        )
        linear = np.ones(m, bool)
        bad = off > LINEAR_TOL * scale
        if bad.any():  # those panels' faces each their own
            linear[np.isin(leader, leader[bad])] = False
            leader = np.where(linear, leader, np.arange(m))
            carry[~linear] = np.eye(3)
        self.faces = m
        self.leader = leader
        self.carry = carry
        heads = np.flatnonzero(leader == np.arange(m))
        self.heads = heads
        column = np.full(m, -1)
        column[heads] = np.arange(len(heads))
        # rows as the terms give them, coefficient by coefficient (`charge_terms`)
        cols = 3 * column[leader][:, None, None] + np.arange(3)[None, None, :]
        rows = np.arange(m)[:, None, None] + m * np.arange(3)[None, :, None]
        self.map = csr_matrix(
            (
                carry.ravel(),
                (
                    np.broadcast_to(rows, carry.shape).ravel(),
                    np.broadcast_to(cols, carry.shape).ravel(),
                ),
            ),
            shape=(3 * m, 3 * len(heads)),
        )

    @property
    def count(self):
        return 3 * len(self.heads)

    def of(self, sheet):
        """The unknowns of `sheet`'s charge, which must be linear where this
        sheet's was."""
        c = sheet._c
        scale = np.abs(c).max() if c.size else 0.0
        off = np.abs(np.einsum("fkj,fj->fk", self.carry, c[self.leader]) - c).max(
            initial=0
        )
        if off > LINEAR_TOL * max(scale, np.finfo(float).tiny):
            msg = (
                f"The charge of {sheet!r} is not linear across the panels it was "
                "when the matrix was built: build it again."
            )
            raise ValueError(msg)
        return c[self.heads].ravel()


class FieldMatrix:
    """The field of solved sheets at fixed observers as one matrix, made by
    :func:`field_matrix`: ``B = matrix @ charges(sheets)``.

    Valid for sheets with the mesh and pose it was built for, whatever their
    charge: those of later solves with the same parts where they were, such
    as a magnet moved or a ``UniformField`` changed between solves. A sheet
    whose mesh or pose differs is refused.

    Attributes
    ----------
    matrix : ndarray, shape (3 M, n)
        One row per observer and component, observer by observer, one column
        per unknown of the sheets' charge.
    field : str
        ``"B"`` in T, or ``"H"`` in A/m.
    """

    def __init__(self, sheets, unknowns, poses, shape, matrix, field):
        self._kinds = [(s._V.copy(), *p) for s, p in zip(sheets, poses, strict=True)]
        self._unknowns = unknowns
        # not the sheets: each holds its solve, and that solve's kept factorization
        self._sheets = [weakref.ref(s) for s in sheets]
        self._shape = shape
        self.matrix = matrix
        self.field = field

    def charges(self, sources=None):
        """The unknowns of the charge of `sources`, the sheets the matrix was
        built from by default, while they exist: ``matrix @ charges`` is their
        field. The matrix does not keep those sheets, nor the solve they
        belong to."""
        if sources is None:
            sheets = [ref() for ref in self._sheets]
            if any(sheet is None for sheet in sheets):
                msg = (
                    "The sheets the matrix was built from are gone: pass the "
                    "sheets whose field you want."
                )
                raise ValueError(msg)
        else:
            sheets = _sheets_of(sources)
        if len(sheets) != len(self._kinds):
            msg = (
                f"The matrix was built for {len(self._kinds)} sheets; got "
                f"{len(sheets)}."
            )
            raise ValueError(msg)
        out = []
        for sheet, (V, position, rotation), unknowns in zip(
            sheets, self._kinds, self._unknowns, strict=True
        ):
            now, turn = _pose(sheet)
            size = np.abs(V).max() + np.abs(position).max() if V.size else 1.0
            if (
                sheet._V.shape != V.shape
                or np.abs(sheet._V - V).max(initial=0) > SAME_TOL * size
                or np.abs(now - position).max() > SAME_TOL * size
                or np.abs(turn - rotation).max() > SAME_TOL
            ):
                msg = (
                    f"{sheet!r} has another mesh or pose than the one the matrix "
                    "was built for: build it again."
                )
                raise ValueError(msg)
            out.append(unknowns.of(sheet))
        return np.concatenate(out)

    def __call__(self, sources=None):
        """The field of `sources` at the observers, shaped as they were given
        with a last axis of 3: ``matrix @ charges(sources)``."""
        q = self.charges(sources).astype(self.matrix.dtype)
        return (self.matrix @ q).astype(float).reshape(*self._shape, 3)


def field_matrix(sources, observers, field="B", dtype=np.float64):
    """The field of solved sheets at fixed observers as one matrix: built once
    for a mesh, it gives the field of each new charge on that mesh by one
    product.

    A loop that solves the same parts again, where they stand, with only the
    magnets or ``UniformField`` that drive them changed, gives sheets of the
    same mesh and new charge each time. Their field at many fixed observers,
    a field map, costs a kernel evaluation per observer and triangle in
    ``getB``; from the matrix it is one product. The charge enters the field
    linearly, so the two agree to rounding.

    The unknowns are the solve's: three per panel of a solved cuboid, three
    per face of a solved triangle mesh. A panel across which a sheet's charge
    is not linear, as in a sheet built by hand, takes three per face.

    Parameters
    ----------
    sources : ChargeSheet, Collection or list
        Solved sheets, at one pose each. A Collection gives the sheets in it;
        any other source is refused: add its field with ``getB``.
    observers : array-like, shape (..., 3)
        Positions in m, in the global frame.
    field : {"B", "H"}, default "B"
        B in T, or H in A/m. As in ``getB``, B inside a solved part is
        ``mu0 H``, without the part's magnetization, and warns.
    dtype : numpy dtype, default float64
        Of the matrix. ``float32`` halves its memory, within about 1e-6 of the
        largest field around the sheets. Far away their charges' fields
        cancel, and its rounding shows: up to 4e-3 of the field there at 2,000
        times a plate's size. Use float64 for observers far away.

    Returns
    -------
    FieldMatrix
        ``fm.matrix`` (3 M, n) for M observers and n unknowns, ``fm.charges(sheets)``
        the unknowns of sheets of the same mesh and pose, and ``fm(sheets)``
        their field, shaped as `observers` with a last axis of 3.

    Notes
    -----
    The matrix takes ``3 M n`` numbers: 31 MB in float32 for 2,601 observers
    around a plate solved at level 1 (996 unknowns). Building it costs under two
    ``getB`` calls of the same observers, and about as much memory at once.

    Examples
    --------
    >>> import magpylib as magpy
    >>> import numpy as np
    >>> from magpylib_material_response.face_charge import (
    ...     ChargeSheet, field_matrix, solve_face_charge)
    >>> def sheets_with(magnet_z):
    ...     magnet = magpy.magnet.Cuboid(
    ...         polarization=(0, 0, 1), dimension=(0.01, 0.01, 0.005),
    ...         position=(0, 0, magnet_z))
    ...     plate = magpy.magnet.Cuboid(
    ...         polarization=(0, 0, 0), dimension=(0.06, 0.03, 0.004))
    ...     plate.susceptibility = 1000
    ...     solved = solve_face_charge(magpy.Collection(magnet, plate), level=1)
    ...     return [s for s in solved.sources_all if isinstance(s, ChargeSheet)]
    >>> points = np.column_stack(
    ...     [np.linspace(-0.03, 0.03, 50), np.zeros(50), np.full(50, 0.02)])
    >>> fm = field_matrix(sheets_with(0.0085), points)
    >>> fm.matrix.shape
    (150, 996)
    >>> moved = sheets_with(0.012)
    >>> B = magpy.getB(moved, points, sumup=True)
    >>> bool(np.allclose(fm(moved), B, rtol=0, atol=1e-12 * np.abs(B).max()))
    True
    """
    if field not in ("B", "H"):
        msg = f"field is 'B' or 'H'; got {field!r}."
        raise ValueError(msg)
    sheets = _sheets_of(sources)
    X = np.asarray(observers, float)
    if X.shape[-1:] != (3,):
        msg = f"observers need a last axis of 3; got shape {X.shape}."
        raise ValueError(msg)
    shape = X.shape[:-1]
    X = X.reshape(-1, 3)
    poses = [_pose(s) for s in sheets]
    unknowns = [_Unknowns(s) for s in sheets]
    offsets = np.cumsum([0] + [u.count for u in unknowns])
    matrix = np.zeros((len(X), 3, offsets[-1]), dtype)
    factor = (magpy.mu_0 if field == "B" else 1.0) / FOUR_PI
    for sheet, (position, rotation), u, a, b in zip(
        sheets, poses, unknowns, offsets[:-1], offsets[1:], strict=True
    ):
        if not u.faces:  # a part inside parts of its material: no field
            continue
        # in the sheet's own frame, as getB computes it, so that observers on a
        # face fall on the same side of it in both; the field turned back after
        turn = sheet.orientation if sheet.orientation.single else sheet.orientation[-1]
        XL = turn.apply(X - position, inverse=True)
        if field != "H" and sheet._part is not None:
            sheet._warn_if_inside(XL, field)
        V, fr = sheet._V, sheet._frames
        with np.errstate(over="ignore"):  # past 1e154 m: inf, and so far
            far = np.linalg.norm(XL - sheet._center, axis=1) >= sheet._far_from
        # the kernel's chunks: the terms add 72 bytes an evaluation to its 250 or
        # so, and a third of them took 1.3 times as long, the threads waiting on
        # one another between small arrays
        step = max(1, CHUNK // (BUILD_SHARE * max(u.faces, 1) * threads()))

        turned = not np.array_equal(rotation, np.eye(3))

        def fill(
            job, V=V, fr=fr, u=u, a=a, b=b, XL=XL, rotation=rotation, turned=turned
        ):
            terms, rows = job
            block = terms(XL[rows], V, fr).reshape(-1, 3 * u.faces) @ u.map
            block = np.asarray(block).reshape(len(rows), 3, -1)
            if turned:  # back to the global frame; a sheet not turned is in it
                block = np.einsum("ij,kjn->kin", rotation, block)
            block *= factor
            matrix[rows, :, a:b] = block

        jobs = [
            (terms, rows[k : k + step])
            for terms, rows in (
                (charge_terms, np.flatnonzero(~far)),
                (far_terms, np.flatnonzero(far)),
            )
            for k in range(0, len(rows), step)
        ]
        run(fill, jobs, work=len(X) * u.faces)
    return FieldMatrix(
        sheets, unknowns, poses, shape, matrix.reshape(3 * len(X), offsets[-1]), field
    )
