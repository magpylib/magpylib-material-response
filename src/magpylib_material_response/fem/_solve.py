"""The finite-element model: geometry, mesh, solve and the field it gives.

Reduced scalar potential: ``mu0 H = B_ext - grad u`` with ``u`` continuous,
``div(mu_r grad u) = div(chi B_ext + J_r)`` in all space, ``u -> 0`` far away.
A uniform ``B_ext`` is linear in ``u``, which every order holds exactly, so a
high susceptibility costs no accuracy.

Space is cut by a sphere of radius ``R`` around the magnets. The outside is
mapped onto a second ball by the Kelvin transform ``w(y) = (R / |y|) u(x)``,
``y = R² x / |x|²``, which keeps ``w`` harmonic, and the two balls are glued
along the sphere: the open boundary is exact (Freeman and Lowther 1988 do this
in two dimensions; Chen and Konrad 1997 review it). In three dimensions the
energy outside gains a surface term,
``∫_outside |grad u|² dx = ∫_ball |grad w|² dy + (1/R) ∫_sphere w² ds``; with it
a soft sphere's moment comes out within 1e-7 of the exact one.

References
----------
- E. M. Freeman, D. A. Lowther (1988) "A novel mapping technique for open
  boundary finite element solutions to Poisson's equation", IEEE Trans. Magn.
  24(6), 2934-2936.
- Q. Chen, A. Konrad (1997) "A review of finite element open boundary
  techniques for static and quasi-static electromagnetic field problems", IEEE
  Trans. Magn. 33(1), 663-676.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from itertools import combinations

import magpylib as magpy
import ngsolve as ng
import numpy as np
from netgen import meshing, occ

from magpylib_material_response.fem._geometry import distance, parts_of, solid
from magpylib_material_response.utils import timelog

MU0 = magpy.mu_0
#: Radius of the meshed ball, in units of the radius of the box around the
#: magnets.
BALL = 1.5
#: Element size on the sphere and in the far air, in ball radii. Netgen's
#: periodic surface mesh fails on coarser spheres.
FAR_MAXH = 0.25
#: Ratio of successive layers of geometric refinement toward edges. 0.2 gives
#: thinner layers whose conjugate gradients take five times the iterations; 0.3
#: is as accurate on the cube at order 5 (2e-5).
LAYER_RATIO = 0.3
#: How far, in ball radii, a point the mesh cannot find is moved toward the
#: ball's centre, one step after the other: points on the sphere fall between
#: it and its elements, about 1e-4 at order 2 and 1e-2 at order 1.
NUDGES = (1e-7, 1e-4, 1e-3, 1e-2, 3e-2)
#: The default element size at an observer, as a share of its distance to the
#: nearest magnet: 0.03 mm at 0.1 mm from a 1 mm cube's edge converged the field
#: there to 0.006 mT, where the magnets' element size left it 1 mT off.
OBSERVER_SHARE = 0.3
#: Observers inside or on a magnet get this share of the magnets' element size.
OBSERVER_FLOOR = 0.1
#: Element size, in ball radii, at most at observers within a quarter radius of
#: the sphere. There the elements curved onto it go wrong at the far size: two
#: of twenty points 0.95 R out read 30 mT as 36 and 71, at order 3. With this
#: cap the field 0.85-0.97 R out fell from 38 % to 1.2 % off, 1.03-1.2 R out
#: from 2 % to 0.24 %.
SHELL_MAXH = 0.125


@contextmanager
def _quiet():
    """NGSolve and netgen without their progress messages, only while we work."""
    level = ng.ngsglobals.msg_level
    ng.ngsglobals.msg_level = 0
    try:
        yield
    finally:
        ng.ngsglobals.msg_level = level


class FemSolution:
    """A solved finite-element model: its field and what the solve took.

    Made by :func:`fem_solve` or :func:`fem_study`.

    Attributes
    ----------
    order : int
        Polynomial order of the elements.
    ndof : int
        Number of unknowns.
    elements : int
        Number of elements of the mesh.
    seconds : dict
        Wall time in s: ``"mesh"`` (geometry, mesh, refinement), ``"assemble"``
        and ``"solve"``.
    moment : ndarray, shape (3,)
        Magnetic moment in A m² of all magnetization: remanent and induced.
    """

    def __init__(self, model, order, gf, ndof, seconds):
        self._model = model
        self._gf = gf
        self.order = order
        self.ndof = ndof
        self.elements = model.mesh.ne
        self.seconds = seconds
        normal = ng.CF((ng.x, ng.y, ng.z)) / model.radius
        boundary = model.mesh.Boundaries("gamma")
        # the dipole term of u on any sphere around the sources: 3/(4 pi) of
        # the moment over the sphere's area, all higher terms integrate to zero
        flux = ng.Integrate(
            gf * normal, model.mesh, definedon=boundary, order=2 * order + 2
        )
        self.moment = 3 / MU0 * model.length**3 * np.asarray(flux)

    def getB(self, observers):
        """Flux density B in T at `observers`, positions (..., 3) in m.

        The field of the model's magnets, remanent and induced, without the
        applied field: what :func:`magpylib.getB` gives on a collection solved
        by :func:`~magpylib_material_response.demag.apply_demag`.
        """
        return self._field(observers, "B")

    def getH(self, observers):
        """Field H in A/m at `observers`, positions (..., 3) in m, without the
        applied field."""
        return self._field(observers, "H")

    def _field(self, observers, kind):
        obs = np.asarray(observers, dtype=float)
        if obs.shape[-1:] != (3,):
            msg = f"observers must have shape (..., 3); got {obs.shape}."
            raise ValueError(msg)
        model = self._model
        s = (obs.reshape(-1, 3) - model.center) / model.length
        out = np.empty_like(s)
        rho = np.linalg.norm(s, axis=1)
        inside = rho < model.radius
        # a later solve of the same model curves its mesh for another order
        model.curve(self.order)
        if inside.any():
            cf = model.flux_density(self._gf) if kind == "B" else -ng.grad(self._gf)
            mp = model.locate(s[inside], np.zeros(3))
            out[inside] = np.asarray(cf(mp)).reshape(-1, 3)
        if (~inside).any():
            out[~inside] = -self._kelvin_gradient(s[~inside])
        if kind == "H":
            out /= MU0
        return out.reshape(obs.shape)

    def _kelvin_gradient(self, s):
        """grad u at points outside the ball, from its image in the Kelvin ball:
        u(s) = (R / rho) w(y), y = R² s / rho²."""
        model = self._model
        R = model.radius
        rho = np.linalg.norm(s, axis=1)[:, None]
        n = s / rho
        y = R**2 * s / rho**2 + model.kelvin_center
        mp = model.locate(y, model.kelvin_center)
        w = np.asarray(self._gf(mp)).reshape(-1, 1)
        gw = np.asarray(ng.grad(self._gf)(mp)).reshape(-1, 3)
        reflected = gw - 2 * n * np.sum(n * gw, axis=1, keepdims=True)
        return -(R / rho**2) * w * n + (R / rho) ** 3 * reflected


class Model:
    """The geometry and mesh of a magpylib model, solved at any order."""

    def __init__(
        self,
        collection,
        *,
        maxh=None,
        edge_layers=4,
        observers=None,
        observer_maxh=None,
        min_log_time=None,
    ):
        start = time.perf_counter()
        self.parts, self.B_ext = parts_of(collection)
        lo = np.min([p.bounds[0] for p in self.parts], axis=0)
        hi = np.max([p.bounds[1] for p in self.parts], axis=0)
        self.center = 0.5 * (lo + hi)
        self.length = 0.5 * float(np.linalg.norm(hi - lo))
        self.radius = BALL
        self.kelvin_center = np.array([3 * BALL, 0.0, 0.0])
        self.min_log_time = min_log_time

        self.curved = None
        sizes = self._sizes(maxh)
        if observer_maxh is not None and observer_maxh <= 0:
            msg = f"observer_maxh must be positive; got {observer_maxh}."
            raise ValueError(msg)
        obs = None
        if observers is not None:
            obs = np.asarray(observers, dtype=float)
            if obs.shape[-1:] != (3,):
                msg = f"observers must have shape (..., 3); got {obs.shape}."
                raise ValueError(msg)
            obs = (obs.reshape(-1, 3) - self.center) / self.length
        with (
            timelog(f"FEM mesh of {len(self.parts)} parts", min_log_time=min_log_time),
            _quiet(),
        ):
            self.mesh = self._mesh(sizes, obs, observer_maxh, edge_layers)
        self.mesh_seconds = time.perf_counter() - start

    def _sizes(self, maxh):
        """Element size of each part in scaled units: `maxh` in m, one value or
        one per part; by default each part's thinnest extent."""
        if maxh is None:
            sizes = np.array([p.thinnest for p in self.parts])
        else:
            sizes = np.broadcast_to(np.asarray(maxh, dtype=float), (len(self.parts),))
        if np.any(sizes <= 0):
            msg = f"maxh must be positive; got {maxh}."
            raise ValueError(msg)
        return sizes / self.length

    def _observer_sizes(self, s, observer_maxh, sizes):
        """Element size at scaled observers `s`: `observer_maxh` in m, or by
        default ``OBSERVER_SHARE`` of the distance to the nearest magnet,
        between ``OBSERVER_FLOOR`` and one times the magnets' element size, and
        at most ``SHELL_MAXH`` close to the sphere."""
        top = min(float(sizes.min()), FAR_MAXH * self.radius)
        if observer_maxh is not None:
            return np.full(len(s), observer_maxh / self.length)
        points = s * self.length + self.center
        gap = np.min([distance(p.source, points) for p in self.parts], axis=0)
        h = np.clip(OBSERVER_SHARE * gap / self.length, OBSERVER_FLOOR * top, top)
        rho = np.linalg.norm(s, axis=1) / self.radius
        shell = (rho > 0.75) & (rho < 1 / 0.75)
        h[shell] = np.minimum(h[shell], SHELL_MAXH * self.radius)
        return h

    def _mesh(self, sizes, observers, observer_maxh, edge_layers):
        R, far = self.radius, FAR_MAXH * self.radius
        solids = []
        for i, (part, h) in enumerate(zip(self.parts, sizes, strict=True)):
            shape = solid(part.source, self.center, self.length)
            shape.mat(f"part{i}")
            shape.maxh = min(h, far)
            solids.append(shape)
        self._check_overlap(solids)
        ball = occ.Sphere(occ.Pnt(0, 0, 0), R)
        ball.faces.name = "gamma"
        ball.faces.maxh = far
        kelvin = occ.Sphere(occ.Pnt(*self.kelvin_center), R).mat("kelvin")
        kelvin.faces.name = "gammak"
        ball.faces[0].Identify(
            kelvin.faces[0],
            "kelvin",
            occ.IdentificationType.PERIODIC,
            occ.Translation(occ.Vec(*self.kelvin_center)),
        )
        air = (ball - occ.Glue(solids)).mat("air")
        params = meshing.MeshingParameters(maxh=far, grading=0.3)
        if observers is not None:
            h = self._observer_sizes(observers, observer_maxh, sizes)
            rho = np.linalg.norm(observers, axis=1)
            for (x, y, z), hi in zip(observers[rho < R], h[rho < R], strict=True):
                params.RestrictH(x, y, z, hi)
            # just outside, where the field still varies on the magnets' scale:
            # at their images in the Kelvin ball, where lengths scale by
            # (R / rho)², and inside the sphere in their direction, which the
            # field at them depends on as much
            near = (rho >= R) & (rho < 2 * R)
            direction = observers[near] / rho[near, None]
            images = R**2 * direction / rho[near, None] + self.kelvin_center
            scale = (R / rho[near]) ** 2
            for (x, y, z), hi in zip(images, h[near] * scale, strict=True):
                params.RestrictH(x, y, z, hi)
            for (x, y, z), hi in zip(0.98 * R * direction, h[near], strict=True):
                params.RestrictH(x, y, z, hi)
        ngmesh = occ.OCCGeometry(occ.Glue([air, *solids, kelvin])).GenerateMesh(params)
        mesh = ng.Mesh(ngmesh)
        if edge_layers:
            mesh.RefineHP(edge_layers, LAYER_RATIO)
        return mesh

    def _check_overlap(self, solids):
        """Refuse magnets that overlap: the mesh would hold their union, where
        magpylib and apply_demag add up their fields."""
        for (i, a), (j, b) in combinations(enumerate(solids), 2):
            lo_a, hi_a = self.parts[i].bounds
            lo_b, hi_b = self.parts[j].bounds
            if np.any(hi_a <= lo_b) or np.any(hi_b <= lo_a):
                continue
            shared = (a * b).mass
            if shared > 1e-9 * min(a.mass, b.mass):
                msg = (
                    f"Magnets {i} and {j} overlap ({shared * self.length**3:.3g} "
                    "m³): the finite elements would model their union, where "
                    "magpylib and apply_demag add up their fields."
                )
                raise ValueError(msg)

    def curve(self, order):
        """Curve the mesh's elements for `order`, unless they are already."""
        if self.curved != order:
            with _quiet():
                self.mesh.Curve(order)
            self.curved = order

    def locate(self, points, center):
        """Mesh points of scaled `points` in the ball around `center`; a point
        the mesh cannot find is moved toward the centre by ``NUDGES`` in turn."""
        mp = self.mesh(points[:, 0], points[:, 1], points[:, 2])
        for nudge in NUDGES:
            lost = mp["nr"] < 0
            if not lost.any():
                return mp
            moved = center + (points[lost] - center) * (1 - nudge)
            mp[lost] = self.mesh(moved[:, 0], moved[:, 1], moved[:, 2])
        if (mp["nr"] < 0).any():
            msg = "Some observers fall outside the finite-element mesh."
            raise RuntimeError(msg)
        return mp

    def coefficients(self):
        """Per part: mu_r = I + chi, and the source chi B_ext + J_r, in T."""
        mu, source = {}, {}
        for i, part in enumerate(self.parts):
            mu[f"part{i}"] = ng.CF(tuple((np.eye(3) + part.chi).ravel()), dims=(3, 3))
            source[f"part{i}"] = ng.CF(tuple(part.chi @ self.B_ext + part.polarization))
        return mu, source

    def flux_density(self, gf):
        """B without the applied field, in T: -grad u in the air, chi B_ext + J_r
        - mu_r grad u in a part."""
        grad = ng.grad(gf)
        mu, source = self.coefficients()
        values = {name: source[name] - m * grad for name, m in mu.items()}
        return self.mesh.MaterialCF(values, default=-grad)

    def solve(self, order, *, solver_tol=1e-10, max_iter=5000):
        """Solve at polynomial `order` on this mesh."""
        if int(order) != order or order < 1:
            msg = f"order must be a positive integer; got {order}."
            raise ValueError(msg)
        order = int(order)
        mesh = self.mesh
        with (
            timelog(f"FEM solve at order {order}", min_log_time=self.min_log_time),
            _quiet(),
        ):
            start = time.perf_counter()
            self.curve(order)
            fes = ng.Periodic(ng.H1(mesh, order=order))
            u, v = fes.TnT()
            mu, source = self.coefficients()
            mu_cf = mesh.MaterialCF(
                mu, default=ng.CF((1, 0, 0, 0, 1, 0, 0, 0, 1), dims=(3, 3))
            )
            src_cf = mesh.MaterialCF(source, default=ng.CF((0, 0, 0)))
            a = ng.BilinearForm(fes, condense=True)
            a += (mu_cf * ng.grad(u)) * ng.grad(v) * ng.dx
            # the surface term of the energy outside, see the module docstring
            a += (1 / self.radius) * u * v * ng.ds("gammak")
            f = ng.LinearForm(src_cf * ng.grad(v) * ng.dx)
            pre = ng.Preconditioner(a, "bddc")
            gf = ng.GridFunction(fes)
            with ng.TaskManager():
                a.Assemble()
                f.Assemble()
                assembled = time.perf_counter()
                solver = ng.CGSolver(a.mat, pre.mat, tol=solver_tol, maxiter=max_iter)
                rhs = f.vec.CreateVector()
                rhs.data = f.vec + a.harmonic_extension_trans * f.vec
                gf.vec.data = solver * rhs
                gf.vec.data += a.harmonic_extension * gf.vec
                gf.vec.data += a.inner_solve * rhs
            first, last = solver.residuals[0], solver.residuals[-1]
            if last > solver_tol * first:
                msg = (
                    f"Conjugate gradients stopped at a relative residual of "
                    f"{last / first:.1e}, above solver_tol={solver_tol:g}, after "
                    f"{solver.iterations} iterations at order {order}."
                )
                raise RuntimeError(msg)
            done = time.perf_counter()
        seconds = {
            "mesh": self.mesh_seconds,
            "assemble": assembled - start,
            "solve": done - assembled,
        }
        return FemSolution(self, order, gf, fes.ndof, seconds)


def fem_solve(
    collection,
    order=3,
    *,
    maxh=None,
    edge_layers=4,
    observers=None,
    observer_maxh=None,
    solver_tol=1e-10,
    min_log_time=None,
):
    """Solve a magpylib model by finite elements.

    An independent check of :func:`~magpylib_material_response.demag.apply_demag`
    on the same objects: the magnets keep their shapes, not cells, and the
    field is a polynomial of degree `order` on each element.

    Parameters
    ----------
    collection : magpylib.Collection
        The model, as :func:`~magpylib_material_response.demag.apply_demag`
        takes it before meshing: Cuboid, Cylinder, CylinderSegment, Sphere,
        Tetrahedron and TriangularMesh magnets with their polarization (the
        remanence) and susceptibility, isotropic or along their own axes, in a
        uniform applied field ``H_ext`` (mu0 H in T). The magnets must not
        overlap. Current sources are not supported yet.
    order : int, default 3
        Polynomial order of the elements.
    maxh : float or sequence of float, optional
        Element size in the magnets in m, one value or one per magnet; by
        default each magnet's thinnest extent. At most a quarter of the meshed
        ball's radius, 1.5 times the radius of the box around the magnets. The
        air grades from it.
    edge_layers : int, default 4
        Layers of geometric refinement toward the magnets' edges and corners,
        where the field is singular. Each layer is 0.3 times as thick as the
        last.
    observers : array_like, shape (..., 3), optional
        Positions in m where the field will be read: the mesh is refined
        there.
    observer_maxh : float, optional
        Element size at the observers in m. By default 0.3 times each
        observer's distance to the nearest magnet, at most the magnets'
        element size, and at most an eighth of the meshed ball's radius close
        to its surface: observers close to an edge, where the field is
        singular, need elements smaller than their distance to it.
    solver_tol : float, default 1e-10
        Relative residual of the conjugate gradients.
    min_log_time : float, optional
        Steps shorter than this many seconds are not logged.

    Returns
    -------
    FemSolution
        The field (``getB``, ``getH``), the moment, and the size and wall time
        of the solve.
    """
    model = Model(
        collection,
        maxh=maxh,
        edge_layers=edge_layers,
        observers=observers,
        observer_maxh=observer_maxh,
        min_log_time=min_log_time,
    )
    return model.solve(order, solver_tol=solver_tol)
