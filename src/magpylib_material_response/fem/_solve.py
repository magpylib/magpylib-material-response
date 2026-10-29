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

Currents are not meshed: their field ``B_c`` is magpylib's. With them, the
magnets take the total potential ``psi``, ``mu0 H = -grad psi``, and the air
keeps the reduced one, ``mu0 H = B_ext + B_c - grad u`` (Simkin and Trowbridge
1979). In soft iron the field is small against a coil's, and the reduced
potential would have to cancel the coil's field there almost exactly; the
total potential carries only what is left. On a magnet's surface
``psi = u + Lambda``, where ``-grad Lambda = B_ext + B_c`` inside it: the
gradient part of the applied field, projected on each solve's elements. So
``psi = u + L`` in a magnet, with ``L`` equal to ``Lambda`` on its surface and
zero inside, and ``u`` stays continuous: the same unknowns and matrix as
without currents, and a right-hand side ``(J_r - B_a - mu_r grad L)`` in the
magnets.

References
----------
- E. M. Freeman, D. A. Lowther (1988) "A novel mapping technique for open
  boundary finite element solutions to Poisson's equation", IEEE Trans. Magn.
  24(6), 2934-2936.
- Q. Chen, A. Konrad (1997) "A review of finite element open boundary
  techniques for static and quasi-static electromagnetic field problems", IEEE
  Trans. Magn. 33(1), 663-676.
- J. Simkin, C. W. Trowbridge (1979) "On the use of the total scalar potential
  in the numerical solution of field problems in electromagnetics",
  International Journal for Numerical Methods in Engineering 14(3), 423-440.
"""

from __future__ import annotations

import time
import warnings
from collections import deque
from contextlib import contextmanager
from itertools import combinations

import magpylib as magpy
import ngsolve as ng
import numpy as np
from netgen import meshing, occ
from ngsolve.comp import IntegrationRuleSpaceSurface
from scipy.spatial import cKDTree

from magpylib_material_response.fem._geometry import (
    contains,
    current_segments,
    distance,
    parts_of,
    solid,
    wire_points,
)
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
#: Grading of the volume mesh, then the gradings tried in turn when NGSolve's
#: refinement toward the edges leaves elements that do not match across a face
#: (#98): a few meshes only, such as a 1.09 x 1.4 x 0.9 mm cuboid with an
#: observer 2 mm above it, at two layers or more.
GRADINGS = (0.3, 0.35, 0.25, 0.4)
#: Segment-point pairs per call when magpylib computes the currents' field: it
#: takes every segment against every point at once, and 20,000 points against
#: a 400-segment coil peaked at 3.7 GB.
CURRENT_PAIRS = 1_000_000
#: Spacing of the points along the currents checked against the magnets, as a
#: share of the thinnest magnet; 50 times closer near a magnet.
WIRE_SPACING = 0.25
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


def _positions(observers):
    """Observers as an (n, 3) array in m, and the shape they came in."""
    obs = np.asarray(observers, dtype=float)
    if obs.shape[-1:] != (3,):
        msg = f"observers must have shape (..., 3); got {obs.shape}."
        raise ValueError(msg)
    return obs.reshape(-1, 3), obs.shape


@contextmanager
def _quiet():
    """NGSolve and netgen without their progress messages, only while we work."""
    level = ng.ngsglobals.msg_level
    ng.ngsglobals.msg_level = 0
    try:
        yield
    finally:
        ng.ngsglobals.msg_level = level


def _coordinates(mapped):
    """Coordinates (n, 3) of mapped mesh points."""
    return np.asarray(ng.CF((ng.x, ng.y, ng.z))(mapped)).reshape(-1, 3)


def _name_magnet_faces(ngmesh):
    """Name each face on a magnet by its sides and the group of touching
    magnets it bounds: ``magnet_in{k}`` where the magnet is the face's inner
    region, ``magnet_out{k}`` where it is the outer one, both against the air,
    and ``between`` where two magnets meet. Returns the number of groups."""
    regions = ngmesh.GetRegionNames(dim=3)

    def part(d):
        name = regions[d - 1] if 0 < d <= len(regions) else ""
        return int(name[4:]) if name.startswith("part") else None

    faces = [ngmesh.FaceDescriptor(k + 1) for k in range(ngmesh.GetNFaceDescriptors())]
    group = {}  # touching magnets share a group: union-find on the parts

    def find(i):
        while group.setdefault(i, i) != i:
            i = group[i]
        return i

    for face in faces:
        inner, outer = part(face.domin), part(face.domout)
        if face.bcname == "surface" and inner is not None and outer is not None:
            group[find(inner)] = find(outer)
    for face in faces:
        for i in (part(face.domin), part(face.domout)):
            if face.bcname == "surface" and i is not None:
                find(i)
    roots = sorted({find(i) for i in group})
    for k, face in enumerate(faces):
        inner, outer = part(face.domin), part(face.domout)
        if face.bcname != "surface":
            continue
        if inner is not None and outer is not None:
            ngmesh.SetBCName(k, "between")
        else:
            side, i = ("in", inner) if inner is not None else ("out", outer)
            ngmesh.SetBCName(k, f"magnet_{side}{roots.index(find(i))}")
    return len(roots)


def _hole_loops(ngmesh):
    """Closed loops (k + 1, 3), in the mesh's coordinates, on the surfaces of
    the groups of magnets that go round their holes: the surfaces' homology
    generators by tree and cotree (Eppstein 2003), two per hole, none on a
    group without holes. Magnets that touch only along an edge leave edges
    with more than two faces, which add loops round no hole: harmless, as a
    current goes through them zero times."""
    coords = np.asarray(ngmesh.Coordinates())
    faces = [
        [v.nr - 1 for v in element.vertices]
        for element in ngmesh.Elements2D()
        if ngmesh.FaceDescriptor(element.index).bcname.startswith("magnet_")
    ]
    edge_of = {}
    face_edges = [
        [
            edge_of.setdefault((min(a, b), max(a, b)), len(edge_of))
            for a, b in zip(face, face[1:] + face[:1], strict=True)
        ]
        for face in faces
    ]
    edges = list(edge_of)
    faces_of = [[] for _ in edges]
    for k, ids in enumerate(face_edges):
        for e in ids:
            faces_of[e].append(k)
    neighbours = {}
    for e, (a, b) in enumerate(edges):
        neighbours.setdefault(a, []).append((b, e))
        neighbours.setdefault(b, []).append((a, e))
    # a spanning tree of the vertices, then one of the faces across the rest
    parent, tree = {}, set()
    for root in neighbours:
        if root in parent:
            continue
        parent[root] = None
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for w, e in neighbours[u]:
                if w not in parent:
                    parent[w] = u
                    tree.add(e)
                    queue.append(w)
    seen, cotree = set(), set()
    for first in range(len(faces)):
        if first in seen:
            continue
        seen.add(first)
        queue = deque([first])
        while queue:
            face = queue.popleft()
            for e in face_edges[face]:
                if e in tree:
                    continue
                for other in faces_of[e]:
                    if other not in seen:
                        seen.add(other)
                        cotree.add(e)
                        queue.append(other)

    def to_root(v):
        path = [v]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path

    def loop(a, b):
        """The tree's paths from a and b, joined where they meet, and the edge."""
        up, down = to_root(a), to_root(b)
        while len(up) > 1 and len(down) > 1 and up[-2] == down[-2]:
            up.pop()
            down.pop()
        return coords[up[::-1] + down]

    return [
        loop(a, b)
        for e, (a, b) in enumerate(edges)
        if e not in tree and e not in cotree
    ]


def outward(mesh, groups):
    """The unit normal on the magnets' faces, pointing out of the magnets:
    NGSolve's normal points from a face's inner region to its outer one."""
    side = {f"magnet_in{k}": 1 for k in range(groups)}
    side |= {f"magnet_out{k}": -1 for k in range(groups)}
    return mesh.BoundaryCF(side, default=0) * ng.specialcf.normal(3)


def _conforming(mesh):
    """Whether the elements of `mesh` match across every face. Assembling with
    unknowns on the faces, from order 2, finds a square face of a prism
    against two triangles (#98); a mass matrix takes about 0.2 s on 50,000
    elements."""
    space = ng.H1(mesh, order=2)
    u, v = space.TnT()
    mass = ng.BilinearForm(space)
    mass += u * v * ng.dx
    try:
        mass.Assemble()
    except meshing.NgException as error:
        if "Inconsistent number of degrees of freedom" not in str(error):
            raise
        return False
    return True


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

    def __init__(self, model, order, gf, ndof, seconds, lift=None):
        self._model = model
        self._gf = gf
        self._lift = lift
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

        The field of the model's magnets, remanent and induced, and of its
        currents, without the uniform applied field: what :func:`magpylib.getB`
        gives on a collection solved by
        :func:`~magpylib_material_response.demag.apply_demag`.
        """
        return self._field(observers, "B")

    def getH(self, observers):
        """Field H in A/m at `observers`, positions (..., 3) in m, without the
        uniform applied field."""
        return self._field(observers, "H")

    def _field(self, observers, kind):
        obs, shape = _positions(observers)
        model = self._model
        s = (obs - model.center) / model.length
        out = np.empty_like(s)
        rho = np.linalg.norm(s, axis=1)
        inside = rho < model.radius
        # a later solve of the same model curves its mesh for another order
        model.curve(self.order)
        air = ~inside
        if inside.any():
            if kind == "B":
                cf = model.flux_density(self._gf, self._lift)
            else:
                cf = model.field_strength(self._gf, self._lift)
            mp = model.locate(s[inside], np.zeros(3))
            out[inside] = np.asarray(cf(mp)).reshape(-1, 3)
            if model.currents:
                air[inside] = model.in_air(mp)
        if (~inside).any():
            out[~inside] = -self._kelvin_gradient(s[~inside])
        if model.currents and air.any():
            # the currents' own field, where the air's reduced potential leaves it
            out[air] += model.applied(s[air]) - model.B_ext
        if kind == "H":
            out /= MU0
        return out.reshape(shape)

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
        self.parts, self.B_ext, self.currents = parts_of(collection)
        lo = np.min([p.bounds[0] for p in self.parts], axis=0)
        hi = np.max([p.bounds[1] for p in self.parts], axis=0)
        self.center = 0.5 * (lo + hi)
        self.length = 0.5 * float(np.linalg.norm(hi - lo))
        self.radius = BALL
        self.kelvin_center = np.array([3 * BALL, 0.0, 0.0])
        self.min_log_time = min_log_time

        self.curved = None
        sizes = self._sizes(maxh)
        if int(edge_layers) != edge_layers or edge_layers < 0:
            msg = f"edge_layers must be an integer, 0 or more; got {edge_layers}."
            raise ValueError(msg)
        if observer_maxh is not None:
            if observers is None:
                msg = "observer_maxh needs observers: the positions to refine."
                raise ValueError(msg)
            if not observer_maxh > 0 or not np.isfinite(observer_maxh):
                msg = f"observer_maxh must be positive; got {observer_maxh}."
                raise ValueError(msg)
        obs = None
        if observers is not None:
            obs = (_positions(observers)[0] - self.center) / self.length
        if self.currents:
            wires = self._wire_points()
            self._check_wires(wires)
        with (
            timelog(f"FEM mesh of {len(self.parts)} parts", min_log_time=min_log_time),
            _quiet(),
        ):
            self.mesh = self._mesh(sizes, obs, observer_maxh, edge_layers)
        if self.currents:
            self._check_linking(wires)
        self.mesh_seconds = time.perf_counter() - start

    def _sizes(self, maxh):
        """Element size of each part in scaled units: `maxh` in m, one value or
        one per part; by default each part's thinnest extent."""
        if maxh is None:
            sizes = np.array([p.thinnest for p in self.parts])
        else:
            sizes = np.broadcast_to(np.asarray(maxh, dtype=float), (len(self.parts),))
        if not np.all(sizes > 0) or not np.all(np.isfinite(sizes)):
            msg = f"maxh must be positive and finite; got {maxh}."
            raise ValueError(msg)
        return sizes / self.length

    def _observer_sizes(self, s, observer_maxh, sizes):
        """Element size at scaled observers `s`: `observer_maxh` in m, or by
        default ``OBSERVER_SHARE`` of the distance to the nearest magnet,
        between ``OBSERVER_FLOOR`` and one times that magnet's element size, and
        at most ``SHELL_MAXH`` close to the sphere."""
        if observer_maxh is not None:
            return np.full(len(s), observer_maxh / self.length)
        points = s * self.length + self.center
        gaps = np.array([distance(p.source, points) for p in self.parts])
        nearest = gaps.argmin(axis=0)
        top = np.minimum(sizes[nearest], FAR_MAXH * self.radius)
        gap = gaps[nearest, np.arange(len(s))] / self.length
        h = np.clip(OBSERVER_SHARE * gap, OBSERVER_FLOOR * top, top)
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
            # the magnets' surfaces, where the total potential meets the reduced
            shape.faces.name = "surface"
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
        geometry = occ.OCCGeometry(occ.Glue([air, *solids, kelvin]))
        for grading in GRADINGS:
            params = self._parameters(grading, sizes, observers, observer_maxh)
            ngmesh = geometry.GenerateMesh(params)
            self.groups = _name_magnet_faces(ngmesh)
            mesh = ng.Mesh(ngmesh)
            if not edge_layers:
                return mesh
            mesh.RefineHP(edge_layers, LAYER_RATIO)
            if _conforming(mesh):
                return mesh
        msg = (
            f"NGSolve's refinement toward the edges left elements that do not "
            f"match across a face at every grading of {GRADINGS} (issue #98); "
            "edge_layers=0 avoids it, at the cost of accuracy near the edges."
        )
        raise RuntimeError(msg)

    def _parameters(self, grading, sizes, observers, observer_maxh):
        """Netgen's meshing parameters: the far size, the grading, and the
        element sizes at and around the observers."""
        R = self.radius
        params = meshing.MeshingParameters(maxh=FAR_MAXH * R, grading=grading)
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
        return params

    def _check_overlap(self, solids):
        """Refuse magnets that overlap: the mesh would hold their union, where
        magpylib and apply_demag add up their fields."""
        volumes = [shape.mass for shape in solids]
        for (i, a), (j, b) in combinations(enumerate(solids), 2):
            lo_a, hi_a = self.parts[i].bounds
            lo_b, hi_b = self.parts[j].bounds
            if np.any(hi_a <= lo_b) or np.any(hi_b <= lo_a):
                continue
            shared = (a * b).mass
            if shared > 1e-9 * min(volumes[i], volumes[j]):
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
        the mesh cannot find is moved toward the centre by ``NUDGES`` in turn,
        with a warning from a thousandth of the ball's radius on."""
        mp = self.mesh(points[:, 0], points[:, 1], points[:, 2])
        moved_far = 0
        for nudge in NUDGES:
            lost = mp["nr"] < 0
            if not lost.any():
                break
            moved = center + (points[lost] - center) * (1 - nudge)
            mp[lost] = self.mesh(moved[:, 0], moved[:, 1], moved[:, 2])
            if nudge >= 1e-3:
                moved_far = max(moved_far, nudge)
        if (mp["nr"] < 0).any():
            msg = "Some observers fall outside the finite-element mesh."
            raise RuntimeError(msg)
        if moved_far:
            msg = (
                "Some observers lie on the meshed sphere, between it and its "
                f"elements, and are read up to {moved_far:.0%} of its radius "
                "closer to its centre; a higher order fits the elements to it."
            )
            warnings.warn(msg, stacklevel=4)
        return mp

    def coefficients(self):
        """Per part: mu_r = I + chi, and the source chi B_ext + J_r, in T."""
        mu, source = {}, {}
        for i, part in enumerate(self.parts):
            mu[f"part{i}"] = ng.CF(tuple((np.eye(3) + part.chi).ravel()), dims=(3, 3))
            source[f"part{i}"] = ng.CF(tuple(part.chi @ self.B_ext + part.polarization))
        return mu, source

    def flux_density(self, gf, lift=None):
        """B without the uniform applied field, in T: -grad u in the air, the
        currents' own field left out; in a part chi B_ext + J_r - mu_r grad u,
        or with currents J_r - mu_r grad(u + L) - B_ext."""
        grad = ng.grad(gf)
        mu, source = self.coefficients()
        if lift is None:
            values = {name: source[name] - m * grad for name, m in mu.items()}
        else:
            total, b_ext = grad + ng.grad(lift), ng.CF(tuple(self.B_ext))
            values = {
                f"part{i}": ng.CF(tuple(p.polarization))
                - mu[f"part{i}"] * total
                - b_ext
                for i, p in enumerate(self.parts)
            }
        return self.mesh.MaterialCF(values, default=-grad)

    def field_strength(self, gf, lift=None):
        """mu0 H without the uniform applied field, in T: -grad u, or with
        currents -grad(u + L) - B_ext in a part."""
        grad = ng.grad(gf)
        if lift is None:
            return -grad
        inside = -grad - ng.grad(lift) - ng.CF(tuple(self.B_ext))
        return self.mesh.MaterialCF(
            {f"part{i}": inside for i in range(len(self.parts))}, default=-grad
        )

    def _magnets(self):
        """The region of all magnets."""
        return self.mesh.Materials("|".join(f"part{i}" for i in range(len(self.parts))))

    def in_air(self, mp):
        """Whether each mesh point of `mp` lies outside the magnets."""
        air = self.mesh.MaterialCF(
            {f"part{i}": 0 for i in range(len(self.parts))}, default=1
        )
        return np.asarray(air(mp)).ravel() > 0.5

    def _wire_points(self):
        """Scaled points along all currents, ``WIRE_SPACING`` of the thinnest
        magnet apart and 50 times closer near the magnets."""
        spacing = WIRE_SPACING * min(p.thinnest for p in self.parts)
        points = np.concatenate(
            [wire_points(c, self.parts, spacing) for c in self.currents]
        )
        return (points - self.center) / self.length

    def _check_wires(self, s):
        """Refuse currents that run through a magnet: their field is taken as
        given in the air only."""
        points = s * self.length + self.center
        if any(np.any(contains(p.source, points)) for p in self.parts):
            msg = (
                "A current runs through a magnet; fem takes the currents' field "
                "as given around the magnets, so wires must stay outside them."
            )
            raise ValueError(msg)

    def _check_linking(self, wires):
        """Refuse currents that link a group of magnets, running through a hole
        in it: the total potential would take many values there. Each current's
        field is integrated round every loop of the groups' surfaces that goes
        round a hole; by Ampère's law the result over mu0 times the current is
        the number of times it goes through, a whole number."""
        loops = _hole_loops(self.mesh.ngmesh)
        if not loops:
            return
        spacing = WIRE_SPACING * min(p.thinnest for p in self.parts) / self.length
        tree = cKDTree(wires)
        nodes, weights = np.polynomial.legendre.leggauss(4)
        nodes = (nodes + 1) / 2
        points, steps, owner = [], [], []
        for k, loop in enumerate(loops):
            a, b = loop[:-1], loop[1:]
            # pieces short against the distance to the nearest wire, taken at
            # the edge's ends and middle and known to within the wires' spacing
            near = (
                np.min([tree.query(c)[0] for c in (a, b, 0.5 * (a + b))], axis=0)
                - 0.5 * spacing / 50
            )
            near = np.maximum(near, 1e-3 * spacing)
            length = np.linalg.norm(b - a, axis=1)
            pieces = np.clip(np.ceil(8 * length / near), 1, 5000).astype(int)
            start = np.repeat(np.arange(len(a)), pieces)
            offset = np.concatenate([np.arange(n) for n in pieces])
            t = (offset[:, None] + nodes[None, :]) / pieces[start, None]
            points.append(a[start, None, :] + t[..., None] * (b - a)[start, None, :])
            steps.append(
                np.repeat(((b - a)[start] / pieces[start, None])[:, None, :], 4, axis=1)
                * weights[None, :, None]
                / 2
            )
            owner.append(np.full(len(start) * 4, k))
        points = np.concatenate(points).reshape(-1, 3)
        steps = np.concatenate(steps).reshape(-1, 3) * self.length
        owner = np.concatenate(owner)
        x = points * self.length + self.center
        chunk = max(1, CURRENT_PAIRS // sum(current_segments(c) for c in self.currents))
        circulation = np.zeros((len(self.currents), len(loops)))
        for first in range(0, len(x), chunk):
            rows = slice(first, first + chunk)
            field = magpy.getB(self.currents, x[rows], sumup=False)
            field = np.reshape(field, (len(self.currents), -1, 3))
            for k in range(len(self.currents)):
                np.add.at(
                    circulation[k], owner[rows], np.sum(field[k] * steps[rows], axis=1)
                )
        for k, current in enumerate(self.currents):
            amps = float(np.max(np.abs(current.current)))
            if amps > 0 and np.max(np.abs(circulation[k])) > 0.5 * MU0 * amps:
                times = np.max(np.abs(circulation[k])) / (MU0 * amps)
                msg = (
                    f"A current ({type(current).__name__}, {amps:g} A) links a "
                    f"magnet, running {times:.0f} times through a hole in it: its "
                    "field has no potential there, which fem cannot model yet."
                )
                raise ValueError(msg)

    def applied(self, s):
        """B_ext plus the currents' field, in T, at scaled points `s` (n, 3)."""
        points = s * self.length + self.center
        field = np.tile(self.B_ext, (len(points), 1))
        segments = sum(current_segments(c) for c in self.currents)
        chunk = max(1, CURRENT_PAIRS // segments)
        for first in range(0, len(points), chunk):
            rows = slice(first, first + chunk)
            field[rows] += magpy.getB(self.currents, points[rows], sumup=True)
        return field

    def _two_potentials(self, order, solver_tol, max_iter):
        """For a solve at `order` with currents: the applied field's flux out of
        the magnets, its integral, and the lift L of the applied field's
        potential onto their surfaces.

        The applied field has no divergence in the magnets, so it enters only
        through their surfaces, ``-int B_a . grad v dx = -oint (B_a . n) v ds``,
        and magpylib computes it at surface integration points only."""
        mesh, magnets = self.mesh, self._magnets()
        surface = mesh.Boundaries("magnet_.*")
        points = IntegrationRuleSpaceSurface(mesh, order=2 * order, definedon=surface)
        rules = points.GetIntegrationRules()
        ds = ng.ds(definedon=surface, intrules=rules)
        # the integration points, element by element as the space holds them
        mapped = mesh.MapToAllElements(rules, surface)
        values = self.applied(_coordinates(mapped))
        components = []
        for k in range(3):
            component = ng.GridFunction(points)
            component.vec.FV().NumPy()[:] = values[:, k]
            components.append(component)
        flux = ng.InnerProduct(ng.CF(tuple(components)), outward(mesh, self.groups))
        # no net flux out of a group of magnets, as without divergence; the
        # quadrature leaves some, which the Neumann problem cannot take
        net = {}
        for k in range(self.groups):
            faces = mesh.Boundaries(f"magnet_(in|out){k}")
            total = ng.Integrate(flux * ng.ds(definedon=faces, intrules=rules), mesh)
            area = ng.Integrate(ng.CF(1) * ng.ds(definedon=faces), mesh)
            net[f"magnet_in{k}"] = net[f"magnet_out{k}"] = total / area
        flux = flux - mesh.BoundaryCF(net, default=0)
        # -grad Lambda = B_a: the Neumann problem in each group, its constant
        # fixed by a small mass term
        space = ng.H1(mesh, order=order, definedon=magnets)
        lam, w = space.TnT()
        a = ng.BilinearForm(space, symmetric=True)
        a += (ng.grad(lam) * ng.grad(w) + 1e-6 * lam * w) * ng.dx(definedon=magnets)
        f = ng.LinearForm(space)
        f += -flux * w * ds
        pre = ng.Preconditioner(a, "bddc")
        potential = ng.GridFunction(space)
        with ng.TaskManager():
            a.Assemble()
            f.Assemble()
            solver = ng.CGSolver(a.mat, pre.mat, tol=solver_tol, maxiter=max_iter)
            potential.vec.data = solver * f.vec
        if not solver.residuals[-1] <= solver_tol * solver.residuals[0]:
            msg = (
                "Conjugate gradients did not converge on the potential of the "
                f"applied field at order {order}."
            )
            raise RuntimeError(msg)
        on_surface = ~ng.H1(
            mesh, order=order, definedon=magnets, dirichlet="magnet_.*"
        ).FreeDofs()
        lift = ng.GridFunction(space)
        lift.vec.data = ng.Projector(on_surface, True) * potential.vec
        return flux, ds, lift

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
            f = ng.LinearForm(fes)
            lift = None
            if self.currents:
                flux, ds, lift = self._two_potentials(order, solver_tol, max_iter)
                remanence = mesh.MaterialCF(
                    {
                        f"part{i}": ng.CF(tuple(p.polarization))
                        for i, p in enumerate(self.parts)
                    },
                    default=ng.CF((0, 0, 0)),
                )
                inside = ng.dx(definedon=self._magnets())
                f += (remanence - mu_cf * ng.grad(lift)) * ng.grad(v) * inside
                f += -flux * v * ds
            # with no remanence and no applied field the field is zero, and a
            # zero source would leave NGSolve a form without a test function
            elif any(np.any(p.chi @ self.B_ext + p.polarization) for p in self.parts):
                f += src_cf * ng.grad(v) * ng.dx
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
            # written so that a NaN residual fails too
            if not last <= solver_tol * first:
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
        return FemSolution(self, order, gf, fes.ndof, seconds, lift)


def fem_solve(
    collection,
    order=3,
    *,
    maxh=None,
    edge_layers=4,
    observers=None,
    observer_maxh=None,
    solver_tol=1e-10,
    max_iter=5000,
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
        remanence) and susceptibility, isotropic or along their own axes (none
        on a magnet or its parents: fixed), in a uniform applied field ``H_ext`` (mu0 H in T) and the field of current
        sources, which are not meshed. The magnets must not overlap; a current
        must not run through a magnet or through a hole in one.
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
    max_iter : int, default 5000
        Iterations of the conjugate gradients before the solve fails.
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
    return model.solve(order, solver_tol=solver_tol, max_iter=max_iter)
