# API

Every public function and class, from its docstring, in the order a model is
built and solved: the solvers, what goes into a model, meshing for cells, error
estimates, what the solvers return, saving and logging. The building blocks of
the cells solver come last.

## Solvers

`solve_fem` needs the `fem` extra: see [Getting started](getting_started.md).

```{eval-rst}
.. autofunction:: magpylib_material_response.face_charge.solve_face_charge

.. autofunction:: magpylib_material_response.cells.solve_cells

.. autofunction:: magpylib_material_response.fem.solve_fem
```

## Model inputs

A `BHCurve` goes where a susceptibility goes; a `UniformField` is an applied
field, a child of the model like any source.

```{eval-rst}
.. autoclass:: magpylib_material_response.BHCurve

.. autoclass:: magpylib_material_response.UniformField
```

### Parts and coils

A closed triangle mesh from a Magpylib `model3d`, and a coil: a polyline with
rounded corners, a winding pack moved along it, and the frame the pack is drawn
in.

```{eval-rst}
.. autofunction:: magpylib_material_response.meshing_utils.trimesh_from_model3d

.. autofunction:: magpylib_material_response.polyline.create_polyline_fillet

.. autofunction:: magpylib_material_response.polyline.move_grid_along_polyline

.. autofunction:: magpylib_material_response.polyline.cross_section_frame
```

## Meshing for cells

```{eval-rst}
.. autofunction:: magpylib_material_response.meshing.mesh_uniform

.. autofunction:: magpylib_material_response.meshing.mesh_all

.. autofunction:: magpylib_material_response.meshing.mesh_Cuboid

.. autofunction:: magpylib_material_response.meshing.slice_Cuboid

.. autofunction:: magpylib_material_response.meshing.mesh_Cylinder

.. autofunction:: magpylib_material_response.meshing.mesh_TriangularMesh

.. autofunction:: magpylib_material_response.meshing.mesh_thin_CylinderSegment_with_cuboids

.. autofunction:: magpylib_material_response.meshing.voxelize
```

## Error estimates and comparison

```{eval-rst}
.. autofunction:: magpylib_material_response.convergence.study_cells

.. autoclass:: magpylib_material_response.convergence.MeshStudy
   :members: from_values

.. autofunction:: magpylib_material_response.face_charge.study_face_charge

.. autoclass:: magpylib_material_response.face_charge.FaceChargeStudy

.. autofunction:: magpylib_material_response.fem.study_fem

.. autoclass:: magpylib_material_response.fem.FemStudy

.. autofunction:: magpylib_material_response.convergence.compare

.. autoclass:: magpylib_material_response.convergence.Comparison
```

## Face charge results

What `solve_face_charge` puts in place of each solved part, and a fast reader
for many solves at the same points.

```{eval-rst}
.. autoclass:: magpylib_material_response.face_charge.ChargeSheet
   :members: from_triangles, part, mean_polarization, jump_indicator, panel_indicator

.. autofunction:: magpylib_material_response.face_charge.field_matrix

.. autoclass:: magpylib_material_response.face_charge.FieldMatrix
   :members: charges, __call__
```

## Finite-element results

```{eval-rst}
.. autoclass:: magpylib_material_response.fem.FemSolution
   :members: getB, getH
```

## Saving and logging

```{eval-rst}
.. autofunction:: magpylib_material_response.utils.to_json

.. autofunction:: magpylib_material_response.utils.from_json

.. autofunction:: magpylib_material_response.configure_logging

.. autofunction:: magpylib_material_response.disable_logging
```

## Internals

The pieces `solve_cells` is built from: the interaction tensors, the structure
analysis and FFT products, and the readers of a model's susceptibilities and
applied fields. A model never needs them; they are here for tests and for
building on the solver. `get_dataset` loads the finite-element reference fields
stored with the package for two small models: three turned cuboids, and a magnet
beside a soft cube.

```{eval-rst}
.. autofunction:: magpylib_material_response.cells.demag_tensor

.. autofunction:: magpylib_material_response.cells.get_susceptibilities

.. autofunction:: magpylib_material_response.cells.get_H_ext

.. autofunction:: magpylib_material_response.get_dataset
```

### Interaction tensors: `newell`

The two interaction rules of `solve_cells`: the field at a point (`point_block`,
the default) and the field averaged over a cell (the Newell tensors).

```{eval-rst}
.. autofunction:: magpylib_material_response.newell.point_block

.. autofunction:: magpylib_material_response.newell.demag_block

.. autofunction:: magpylib_material_response.newell.demag_block_general

.. autofunction:: magpylib_material_response.newell.demag_tensor_newell

.. autofunction:: magpylib_material_response.newell.self_demag_factors
```

### Structure analysis and FFT products: `demag_fft`

```{eval-rst}
.. autofunction:: magpylib_material_response.demag_fft.analyze_collection

.. autofunction:: magpylib_material_response.demag_fft.analyze_structure

.. autofunction:: magpylib_material_response.demag_fft.detect_uniform_grid

.. autofunction:: magpylib_material_response.demag_fft.build_fft_kernel

.. autofunction:: magpylib_material_response.demag_fft.demag_fft_matvec
```
