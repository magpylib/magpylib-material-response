# API Reference

## Cells — `cells`

```{eval-rst}
.. autofunction:: magpylib_material_response.cells.solve_cells

.. autoclass:: magpylib_material_response.BHCurve

.. autoclass:: magpylib_material_response.UniformField

.. autofunction:: magpylib_material_response.cells.demag_tensor

.. autofunction:: magpylib_material_response.cells.get_susceptibilities

.. autofunction:: magpylib_material_response.cells.get_H_ext
```

## Face charge — `face_charge`

```{eval-rst}
.. autofunction:: magpylib_material_response.face_charge.solve_face_charge

.. autoclass:: magpylib_material_response.face_charge.ChargeSheet
   :members: from_triangles, mean_polarization, jump_indicator, panel_indicator
```

## Mesh convergence — `convergence`

```{eval-rst}
.. autofunction:: magpylib_material_response.convergence.study_cells

.. autoclass:: magpylib_material_response.convergence.MeshStudy
   :members: from_values

.. autofunction:: magpylib_material_response.convergence.compare

.. autoclass:: magpylib_material_response.convergence.Comparison
```

## Finite elements — `fem`

Needs the `fem` extra: `pip install "magpylib-material-response[fem]"`.

```{eval-rst}
.. autofunction:: magpylib_material_response.fem.solve_fem

.. autofunction:: magpylib_material_response.fem.study_fem

.. autoclass:: magpylib_material_response.fem.FemSolution
   :members: getB, getH

.. autoclass:: magpylib_material_response.fem.FemStudy
```

## Meshing — `meshing`

```{eval-rst}
.. autofunction:: magpylib_material_response.meshing.mesh_all

.. autofunction:: magpylib_material_response.meshing.mesh_uniform

.. autofunction:: magpylib_material_response.meshing.mesh_Cuboid

.. autofunction:: magpylib_material_response.meshing.slice_Cuboid

.. autofunction:: magpylib_material_response.meshing.mesh_Cylinder

.. autofunction:: magpylib_material_response.meshing.mesh_TriangularMesh

.. autofunction:: magpylib_material_response.meshing.mesh_thin_CylinderSegment_with_cuboids

.. autofunction:: magpylib_material_response.meshing.voxelize

.. autofunction:: magpylib_material_response.meshing_utils.trimesh_from_model3d
```

## Coil construction — `polyline`

```{eval-rst}
.. autofunction:: magpylib_material_response.polyline.create_polyline_fillet

.. autofunction:: magpylib_material_response.polyline.move_grid_along_polyline
```

## Structure analysis and FFT kernels — `demag_fft`

```{eval-rst}
.. autofunction:: magpylib_material_response.demag_fft.analyze_collection

.. autofunction:: magpylib_material_response.demag_fft.analyze_structure

.. autofunction:: magpylib_material_response.demag_fft.detect_uniform_grid

.. autofunction:: magpylib_material_response.demag_fft.build_fft_kernel

.. autofunction:: magpylib_material_response.demag_fft.demag_fft_matvec
```

## Interaction tensors — `newell`

The two interaction rules of `solve_cells`: the field at a point (`point_block`,
the default) and the field averaged over a cell (the Newell tensors).

```{eval-rst}
.. autofunction:: magpylib_material_response.newell.point_block

.. autofunction:: magpylib_material_response.newell.demag_block

.. autofunction:: magpylib_material_response.newell.demag_block_general

.. autofunction:: magpylib_material_response.newell.demag_tensor_newell

.. autofunction:: magpylib_material_response.newell.self_demag_factors
```

## Utilities

```{eval-rst}
.. autofunction:: magpylib_material_response.utils.to_json

.. autofunction:: magpylib_material_response.utils.from_json

.. autofunction:: magpylib_material_response.get_dataset

.. autofunction:: magpylib_material_response.configure_logging

.. autofunction:: magpylib_material_response.disable_logging
```
