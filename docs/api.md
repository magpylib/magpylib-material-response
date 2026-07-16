# API Reference

The public functions of `magpylib-material-response`, grouped by module.

## Demagnetization solver — `demag`

```{eval-rst}
.. autofunction:: magpylib_material_response.demag.apply_demag

.. autofunction:: magpylib_material_response.demag.demag_tensor

.. autofunction:: magpylib_material_response.demag.get_susceptibilities

.. autofunction:: magpylib_material_response.demag.get_H_ext
```

Legacy helpers of the historical all-point-matching path (used by the
`pairs_matching` and `max_dist` options of `apply_demag`):

```{eval-rst}
.. autofunction:: magpylib_material_response.demag.filter_distance

.. autofunction:: magpylib_material_response.demag.match_pairs
```

## Meshing — `meshing`

```{eval-rst}
.. autofunction:: magpylib_material_response.meshing.mesh_all

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

## Analytical Newell tensors — `newell`

```{eval-rst}
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
