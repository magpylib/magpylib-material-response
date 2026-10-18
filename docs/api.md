# API Reference

## Demagnetization solver — `demag`

```{eval-rst}
.. autofunction:: magpylib_material_response.demag.apply_demag

.. autofunction:: magpylib_material_response.demag.demag_tensor

.. autofunction:: magpylib_material_response.demag.get_susceptibilities

.. autofunction:: magpylib_material_response.demag.get_H_ext
```

Helpers behind the `pairs_matching` and `max_dist` options of `apply_demag`:

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

## Utilities

```{eval-rst}
.. autofunction:: magpylib_material_response.utils.to_json

.. autofunction:: magpylib_material_response.utils.from_json

.. autofunction:: magpylib_material_response.get_dataset

.. autofunction:: magpylib_material_response.configure_logging

.. autofunction:: magpylib_material_response.disable_logging
```
