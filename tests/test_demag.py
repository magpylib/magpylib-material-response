# import magpylib as magpy

# import magpylib_material_response

# print(magpylib_material_response.__version__)
# import matplotlib.pyplot as plt
# import numpy as np

# from magpylib_material_response.demag import apply_demag
# from magpylib_material_response.meshing import mesh_Cuboid

# elements = 4

# # hollow cylinder magnet
# cuboid = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(2, 2, 2))
# coll = mesh_Cuboid(cuboid, elements)
# # coll.susceptibility = (1,2,3,4)
# for i in range(len(coll)):
#     coll[i].susceptibility = (i, i, i * 10)
#     # coll[i].susceptibility = i
# # coll.H_ext = (-10,-10,-10)
# coll = apply_demag(coll)

# fig, ax = plt.subplots()

# ts = np.linspace(0, 1, 9)
# grid = np.array([[(x, 0.1, z) for x in ts] for z in ts])

# B = coll.getM(grid)

# # Display the B-field with streamplot using log10-scaled
# # color function and linewidth
# splt = ax.quiver(grid[:, :, 0], grid[:, :, 2], B[:, :, 0], B[:, :, 2])

# print(B)


# # Figure styling
# ax.set(
#     xlabel="x-position (mm)",
#     ylabel="z-position (mm)",
# )

# plt.tight_layout()
# plt.show()
