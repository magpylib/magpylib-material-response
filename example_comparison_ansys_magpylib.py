import matplotlib.pyplot as plt
import numpy as np

isotropic_results_ansys = np.loadtxt(
    "tests/testdata/isotropic_results_ansys.txt", skiprows=1
)
isotropic_results_ansys = isotropic_results_ansys[:, 3:]
anisotropic_results_ansys = np.loadtxt(
    "tests/testdata/anisotropic_results_ansys.txt", skiprows=1
)
anisotropic_results_ansys = anisotropic_results_ansys[:, 3:]
isotropic_results_magpylib = np.load("isotropic_results_magpylib_15625.npy")
anisotropic_results_magpylib = np.load("anisotropic_results_magpylib_15625.npy")


isotropic_results_ansys = isotropic_results_ansys.reshape((6, -1, 3))
anisotropic_results_ansys = anisotropic_results_ansys.reshape((6, -1, 3))
isotropic_results_magpylib = isotropic_results_magpylib.reshape((6, -1, 3))
anisotropic_results_magpylib = anisotropic_results_magpylib.reshape((6, -1, 3))


isotropic_results_ansys_abs = np.linalg.norm(isotropic_results_ansys, axis=-1)
anisotropic_results_ansys_abs = np.linalg.norm(anisotropic_results_ansys, axis=-1)
isotropic_results_magpylib_abs = np.linalg.norm(isotropic_results_magpylib, axis=-1)
anisotropic_results_magpylib_abs = np.linalg.norm(anisotropic_results_magpylib, axis=-1)


for i in range(6):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    print("evaluation line ", i)
    ax1.plot(
        isotropic_results_ansys[i, :, 0],
        label="isotropic ansys x",
        color="C0",
        linestyle="-",
    )
    ax1.plot(
        isotropic_results_ansys[i, :, 1],
        label="isotropic ansys y",
        color="C0",
        linestyle="--",
    )
    ax1.plot(
        isotropic_results_ansys[i, :, 2],
        label="isotropic ansys z",
        color="C0",
        linestyle="-.",
    )
    ax1.plot(
        isotropic_results_magpylib[i, :, 0],
        label="isotropic magpylib x",
        color="C1",
        linestyle="-",
    )
    ax1.plot(
        isotropic_results_magpylib[i, :, 1],
        label="isotropic magpylib y",
        color="C1",
        linestyle="--",
    )
    ax1.plot(
        isotropic_results_magpylib[i, :, 2],
        label="isotropic magpylib z",
        color="C1",
        linestyle="-.",
    )
    ax1.plot(
        anisotropic_results_ansys[i, :, 0],
        label="anisotropic ansys x",
        color="C2",
        linestyle="-",
    )
    ax1.plot(
        anisotropic_results_ansys[i, :, 1],
        label="anisotropic ansys y",
        color="C2",
        linestyle="--",
    )
    ax1.plot(
        anisotropic_results_ansys[i, :, 2],
        label="anisotropic ansys z",
        color="C2",
        linestyle="-.",
    )
    ax1.plot(
        anisotropic_results_magpylib[i, :, 0],
        label="anisotropic magpylib x",
        color="C3",
        linestyle="-",
    )
    ax1.plot(
        anisotropic_results_magpylib[i, :, 1],
        label="anisotropic magpylib y",
        color="C3",
        linestyle="--",
    )
    ax1.plot(
        anisotropic_results_magpylib[i, :, 2],
        label="anisotropic magpylib z",
        color="C3",
        linestyle="-.",
    )
    ax1.set_xlabel("point along avaluation line")
    ax1.set_ylabel("field components [T]")
    ax1.grid()
    ax1.legend()

    ax2.plot(
        isotropic_results_magpylib[i, :, 0] - isotropic_results_ansys[i, :, 0],
        label="isotropic error x",
        color="C4",
        linestyle="-",
    )
    ax2.plot(
        isotropic_results_magpylib[i, :, 1] - isotropic_results_ansys[i, :, 1],
        label="isotropic error y",
        color="C4",
        linestyle="--",
    )
    ax2.plot(
        isotropic_results_magpylib[i, :, 2] - isotropic_results_ansys[i, :, 2],
        label="isotropic error z",
        color="C4",
        linestyle="-.",
    )
    ax2.plot(
        anisotropic_results_magpylib[i, :, 0] - anisotropic_results_ansys[i, :, 0],
        label="anisotropic error x",
        color="C5",
        linestyle="-",
    )
    ax2.plot(
        anisotropic_results_magpylib[i, :, 1] - anisotropic_results_ansys[i, :, 1],
        label="anisotropic error y",
        color="C5",
        linestyle="--",
    )
    ax2.plot(
        anisotropic_results_magpylib[i, :, 2] - anisotropic_results_ansys[i, :, 2],
        label="anisotropic error z",
        color="C5",
        linestyle="-.",
    )
    ax2.set_xlabel("point along avaluation line")
    ax2.set_ylabel("field components difference [T]")
    ax2.grid()
    ax2.legend()
    fig.suptitle("evaluation line %d" % i)
    plt.show()


for i in range(6):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    print("evaluation line ", i)
    ax1.plot(isotropic_results_ansys_abs[i, :], label="isotropic_results", color="C0")
    ax1.plot(
        isotropic_results_magpylib_abs[i, :],
        label="isotropic_results_magpylib",
        color="C1",
    )
    ax1.plot(
        anisotropic_results_ansys_abs[i, :], label="anisotropic_results", color="C2"
    )
    ax1.plot(
        anisotropic_results_magpylib_abs[i, :],
        label="anisotropic_results_magpylib",
        color="C3",
    )
    ax1.set_xlabel("point along avaluation line")
    ax1.set_ylabel("field amplitude [T]")
    ax1.grid()
    ax1.legend()

    ax2.plot(
        (isotropic_results_magpylib_abs[i, :] - isotropic_results_ansys_abs[i, :])
        / isotropic_results_ansys_abs[i, :]
        * 100,
        label="isotropic_results_magpylib",
        color="C4",
    )
    ax2.plot(
        (anisotropic_results_magpylib_abs[i, :] - anisotropic_results_ansys_abs[i, :])
        / anisotropic_results_ansys_abs[i, :]
        * 100,
        label="anisotropic_results_magpylib",
        color="C5",
    )
    ax2.set_xlabel("point along avaluation line")
    ax2.set_ylabel("field amplitude difference [%]")
    ax2.grid()
    ax2.legend()
    fig.suptitle("evaluation line %d" % i)
    plt.show()
