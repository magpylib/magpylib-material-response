References: U-core peak 2.152 mT ±0.2 %; cube moment 0.02888 A m² ±0.1 %; plate
moment 0.5633 A m² (-0.23..+0.11 %)

## U-core, uniform cells (apply_demag on #52, direct)

| mesh                               | cells | peak error | solve (s) | GB  |
| ---------------------------------- | ----- | ---------- | --------- | --- |
| cuboids, uniform                   | 88    | -36.0 %    | 0.0       | 0.1 |
| cuboids, uniform                   | 297   | -22.3 %    | 0.1       | 0.1 |
| cuboids, uniform                   | 704   | -14.8 %    | 0.4       | 0.4 |
| cuboids, uniform                   | 1,375 | -10.5 %    | 1.2       | 0.9 |
| cuboids, uniform                   | 2,376 | -7.8 %     | 3.5       | 2.1 |
| cuboids, uniform                   | 3,773 | -6.0 %     | 9.3       | 5.0 |
| tetrahedra, U-core as one surface  | 89    | -1.7 %     | 0.0       | 0.1 |
| tetrahedra, U-core as one surface  | 417   | +0.2 %     | 0.5       | 0.9 |
| tetrahedra, U-core as one surface  | 989   | -0.5 %     | 2.3       | 1.2 |
| tetrahedra, U-core as one surface  | 1,928 | -0.4 %     | 9.0       | 2.8 |
| tetrahedra, U-core as one surface  | 4,016 | -0.2 %     | 43.5      | 7.1 |
| tetrahedra, each cuboid on its own | 72    | +2.7 %     | 0.0       | 0.1 |
| tetrahedra, each cuboid on its own | 359   | +1.8 %     | 0.4       | 0.6 |
| tetrahedra, each cuboid on its own | 1,033 | -0.7 %     | 2.6       | 1.3 |
| tetrahedra, each cuboid on its own | 1,968 | +2.2 %     | 9.5       | 2.6 |

## U-core, #52 direct vs iterative on the same uniform cuboid grids

| cells | direct (s) | GB  | iterative (s) | GB  |
| ----- | ---------- | --- | ------------- | --- |
| 88    | 0.01       | 0.1 | 0.02          | 0.1 |
| 297   | 0.08       | 0.1 | 0.08          | 0.1 |
| 704   | 0.38       | 0.4 | 0.34          | 0.4 |
| 1,375 | 1.19       | 0.9 | 1.14          | 0.6 |
| 2,376 | 3.53       | 2.1 | 12.57         | 3.2 |
| 3,773 | 9.27       | 5.0 | –             | –   |

## U-core, adaptive refinement (residual indicator)

| mesh                 | cells | peak error | solve (s) | indicator (s) |
| -------------------- | ----- | ---------- | --------- | ------------- |
| cuboids, adaptive    | 88    | -36.0 %    | 0.0       | 0.1           |
| cuboids, adaptive    | 242   | -16.5 %    | 0.1       | 0.2           |
| cuboids, adaptive    | 620   | -14.2 %    | 0.3       | 1.2           |
| cuboids, adaptive    | 1,572 | -5.9 %     | 2.0       | 6.6           |
| cuboids, adaptive    | 3,994 | -4.3 %     | 11.2      | 41.6          |
| tetrahedra, adaptive | 89    | -1.7 %     | 0.0       | 0.1           |
| tetrahedra, adaptive | 292   | -1.2 %     | 0.2       | 0.6           |
| tetrahedra, adaptive | 593   | -1.1 %     | 0.9       | 2.3           |
| tetrahedra, adaptive | 1,377 | -0.5 %     | 4.7       | 11.8          |
| tetrahedra, adaptive | 1,496 | -0.5 %     | 5.5       | 13.9          |

## U-core, linear cells (12 unknowns per cell)

| scheme                                                    | cells | unknowns | peak error | solve (s) | GB  |
| --------------------------------------------------------- | ----- | -------- | ---------- | --------- | --- |
| linear cuboids, LSQ at 8 points (±0.4)                    | 88    | 1,056    | -2.14 %    | 0.16      | 0.2 |
| linear cuboids, LSQ at 8 points (±0.4)                    | 297   | 3,564    | -1.00 %    | 1.62      | 1.2 |
| linear cuboids, LSQ at 8 points (±0.4)                    | 704   | 8,448    | -0.60 %    | 10.68     | 3.6 |
| linear cuboids, Galerkin, 8-point quadrature              | 88    | 1,056    | -1.09 %    | 0.15      | –   |
| linear cuboids, Galerkin, 8-point quadrature              | 297   | 3,564    | -0.58 %    | 1.46      | –   |
| linear cuboids, Galerkin, converged quadrature (reviewer) | 88    | 1,056    | -0.79 %    | 3.8       | –   |
| linear cuboids, Galerkin, converged quadrature (reviewer) | 297   | 3,564    | -0.41 %    | 20.5      | –   |
| linear tetrahedra, LSQ at 8 child centroids               | 89    | 1,068    | -3.16 %    | 0.26      | 0.2 |
| linear tetrahedra, LSQ at 8 child centroids               | 417   | 5,004    | -1.41 %    | 5.29      | 1.8 |
| linear tetrahedra, 4 points at f = 0.447                  | 89    | 1,068    | -0.78 %    | 0.13      | –   |
| linear tetrahedra, 4 points at f = 0.447                  | 417   | 5,004    | -0.22 %    | 2.72      | –   |
| linear tetrahedra, 4 points at f = 0.5                    | 89    | 1,068    | +0.04 %    | 0.13      | –   |
| linear tetrahedra, 4 points at f = 0.5                    | 417   | 5,004    | +0.17 %    | 2.71      | –   |
| linear tetrahedra, 4 points at f = 0.5 + centroid, LSQ    | 89    | 1,068    | -0.60 %    | 0.17      | –   |
| linear tetrahedra, 4 points at f = 0.5 + centroid, LSQ    | 417   | 5,004    | -0.18 %    | 3.45      | –   |

## Iron cube moment

| scheme                                                    | cells | unknowns | moment error | solve (s) |
| --------------------------------------------------------- | ----- | -------- | ------------ | --------- |
| cuboids, uniform                                          | 8     | 24       | -13.82 %     | 0.00      |
| cuboids, uniform                                          | 27    | 81       | -8.80 %      | 0.10      |
| cuboids, uniform                                          | 64    | 192      | -6.09 %      | 0.50      |
| cuboids, uniform                                          | 216   | 648      | -3.43 %      | 5.10      |
| cuboids, uniform                                          | 512   | 1,536    | -2.23 %      | 28.60     |
| linear cuboids, LSQ at 8 points                           | 8     | 96       | -3.77 %      | 0.00      |
| linear cuboids, LSQ at 8 points                           | 27    | 324      | -1.87 %      | 0.01      |
| linear cuboids, LSQ at 8 points                           | 64    | 768      | -1.14 %      | 0.06      |
| linear cuboids, LSQ at 8 points                           | 216   | 2,592    | -0.60 %      | 0.76      |
| linear cuboids, Galerkin, 8-point quadrature              | 27    | 324      | -2.09 %      | 0.01      |
| linear cuboids, Galerkin, 8-point quadrature              | 64    | 768      | -1.30 %      | 0.06      |
| linear cuboids, Galerkin, converged quadrature (reviewer) | 8     | 96       | -3.54 %      | –         |
| linear cuboids, Galerkin, converged quadrature (reviewer) | 27    | 324      | -1.68 %      | –         |
| linear cuboids, Galerkin, converged quadrature (reviewer) | 64    | 768      | -1.02 %      | –         |
| linear tetrahedra, LSQ at 8 child centroids               | 108   | 1,296    | -1.03 %      | 0.33      |
| linear tetrahedra, LSQ at 8 child centroids               | 408   | 4,896    | -0.81 %      | 4.88      |
| linear tetrahedra, 4 points at f = 0.447                  | 108   | 1,296    | -0.47 %      | 0.16      |
| linear tetrahedra, 4 points at f = 0.447                  | 408   | 4,896    | -0.41 %      | 2.44      |
| linear tetrahedra, 4 points at f = 0.5                    | 108   | 1,296    | +0.07 %      | 0.16      |
| linear tetrahedra, 4 points at f = 0.5                    | 408   | 4,896    | -0.03 %      | 2.46      |
| linear tetrahedra, 4 points at f = 0.5 + centroid, LSQ    | 108   | 1,296    | -0.24 %      | 0.21      |

## Thin plate moment

| mesh, code                                  | cells  | moment error | solve (s) | GB   |
| ------------------------------------------- | ------ | ------------ | --------- | ---- |
| cuboids 5×1, main                           | 25     | +9.55 %      | 1.2       | 0.09 |
| cuboids 10×1, main                          | 100    | +0.96 %      | 1.2       | 0.11 |
| cuboids 20×1, main                          | 400    | -2.87 %      | 1.2       | 0.28 |
| cuboids 40×1, main                          | 1,600  | -3.92 %      | 3.5       | 2.47 |
| cuboids 20×2, main                          | 800    | -1.49 %      | 1.2       | 0.81 |
| cuboids 20×3, main                          | 1,200  | -0.84 %      | 2.1       | 1.69 |
| cuboids 20×3, #52 direct                    | 1,200  | -0.84 %      | 0.8       | 0.57 |
| cuboids 20×3, #52 iterative                 | 1,200  | -0.84 %      | 0.2       | 0.11 |
| cuboids 40×3, #52 iterative                 | 4,800  | -1.12 %      | 1.1       | 0.16 |
| cuboids 40×6, #52 iterative                 | 9,600  | -0.70 %      | 1.6       | 0.23 |
| cuboids 40×12, #52 iterative                | 19,200 | -0.59 %      | 2.6       | 0.37 |
| cuboids 80×6, #52 iterative                 | 38,400 | -0.50 %      | 6.7       | 0.54 |
| cuboids 80×12, #52 iterative                | 76,800 | -0.34 %      | 11.5      | 0.98 |
| tetrahedra (relaxed quality), #52 iterative | 1,728  | +0.72 %      | 10.2      | 2.30 |
| tetrahedra (relaxed quality), #52 direct    | 1,728  | +0.72 %      | 7.1       | 1.63 |
| tetrahedra, #52 direct                      | 3,328  | +0.24 %      | 28.6      | 4.00 |
