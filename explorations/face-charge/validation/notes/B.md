**Reading.** The dataset interleaves the three sensor lines step by step, as the
page's dataframe does; read that way, face charge is within 0.34 % of ANSYS at
worst and 0.027 % RMS, converged from m = 3 (1.2 s, within 6e-5 of m = 6). The
MoM's finest, 466 cells in 0.2 s, is at 0.75 % and 0.12 %, and still moves by
3e-3 between its last two levels. On every line its error is 3 to 10 times the
face-charge one and has a shape: bumps beside the magnets. The face-charge error
has none, it looks like the FEM's mesh noise. The worst point for both is 0.5 mm
under the hard magnet's edge: Bx is 60.12 mT by face charge, 59.68 by ANSYS,
60.52 by the MoM, which approaches face charge as it refines.
