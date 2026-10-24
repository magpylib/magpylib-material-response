**Summary, all six cases.**

- **Accuracy.** Face charge agrees with every reference to within that
  reference's own uncertainty:
  - ANSYS: 0.64 % in A, at the end of the sensor line where the MoM agrees with
    face charge, not with ANSYS; 0.34 % in B;
  - the pages' MoM references: 0.023 % in C (good to 0.14 %), 0.010 % in D (0.03
    %), 0.19 % in E (3.3 %);
  - FEM: 0.12 % in F (0.2 %).

  In C, E and F it narrows the value further than the reference can.

- **The MoM at the pages' settings** is 0.64, 0.75, 0.29, 0.027, 10 and 23 %
  off. It is good on magnets and arrays (A, D) and poor near thin or large soft
  parts (E, F).
- **Time.** Face charge wins on soft iron, where the MoM needs many cells: 0.4 s
  against 8 s for four times the accuracy in C. It loses on weak magnets (A: 0.9
  s against 0.5 s at equal accuracy), on many touching bodies (D: 6.5 s against
  5.1 s), and on thin sheets (E: 8.6 s or more). In the last two the costs are
  the dense coupled system and panels sized to the thickness across a whole
  sheet.
