**Reading.** Face charge is converged from m = 2: 2,016 unknowns in 0.9 s,
within 1.4e-4 of m = 4. The MoM converges toward it, 2.2e-3 apart at 682 cells.
Where both differ most from ANSYS, they agree with each other: at x = −4 mm, the
end of the sensor line, both give Bx = 5.24 mT and ANSYS 6.05 mT. The jagged
error along the rest of the line is the FEM's mesh noise. So this case resolves
ANSYS's error, about 0.6 % at worst, and no less. For these weak magnets the
MoM is the cheaper of the two: 0.5 s at 682 cells.
