"""Validation of the methods against finite elements (issue #82, step 2).

Two lists: the cases (``cases.py``), each a model and its sensors, and the
methods (``methods.py``), each an adapter with settings from coarse to fine,
with magpylib alone, the model without its material response, as the floor.
Every case has a finite-element reference stored in ``references/``, made once
by ``python -m validation.references`` (needs the fem extra). ``python -m
validation.run`` solves cases with methods and prints the error of B in each
sensor group against the reference, with the time of each phase: prepare,
solve and getB. ``tests/test_validation.py`` runs each method at the setting
``pr_check.json`` stores for it and fails when a group's error or the size of
the problem grows.

In CI, ``timing.py`` times that check and measures its peak memory against the
PR's base branch for a comment (``.github/workflows/timing.yml``), and every night the references are
made again and every method climbs its settings; ``report.py`` turns the
records into a table of time to accuracy and lists what got worse against the
baseline, the best of every night that passed
(``.github/workflows/nightly.yml``).
"""
