"""Validation of the methods against finite elements (issue #82, step 2).

Two lists: the cases (``cases.py``), each a model and its sensors, and the
methods (``methods.py``), each an adapter with settings from coarse to fine.
Every case has a finite-element reference stored in ``references/``, made once
by ``python -m validation.references`` (needs the fem extra). ``python -m
validation.run`` solves cases with methods and prints the error of B in each
sensor group against the reference, with the time of each phase: prepare,
solve and getB. ``tests/test_validation.py`` runs each method at the setting
``pr_check.json`` stores for it and fails when a group's error or the size of
the problem grows.
"""
