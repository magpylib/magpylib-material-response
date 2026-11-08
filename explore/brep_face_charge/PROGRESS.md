# B-rep parts for face charge: study

**Question.** A STEP file holds a part's exact geometry (B-rep); an STL export
cuts its curved faces into facets of the exporter's choosing. Would a source
that keeps the B-rep, and lets the solver facet curved faces itself, as it does
a `Cylinder` today, beat solving the export? In unknowns and time for the same
accuracy, against a reference on the exact geometry.

**Parts** (`geometry.py`), soft iron χ = 1000 in a 10 mT field along
(0.8, 0.36, 0.48), sensors near the curves and 60 mm away:

- `plate`: the validation's CAD plate, 20 × 12 × 4 mm, 3 mm fillets on its
  vertical edges, a 3 mm hole: curved faces are a small part of its surface.
- `pole`: a Ø20 × 10 mm bar with its top rim filleted 3 mm: curved faces are
  most of it.

**Arms** (`arms.py`): an export at 12, 24, 48, 96 facets per turn, fixed,
solved at levels 1 to 3; the solver's own faceting, `12 (level + 1)` per turn,
widened to keep the volume or not.

**Reference** (`reference.py`): finite elements on the exact OpenCASCADE solid,
curved elements, orders 3 to 5, as `validation/references.py` makes one.

## Steps

| Step | What | Status | Result |
| --- | --- | --- | --- |
| 1 | Geometry: exact solids, facetings, volumes | done | widened facetings keep the exact volume; unwidened lose 0.4 % (plate) to 5 % (pole) at 12/turn |
| 2 | Exact references, plate and pole | running | |
| 3 | Arms, plate and pole | | |
| 4 | Figures and conclusion | | |
