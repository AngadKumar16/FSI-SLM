"""UNet CFD surrogate ("sensor") for airfoil velocity/pressure fields.

Upstream data-producer for the FSI-SLM interpreter: this package turns an
airfoil shape + flow condition into velocity/pressure fields (and integrated
force coefficients), which downstream become ``RawExperiment`` /
``FeatureRecord`` objects (see ``Data/schema.py``).

v1 scope: rigid foil, steady RANS mean fields. Transient/2-way FSI is a planned
extension -- the input representation reserves scalar channels and the dataset
keeps per-case metadata so a temporal axis can be added without a rewrite.
"""
