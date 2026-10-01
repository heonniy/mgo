"""Independent reference oracle for MoE expert-cache controller verification.

This package is the *new ground truth* for controller-side eviction / owner /
fetch planning.  It is written from the design spec, NOT from the production
``moe_infinity_ep.controller`` code and NOT from the (suspect) golden JSONL
files under ``실험/moe_cache_golden``.  It deliberately imports nothing from
``moe_infinity_ep`` so that a differential test against the production
controller is a genuine cross-check rather than a tautology.

See ``reference.py`` for the model and ``SPEC.md`` for the encoded semantics.
"""
