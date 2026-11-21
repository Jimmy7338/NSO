# Offline map resampling validation

No production edit, ROS launch, sensor capture or motion was performed.

All 66 real saved maps (25 B1_C + 41 S40) exactly match current production and the corresponding saved joint belief: shape, int8 dtype and every value. Current production function AST matches preserved reference. Per-input source/output hashes and source-code hashes are in `offline_validation_compact.json`. Original tiling candidate also passes, but is slower (mean 0.644 vs 0.614 s) with higher traced allocation (28.0 vs 19.0 MB); keep this negative result.

`candidate_compact.py` preserves original floor, indexing/clipping, SAT tests, epsilon, unknown/intermediate semantics, transform calculations and output row order. It only compacts destination cells whose transformed square intersects the raw source-map AABB, keeping even touching cells. A cell entirely outside the source cannot overlap an in-bounds source cell: occupied cannot win, and free is impossible, so output remains -1. This is output computation pruning, not research-policy candidate filtering. Remaining cells run the original loop unchanged. It also passes 200 seeded synthetic cases including rotations, negative coordinates, edge perturbations, unknown/intermediate values, multiple integer dtypes and allowed small transform deviations.

Serial five-repeat representative medians: production 0.6032 s, compact 0.04227 s (14.27x; 0.561 s saved offline). Tracemalloc peaks: 19.05 MB vs 9.28 MB. Full-map aggregate means: 0.6300 s vs 0.05304 s, but early samples had other offline work overlapping; prefer serial repeat medians for speed claims. NPZ reading and equality checking are outside timed function calls.

Historical live cycles have roughly 6–7 s ROS goal execution, 3.3–4.1 s capture and 2.3 s observe. Resampling inside capture was 1.06–1.23 s. This candidate targets that actual cost but has not yet demonstrated a real 1 s reduction: idle baseline is only 0.60 s, and no live integrated capture was run. Do not claim physical experiment success or new method evidence.

`proposed_production.patch` contains the minimal source change for review, generated but not applied. Current source hash is retained in `summary.json`. `offline_benchmark*.py` are reproducible offline runners. Memory is tracked NumPy/Python allocation, not RSS.
