# Full dual-template gain in unknown-space exploration

This is a declared new joint implementation. It preserves the v10_3_1 unknown-coverage term, measured-surface-quality term, shared instance frontend, candidate generator, execution and return accounting. It replaces the entire analytic measured-AABB prior with expected visible, unexplained template surface area over committed outbound states. It does not add both priors and does not reproduce the full V3 planner or V35 public-pose-graph system.

## Source and computation

The source is the frozen `nso/semantic_completion_v3.py:ObjectCompletionModel`. `FullTemplateModel` preserves its box/shelf construction, dimensions from measured quantiles plus fixed minimum dimensions, pseudo-posterior fit, full-scene depth-history rejection, global 3,500-point cap, and per-hypothesis self-occlusion. `_box` and `gain` are inherited directly. The model never updates the measured map or TSDF.

The input grouping adaptation partitions the original globally capped 10,000 measured quality points by their shared measured support tracks, including the unknown partition. It retains original quality voxel keys. Original XY closing, minimum support, 2.2 m horizontal extent and 0.25 m vertical extent tests operate within each partition. No full instance bounds, hidden mesh, world ID or reference surface is supplied to this model.

Current labels are `2=open, 3=closed`; V3 used `2=box, 3=shelf`. Only the local template-prior label view is converted. The semantic shelf prior remains the original 0.9 for open and 0.1 for closed; the geometry arm begins at 0.5. Both use the original mean clipped-distance fit and 6 cm model-discrepancy scale. These are heuristic pseudo-posteriors, not calibrated probabilities. Each model is reconstructed from the original prior and current accumulated measurements, so repeated planning does not repeatedly apply the same evidence to an old posterior.

History uses the original mapper's retained, **full-scene** RGB-D keyframes, never depth cropped to an instance. Points contradicted by measured free space or already explained by measured depth are removed. The retained hypotheses are checked against measured TSDF occlusion and separately against their own alternative template meshes. Mutually exclusive box/shelf meshes do not occlude one another.

For a route, source `gain` is evaluated at each outbound successor; the pointwise maximum is summed to form the union of visible expected area. The initial observation and return route add no predicted utility. The replacement coefficient is 1, matching the existing v10 prior's outer coefficient and area convention. V3's old `hypothesis_weight=4` belonged to its separate normalized quality objective and is not transferred here.

## Six arms

| Arm | Selected score |
| --- | --- |
| C | Original coverage only |
| J | Original coverage + measured surface quality |
| G | J + full box/shelf gain; no category prior; geometric fit and history active |
| S | J + full gain; category prior, geometric fit and history active |
| S_no_fit | S with geometry-fit log-likelihood ratio zero; geometry, prior, history and visibility retained |
| S_no_history | S with both historical free-space contradiction and already-measured surface removal disabled; fit and visibility retained |

The latter is a combined depth-history mechanism ablation. It is not only a memory ablation. G and S here denote the full-template version, distinct from prior analytic G/S runs. All arms use the same instance support and candidate-generation algorithm at the same observation history; paths may diverge afterward. C/J still compute template counterfactual diagnostics, which must be accounted for if reporting wall-clock performance.

## Runner API

Use the scientific source path and study root on `sys.path`, then `from method import adapter`.

1. Call `adapter.install()` once after the shared instance frontend installation. It binds only the checked in-process `ExplorationJointCPUFourModules._scores_v10_1`; absent `args.dual_template_arm` delegates to the original scorer.
2. Set `args.dual_template_arm` to the six-arm name. Set `cpu_score_mode` to C, N, G, S, S, S respectively.
3. After cold-start fusion and shared instance association, call `adapter.initialize(mapper, raw_frame0)` before selecting a target.
4. After associating each new raw frame with shared instance support, call `adapter.observe(mapper, raw_frame)` before original `runtime.observe`. This records a hash and invalidates the cache; it does not fuse, advance time or consume budget. Scoring rejects a registered frame that has not yet been fused.
5. Preserve `adapter.receipt(mapper)` at decisions and `adapter.audit_log(mapper)` plus `adapter.frame_receipts(mapper)` at termination or failure. These include template construction, fit, depth-removal counts, actual source `gain` call counts, outbound area and excluded return actions. Cache reuse at the same fused map state is explicit.

No matrix is started by this package. Code parity/mechanism checks do not establish exploration or reconstruction superiority.
