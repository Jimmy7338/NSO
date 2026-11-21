# B5: observed-object support and outer-plane fitting

This is an explicitly revised research method, not an engineering-equivalent
implementation of B/B2/B3/B4. Robot hardware remains paused. No source mesh,
true object dimensions, future observations or evaluator labels are read by
the model. All evaluation comparisons retain failures and negative results.

## Three isolated stages

| Stage | Object support | Geometry input | Template envelope |
|---|---|---|---|
| object_support (O) | Acquired track > 0, pooled | Legacy .12 < z < 1.8 m quality crop | Original B initialization |
| object_full_support (O_full) | Acquired track > 0, pooled | All already acquired quality points of the track | Original B initialization |
| object_fit (OF) | Acquired track > 0, pooled | Same all-acquired points as O_full | Measured outer-plane fit |

API: `configure(object_support=True, full_observed_support=False,
geometry_fit=False)` returns the class used by the unchanged adapter. The last
two flags are independent, but only the three declared stages are evaluated.

Track 0 no longer generates full furniture hypotheses. Its measured map,
generic measured-surface reward and unknown-coverage reward remain intact;
no map points or planning candidates are removed. Track ownership is acquired
online. The current controlled scene associates one observed shelf and one
cabinet; this is not evidence of a general instance segmentation system.

G and S use the same acquired object support and geometry. G has a neutral
50/50 class prior and therefore is **not a completely semantics-free baseline**.
The original S class vote changes the conditional shelf/box prior only. Shared
track support makes objectness 1 for both methods; an inherited S-only generic
label multiplier is disabled in this mode. Both receive the same original
measured surface-distance likelihood. Neither V1's additional free likelihood
nor V2's absolute applicability multiplier Z is enabled.

## Measured envelope and identifiable dimensions

The old depth-difference normals can span depth discontinuities and their
voxel average preserves the first sign. They are not reliable orientation or
inside/outside measurements by themselves. OF estimates vertical planes from
the acquired 3-D point positions. Normals only impose a broad unsigned
consistency check, abs(horizontal dot plane normal) > .7 and abs(n_z) < .7.

Deterministic pair proposals use at most 64 evenly spaced input points; every
proposal is evaluated against all input points. At most six planes are
extracted. Inliers lie within **0.025 m** of a plane and require at least
**4 points**, **0.15 m tangential horizontal span**, and **0.25 m vertical span**.
Three TLS refinement iterations operate on the observed XY coordinates.
Unsupported refinement is discarded. The strongest plane establishes a
Manhattan orientation; other plane normals within **10 degrees** of its
orthogonal axes contribute their point-count-times-spatial-span weighted
fourth angular moment. This estimates orientation modulo 90 degrees, retaining
the inherited axis-specific dimension priors; it does not resolve length/width
identity from an unseen side.

Actual acquired keyframes orient an observed face. A point contributes a
camera-to-point direction only if it projects into a finite positive measured
depth within **0.10 m**. Average directions never use future frames. A plane is
an outer boundary only if all acquired points are on its interior side (within
0.025 m) and observed camera directions face outward. With only a single
coplanar face, camera directions choose the supported endpoint. An internal
plane with observed points on both sides is not an envelope boundary.

Two supported opposite outer planes determine that axis's width and center.
A single supported plane anchors the visible endpoint while the unobserved
extent uses max(inherited prior, observed full span). With neither endpoint,
the prior envelope is centered on the acquired span. The inherited horizontal
priors are 0.9 and 0.65 m. No 5/95-percentile shrinkage is interpreted as a
physical object boundary. Extent audit fields report a lower observed span
and no finite upper bound for hidden dimensions, rather than presenting a
point estimate as resolved object size.

A horizontal top candidate requires at least four spatially two-dimensional
points (both XY spans >=0.15 m), a measured normal consistent with a horizontal
surface, and acquired views from above. No acquired point may lie more than
0.025 m above it. Otherwise height is max(inherited height prior, observed max
height +0.08 m). Grounded-object z=0 remains an assumption. Even an apparent top
does not prove no unseen higher layer; vertical completeness remains false.

Full support only removes the model input height crop. It changes no TSDF,
sensor acquisition, map, route, policy coefficient, or stop rule. O_full
retains the original quantile-based envelope initializer, including its
1.5 m height cap, to isolate the new geometry estimator in OF.

## Preserved behavior and audit

Full box and shelf constructions, original measured-distance Gibbs scale and
logodds clipping, candidate routes, reward coefficients and stopping behavior
are preserved. B2's finite-positive strict-free classification, measured
surface veto, hypothetical visibility repair and measured TSDF occlusion are
preserved under the frozen repair-file SHA. Historical samples are zero
weighted on an unchanged full quadrature basis; no global point cap is used.
The clean no-consumption switch preserves verified-free rejection.

Each object records the input count and SHA256 of ordered, contiguous float64
XYZ and normals and original int64 labels. O_full and OF must have identical
input hashes; G/S in a stage must have identical template points and hypothesis
IDs. O must reproduce B4 pooling's positive-track points, weights and route
gain pointwise. Posterior values are rebuilt from current observations and
are not recursively accumulated as independent evidence.

Limitations: plane thresholds are fixed modelling choices, not a calibrated
sensor likelihood; min/max containment is sensitive to observed outliers;
Manhattan orientation and grounded rectangular furniture remain assumptions;
unseen extents and unseen shelf layout remain prior dominated. Repeated planes
and paired branch outcomes do not increase the independent scene count. Method
promotion requires fixed-replay evidence before any conditional new rollout.

Constants were selected from the existing observed quality representation and
synthetic geometry tests before outcome evaluation; no GT fitting or parameter
search was performed.
