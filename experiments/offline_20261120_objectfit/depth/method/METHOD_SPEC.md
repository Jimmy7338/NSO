# B7: acquired-data depth profile, diagnostic only

`profile_model(mapper, frozen_B6_AB_G_model)` returns JSON without mutating the
mapper/model or executing actions. G/S share the same geometry, and no S prior
or reward weights are read. Only local axis 1 with exactly one observed outer
endpoint is profiled. No-anchor, two-anchor and degenerate-domain objects retain
their current dimensions. Width, yaw, box/leg height, absolute slab heights and
backboard orientation remain fixed; slab/leg/backboard thickness does not scale
with depth.

For each eligible object, the grid has 21 equally spaced depths from its acquired
span to max(2*0.65 m, span), plus the original depth explicitly. The measured
endpoint is fixed and only the other endpoint moves. The maximum-depth search
box contains all B6 components, including slabs outside the box/leg height.
All acquired finite positive RGB-D pixels on a fixed stride-3 lattice whose
rays intersect this box are used. Exact pose/intrinsic/pixel/depth duplicates
are removed. This **same ray set** is used for every depth and both hypotheses.
Ray directions retain camera z=1, so ray-casting t and measured RGB-D depth have
the same camera-depth units. Hashes and counts record the selected ray basis.

For hypothesis h and candidate depth d:

```
E_surface(h,d) = mean over fixed acquired object points
                 min(Euclidean distance to h(d), 0.15 m)^2
E_free(h,d) = mean over the fixed acquired ray set
              I(strict-free first hit) * min(observed_depth - predicted_t, 0.15 m)^2
L(d) = 0.5 exp(-(E_surface(box,d)+E_free(box,d))/(2*0.06^2))
     + 0.5 exp(-(E_surface(shelf,d)+E_free(shelf,d))/(2*0.06^2))
J(d) = -log L(d)
```

A first hit can be a free-space contradiction only when t < measured depth
minus the inherited 0.10 m threshold **and** the frozen B2 history classifier
reports strict free. Cross-history measured-surface evidence vetoes a free
classification. An actual hit closer than the hypothesis is occlusion, not
counterevidence. Invalid depth, no hypothetical hit, or unknown/occluded space
contributes no free penalty. The neutral mixture marginalizes the mutually
exclusive box/shelf hypotheses; it does not require both to explain every ray.
These mean squared discrepancies are heuristic model evidence, not calibrated
sensor noise or a statistical likelihood.

The declared plausible grid band is J <= min(J)+1. It is not a confidence
interval. If the prior remains inside, it is retained. A band reaching the
artificial upper search limit, disconnected modes, or no reduction of measured
free contradiction in either hypothesis also retains the prior. Otherwise the
band point closest to the prior is reported as a proposed point estimate. A
minimum at the actual measured-span lower bound is allowed when an internal
upper crossing excludes the prior. The output retains all plausible grid
values, the crossing bracket and unresolved endpoints; it never identifies a
grid boundary as exact hidden object depth. No band width is adjusted to obtain
an update.

All primitive meshes use B6's acquired parameters. No source mesh, true object
size, future observation or evaluator value enters profiling. Source-parity,
flat-front-view, valid-free/veto, lower/upper-bound and zero-evidence tests precede
the six fixed-prefix profile replay. No new candidate rollout or robot motion
is part of this diagnostic.
