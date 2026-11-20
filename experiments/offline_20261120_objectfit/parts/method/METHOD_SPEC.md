# B6: qualified outer-plane orientation and observed absolute layer heights

This explicitly revised template model starts from B5 OF's full acquired
object support. B5, its negative outcomes and the O-only policy experiments
remain frozen. No candidate, coverage/quality coefficient, stop rule, learned
reachability model or robot control is changed. No source mesh, true object
dimensions, hidden categories, evaluator distances or future frames enter
these routines.

## Independent switches and stages

`configure(object_support=True, full_observed_support=True,
geometry_fit=True, outer_orientation=False, observed_layers=False)` returns
the model class. The new switches are independent:

| Stage | outer_orientation | observed_layers |
|---|---:|---:|
| outer_orientation | true | false |
| observed_layers | false | true |
| outer_and_layers | true | true |

Both switches disabled must reproduce B5 OF's predicted points, weights,
hypothesis IDs and route gains exactly. All three revisions use the same
ordered acquired object support and per-object point/normal/label hashes.
G/S share object support and geometry; G has a neutral class prior, rather
than being a no-semantic frontend. Neither the added free-space likelihood
nor the absolute applicability multiplier Z is enabled.

## A: qualify an outside face before using its orientation

B5's vertical-plane proposals, 2.5 cm inlier residual, four-point minimum,
15 cm horizontal span and 25 cm vertical span remain unchanged. Each candidate
is first evaluated in its own world-space normal coordinates. All observed
object points must lie on one side within the 2.5 cm tolerance, and the mean
acquired view direction of the supporting points must face outward with
normal component magnitude greater than 0.1. View directions use B5's actual
RGB-D depth agreement; averaged normal sign cannot establish the outside.

Only these qualified planes participate in the existing strongest-plane and
Manhattan orientation estimate. If none qualifies, yaw remains the explicit
zero-radian prior and `orientation_observed=false`. Rejected candidates remain
in the orientation audit. This changes the order of evidence use: an internal
or unsupported plane cannot rotate the box/leg axes before later being rejected
as an envelope endpoint.

Outside qualification still does not prove that a sparsely sampled plane is a
continuous physical face; a valid-looking two-column fit remains a limitation.

This does not resolve an unseen side. The original horizontal size priors,
partial-depth uncertainty, measured-envelope endpoint rules and observed
centering behavior remain. Min/max containment can still reject a valid face
because of an acquired outlier; zero yaw is a prior, not a recovered orientation.

## B: absolute slab priors with measured horizontal-face constraints

The original three layer priors are instantiated at absolute heights
`[0.15, 0.50, 0.85] * H_prior`, with inherited 0.08 m thickness. `H_prior` is
the original initializer's height **before** the new envelope fit. Increasing
the estimated box/leg height therefore does not stretch every internal slab.

Horizontal candidates use only acquired points with abs(normal_z)>0.7 and a
depth-validated view direction whose vertical component magnitude exceeds
0.1. Upward and downward viewing directions are grouped separately. A candidate
height is the median of a 2.5 cm z band. It must retain at least four points,
span at least 15 cm in each horizontal coordinate and have minimum XY covariance
eigenvalue at least (2.5 cm)^2. The last condition prevents a diagonal line with
large coordinate bounds from impersonating a two-dimensional patch. Candidates
are extracted deterministically by supported point count times XY covariance
area; their point indices are consumed once.

For a face observed from above, the measured height anchors the slab's top and
the unobserved thickness extends 0.08 m downward. For a face observed from below,
it anchors the bottom and thickness extends upward. When an observed bottom
and a higher observed top have overlapping 0.08 m proposed intervals, the
closest available pair constrains the interval directly between those two
heights. This is a declared local association heuristic; it does not establish
hidden component identity. An unpaired face retains the inherited thickness.
Each partial horizontal patch extends across the current assumed XY extent;
the observations do not establish those unobserved lateral endpoints.

An observed component replaces an original prior slab only when their vertical
intervals overlap. Non-overlapping observed components are added, and every
unconstrained original prior slab remains. Overlapping component intervals are
united before constructing the mesh and quadrature, avoiding duplicated slab
volume and internal interval faces. Each final interval records contributing
observed-plane IDs and residual prior IDs. Same-side stepped faces can still
be simplified by this union; it is a restricted slab model, not a free-form
surface reconstruction. The measured TSDF retains the original observations.

No unobserved slab is deleted merely because it lies above the current height
estimate. `slab_bounds_exceed_envelope` explicitly records such cases. The
reported H and XY envelope describe the box and leg estimate; they are not
asserted to bound every open-structure slab prior. This avoids silently turning
a partial apparent top into proof that no higher layer exists.

## Preserved visibility, posterior and remaining limitations

Original box/leg/backboard construction and class-conditioned versus neutral
prior remain, apart from the explicitly changed orientation and slab geometry.
Both methods use the same measured surface-distance likelihood and historical
zero-weight masks on the complete quadrature basis. B2's finite-depth free
evidence, measured-surface veto and hypothetical visibility repair keep their
frozen implementation. The clean history ablation continues rejecting verified
free space while disabling measured-surface reward consumption only.

The observed object depth may still be incomplete; the 0.65 m depth prior and
other unresolved extents remain. Misplaced legs and unsupported backboards were
also present in B5's negative evaluation. This revision does not attribute all
error to slab height, claim those residual problems are solved, or use true
dimensions to remove them. Added components change hypothetical surface area
through declared geometry, not through a new reward coefficient. Complete
unknown-space exploration and G/S efficiency require separate policy evidence.

All numeric evidence thresholds are inherited from B5 or the original 0.08 m
slab model. No GT-based parameter search is performed. Unit tests cover the
specific structural failures and source parity; performance is assessed only
after freezing through the same six acquired prefixes and 66 reused candidate
outcomes. These are dependent development states, not new independent scenes.
