# B8 conditional depth and placement particles

This new module preserves the frozen B6 AB object support, qualified outside
orientation, width, box/leg height, absolute layers and primitive construction.
It changes only the distribution over local-axis-1 depth and, without an
observed endpoint, the allocation of unseen extent around acquired support.
It is not a posterior over complete shape, rotation, layer number or pose.
No source mesh, hidden dimensions, evaluation labels or future frame is read.

## Common acquired basis

G and S use the same full acquired quality rows and observed-instance tracks.
The B6 conditional model is constructed with the neutral category prior; its
old surface likelihood may be computed by that constructor but is never used
in the B8 posterior or multiplied into B8 weights. Existing B6 object eligibility
is inherited, with unmodeled observed tracks reported explicitly. Unknown-track
measured geometry and coverage remain outside this furniture-template term.

The shared bank is cached by actual map shape, quality keys, instance
memberships, points, normals, labels, view bits, and all retained keyframe poses,
intrinsics and depths. G/S reuse the bank and ray scenes. Changes to observations
invalidate the bank; no recursive posterior multiplication is performed.

## Depth and position prior

The class-neutral prior is uniform over depth [0.20, 1.40] m, conditioned on the
actual local-axis-1 span of the acquired object points. The shared deterministic
nodes are 0.20, 0.30, ..., 1.40 m, plus the measured lower boundary when inside
the prior interval. Node masses are their clipped Voronoi cell lengths divided
by the conditioned interval length, not equal masses for irregular nodes.

With a credible low or high outside face, that coordinate stays fixed. With
neither, allocation a has nodes [0, 0.5, 1] and masses [0.25, 0.5, 0.25]:

    lower = measured_min - a * (depth - measured_span).

Thus a particle contains the acquired local depth interval. This is a finite
placement prior, not knowledge of an unseen endpoint. Credible opposite faces
use their measured distance directly. Inconsistent ordered anchors fall back
to the declared unanchored construction and are flagged. When the measured
span exceeds 1.40 m, a boundary particle preserves the object and records
out_of_support / distribution_calibratable=false. That numerical fallback is
not calibrated certainty and must remain an evaluation failure/fallback case.

The fixed-depth ablation uses max(0.65 m, acquired span), or the credible
opposite-face distance, while retaining the same unanchored allocation nodes
and their observation update. It does not restore the old guessed center.

## Single observation update

For particle k and structure h in [box, shelf], the error is the unchanged B7
mean acquired-point-to-mesh clipped squared distance plus mean strict-free
first-hit depth-gap squared error. Both are in m² and clipped at 0.15 m before
squaring. A single maximum ROI encloses all depth/position alternatives, both
structures, both fixed/uncertainty modes and every retained slab height, even
slabs outside the box/leg height estimate. All alternatives use the same finite,
positive acquired depth rays on the inherited stride-3 pixel lattice.

Rays have camera z=1, so the intersection parameter is axial RGB-D depth.
A model hit behind a closer actual hit is occluded, not free evidence. A hit
in front of actual depth by more than the inherited 0.10 m history tolerance
must also pass B2's cross-history strict-free classifier. Any acquired measured
surface support vetoes conflicting free evidence. Exact duplicate acquired rays
are removed. Invalid or missing depths provide no free evidence. Mean-loss
normalization and the B7 discrepancy definitions are unchanged.

    w_m(h,k) ∝ pi_m(h) q0(k) exp[-E(h,k)/(2 sigma²)].

Normalize once over both structures and all geometric particles for each
object. G uses pi=[0.5,0.5]. S uses the inherited visible-label category prior:
current label 2=open gives shelf prior 0.1+0.8*open_vote; unrecognized labels
leave the prior neutral. G and S have the same geometry and error bank but may
have different marginal depth weights because their class priors differ.
The old 6 cm surface likelihood is not applied again. Nonfinite energies or
nonpositive/nonfinite sigma raise errors rather than producing NaN weights.

Sigma is selected externally from [0.01,0.015,0.025,0.04,0.06,0.09,0.14,0.22] m
using the separately frozen independent calibration protocol. Inference without
sigma exports energies and priors only; it neither selects a scale nor binds a
policy. These are generalized loss-based weights, not an exact sensor posterior.

## Expected area and the five declared arms

Each particle retains the original B6 parts and 0.22 m face-area quadrature.
No boolean-union removal, overlap correction, point cap, candidate filtering,
score coefficient or stopping change is introduced. Existing primitive overlap
and unconfirmed layer/backboard assumptions remain limitations.

A sample weight equals its inherited area times its particle/structure mass.
B2 history masks zero already observed or verified-free samples on the same
complete basis. Each route takes per-sample maximum visibility before summing;
particles are mutually exclusive alternatives, not multiple physical objects.
Duplicating a particle with half its original prior mass leaves expected area
unchanged. Repeated route viewpoints cannot add area. Zero-weight samples are
skipped in B2 ray queries and then scattered back into the unchanged full basis;
the resulting vector is exactly equal to the unoptimized query. B2 trace counts
are retained and explicitly sum gain calls rather than unique measurements.

Declared interfaces support G_fixed, S_fixed, G_uncertainty, S_uncertainty and
S_mean-collapse. Mean-collapse first computes the complete S joint posterior,
then for each structure uses its conditional expected depth and expected lower
coordinate, retaining that structure's marginal probability. It does not
recompute likelihoods or class weights at the collapsed geometry.

## API and output

- build_particle_bank(mapper, neutral_model=None) returns the shared bank. A
  supplied neutral model must be frozen B6 AB with full acquired support.
- bank.distribution(sigma_m=None, semantic=False, fine_categories=True,
  use_likelihood=True, extent_mode='uncertainty'|'fixed') returns sensor-only JSON.
- infer_depth_distribution(mapper, neutral_model=None, frozen_scale_config=None)
  is the calibration entry point; the optional configuration contains sigma_m.
- configure(sigma_m=..., extent_mode='fixed'|'uncertainty'|'mean_collapse') returns
  a FullTemplateModel class retaining points, weights, hypothesis_ids,
  hypothesis_rays and gain interfaces. Policy binding requires an explicit
  calibrated scale. measured_surface_consumption=false retains the common basis;
  fit_update=false means prior-only for the entire B8 observation likelihood.

Each track exports particles (depth_m, allocation, lower_m, upper_m, prior_mass),
prior_mass[K], energy_m2[K,2], separate surface/free energies, shape prior,
conditional geometry, world depth axis, actual support hashes and shared-ray
hashes. With sigma it additionally exports joint_probability_mass[K,2], unique
depth_grid_m, marginal probability_mass and structure masses. Missing templates
remain in unmodeled_tracks with an explicitly declared world-y fallback axis;
no source-aware axis choice or silent target exclusion is allowed.

The old six prefixes remain development data. This freeze precedes independent
geometry calibration, all new candidate rescoring, and all new closed-loop
runs. Calibration failures, finite prior support, unknown orientation/width,
conditional parts, ideal instance masks, and the limited synthetic calibration
split must remain visible in subsequent claims.
