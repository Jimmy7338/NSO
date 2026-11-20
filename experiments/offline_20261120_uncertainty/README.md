# B8 conditional dimension uncertainty — complete development evidence

B8 replaces an unresolved fixed depth with a class-neutral depth/placement
particle bank. G/S share acquired instance support, particle geometry and
surface/free-space losses; only their structural class prior differs. Marginal
depth beliefs can consequently differ. Width, orientation, height and parts
remain conditional B6 AB estimates. This is not a full shape posterior or a
new navigation algorithm. DESIGN.md, ARCHITECTURE_DELTA.json and method/ state
all changes; earlier methods and their negative findings remain unchanged.

Independent procedural calibration contains 24 geometry units split 12/6/6,
with four observation conditions per unit (96 dependent condition cases), plus
two identical-sensor counterfactual fixtures. The shared sigma=0.01 m was chosen
by training-only CRPS and frozen. Every case, energy table, fallback, training
score, validation and held-out result is retained. The declared informative-view
CRPS progression criterion passed; 95% intervals covered only 9/12 validation
and 8/12 held-out informative cases. Nominal probability calibration is therefore
not established. Validation/held-out outcomes did not trigger retuning.

The small evaluator_private JSON archive contains the independently generated
procedural truth/part specifications needed to reproduce calibration. It is
explicitly evaluator/generator-only input. Isolated inference opened only
sanitized sensor manifests and acquired measurements; open-file receipts and
counterfactual checks preserve that boundary. Procedural geometry generation
code is included; third-party furniture assets are not included.

Five arms (G_fixed, S_fixed, G_uncertainty, S_uncertainty, S_mean_collapse) are
scored on six previously inspected prefixes, reusing exactly 66 old acquired
candidate outcomes. The 330 arm/candidate rows are dependent measurements, not
330 acquisitions or independent environments. All matrices, object/particle
IDs, input hashes and independent full evaluations are retained. Mean-collapse
uses the S posterior's per-structure means and masses without refitting. Fixed
controls retain the same position integration and calibrated loss, and must not
be relabeled as historical O. All ten new policy runs and independent evaluations
are complete: both furniture permutations attained full correct-free coverage
with zero simulated collisions. In the left layout, extent updating first reached
full coverage at 88 instead of 97 actions and raised final global F1 by 0.3773
percentage points, while shelf recall decreased and cabinet recall increased.
The right layout tied exactly. G/S trajectories and native meshes tied within
both matched representations, and S mean-collapse reproduced full uncertainty.
These results do not establish semantic planning benefit or superiority of
planning with the full distribution. Timing records preserve the substantial
particle evaluation cost and shared-compute caveat. There is one floorplan
family, two inspected furniture permutations and no new random repetitions.
Frozen readiness receipts retain their historical pre-run status; final run
status and claim boundaries are in EVIDENCE_SCOPE.json and independent results.

Raw RGB-D/owner arrays, native/source meshes and large TSDF/GT distance arrays
remain local with hash manifests. Earlier public evidence packages contain
compact branch execution receipts and hashes, not the native or licensed source
meshes. Regenerating the old exploration evaluation requires those available
licensed assets, previous frozen source packages and the pinned numerical
runtime. This payload is not a standalone simulator. Reused branch references
do not imply that unpublished meshes are downloadable from GitHub.

Editable scientific SVG/PDF, plotting source and CSV inputs are included; figures
are computed evidence, not generated experimental imagery. All failures, ties
and negative effects are retained. This builder performs no robot motion, no
new fitting and no remote upload.
