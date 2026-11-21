# B9 semantic decision diagnosis and proposed controlled mechanism test

Status: completed read-only diagnosis; the scene below is a proposal, not an
executed experiment. No B8 source, weight, candidate generator, stopping rule or
paper was changed. All previous negative results remain.

## What is established from B8

The 10 completed runs contain 274 recorded planning calls. These are repeated
observations from two layouts, not 274 independent trials. Deduplicating exact
acquired input signatures within mode gives 52 fixed, 55 uncertainty and 55
mean-collapse states. Each mode includes 33 states with only one candidate.
Among states with multiple candidates, G/S scores differ in 9/19 fixed, 13/22
uncertainty and 13/22 mean-collapse states; their selected target never differs.

The semantic prior is connected correctly. The bank, acquired support and
likelihood are shared. Within each structure h, G and S have identical
conditional geometry distributions q(k|h,D). On all 66 frozen branches in
fixed and uncertainty modes, the saved per-point gains satisfy

    gain_S(r) - gain_G(r)
      = sum_objects [p_S(shelf)-p_G(shelf)]
          * [E(gain(r)|shelf)-E(gain(r)|box)].

Both the posterior change and the contrast between a route's two conditional
gains must therefore exist. Merely adding more category labels need not change
an action. The shared total is N + template gain, with N=C+Q for the unchanged
coverage and measured-quality terms.

For candidate j to displace G's winner w, its semantic relative advantage
must exceed G's original margin:

    delta(j)-delta(w) > score_G(w)-score_G(j),
    delta(r)=score_S(r)-score_G(r).

The maximum observed advantage/margin is 0.350609 for fixed, 0.233058 for
uncertainty and 0.239599 for mean-collapse. Thus none crosses the original
margin. At the strongest fixed case (right layout, action162), the original
margin is0.0234178, the semantic relative advantage is0.00821049, and the
remaining margin is0.0152073. This is an exact score decomposition, not a
claim that one common term universally dominates.

For neutral odds r=exp(L), changing to a shelf-favouring9:1 prior changes its
probability by8r/[(1+r)(1+9r)], at most0.5. A negative cue is symmetric. We
exchanged the two category priors algebraically on all six old prefixes in
both modes, keeping geometry, measured history and visibility fixed. No
winner changes. Even a conservative bound allowing each object's prior to
take either extreme independently stays below every winning margin: the
largest bound/margin is0.812206. This diagnostic does not constitute a new
policy trial, and these reused prefixes are not independent new evidence.

There are multiple causes, not one missing semantic connection:

* Single-candidate calls provide no action choice at all. They should not be
  counted as repeated opportunities where semantic choice failed.
* Geometry sometimes already agrees strongly with category, making its
  additional prior redundant. It sometimes disagrees strongly because the
  conditional shape is wrong. In fixed mode,25/102 deduplicated object states
  have >99% neutral structural confidence opposing the acquired category;
  uncertainty reduces this to3/108. These are repeated object observations.
* All three uncertainty errors occur at left24/54, before any accepted yaw or
  horizontal layer observation. The surface-only likelihood already has the
  wrong sign. The left24 closed object has only17 supporting points; its
  surface-only shelf/box log evidence ratio is+25.39 and combined ratio+23.12.
  The retained width, height and placement priors are not complete geometry.
  An open template can put an internal surface near a measured fragment more
  easily than the mispositioned box. A depth-CRPS-selected sigma does not
  independently calibrate this structural comparison.
* History reduces predicted area substantially, but does not eliminate it.
  At the strongest fixed case,8.0% of G's area remains after history; at the
  strongest uncertainty case,21.0% remains. Stored masks combine measured
  consumption and strict-free rejection; this audit cannot attribute the
  reduction to either mechanism alone.
* Several multiple-candidate states have equal template gains across routes
  (7 fixed,6 uncertainty). This can remove semantic route contrast even when
  structure probabilities differ. Candidate-level tables preserve their
  remaining area, common scores and gains; equal gains do not by themselves
  imply that the entire hidden object has been observed.

Agreement/disagreement above uses the controlled acquired category channel,
not an additional truth query. Pure priors0.1/0.9 and two separate supported
tracks give no evidence of cross-class label mixing in these cases. They do
not prove perfect instance association or template validity.

## Minimal next mechanism scene

Use the original unknown-exploration mapper, original candidate generator,
same motion interface and unchanged common scores. Do not inject two chosen
goals into the planner.

Build two geometrically symmetric, equal-length branches with room to pass
beside and behind their objects. Use an open shelf and closed cabinet with a
common outer envelope and common top board/top rim. A real low opaque screen
in front of each object hides the lower body from the initial view while
allowing the shared top board/rim to be measured. This makes hidden structure
different without making the first visible depth already classify it. The
screen is part of the physical simulator geometry and is observed normally.
It must not become a policy mask or evaluator-only occluder.

Create two layouts by swapping the open and closed objects between branches.
For each layout, retain three cue conditions: the correct category, neutral
0.5 category and deliberately exchanged category. G always uses neutral
structural prior with the same acquired instance support. Correct and wrong
cues test information value and failure respectively. Category is an ideal
controlled visible-instance input, as in the existing virtual interface;
identical RGB/depth plus different labels must not be called a natural image
detector evaluation.

Before any gain or outcome is inspected, freeze the scene dimensions,
occlusion placement, two permutations, starting pose, acquisition prefix and
all cue conditions. Geometry-only checks may verify collision-free side/back
access and byte-identical initial depth/mask/scan across paired hidden
structures. If equality fails, retain the construction failure; do not adjust
the scene after seeing S win or lose. Expose the full source family and its
departure from natural furniture distribution.

Then acquire the common prefix, let the original policy generate every
candidate, freeze that candidate set, and execute each candidate's original
outbound route from the same prefix. Retain all actual branch observations,
failures, new source-supported object surface, global surface, coverage,
action cost and path length. Report every condition and both permutations,
not just a representative successful screenshot. Use the exact margin and
posterior decomposition above to connect a target change to semantic input.

Primary mechanism questions are whether category changes posterior mass,
whether candidates have distinct conditional gains, whether these changes
alter the original choice, and whether the selected branch realizes more
new correct surface at its measured cost. A changed winner alone is not a
benefit. If no distinguishing candidates are generated or choices remain
equal, report that failure and stop this mechanism attempt. Do not introduce
hand-selected goals or tune weights to create a positive result.

This is a controlled local mechanism test. It cannot establish complete
unknown-environment exploration efficiency. Only a later frozen full-policy
comparison could support that separate claim.

## Optional structural calibration, not yet a deployed change

The proposed factorization can calibrate structural confidence independently:

    p_G,T(shelf|D)=logistic(L/T),
    p_S,T(shelf|D)=logistic(L/T+logit(category_prior)),
    q_T(h,k|D)=p_T(h|D)*q_B8(k|h,D).

This is a modular calibrated mixture, not the original single-likelihood
Bayesian update. T=1 must exactly recover B8. A positive T cannot reverse the
neutral class ordering or fix its geometry; it can only change confidence.
The class-conditional geometry remains unchanged, but its mixture weights
and therefore marginal depth can change. Its depth CRPS must be reevaluated.

Freeze a finite temperature grid and tie rule before new results. Select T
using equal-object/equal-condition neutral class NLL on the new calibration
objects, then evaluate once on disjoint new test objects. Preserve fallback
cases in the full denominator and report NLL, Brier score, accuracy, high-
confidence errors and depth CRPS. Do not choose T using route outcomes. A
test improvement would justify a calibrated-confidence claim, not a claim
that template geometry, semantic detection or exploration is solved.

## Evidence files

`summary.json` and `rollout_*` retain all ten original runs.
`fixed_branch_details.json` and `fixed_object_candidates.csv` retain per-point
gain decompositions. `prior_exchange.json` contains finite-prior bounds and
explicitly counterfactual swapped-cue results. `evidence_terms.json` contains
surface/free diagnostic evidences. `input_receipt.json` verifies unchanged B8
inputs. All scripts and outputs are under this new B9 semantic directory.
