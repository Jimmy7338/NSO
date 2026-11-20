# Independent structural-temperature evaluation

Freeze the evaluator and the already declared structural protocol before opening
new calibration/test truth. B9's 48 newly generated objects retain their fixed
24 calibration / 24 test split. All four conditions of each object remain
dependent measurements; aggregate conditions equally, then objects equally.

Compute each structure's log evidence with a separate stable log-sum-exp of
`log(q0)-E/(2*.01^2)`. Keep the conditional geometry probabilities unchanged.
For neutral G, the class logit is `(logZ_shelf-logZ_box)/T`. The controlled
correct-category S diagnostic adds `+log(9)` for an open object and `-log(9)`
for a closed object **after** temperature scaling. Correct-category S uses
private class truth only in this evaluator to define an oracle category-prior
condition; it does not evaluate a natural detector or change actor observations.

Select T using only neutral calibration NLL on the protocol's ten-value grid.
Use `logaddexp(0,(1-2*y)*logit)` without probability clipping. Exact objective
ties choose the temperature closest to one in log2 distance, then the larger
temperature. Check the independent scalar implementation against the frozen
method and retain the complete selection curve. Save and hash
`selected_temperature.json` before opening test truth. Test is evaluated once
with no retuning; primary comparison is neutral equal-object NLL against T=1.

Report G and controlled-correct-category S at T=1 and selected T: natural-log
NLL, binary Brier `(p_shelf-y)^2`, accuracy (exact logit-zero ties count half),
counts of confidently wrong predictions with max class probability strictly
greater than .9, and marginal depth CRPS/median error. Depth error uses the true
mesh extent projected onto the unchanged inferred depth axis, with axis-error
diagnostics retained. This is a newly reweighted PMF, distinct from the B9
diagnostic conformal interval. No interval guarantee is transferred to it.

Missing, ambiguous/multiple or noncalibratable/out-of-support banks use the
declared B8 evaluator fallback, never a truth-best track: world-y target axis,
uniform density on [.2,1.4] using its fixed quadrature, and zero log-evidence
difference. G class probability is .5. Controlled S then retains its supplied
correct-category .9 prior. These fallback outputs are independent of T and
remain in all denominators, with reasons preserved. Hash mismatches remain
integrity failures rather than statistical fallback. Directly observed
two-face extents retain B8's original calibratable-status semantics.

The scalar class calibration cannot correct the sign of neutral evidence,
wrong geometry or instance association. Conditional geometry is fixed while
the marginal depth distribution can change through the structure mixture.
Scalar test improvement authorizes only the predeclared fixed-branch replay;
it does not establish G/S exploration efficiency or generalization to physical
sensor data. This evaluator performs no policy execution.
