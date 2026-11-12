# Post-hoc common-support reliability diagnostic

This diagnostic uses all 145 intersecting bootstrap support cells from the two
actual C/S episodes, at the native 0.15 m support resolution and their identical
planning transform. It was designed after inspecting the original results;
`result/protocol.json` records the rule before this additional calculation.
It is not a preregistered or causal comparison.

The original mapper was evaluated without the policy, ROS or detector. Its
original own-support counts, key hashes and quality values matched all 41 C and
33 S recorded observations to 1e-12. The native quantity is
`Q = I / (I + 0.25 + 100 R)`, where `I` records the best incidence/range-derived
information and `R` is a running squared point-to-plane residual. It is an
internal reliability proxy, not TSDF accuracy or measured surface completeness.

| Actual observations evaluated | Initial Q | Endpoint Q | Change | Information component | Residual component |
|---|---:|---:|---:|---:|---:|
| C through action 32 | 0.279212 | 0.253034 | -0.026179 | +0.010359 | -0.036538 |
| S through action 32 (failed episode endpoint) | 0.265727 | 0.335260 | +0.069532 | +0.165932 | -0.096400 |
| C through action 40 (actual returned endpoint) | 0.279212 | 0.215996 | -0.063216 | +0.038661 | -0.101877 |

The exact decomposition holds initial residual fixed while changing information,
then changes residual to its measured endpoint value. Thus the decrease in C's
proxy is not a decrease in its best recorded information: the residual penalty
outweighs the information improvement. S's recorded path has a larger information
increase on these cells, alongside a residual penalty. Neither observation
establishes the cause of the residual or a semantic/policy benefit.

Equal cell keys do not establish identical physical surface correspondence:
initial representative-point separation is 0.0800 m median and 0.1980 m maximum.
Initial sensor states differ, the camera extrinsics remain unverified, and this
is one ordered pair in one scene. S failed its original footprint check at 32
actions and did not return; the action-32 view does not remove that outcome or
replace C's actual 40-action endpoint. No run, support subset, scoring weight or
robot decision was selected or changed to improve the result.

Reproduce with the preserved canonical package and matching source snapshot:

```bash
python analyze.py --package /absolute/path/to/joint_coverage_pair_20261107 \
  --code /absolute/path/to/preserved/code --output /absolute/path/to/new-output
```

The code directory must contain `nso_bridge/` and `joint_vendor/`; source and
input hashes are checked against each arm's recorded manifests. Outputs are
small JSON/CSV files; existing results are not overwritten.
