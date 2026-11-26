# Prediction-only reconstruction acquisition operator

`nso/acquisition_forecast.py` provides an optional, offline experimental acquisition model. The default exploration policy is unchanged. The current physical robot remains paused.

Pass all actually acquired RGB-D depth/pose/intrinsic frames as `DepthFrame(..., provenance="acquired")`. Render candidate future depth from **one mutually exclusive model hypothesis**, optionally combined with the current acquired scene, and label it `predicted_hypothesis`. Source-reference geometry and actual future observations must never be used to build these frames. Candidate generation, probability fitting and execution are outside this module.

`forecast(past_frames, future_depth_frames, template_surface_points, template_area_weights)` creates two fresh local TSDF volumes. It integrates acquired past into the first, and acquired past plus synthetic future into the second. It returns the hypothesis surface area that crosses from farther than5cm to within5cm of the predicted reconstruction, alongside lost and net area. Template points/area weights must come from the model, independent of evaluation reference samples. Preserve existing verified-free rejection. Do not also consume these points with the old10cm axial measured-surface band: the new before/after metric supplies measurement consumption.

The operator does not accept or modify the actual measured mapper. Its default sensor helper uses a strict0.15m near boundary; the fitted controlled runtime uses an explicitly tested inclusive0.15m rendering adapter. Render depth according to the actual sensor contract. TSDF scales default to6cm voxels/12cm truncation; changing them requires declaring the evaluated configuration.

The known-geometry twelve-object/36-route test reduced same-basis gross-area MAE from0.153218 to0.014258m², with zero revisit gain. Here the hypothesis is exactly the object, so this supports acquisition-operator consistency. In a fitted left-layout state, the conditional-mean approximation improves ranking but does not improve absolute area error. It has not been promoted to the default planner or validated through complete C-policy rollouts. Full reports, all failures and independent counts are in `experiments/offline_20261120_gain_alignment` on the research evidence branch.

Example:

```python
from nso.acquisition_forecast import DepthFrame, forecast
# observed_frames: acquired past; candidate_frames: hypothesis-rendered future
# points/areas: hypothesis-owned surface quadrature, not evaluation truth
result = forecast(observed_frames, candidate_frames, points, areas)
expected_new_area = result["receipt"]["predicted_gross_new_m2"]
```

Apply class/geometry mixture masses outside this operator. The current fitted probe uses a declared conditional-mean geometry approximation per structural hypothesis; this is distinct from integrating every geometric particle. Its uncertainty and unknown-occlusion limits remain open.
