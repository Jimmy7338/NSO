# Offline CSM correspondence audit

This audit reuses the 096/098 references, target099, previous pose098, first-guess construction, original parameters and unchanged installed CSM/matcher libraries from `../instrumented_csm_candidate/STATE_FIXTURE.md`. No points, thresholds or parameters were changed. Both results reproduce the earlier fixture: reference096 yielded 50.105966 mm planar step and exact recorded target XY, while reference098 yielded 10.173318 mm. This identifies reference/initial-state dependence, not the true physical surface positions or a validated fix.

The library's retained valid correspondence arrays after `sm_icp` contain 1,652 and 1,630 entries respectively, matching reported `nvalid`. For each retained correspondence, the script transforms its target point using the returned final pose and measures signed perpendicular distance to the line through associated reference points j1/j2. Reference096 has 44.63 mm RMS and reference098 7.52 mm RMS. About 75.5% of 096's squared residual lies in measured target beam angles +45° to +180°, and about 90.2% in measured target ranges 2–6 m. Angles are sensor-frame beam angles; these sectors have not been identified with physical surfaces. Among 1,442 common target rays, +90° to +135° (81 rays) has RMS 94.9/10.1 mm (096/098), and +135° to +180° (126 rays) 68.9/6.2 mm. The two solutions use different references and different first guesses; neither residual is truth, and one cannot attribute the discrepancy uniquely to range, surface, sensor error or optimizer initialization.

`summary.json` has sector/range breakdowns and an unweighted local geometric JᵀJ estimate. Its eigenvalue ratios 13.66/13.90 depend on parameter units and coordinate origin and do not establish or exclude degeneracy. This matrix is calculated from retained point-to-line normals, not obtained from the solver or an uncertainty estimate. Associations may represent the last iteration's linearization rather than a fresh correspondence search at the returned pose; no extra association search is performed. Our unweighted squared residual and library-reported `error` use different definitions and should not be equated. No correspondence labels identify actual objects.

To reproduce on the originating machine, first run the sibling `instrumented_csm_candidate/STATE_FIXTURE.md` procedure with `FIXTURE_OUTPUT_DIR` exported. From `integrations/forward_controller_20261107`, use a separate output directory and the same native matcher compile flags:

```bash
export AUDIT_OUTPUT_DIR="$(mktemp -d)"
g++ -std=c++14 -O2 -fno-access-control correspondence_audit/correspondence_fixture.cpp -o "$AUDIT_OUTPUT_DIR/correspondence_fixture" -L/opt/ros/noetic/lib -llaser_scan_matcher $(pkg-config --cflags --libs roscpp tf pcl_ros gsl)
/usr/bin/python3 -B correspondence_audit/run.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 correspondence_audit/summarize.py
```

The runner refuses occupied localhost port11401, uses its own temporary master and does not connect to live11382. Publication-copy paths write generated files only in the two explicitly supplied output directories. The derived correspondence table is deliberately not bundled; it is reconstructed by the above commands from the already-published probe06 scan ring. The portable copy reproduced the full archived summary on the originating machine, not a clean-room installation. Each generated correspondence line holds reference index, target ray index, reference j1/j2, type, target beam angle/range, reference range, signed residual, nearest-point distance, transformed target XY, and unit normal.

Installed library SHA-256: `liblaser_scan_matcher.so` 0f3e2816b3f41e497793a129a61d046e33d85e09eca1b634660ebee07426afe5; `libcsm.so` fc446d14e987930d9874426c9d581567449de663475a1011012c68828b8a1994. No live runtime was modified or sensor/motion command issued.
