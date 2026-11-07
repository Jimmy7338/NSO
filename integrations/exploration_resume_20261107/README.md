# Completed S40 physical exploration: derived-input evidence

One physical run of `exploration_joint_v1` / `v10_3_1` with real YOLOE-derived labels completed 40 paid actions and 40 verified arrivals, traveled 7.3935 m by CAN, and ended at the declared initial grid/heading anchor when the budget was exhausted. The ROS live navigation map already included prior observations at initialization; this is not a fresh empty-map trial. The observed-area proxy increased from 72.08 to 128.28 m². The fixed old-surface quality proxy **decreased** from 0.213760 to 0.199107; this is not ground-truth reconstruction accuracy. The completed run records 12 distinct global-choice events (plan numbers 1–12), with equal same-candidate S and G scores. The 41 next-action returns include repeated use of earlier selections: the first two plans have 5 and 2 candidates, respectively, and the remaining ten have one candidate each. This is not a paired physical comparison, proof of semantic benefit or an efficiency result.

The original camera extrinsics remain unverified after relocation. Actual camera and laser transforms are retained as observed inputs, without an accuracy claim. Execution used original default-keyframe CSM pose with a fixed SE(2) coordinate-continuity alignment and CAN control-velocity adaptation; ordinary ROS navigation and hardware protections were retained. A logical grid/heading return and ROS acceptance are not independent physical localization accuracy. The existing live map already included earlier data; the paid run is not an independent scene. YOLOE labels are predictions, not verified object identities or class accuracy. The final receipt reports `native_driver_fault:8` after the last goal during external-guard cleanup; this record does not establish an independent chassis hardware fault.

`frames/` contains losslessly selected canonical 320×180 depth, RGB and cached semantic arrays, observed navigation grids, measured transforms, base poses, scans, timestamps and metadata for all 41 accepted observations. Each source packet's name, byte size and SHA-256 are in `frames_manifest.json`; all original 41 source packets total 424,835,303 bytes and remain locally at the report path, not in this package. Full-resolution RGB-D, ROS serialized messages, detector weights and full instance masks are omitted. Semantic receipts preserve provenance and source hashes, but this package cannot independently rerun or verify the original detector on raw inputs. Its arrays are exact relative to the local source and validated cached semantics at conversion time. These are derived sensor inputs, not a complete raw-sensor dataset.

`source/` includes the historical configuration, source manifest, original events and result/summary, plus the read-only completed summary. `decisions.jsonl` retains the actual recorded requests and results, including 12 distinct recorded planning selections; source paths are historical provenance and need not resolve elsewhere. `code/` is a snapshot of the Bridge and vendored joint source; 110 entries in the recorded source manifest matched the local workspace at packaging time. Dependencies, original policy variants and MIT license/source provenance are retained. The cached-semantic downstream S policy replay has been run: the original engine processes all 41 canonical frames and each original observed grid in recorded order, without a prefix or detector. It recomputed all 40 actions and final closure, 12 global choices, candidate paths and scores, feedback and gain histories against the archived run. This is a downstream replay, not detector reproduction or independent physical validation. Per-frame observed grids are preserved; prefix initialization and live observation paths can handle historical grids differently.

`semantic_aperture_audit.json` and `audit.py` reproduce two recorded selection-state scalar checks, without running detection or planning. At selection 4, the group with 78 class-3 points has no candidate passing the hypothesized rear-plane depth gate. At selection 85 the same measured geometry again fails that gate; a second marked group has candidate states passing depth, but all sampled rear-plane points fail horizontal FOV. The group geometry and candidate paths are logged in the original selection audit. This is a model/candidate-pool result, not proof of physical inaccessibility, object identity, grouping correctness or a decision effect. Group IDs can change between frames and do not establish tracking identity.

In a Python environment with NumPy (tested Python 3.12.15, NumPy 1.26.4), run from this directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B verify.py
```

This verifies every packed file and canonical array against its manifest, binds the cached semantic receipt, recomputes the two recorded scalar audits, and counts actual global-choice events separately from repeated next-action returns. It does not load ROS, run a detector or planner, or prove the source experiment correct. `convert_inputs.py --report <local-original-report> --output <new-empty-directory>` documents the one-time lossless conversion and checks original packet/mask hashes when those local inputs are available; do not point it at an irreplaceable directory.

For an independent functional cross-check of the scalar geometry against the preserved original `camera_pose` and `aperture_support` implementations, use an environment with NumPy, SciPy and Open3D (tested 1.26.4, 1.11.4 and 0.19.0):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B original_function_check.py --decisions decisions.jsonl --audit semantic_aperture_audit.json --vendor code/joint_vendor
```

This checks 395 and 44 recorded state/asset combinations and 25 and 8 candidate/asset factors against the original functions and logged factors, without detection, mapping or planning.

The editable `figure/plot_episode.py` reproduces the recorded trajectory and measured-area/quality plots from the bundled inputs, without the local original RGB-D packets. With Matplotlib and NumPy installed, run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B figure/plot_episode.py --bundle . --output /tmp/nso_episode_figure
```

The source CSV, provenance and SVG/PDF/PNG outputs are retained. The bundle mode was independently checked from a different working directory against the saved CSV and rendered image. It does not validate pose accuracy or convert the quality proxy into ground truth.

## Cached-semantic downstream policy replay

`replay.py` instantiates the bundled original `JointSession` with recorded configuration and S mode, `execution_allowed=false`, and semantic frontend disabled because canonical cached labels replace inference. It supplies each real frame, scan, transform, base pose and contemporaneous observed grid in the recorded order. It calls the engine for every observation and next action; it does not force logged actions or copy logged decisions into the engine. On the original machine, the strict run matched 41 observations and 41 next-action calls (40 actions plus terminal closure), including 12 global-choice events, all candidate paths and mode scores, selection audits, feedback ledger, quality metrics, gain posteriors and termination. The 131 candidate comparisons include repeated returns of cached selections; they are not 131 independent candidates. The complete observe and next-action response objects, including runtime guard events, are compared after only the explicit frontend substitution: recorded frontend enabled and its actual inference receipt are replaced by cached-semantic disabled-frontend metadata, and the known system-variant text is normalized. Other keys and hashes are not broadly excluded. Integers, numeric types, timestamps and strings are compared exactly; other floating-point values use `atol=rtol=1e-8`. Source-container file hashes are separately verified against the derived-input manifest. The archived original inference receipt is not re-executed. `policy_replay_result.json` records the result.

```sh
work=$(mktemp -d)
nice -n 10 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -B replay.py --output "$work"
```

The tested environment is Python 3.12.15, NumPy 1.26.4, SciPy 1.11.4 and Open3D 0.19.0. Results can be sensitive to numeric library versions. This verifies downstream replay for this one recorded run and cached semantic labels; it neither reruns YOLOE nor proves the detector, camera calibration, original environment, independent accuracy or physical method efficacy.

An independent review repeated the full replay from `/tmp`, verified 49 bundled Python sources against the recorded source manifest, and exercised 12 negative comparator injections against actual decision records (including candidate, frame, score, guard, timestamps and unexpected keys). Its receipt and portable comparator check are under `policy_replay_independent/`. Run the negative checks without a planner using `python -B policy_replay_independent/test_comparator.py`.

A separate optional post-replay diagnostic and its qualified final-map results are in `frozen_semantic_availability/`. Its additional-budget calculations are counterfactual availability on the accumulated map, not part of the replayed S decision or an additional physical trial.
