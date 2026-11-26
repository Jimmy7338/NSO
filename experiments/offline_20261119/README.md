# NSO offline experiments and evidence — 2026-11-19

This is a compact code-and-evidence release prepared for GitHub. It records offline simulations only; packaging and verification do not run a robot or simulation actor. Manuscript source, PDFs and cited figures are added separately by the project maintainer. Local preparation receipts record the pre-upload state; publishing is a separate step.

## What was tested

The newest study has one authored three-branch floorplan, two furniture-position permutations (`shelf_left`, `shelf_right`), and six strategies per layout: C, J, G, S, S_no_fit and S_no_history. These are 12 deterministic episodes, **not 12 independent environments**, and the two permutations are not independent natural scenes. The shared 256-action budget includes the original return requirement. Mapping uses measured simulator packets; independent geometry is evaluator-only.

The new, explicitly disclosed combination keeps v10_3_1 unknown-coverage and measured-surface terms and its candidate generator, replacing the analytic AABB prior with the original V3 box/shelf template estimator and visible-area gain. The complete V3 candidate proposer is not used, and the result is not an unchanged reproduction of either original planner. Current visible ideal categories and common partial instance masks are used. This is an **oracle perception experiment**, not a demonstration that YOLOE detects structure correctly. No hidden full-object pose, dimensions or reference mesh are supplied to the planner.

The implementation and parameter provenance are in `analyses/dual_template_exploration_20261119/method/METHOD.md`. Per-episode template receipts record geometry fitting, retained real depth history, surface-point rejection and gain calls. The shape probability is an uncalibrated model-discrepancy pseudo-posterior. Historical depth removes reward samples near measured surfaces or in observed free space; it does not update the geometric likelihood or remove contradicted parts from the complete hypothesis self-occlusion mesh.

## Outcomes and limits

All 12 declared episodes completed and returned, with 100% start-reachable-reference coverage and zero simulator collisions. Full-template G and S have identical complete action/pose trajectories and native TSDF meshes in both layouts. Semantic scores differ at some actual planning states, but no G/S first-choice change occurs. First full coverage is 67/66 actions for C, 88/100 for J and 97/100 for G and S (left/right). These outcomes do not support a semantic coverage-efficiency improvement.

S_no_fit actually sets the consumed geometry likelihood to zero, but its trajectories remain identical to S. S_no_history disables both historical free-space rejection and measured-surface consumption. Its left-layout first full coverage is 69 actions, earlier than S's 97; the right layout is 100 for both. This ablation also activates the unchanged 3,500-point weighted thinning cap in 28/14 planning snapshots, whereas the other arms do not. It therefore cannot isolate a pure history-only effect from the induced sampling approximation. Retain surface precision/recall/F1 and per-object completeness from `audit/summary.json` alongside coverage; do not turn a completion checkpoint into an efficiency or reconstruction-quality claim.

The package also retains the earlier natural-apartment and four-room controls. The natural apartment has only partial verified ideal structure labels and is a negative case: measured-instance grouping restores nonzero semantic scores without improving the closed-loop outcomes. The four-room matrix is three layouts by six arms, with its original oracle labels and synthetic geometry; C outperforms G/S on coverage and F1 in these recorded layouts. The S/1024 diagnostic stopped after an external SIGTERM at action 398. Its partial curve and interruption receipt are preserved, with no invented terminal result or algorithm-failure attribution.

## Contents and verification

- `analyses/dual_template_exploration_20261119/`: exact new method, scene, runner, evaluation and diagnostic sources; setup/freeze receipts; all 12 final results and independent surface evaluations.
- `archives/frozen_scientific_source_20261104.zip`: the 145 Python/configuration files matched against the pre-run scientific-source SHA-256 map.
- `archives/runtime_and_evaluation_dependencies.zip`: exact earlier runner helpers, existing measured-instance modules, local rendering/evaluation code and test fixture. These are included to expose the dependency chain, not replaced by stubs.
- `archives/dual_template_episode_logs.zip`: all 12 exact choices, actions, template audits, frame/owner hashes, execution/episode summaries, self-checks and original manifests. No RGB/depth images, TSDF, point-cloud or mesh data are included.
- `archives/four_room_budget_plot_logs.zip`: observed curve/decision logs, including the interrupted S/1024 prefix.
- Earlier study summaries, all fixed-cost terminal outcomes, independent surface results, detector diagnostics, and declared limitations remain under their original `analyses/...` paths.
- `RELEASE_MANIFEST.json` checks packaged files. `ARCHIVE_INDEX.json` additionally lists every archived member's original path, byte count and hash.

Run `python3 verify_package.py` from any location to check packaged bytes without loading an actor, reference mesh or third-party asset. The original run artifact manifests also name excluded meshes and arrays. Their hashes are retained as provenance; those omitted files cannot be revalidated from this compact release alone.

## Reproduction dependencies

This is an auditable snapshot, **not a self-contained, relocatable one-command reproduction**. Frozen Python files retain original absolute paths rooted at `/home/ouc/nso_robot`, including the source tree under `analyses/reproduction_20261104/.../source`, earlier analysis runners and the existing instance-support modules under `workspaces/nso_hybrid_20261029`. Archives preserve paths relative to that root. A new installation must either recreate that directory layout or make and disclose a separate path-portability adaptation; changing frozen files will intentionally invalidate their existing hashes.

`ENVIRONMENT.json` records the installed Python and package versions used when packaging. The runtime requires NumPy, SciPy, Open3D, trimesh, Shapely, Pillow and the retained scientific dependencies; this metadata is not an environment lockfile. Runs and evaluations use one numerical-library thread. The method tests use the included inspection-benchmark fixture.

The textured simulator additionally requires the original ReplicaCAD apartment/furniture assets, configuration, articulations and textures at the paths recorded by the included download receipt and pre-run freezes. The scene builder currently loads the original apartment before extracting the two unscaled furniture objects, so supplying just a screenshot or a replacement box is insufficient. Obtain these assets from their original providers under their own terms; **this release does not redistribute them**. Also omitted are source/reference meshes, reference point archives, reconstruction meshes, raw RGB-D and videos. Independent metric recomputation requires those assets and regenerated native TSDF output, following the included evaluator and fixed-reference protocols. No authorisation to use or redistribute an external dataset is implied by this package.

Current final figures and the manuscript remain working research materials. Completion of these software runs does not establish robust natural semantic perception, real SLAM accuracy, robot safety, learned reachability, or a publication-ready positive method comparison.
