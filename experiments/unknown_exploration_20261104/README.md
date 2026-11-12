# Current-source unknown exploration simulations

This is a new controlled simulation, not recovery or verification of the missing historical Windows frozen results. Q0 shelf_west and shelf_east are two arrangements of one generator context, not independent scene samples. The original 151-frame shared-prefix experiment starts at 100% reachable-cell coverage and is a surface-inspection diagnostic, not evidence of unknown-space coverage gain. The fresh matrices start from one sensor frame and differ from the old 150-action-prefix protocol. Preserve failed or negative results; these observations do not demonstrate physical robot or generalization benefits.

The original v10.3.1 and explicitly revised exploration_joint_v1 matrices are distinct policies. Simulation uses the virtual world's 0.2 m robot radius and synthetic RGB-D/scan and exact discrete odometry; the physical robot's circular radius is 0.721 m and its sensor/odometry execution is different. The deployed-policy matrix is a policy-only simulator experiment with physical adapters disabled, not a physical trial. See evidence/*/protocol.json and choice audits for exact configurations.

## Runtime and offline commands

Tested using Linux Python 3.12.15, NumPy 1.26.4, SciPy 1.11.4, Open3D 0.19.0+597106a. `requirements-observed.txt` records the full environment as observed, rather than promising that its platform-specific wheels are installable everywhere. Install equivalent CPU dependencies beforehand; after that, these commands use local files and do not access ROS or the network. Exact byte equality across platforms/Open3D builds is not guaranteed.

From this directory, choose a new empty output directory:

```bash
mkdir -p /tmp/nso-reproduction
export NSO_OUTPUT_ROOT=/tmp/nso-reproduction
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# One original west/48 G branch, with strict comparison to preserved original:
NSO_ONE=1 python matrix.py
# Full original 8-branch matrix (use a fresh NSO_OUTPUT_ROOT):
python matrix.py
# Revised policy matrix (use a fresh NSO_OUTPUT_ROOT):
python deployed_policy_matrix_runner.py
# New 151-frame current-source G/S pair (fresh NSO_OUTPUT_ROOT):
python current_pair.py
# Rebuild and verify the two west/96 outbound routes (fresh NSO_OUTPUT_ROOT):
python counterfactual.py
```

Each command creates a named subdirectory and refuses to overwrite it. The revised-policy runner compares its initial packet and anchor to the archived original matrix. Reconstructed results are new runs and are not historical Windows evidence. `evidence/` retains the original records, including negative and tied cases, scores, actions, feedback, failed-case diagnostics, and the original parameter/hash receipts. `source/` contains locally loaded or statically imported project dependencies; source files are unchanged. Harnesses were adjusted for portable paths and selectable validation only; research decision logic is unchanged.

Validation in the originating environment: packaged `NSO_ONE=1 python matrix.py` completed west/48 G and asserted equality with the original for initial packet hash, all 48 action/pose/packet/coverage/surface records, and final metrics. The packaged revised-policy west/48 G also ran successfully. The two outbound routes were separately checked against original per-step hashes and scores.

## Coverage ablation and archived prefix

`coverage_ablation_runner.py` evaluates four C-mode branches of exploration_joint_v1; C ranks the existing candidate set by grid-cell area × (predicted radar cells × radar feedback posterior + predicted camera cells × camera feedback posterior). The radar/camera proxies may overlap. C omits the quality and semantic/geometry prior terms; it is a within-policy scoring ablation, not an independent frontier baseline. It retains candidate generation, safety and return contracts. The C experiment was added after the G/S matrices; both use the identical policy-source hash recorded in their protocols.

```bash
# C full matrix; use an empty output root
python coverage_ablation_runner.py
# Or just west/48 C
NSO_ONE=1 python coverage_ablation_runner.py
# Restore the previously stored 151-frame prefix for counterfactual.py
tar -xzf evidence/current_source_pair/prepared.tar.gz -C evidence/current_source_pair
# Recreate the comparison figures and tables
python comparison_artifacts/generate.py
```

The per-file original prefix SHA-256 values remain in evidence/current_source_pair/prepared_hashes.json. Unpacking is not required for the newly generated current_pair.py run.

The east/96 C run stopped after 94 actions with `no_positive_option_at_anchor`; it did not exhaust its budget. All other 19 runs used their full budgets.
