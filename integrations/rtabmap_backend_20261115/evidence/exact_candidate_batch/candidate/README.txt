Optional exact_candidate_batch_v1. No default install, no vendor/runtime edits.
Deploy new nso_bridge/exact_candidate_batch_v1.py alongside package only; root
must explicitly record requested compute_profile and call install(PROFILE) in
that worker process before generating options. A missing/other profile does not
silently select this module. install rejects other strings. Existing API unchanged.

Reviewed function source SHA256:
4aca18992d18808e4457026376bce4157a82d6e3deb95b9741fef7d5ef1df751
inspect source is hash checked. Only the reviewed state-pool loop, coverage sort,
rotation-pool traversal and the existing proven-zero asset shortcut are replaced
when installing. Remaining original function source and globals are cloned;
original function retained on candidate.original_generate_options for rollback.
Default/no install uses original function; finite profiles still original loop.
Only unbounded + existing native integer visibility tables uses the batch pool.
No states/candidates removed, no grid bounds/costs/scores/budgets/stop criteria edit.

Full pool is columnar. Integer lookup tables gathered by heading and real cell;
path unit-cost dijkstra doubles rint ties-to-even same Python round (finite exact
representation guarded). Camera pose matrix constructed only when actually
accessed, from unchanged templates and original arithmetic. Coverage lexsort keys
are precisely (-unknown,outward,cost,row,col,heading); original spatial-bin pick
loop unchanged. Asset bulk precheck is the pre-existing strict > xy_bound_sq
zero-aperture shortcut, same arithmetic. Surviving aperture/routing/scores unchanged.

Validation: 16 random small grids x unbounded and finite + 5 geometry-asset cases:
complete original (routes,audit) recursively equal, including ndarray values.
One frozen actual current0 replay, canonical cached labels (no inference/ROS):
complete selection and complete target exactly equal to original recorded result.
Batch wall47.7684s vs original118.5339s. Different process load/affinity may affect
speed; this single frozen measurement does not guarantee live latency.
Native visibility table build (~40s in previous profile) intentionally unchanged.
CPU4-7 nice10 for offline tests; no deployment/nodes/motion performed here.

Tests:
  taskset -c 4-7 nice -n 10 /home/ouc/nso_robot/envs/science312/bin/python -s -B candidate/tests/test_exact_candidate_batch_v1.py
Actual frozen receipt: ../actual_frozen/receipt.json
Default rollback: do not install in new worker; existing process can restore
exploration_joint_policy.generate_options = installed.original_generate_options.
