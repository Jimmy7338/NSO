# ROS-checked exploration v2

Direct user authorization permits attempting unobserved space and removes research-layer footprint hard stops. Select `configuration.policy_variant = "ros_checked_exploration_v2"` with revision `v10_3_1`. This is a changed research policy, not an equivalent original NSO result.

The measured belief, camera-seen ledger, occupied cells, scoring formulas, category interpretation and configured circular radius remain intact. A separate planning graph treats unknown and free centers as traversable, retaining occupied centers as blocked. The actual current center alone can be virtually admitted even when its coarse cell is occupied, permitting a rotation or escape into an adjacent nonoccupied center. This does not clear any measured cell. Research circular-footprint admission is removed; the existing ROS rectangular footprint/collision checking and hardware protection remain authoritative. The two footprint contracts are not equivalent.

An optimistic return is budget-reserved when it is reachable on the new graph. An unreachable return does not certify impossibility or end the session: outbound options can be offered without a return certificate. If no positive scored option and no return action exists, the runtime records a separate, unscored finite-budget best-effort forward/left/right attempt. Translation still cannot enter an occupied center. Budget exhaustion and actual collision/terminal observations remain terminal. Return success remains actual discrete-anchor acceptance; no return override is introduced.

## Failed ROS goal contract

RPC `observe_failed_action` accepts `path`, `action_id`, `action`, `reason` only for v2 and one pending actual attempt. The caller must capture a fresh real packet at the actual partial pose, retain ROS status independently, and never replace its pose with the intended target. The attempt consumes one primitive budget unit and one acquired observation; it gives no option-endpoint arrival credit, clears the active macro and allows replanning. The result contains normal `state`/`summary` fields and an explicit `failed_execution_attempt` event. ROS verified-arrival counts must remain separate from attempted/paid counts. Original strict `observe` behavior remains unchanged.

## Evidence

`tests.log`: 21 focused original/new tests pass, including untouched-vendor manifest checks, unknown-cell measurement preservation, occupied-start escape, occupied-successor rejection, best-effort budget bounds, no-option continuation and failed-goal partial-pose accounting. `replay_C9.py` forces the ten actual saved C observations at actions0–9, then asks v2 for a next action. It is an offline saved-state diagnostic, not a v2 rollout or executed outcome. `C9_summary.json` preserves its scope, actual map hash and measured-belief nonmutation assertion.
