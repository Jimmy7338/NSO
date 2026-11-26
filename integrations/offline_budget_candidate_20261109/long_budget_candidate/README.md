# Proposed long_budget_1200s_v1 execution profile

Status: isolated candidate, not deployed, no robot/ROS/CAN operation. Production mission.py and visual_velocity_adapter.py remain byte-identical to the originals saved here. Root's planned_protocol.json and planned_configuration.json were preserved.

The default remains standard_600s_v1 with duration600. Only explicit `--duration-profile long_budget_1200s_v1 --mode nso` or `--mode joint` permits duration1..1200 inclusive. Selecting the long profile in navigation, visual or reference mode is rejected even when duration <=600. Unknown profiles, nonfinite durations, values below1 and values above the selected maximum are rejected both at mission entry and independently at adapter configuration/Gate entry. Long profile is an experimental wall-clock change, not equivalence to historical40-action trials.

Reviewable patch: candidate.patch. Exact candidate files: candidate/runtime/{duration_profile.py,mission.py,visual_velocity_adapter.py}. Only these three files would need deployment after the user explicitly authorizes this profile. manifest.json binds originals/candidate hashes. No deployment commands were run. build_candidate.py reconstructs the candidate from the archived originals.

The profile identifier and mode are passed in adapter-config.json to the adapter Gate, including the separate stop-feedback Gate created during cleanup. The validated profile receipt is stored in experiment_receipt.json, authorization.json and final result; adapter started.json and actuator state carry the same receipt. Existing source_manifest.json runtime/*.py enumeration automatically hashes the new duration_profile.py file together with mission.py/adapter. Adapter-config's existing configuration SHA256 also binds the explicit profile and mission mode. The helper records adapter deadline clock (arm) separately from supervisor deadline clock (run_start).

## Effective time bounds after the proposed change

| Layer | Effective bound / clock | Change |
|---|---|---|
| Mission parser/entry, original mission.py:392,425 | Default600; explicit long nso/joint max1200, finite and >=1 | Explicit profile only |
| Adapter configuration and Gate constructor, original adapter:96,181–191 | Independently validates the same profile, mode, selected duration and maximum | Explicit profile only |
| Mission stop-feedback Gate, original mission:974 | Receives same profile/mode; otherwise a1200 mission would fail during cleanup | Explicit forwarding only |
| Adapter Gate.decision, original adapter:495 | At elapsed >=duration since arm: latch duration_complete and output exactly zero | Unchanged code |
| Main mission loops, original mission:872,900,946,953 | Stop/error at args.duration from run_start | Unchanged code |
| Joint coordinator heartbeat, joint_ros_coordinator.py:197 | external_execution_duration_complete at ctx.duration from same run_start, also checked while RPC/goal waits | Unchanged code; receives args.duration |
| Adapter pre-arm pending expiry, original adapter:740,886 | 3600s from adapter startup, before arm only; not an armed-motion cap | Unchanged |
| Pre-arm readiness, original mission:727,758 | 30s readiness deadline; arm service waits6s | Unchanged |
| Reference RPC transport, reference_transport.py:39 | 45s per RPC, with heartbeat; not600s mission cap | Unchanged |
| Observation acquisition, joint_observation.py:191 / reference_observation.py:173 | 5s map wait /10s reference observation wait | Unchanged |
| Native timing-v2 gate header:23,25,50 | Command TTL100ms; sender tick-gap40ms; source stamp age80ms | Byte-identical, no staging writes |
| Adapter Gate constructor/decision | Proposal TTL default0.20s (allowed0.05..0.25), observation TTL default0.50s (allowed0.15..0.50); adapter tick gap80ms; driver status200ms | Unchanged |
| Actuation and navigation | v<=0.1m/s, angular<=0.3rad/s, existing authorized reverse correction, original0.08m position tolerance and circular robot radius | Unchanged |

Search scope: active hybrid runtime Python sources, run_hybrid.sh and native timing-v2 cpp/header. The only hard600s acceptance limits are mission entry and adapter Gate (plus its cleanup invocation). Other matches to600 in export_episode.py are image dimensions/resolution, not runtime limits. run_hybrid.sh forwards all arguments without a timeout. Per-source freshness, hardware feedback guards, planner's own budget/return constraints, navigation failure and hardware/ROS stop remain able to terminate earlier.1200 is an upper execution allowance, not guaranteed completion or a relaxation of those guards. Supervisor may be occupied inside a call; the independent adapter's unchanged armed-clock deadline still rejects output at expiry on its next control tick. This is software verification, not a physical watchdog test.

## Validation

validation.json:8 test methods passed,0 failures/errors; production originals remain unchanged. Tests include default rejection above600,600/1200 boundaries for nso/joint at Gate and configuration layers,1200+epsilon rejection, invalid/nonfinite durations through mission and adapter, wrong modes/unknown profiles, and real unchanged Gate.decision at the exact deadline producing a latched zero output. Freshness-bound rejection remains in force. AST comparisons prove decision, arm, propose, observe, source freshness and hardware_reason implementations are unchanged. Mission forwarding and unchanged elapsed-time checks are also checked. No ROS/CAN sockets or actuators were opened.

Re-run: `/home/ouc/nso_robot/envs/science312/bin/python -B /home/ouc/nso_robot/analyses/long_budget_profile_20261109/test_profile.py`.
