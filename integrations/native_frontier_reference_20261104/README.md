# Local native frontier reference

This package launches the locally deployed `explore_lite` binary as a separate ROS exploration reference and passively records action events. The node is **disabled by default**. It does not launch `move_base`, sensors, localization, a velocity adapter, or the chassis; the recorder publishes nothing and does not select, send, filter, or cancel goals. The package has not itself been tested on the robot.

This package is MIT licensed; the external locally built explore_lite dependency retains its BSD terms and separate provenance (`provenance.json`); its binary is not packaged here.

The binary is a local modified implementation, **not unmodified upstream**. It includes the existing `goal_update_min_distance` active-goal holding patch; this package retains the historic deployed 0.25 m value and the other parameters from `ans-live-20261023-standard-attended-a4/config/exploration.launch`. Frontier scoring, failure handling and blacklisting are native to that binary. It is distinct from the local Python `FrontierPolicy` used in earlier offline analysis.

The reference does not implement the NSO observed-grid circular research guard, discrete paid-action budget, or full-anchor-heading return reservation. ROS costmaps, rectangular footprint, their unknown-space handling and ordinary obstacle avoidance remain the existing navigation configuration. Native results therefore are a separate integration/reference condition; they are not automatically a matched-budget comparison or evidence of identical research policy. ROS/hardware protections remain separately owned and must be recorded explicitly if relevant.

## Entry points

`launch/reference.launch` accepts:

- `enable_exploration`: default `false`; explicitly enabling starts the reference.
- `enable_recorder`: defaults to `enable_exploration`; independently set true with exploration false for passive recording.
- `session_id`: operator-supplied identifier attached to each JSONL record; empty by default generates a fresh UUID for each recorder process.
- `navigation_namespace`: default `nso_visual`; controls the explore node namespace and default paths.
- `native_binary`: defaults to the existing locally built absolute explore executable.
- `costmap_topic`, `costmap_updates_topic`: default to the selected namespace's `move_base/global_costmap/costmap` and `costmap_updates`.
- `action_namespace`: defaults to the selected namespace's `move_base`; the explore node's relative `move_base` action is remapped here.
- `base_frame`: default `base_link`.
- `event_log`: default `/tmp/native_frontier_reference_events.jsonl`; choose a unique durable path per attended run.
- `protection_topic`: default empty. Only connect an explicit protection-intervention event stream of type `std_msgs/String`; content is logged verbatim as `protection_event`. Do not connect high-rate actuator-state telemetry as protection events. An absent or empty stream does not establish that no intervention occurred; unrelated action statuses are never inferred to be protection interventions.

Recorder writes one JSON line for start/stop, action goal, result, cancel, status transitions, and optional explicit protection events. It does not record periodic duplicate status, maps, image frames or raw sensor streams. Action topics are shared: entries may include goals from other clients, and no field claims exclusive authorship. Time fields include ROS and wall clocks; goal IDs, action status, session ID and the observed ROS caller ID are retained for correlation. Caller ID is transport provenance, not proof of exclusive ownership.

## Intended on-site integration

After resolving the outstanding onsite near-body object identity, the primary execution entry remains the existing mission coordinator, which owns the native exploration process, velocity adapter and stop lifecycle. Start this package **only as a passive recorder**; do not independently enable its explore node alongside the coordinator. The shell forwards arguments to `runtime/mission.py`; `--execute` is required for actual execution.

```bash
# Passive recorder: source the workspace and configure the same ROS master first.
roslaunch native_frontier_reference reference.launch enable_exploration:=false enable_recorder:=true session_id:=YOUR_SESSION event_log:=/absolute/path/to/unique-events.jsonl

# Existing separate mission entry, only when the physical run is intended:
/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/run_hybrid.sh --mode visual --execute --report /absolute/path/to/report
```

The mission starts and stops exploration, velocity adapter and navigation execution; this reference package does not add a second chassis owner. Its standalone `enable_exploration:=true` path exists for isolated integration testing and starts autonomous goal publication, but is not a complete physical mission entry. No return-budget adapter is included; any such adapter is a separate disclosed intervention.

**Goal cleanup:** Stopping or killing the standalone native explore node does not guarantee that an active move_base goal is canceled or preempted. Physical runs must use the existing mission owner, whose cleanup publishes stop and calls `cancel_all_goals`; the raw diagnostic launch does not own that cleanup.


## Isolated fixture reproduction

The fixture requires a working ROS Noetic environment and a separately supplied, compatible native explore executable. It starts its own local ROS master and fake action server; it does not require the physical robot. For example:

```bash
/usr/bin/python3 tests/run_isolated.py \
  --ros-setup /opt/ros/noetic/setup.bash \
  --overlay-setup /path/to/optional/overlay/devel/setup.bash \
  --native-binary /absolute/path/to/explore
```

Omit `--overlay-setup` when it is unnecessary. The package root is discovered from the script location; paths containing spaces are supported. Each receipt records the setup paths and hashes, binary hash, package source hashes, isolated port and results. The original runner used for the previously published 25-check receipt is retained as `tests/run_isolated_historical_cc8.py`. The parameterized runner also passed 25 checks from a copied path containing spaces on the same machine, as recorded in `tests/runs/20261104_184551/receipt.json`. Neither result establishes a clean-machine build, dependency portability, physical operation, or a production mission test.
