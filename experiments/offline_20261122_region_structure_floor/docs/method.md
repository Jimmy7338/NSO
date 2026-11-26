# Unknown-floor exploration and acquisition

The experiment starts from a single sensor packet at the return anchor. Each arm owns the resulting observations, occupancy grid and TSDF. The 0.20 m lattice stores unknown, observed free and occupied cells. Each charged action is a translation or a turn. The simulator provides an exact discrete pose in the main matrix; category-aware arms can access supplied labels of currently observed instances. The main study uses 192 charged actions including return.

| Component | Inputs and units | Outputs and update time |
|---|---|---|
| Observations and map | RGB-D, laser ranges in m, poses, camera calibration | Observed RGB-D TSDF, fused measured evidence, occupancy and view history after every accepted sensor packet |
| Native shadow and fallback | Current measured coverage and surface features, return anchor, observed known-free graph, native feedback | Original neutral candidate poses and scored macro target whenever the native executor asks for a choice |
| Observed region layer | Current occupied/free/unknown raster, charged pose history, observed prior rejects | Persistent observed region and entrance IDs, visits, estimated remaining frontier, oriented viewpoints and a ranking at the macro boundary |
| Local choice | One macro pose and heading, the common admitted waypoint set, measured surface evidence and local shape model | One bounded outbound route to the same target, checked with unchanged return commitment and executed one action at a time |
| Native execution | Selected route sidecar and current observed footprint, reserved return and current budget | Authorized action, observation, arrival/rejection, subsequent planning or return |
| Independent evaluation | Verified terminal and checkpoint triangle mesh, occupancy, traced actions and separately held source geometry | `C_free`, scene `F1@5cm`, their product `J`, room transitions, actual path, repeats, arrival, return, time and collision receipts |

## Region selection and feasibility

The detector requires opposing occupied jamb evidence and an observed free gap. Candidate directions point through a usable observed gap or toward a frontier that can still yield new observations. Region IDs follow overlap in the evolving map. Entrance IDs follow observed spatial correspondence. Charged poses update visit and traversal status. An execution rejection is evidence about that attempted route; the door is marked blocked from the measured footprint and occupancy evidence. Region IDs are generated online. The evaluator uses the original floor plan exclusively to score actual room crossings.

For each proposed pose, the adapter uses the same positively observed footprint mask, oriented action graph and directed return as the original selector. A candidate is admitted when the start, goal and anchor are in the observed reachable component, the outbound cost is positive and a return with the fixed anchor heading fits the remaining budget. The candidate list retains the original alternatives. Each new admitted route receives the usual sidecar and its original global scores for audit. Its source ID identifies whether it was newly generated or re-ranked from the original pool.

The frozen structural ranking uses equivalent square metres. It awards 1.25 for a presently unvisited observed endpoint region, adds 0.25 per metre of nearby region frontier up to 3 m, up to 2.0 m² of visible unknown, and 0.35 for a not-yet-traversed entrance viewpoint. It deducts 0.0125 per outbound-plus-return action and 0.35 per previous dispatch of the same target pose. Each quantity is logged at selection time. Equal scores prefer lower round-trip action count and then lexicographic pose. No trustworthy observed entrance or no feasible informative pose selects the unchanged native fallback. Code fixes the geometry thresholds and caps in `policy/observed_regions.py`.

## Local selection and measured surfaces

Both local priors use the frozen common same-goal pool, identical range limits, 8 heading bins, extra outbound cap of 8 actions and native predicted-coverage retention of 95%. The visibility forecaster estimates newly observable surface support on the same underlying shape particle bank. In G the class-dependent weight is neutral, in S it conditions on currently visible structure class. The score adds 0.25 times the difference in predicted observable unknown area and subtracts 0.20 m²-equivalent per extra outbound action. The native deterministic tie rules and safety tests remain active. Source truth remains in the independent evaluator, and predicted hidden area is local scoring input only. The direct-inspection diagnostic uses measured but sparsely observed instance surface voxels and views, with no hypothesized backside geometry.

```text
initialize from a single sealed sensor packet
while executor has budget and returns authorized actions:
    if no current bound outbound route:
        obtain the native neutral shadow target and candidate set
        if regions enabled:
            detect observed entrances and stable regions
            certify candidate poses with observed oriented outbound and return
            rank with observed coverage and visit state, or fall back
        choose a local route to that exact macro pose
    let the native footprint and return guard authorize one paid action
    integrate the new observation and update the observed region history
    preserve arrival, failure and replanning receipts
evaluate checkpoint and actual terminal maps with a separate reference ledger
```

The historical early topology model and its semantic-density priority are background. The implemented global switch uses only structural evidence. Enabled regional ranking and local category-conditioned scoring are separate factors. The current estimated-pose and robot integration evidence is scoped separately from the deterministic simulator matrix.
