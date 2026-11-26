### 6.3 Frontier comparison and coverage timing

An additional four-run comparison uses a frontier policy whose goals are generated directly from observed occupancy, independently of the NSO object candidates. This local implementation retains up to four representatives per frontier component, at most 24 candidates, predicted unknown visibility, and in-place scanning. It receives the observed grid and odometry, with no reference geometry or semantic assets. The policy inflates occupied cells when choosing goals; the shared execution guard treats both unknown and occupied cells as unsafe when checking each action and its reserved return. The adapter permanently switches to return after the first rejected action or policy stop. This differs from NSO candidate feasibility screening, despite the shared execution checks. The implementation is not the full ROS explore_lite system.

The four layout–budget conditions and source hashes were fixed after inspecting a west/48 smoke run, before executing the matrix, with no subsequent parameter tuning. The west/48 matrix run exactly reproduces that smoke trajectory and is not an additional independent sample. Initial sensor packets, anchors and complete initial evaluation metrics match the corresponding original, joint and C runs. All four frontier trajectories return without a simulated collision and exhaust their budgets. Each has one forward action rejected because its reserved return would exceed the remaining budget, after 23, 76, 23 and 53 executed actions for west/48, west/96, east/48 and east/96, respectively. No policy-stop event occurs. The rejection and permanent return behavior remain part of the recorded comparison.

**Table 9. Local observed-frontier baseline with the shared execution guard.** Each row is one deterministic trajectory on the same Q0 conditions as Table 8. The adapter returns permanently after a rejected action; these are not ROS explore_lite trials. All four use their full action budget and return to the initial pose.

| Layout | Budget | Path (m) | Coverage gain (m²) | New visible surface (m²) | F1 @ 5 cm |
|---|---:|---:|---:|---:|---:|
| west | 48 | 8.4 | 15.00 | 84.12 | 0.8158 |
| west | 96 | 17.6 | 16.88 | 107.23 | 0.9359 |
| east | 48 | 8.4 | 15.00 | 84.02 | 0.8130 |
| east | 96 | 16.8 | 16.76 | 96.56 | 0.8751 |

The frontier comparison shows a coverage–surface tradeoff (Tables 8–9). At the 48-action limit, frontier gains 15.00 m² of observed coverage in each layout, versus 14.16 m² for joint G, S and C, with the same 8.4 m translation. Its newly visible surface is about 84 m², versus about 96 m² for the joint policies. At 96 actions, joint G has slightly higher terminal coverage than frontier (16.92 versus 16.88 m² in west and 16.76 m² in east) and more newly visible surface. These fixed-condition outcomes do not establish uniformly faster exploration.

The S–G difference in simulator truth-visible surface appears late in coverage. At the 96-action limit, frontier reaches 95% coverage after 20 actions and 99% after 32; joint G, S and C reach these thresholds after 24 and 29 actions. At the 48-action limit the joint policies never reach 95%, whereas frontier does after 20 actions (Figure 12). In west/96, the only joint-policy condition with different G/S outcomes, both have identical first-hit times for 90%, 95% and 99% coverage: 18, 24 and 29 actions. Their newly visible surface first differs at action 31, when both have already observed 99.55% of the reachable cells. The observed surface difference in this condition concerns late surface observation; it does not support faster early unknown-space coverage. The other three joint G/S conditions have identical coverage and surface histories.

![Coverage timing against an observed-frontier baseline](../figures/unknown_exploration_20261104/coverage_progress.png)

**Figure 12.** Reachable-cell coverage over paid actions in the four Q0 conditions. Lines follow the recorded steps without smoothing; horizontal guides mark 90%, 95% and 99%. The initial packet is action zero. East/96 C ends at its recorded action 94. Original G is shown as a policy-family reference; original S remains reported in Table 8. These are deterministic, related conditions without independent-scene uncertainty estimates.
