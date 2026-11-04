### 6.2 Policy-level unknown-exploration diagnostic

A separate exploration diagnostic evaluates the policy family used by the robot, starting from one simulated sensor packet rather than a supplied complete navigation graph. It compares the original V10.3.1 implementation with the existing exploration_joint_v1 policy core. The latter restricts camera coverage gain to unknown support, diversifies coverage candidates spatially, and replans an invalidated macro-action before returning. These are declared policy changes. The experiment does not evaluate the two-template belief policy of Sections 3–5 or the complete physical adapter stack.

Both implementations use the same Q0 scene generator, fixed sensor seed, exact simulated poses, 0.2 m grid, 0.2 m circular radius, and initial sensor packet within each layout. Initial observed coverage is 61.50% of the start-reachable set. The two arrangements swap the shelf and cabinet positions within the same scene; they are related conditions, not independent environments. G and S run at action-budget limits of 48 and 96. A subsequent four-run C ablation uses the existing joint-policy candidate generator and scores only predicted unknown coverage, omitting the measured-surface quality and category-prior terms. C is a within-family scoring control, not a standard frontier baseline. Each trajectory is deterministic; no between-run confidence interval is estimated.

Table 8 reports all 20 trajectories. All return to the anchor without a simulated collision; C in the east/96 condition stops after 94 actions because no positive option remains at the anchor, while the other runs exhaust their budgets. Newly visible surface is evaluated against the reference mesh after excluding visibility from the initial packet. F1 uses a 0.05 m distance threshold. Translation cost counts 0.2 m per forward action; turns also consume the action budget. The simulated radius and exact poses differ from the approximately 0.721 m physical circular model and measured localization.

**Table 8. Current-source unknown-exploration diagnostic.** V10.3.1 is the original policy; Joint v1 is the declared exploration variant. C is coverage-only scoring, G is geometry-conditioned joint scoring, and S includes category information. Used is the actual number of paid actions. Each row is one deterministic run; layouts and budgets are related conditions within one Q0 scene. Surface and F1 are evaluated in simulation, not on the robot.

| Policy | Layout | Budget | Mode | Used | Path (m) | Coverage gain (m²) | New visible surface (m²) | F1 @ 5 cm |
|---|---|---:|:---:|---:|---:|---:|---:|---:|
| V10.3.1 | west | 48 | G | 48 | 7.6 | 12.52 | 95.52 | 0.8857 |
| V10.3.1 | west | 48 | S | 48 | 7.6 | 12.52 | 95.52 | 0.8857 |
| V10.3.1 | west | 96 | G | 96 | 16.8 | 16.96 | 114.11 | 0.9689 |
| V10.3.1 | west | 96 | S | 96 | 16.0 | 16.92 | 111.44 | 0.9483 |
| V10.3.1 | east | 48 | G | 48 | 7.6 | 12.52 | 95.39 | 0.8830 |
| V10.3.1 | east | 48 | S | 48 | 7.6 | 12.52 | 95.39 | 0.8830 |
| V10.3.1 | east | 96 | G | 96 | 16.8 | 16.96 | 118.71 | 0.9857 |
| V10.3.1 | east | 96 | S | 96 | 16.8 | 16.96 | 118.71 | 0.9857 |
| Joint v1 | west | 48 | C | 48 | 8.4 | 14.16 | 96.58 | 0.8909 |
| Joint v1 | west | 48 | G | 48 | 8.4 | 14.16 | 96.58 | 0.8909 |
| Joint v1 | west | 48 | S | 48 | 8.4 | 14.16 | 96.58 | 0.8909 |
| Joint v1 | west | 96 | C | 96 | 17.2 | 16.92 | 108.56 | 0.9426 |
| Joint v1 | west | 96 | G | 96 | 17.2 | 16.92 | 109.51 | 0.9456 |
| Joint v1 | west | 96 | S | 96 | 16.4 | 16.96 | 118.52 | 0.9829 |
| Joint v1 | east | 48 | C | 48 | 8.4 | 14.16 | 96.46 | 0.8883 |
| Joint v1 | east | 48 | G | 48 | 8.4 | 14.16 | 96.46 | 0.8883 |
| Joint v1 | east | 48 | S | 48 | 8.4 | 14.16 | 96.46 | 0.8883 |
| Joint v1 | east | 96 | C | 94 | 17.2 | 16.92 | 112.11 | 0.9552 |
| Joint v1 | east | 96 | G | 96 | 16.8 | 16.92 | 114.00 | 0.9638 |
| Joint v1 | east | 96 | S | 96 | 16.8 | 16.92 | 114.00 | 0.9638 |

Semantic effects depend on both the layout–budget condition and the policy version. Three of the four G/S pairs have identical actions and outcomes within each implementation. In west/96, the original S policy acquires 2.66 m² less newly visible surface than G and has lower final F1 (0.9483 versus 0.9689). For the joint variant, S instead acquires 9.01 m² more surface, travels 0.8 m less, and increases F1 from 0.9456 to 0.9829. Its additional planar coverage is only one 0.04 m² cell. The F1 gain reflects higher surface recall (0.8972 to 0.9673); mean surface error increases from 3.73 to 4.31 mm, so this result does not establish better pointwise accuracy.

The recorded decisions explain where the semantic cue enters this difference. In the joint west/96 pair, the second planning call follows 21 actions and has identical recorded inputs and candidates. G selects grid pose [5,29,2], while S selects [5,10,2]; their paths first differ at action 31. Switching the category-dependent score at this shared state changes the selected goal. The eventual quality difference also includes later observations and replanning: G leads in newly visible surface at action 60, whereas S leads at actions 80 and 96 (Figure 11). A selected-goal change therefore does not by itself predict the sign of the terminal result.

The C ablation separates coverage-only scoring from the joint objective within the same candidate architecture. At budget 48, C, G and S have identical outcomes in both arrangements. At west/96, newly visible surface is 108.56, 109.51 and 118.52 m² for C, G and S; at east/96 it is 112.11, 114.00 and 114.00 m². Final coverage differs by at most one cell. These observations provide a bounded example of surface gains at nearly unchanged terminal coverage, but do not show a general improvement in unknown-space exploration efficiency. The matrix was fixed for development on one Q0 context, with synthetic sensing and category information. Natural recognition, physical execution, independent environments and comparison against standard exploration methods remain separate validation tasks.

![All paired semantic differences](figures/unknown_exploration_20261104/paired_differences.png)

**Figure 10.** S minus G differences in all eight policy–layout–budget conditions. Zero denotes a tie. In west/96, coverage, surface and F1 differences change sign between versions; both path-length differences are negative. No statistical uncertainty is estimated from these deterministic runs.

![Coverage and surface trajectories](figures/unknown_exploration_20261104/west96_trajectories.png)

**Figure 11.** Observed coverage and newly visible reference surface over paid actions in west/96. Colour distinguishes policy versions and line style distinguishes G and S. The changing surface ranking shows why endpoint and intermediate outcomes must be reported together. The curves are dependent observations from four trajectories.
