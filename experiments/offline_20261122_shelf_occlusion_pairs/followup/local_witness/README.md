# Independent route witnesses for the two archived development scenes

The frozen bank covers the two published `increment_open` and `increment_screen` forks. It reuses the exact original measured fork, oriented goal, legal native routes, return reservation, scores, sensor model and motion guard. The geometry-only witnesses are selected by outbound length and early oriented path diversity. The three actual policy nominations are added for measured regret.

The union has six open and seven screened routes, with five geometry-only references in each condition. Three open and four screened routes have no prior execution. The six archived routes are replayed to materialize the new registered action clocks, with the original full trace, guard evidence and original checkpoints checked byte for byte. All earlier archives remain available and unchanged. The original six runs had a median worker wall time of about 25 s. Thirteen executions at up to three concurrent workers should take about 2–3 min before evaluation on comparable hardware.

Use the source tree paths recorded by `WITNESS_FREEZE.json`. The already-sealed run command requires the formal shelf stages and their full evaluated summary to be complete.

```bash
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
"$NSO_PY" witness.py run --workers 3
"$NSO_PY" witness.py measure
```

`results/routes.csv` records each route, requested and actual clocks, measured area, and native stop reason. `results/summary.json` reports the geometry-only subset and full union, separately marks complete and incomplete witness sets, and reports D, D+ and S regret when the selected route is observed at that clock. The primary paired clock is 16 actions after the fork. The 24-action and minimum nominal route clocks are secondary. Each actual route endpoint has its own measured area. A route cancelled before a fixed clock provides no area at that clock.

These two source layouts serve development. An independent confirmation must prospectively freeze separate room and shelf parents and retain all their scheduled outcomes.
