# Frozen floor comparison

Five arms started from the same initial sensor packet in each of four held-out floors. The 192-action cap included the return. We measured 20/20 terminal maps, with 20 returns and 0 simulated collisions. The mean terminal joint map quality change for the combined method versus the original global and tuned geometric local baseline was +0.06 percentage points. The registered engineering criterion was not met.

Mean correct free-space coverage changed by -0.88 percentage points, and mean whole-scene surface F1 at 5 cm changed by +0.82 percentage points.

| Floor | joint vs. original | joint vs. frontier | regional main effect | local given original global | local given regional global | interaction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | -0.31 | -0.85 | -0.78 | +0.00 | +0.95 | +0.95 |
| branch | +0.80 | +0.09 | +0.80 | +0.00 | +0.00 | +0.00 |
| corridor | -0.95 | +0.68 | -0.95 | +0.00 | +0.00 | +0.00 |
| loop | +0.68 | -4.54 | -0.85 | +0.31 | +2.76 | +2.45 |
| Mean | +0.06 | -1.15 | -0.45 | +0.08 | +0.93 | +0.85 |

Regional global main effect changed the first inter-room arrival by +1.75 [-2.00, +5.00] actions, n=4, repeated-edge travel by -1.48 [-6.60, +2.80] m, n=4, and the 95% coverage clock by -5.17 [-20.00, +11.50] actions, n=3.
Combined method versus original changed the first inter-room arrival by +1.50 [-2.00, +4.00] actions, n=4, repeated-edge travel by -1.65 [-6.60, +2.80] m, n=4, and the 95% coverage clock by -6.00 [-17.00, +6.00] actions, n=3.
Combined method versus strict nearest-reachable frontier changed the first inter-room arrival by +2.50 [-2.00, +6.00] actions, n=4, repeated-edge travel by +1.95 [-2.60, +5.20] m, n=4, and the 95% coverage clock by +15.00 [+7.00, +20.00] actions, n=3.

The smallest combined-versus-original difference occurred in the corridor floor (-0.95 percentage points). Correct free-space coverage was 97.54% versus 92.05%, and scene F1 was 83.85% versus 87.81%.
The first different macro target was dispatched at action 1. The original and combined execution receipts were `arrived` and `arrived`.
The regional layer changed the order of already feasible native candidates for this first decision.
The first executed action diverged at action 20.
At the shared 48-action checkpoint, the combined-minus-original J difference was -14.01 percentage points.

Table values are differences in percentage points. Brackets give the observed building-level range. Threshold times were differenced only for floors on which the relevant arms reached the threshold. One independent reporting unit was a floor. See [episodes](episodes.csv) and [paired traces](first_divergence.csv) for observed arrivals, guard interventions and stops.
