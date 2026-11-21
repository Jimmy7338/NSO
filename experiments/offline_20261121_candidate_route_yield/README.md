# Candidate-route yields in a warehouse and a gallery

The warehouse and gallery each have two placements of an open shelf and a cabinet. The first intervention in each placement exposes four native options. The experiment executes each option once and evaluates the resulting native TSDF meshes. The prior study at commit `1f06bb5d93bbe93a40b9d4499861101aa7555c09` is preserved. [The Chinese results](RESULTS_ZH.md) give the interpretation and manuscript text.

- [Frozen protocol](PROTOCOL.json), [bound inputs](PROTOCOL_FREEZE.json) and [short-run receipt](runner/supervisor_receipt.json).
- [Scores and measured yields by candidate](ranking/results/per_candidate.csv), [paired differences](ranking/results/pairwise.csv), [ranking results by parent](ranking/results/rank_agreement_by_parent.csv), [swap audit](ranking/results/swap_audit.csv) and [short-horizon candidate upper bound](ranking/results/candidate_oracle.csv).
- [A/B/C decision](decision.json), [continuation freeze](runner/branch_b/freeze.json) and [two-route continuation measurements](measurement/results/branch_b_continuation.csv).
- [Paper figure (PDF)](ranking/results/candidate_rank_yield.pdf) and [editable figure (SVG)](ranking/results/candidate_rank_yield.svg).
- [Evidence index](EVIDENCE_INDEX.json) and [completion checks](completion_audit.json).

## Recompute

Use Python 3.12 with NumPy, Open3D and Matplotlib for full evaluation. The execution programs use the original local simulator, compiled runtime, frozen study dependencies and reference files. The native snapshots and exact paths are listed in the evidence index. On the original host, use the copied study scripts and fresh output directories for reproducing the candidate measurements and gate.

```sh
python measurement/measure.py --out /tmp/nso-route-yield-measurement
python ranking/rank_routes.py --measurement /tmp/nso-route-yield-measurement/candidate_rows.json --output /tmp/nso-route-yield-ranking
python ranking/plot_ranking.py --summary /tmp/nso-route-yield-ranking/ranking_summary.json --output /tmp/nso-route-yield-ranking
python decide.py --measurement /tmp/nso-route-yield-measurement/candidate_rows.json --out /tmp/nso-route-yield-decision.json
```

`runner/execute.py <layout> <candidate-id>` executes a single short branch. The supervisor requires a fresh run tree and preserves existing run receipts. `runner/branch_b/continue_pair.py` verifies the two selected prefixes, recovers the original endpoint from the archive and applies the frozen neutral geometry policy to the alternate. `measurement/measure_continuation.py` independently verifies and measures those endpoints. These commands rely on the preserved local source archive.

The aggregate tables and figure can be regenerated using the GitHub package and the frozen prediction CSV. Decompress `measurement/results/candidate_rows.json.gz` to `measurement/results/candidate_rows.json`, then run the ranking, plotting and decision commands above. The compressed [checkpoint receipts](evidence/checkpoint_receipts.json.gz) record native snapshot hashes and recovery metadata. The [two continuation traces](evidence/branch_b_steps.json.gz) contain the action-level records used for the terminal comparison.
