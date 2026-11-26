# Candidate-route ranking audit

This audit compares archived native score predictions with 16 short rollouts across two parent layouts. Each layout was evaluated with left and right object placements. The analysis applies the fixed comparison rules in [the study protocol](../PROTOCOL.json). Agreement counts describe those four placements.

With the archived experiment and measured short rollouts in place, run from the study directory.

```sh
python3 ranking/archive_predictions.py
python3 ranking/rank_routes.py
python3 ranking/plot_ranking.py
```

The figure script requires Matplotlib. The default measurement input is `measurement/results/candidate_rows.json`. All three scripts accept `--help`.

- `archive_predictions.csv` contains the frozen native score components, ranks, nominal action costs and optical-side diagnostics. `archive_prediction_receipt.json` records the archived source hashes.
- `results/per_candidate.csv` joins the 16 native predictions to the start, arrival and common paid samples, preserving unavailable outcomes and stop reasons.
- `results/pairwise.csv` reports lower-ID minus higher-ID differences. It includes strict cost matches at target arrival and again at the common paid sample. Gallery 0–1 has separate unmatched arrival and paid-58 rows.
- `results/rank_agreement_by_parent.csv` reports directional agreement for object components, category corrections and total scores. The material bands are ±0.05 m² for object area and ±0.001 for `J`. `results/swap_audit.csv` keeps the two placements of each nominated pair together.
- `results/candidate_oracle.csv` includes each short branch's `J` and the observed maximum relative to the original choice. The candidate-set upper bound uses rows with four observations at a shared paid step.
- `results/candidate_rank_yield.pdf` and `results/candidate_rank_yield.svg` reproduce the paper figure. The SVG preserves editable text. `results/analysis_receipt.json` and `results/figure_receipt.json` record source and output hashes.
