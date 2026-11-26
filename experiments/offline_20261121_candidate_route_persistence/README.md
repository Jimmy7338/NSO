# Candidate-route persistence at a 256-action horizon

This follow-up starts from the sealed study at commit `7fda64686ecd9b2adbc5aa35bc3e277eb720d00f`. Five candidates were advanced from their exact previously measured short prefixes under the neutral G policy. The other five routes use their original full-run archives. Gallery candidate 1 intermediate measurements are recovered by replaying the archived controls and verifying each recorded sensor packet. No planner runs in those two archival recoveries.

- [Frozen protocol](PROTOCOL.json), [input hashes](PROTOCOL_FREEZE.json), [execution freeze](runner/freeze.json) and [completed-run receipt](runner/supervisor_receipt.json).
- [All measured checkpoints](measurement/results/checkpoints.csv), [paired comparisons by placement](results/paired_layouts.csv), [terminal comparisons](results/paired_terminal.csv) and [terminal interpretation](results/decision.json).
- [Gallery actions 38–58](interval/INTERVAL_ZH.md), [planned and actual route segment](interval/neutral_move_summary.csv), [correct-free cells by scene region](interval/correct_free_by_region.csv), [fixed surface area by region](interval/fixed_surface_by_region.csv) and [coverage/F1 decomposition](interval/J_product_decomposition.csv).
- [Joint quality figure (PDF)](results/joint_quality_by_action.pdf), [editable figure (SVG)](results/joint_quality_by_action.svg) and [paper-ready Chinese result](RESULTS_ZH.md).
- [Evidence index](EVIDENCE_INDEX.json) and [completion checks](completion_audit.json). The larger native meshes, beliefs, mask files and full local traces are indexed there. Action traces and checkpoint/verification receipts are preserved on GitHub as one compressed JSON record per route under `evidence/`.

## Reproduction

Use the original host with Python 3.12, NumPy, Open3D and Matplotlib to remeasure the native evidence and rebuild the reports. The native snapshots, simulator and independent reference arrays remain at their frozen original paths. With the working directory set to this study, use a new analysis output directory to preserve the published results.

```sh
python measurement/evaluate.py --out /tmp/nso-persistence-measurement
python interval/analyze_gallery.py --output /tmp/nso-persistence-interval
python summarize.py --checkpoints /tmp/nso-persistence-measurement/checkpoints.json --terminal /tmp/nso-persistence-measurement/terminal.json --output /tmp/nso-persistence-report
python plot_pairs.py --measurements /tmp/nso-persistence-measurement --output /tmp/nso-persistence-report
```

The paired tables and figure can also be regenerated without a simulator from the published summary JSON tables. Decompress `measurement/results/checkpoints.json.gz` into `measurement/results/checkpoints.json`, then run the last two scripts with their default inputs and a fresh `--output` directory.
