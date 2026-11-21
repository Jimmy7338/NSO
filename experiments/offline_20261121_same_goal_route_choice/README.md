# Same-goal route choice

The optional local policy changes only the route to the native global target pose. In the paired experiment, `newG` and `newS` both use the same neutral global selector and observed-map safety rules. They differ in their local expected object-surface score, using the neutral prior or the acquired visible category prior. The parent workspace and earlier experiments continue to use their original entry points.

## Switch and rules

The switch is the local mode passed to the experiment harness in [runner/full_run.py](runner/full_run.py). Its `ARM_MODES` maps `newG` to local `G`, `newS` to local `S`, and leaves the local hook disabled for `originalS` and `F_strict`. The module [route_choice.py](route_choice.py) is attached only when the selected local mode is enabled. The native global selection and return reserve are independent of that attachment.

The original route is always retained. Up to four alternates visit an alternative observed-safe orientation waypoint and reach exactly the same terminal cell and heading. Admission uses at most eight additional charged outbound actions, the original return path and observed-safe reserve, and at least 95% of the original route's observed-map unknown-visibility union. An alternate must expose a new predicted viewing azimuth for existing observed object support. New G and New S share this support, the route pool, safety, and all costs. The currently available support clusters come from the same visible-instance frontend fed by simulator observations. Independent true geometry is evaluator-only.

## Evidence and reproduction

- [Predeclared criteria](PROTOCOL.json), [development execution freeze](runner/development_freeze.json), [development acceptance decision](measurement/development_decision.json) and [completion audit](completion_audit.json).
- [Episode rows](measurement/results/development/episodes.csv), [actual paired action audit](measurement/results/development/choice_audit.csv) and [paired layout table](results/development/paired_layouts.csv).
- [Paired figure](results/development/paired_quality.pdf), [Chinese manuscript paragraph](RESULTS_ZH.md), and [raw evidence and hash index](EVIDENCE_INDEX.json).

Recompute on the original host with Python 3.12, NumPy, Open3D, SciPy and Matplotlib. The actor and reference receipts bind existing NSO files and native simulator meshes on that host. Use a different result directory to preserve existing records.

```sh
python measurement/evaluate.py development --out /tmp/nso-same-goal-evaluation
python report.py development --out /tmp/nso-same-goal-report
```

The report script can also run from the published summary JSON files without the native simulator. Decompress `episodes.json.gz` and `choice_audit.json.gz` in `measurement/results/development/` to the corresponding `.json` paths. The `evidence/` archives preserve the original bytes of the per-arm steps and local choice records. The native meshes, belief grids and full global candidate histories remain indexed locally. New actor runs require a fresh run area and use the sealed `runner/supervise.py` harness. The held-out arm matrix can only be released by the independently verified development gate. Static held-out maps and reference code are preserved separately from any policy outcome.
