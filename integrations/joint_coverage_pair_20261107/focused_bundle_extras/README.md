# Scope of additional records

`SEMANTIC_CANDIDATE_PROPOSAL.json` records a proposed method variant; it was not implemented or enabled. It is not an experiment or evidence of improved performance. Its referenced `C_prefix21_availability/availability.json` is in the parent directory.

`run_lidar_probe_once.sh` is the exact historical machine-specific entry point. It contains absolute local paths, connects to the live ROS master and uses sudo to capture; it is provenance, not a portable offline reproduction command. The `raw_scan_probe/` README and synthetic fixture describe the separately preserved tooling. No captured real window is included here.

`C_prefix21_availability/audit.py` and `semantic_chain/audit_chain.py` retain original local source/workspace paths. Their results and source hashes are preserved; running them on another machine requires path adaptation and the recorded source dependencies. Figure provenance in `docs/thesis/submission_focus_20261107/figure_source/figure_provenance.json` likewise records original acquisition paths, not public repository paths. The actual public C/S canonical frames and result data are under `integrations/joint_coverage_pair_20261107/`.
