# Physical frontier preparation snapshot

This is an offline same-state selection probe from one recorded S40 physical integration episode, not a frontier trajectory or a real-world efficacy comparison. The initial reconstructed map matched an archived immediate-planning hash; paid actions 20 and 40 have no archived immediate-planning hashes and therefore lack this exact verification. At action 40 the recorded remaining budget is zero.

The frozen script and result are included for provenance. They require 41 original recorded NPZ packets, cached semantic frontend, journal, calibration/self-mask and matching sources. These raw inputs are not distributed here, so this package does not support complete public independent replay. `result.json` retains packet/cache hashes, configuration and state observations; `MANIFEST.json` lists prerequisites and original artifact hashes. This analysis performed no robot movement or ROS interaction.

Independent local verification checked the 41 packet hashes, the initial belief hash and 72.84/120 m² coverage. Of 107 archived sources, 106 matched; the known `shared_observed_prefix.py` difference remains. The mapper, range adapter, frontier, guard and route sources matched the archive. See `root_verification.json`; later snapshots still lack original immediate-planning hashes.
