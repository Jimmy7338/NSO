The original `README.md` and `freeze_manifest.json` describe the original study layout and source hashes. The publication copy of `run_recursive.py` and `build.py` changes paths/output handling and uses isolated master 11403; their hashes therefore differ from the original frozen files. Inputs are the existing published `../../probes/control_can_offset_06/` ring, the accompanying probe07 ring in `../../probes/control_can_offset_07/`, `../../can_source_audit07/intervals.json`, and archived matcher parameters from `../state_fixture_results.json`. No duplicate probe06 or stationary scans are included.

From this directory in an existing compatible ROS Noetic environment:

```bash
export SEQUENCE_OUTPUT_DIR="$(mktemp -d)"
/usr/bin/python3 -B build.py
/usr/bin/python3 -B run_recursive.py 06
/usr/bin/python3 -B run_recursive.py 07
```

All generated files stay in the separate output directory. The script refuses an occupied port and starts only a localhost master, never the robot master. This relocated code was validated on the original machine: reference changes, violations, max steps and final poses match both archived summaries; timings are naturally variable. It does not establish a clean-machine installation or deploy any adaptation.
