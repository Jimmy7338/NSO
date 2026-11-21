# Read-only cross-frame physical diagnostic

The original eight saved RGB-D NPZ packets are not included in this package; public end-to-end reproduction is therefore unavailable. Their filenames and SHA-256 hashes, manual ROIs, recorded intrinsics and transforms are preserved in `result.json` and `audit.py`. The original result retains its historical local directory solely as provenance. A holder of those exact packets can set `NSO_PACKET_DIR` and run `python audit.py`; it writes `result_recomputed.json` and does not connect to ROS.

The manual ROI analysis is consistent with a world-fixed object, but correspondence, calibration error and the action-7 conflict pixel remain unverified. It does not identify the object, establish obstacle safety, justify changing masks or protective stops, or validate reconstruction accuracy. See `result.md`.
