# Recorded cycle timing brackets

Retrospective timestamps from two recorded physical integration episodes, not a new robot run. Navigation, capture and pre-goal intervals are brackets that include multiple costs; they cannot establish a CPU bottleneck or full mission elapsed time. Source event hashes are in `summary.json`; original event files are not distributed here.

Given the exact original `default_S40_02/events.jsonl` and `history_S40_01/events.jsonl`, run `python analyze.py --reports-root /path/to/reports --output-dir /path/to/empty/output`. The 47 rows report `navigation_result_status`; the saved numerical timings remain unchanged. Source instrumentation in the adjacent package was added after S40 and was not present in these historical runs.
