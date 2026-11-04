# Joint timing instrumentation

Existing journal events gain timing fields only. No timestamps are inserted into sensor NPZ or receipt data. No policy, protection, sensor selection, heartbeat or call ordering is changed. Successful observations attach `capture_timings` to `NSO_joint_observation_accepted`; successful `NSO_joint_rpc` entries have `rpc_elapsed_s`, including `observe` and `next_action`. Failure continues to raise through the existing path, so incomplete phases need not have timing events. Every value is a `time.monotonic()` elapsed duration in seconds.

- `reference_capture_total_s`: full live sensor collector call; includes waiting, validation, TF, conversion, provenance, serialization and compressed source NPZ write.
- `reference_wait_and_validate_s`: start through successful iteration reaching historical TF; includes waiting, validation and any previously rejected or retried iterations (including their TF work).
- `reference_historical_tf_s`: seven historical TF queries and associated receipts in the successful iteration only.
- `reference_npz_compress_write_s`: source NPZ compression/write, excluding earlier conversion/receipt serialization.
- `joint_capture_total_s`: complete joint-map capture call, including wait, TF, resampling, source read, metadata, both NPZs and hashing.
- `joint_map_resample_s`: conservative map resampling function only.
- `joint_source_npz_read_s`: source file read and NPZ decompression/array copy, excluding receipt JSON parsing.
- `joint_map_npz_compress_write_s`: original map NPZ compression/write; excludes subsequent read-back SHA.
- `joint_input_npz_compress_write_s`: final joint NPZ compression/write.
- `rpc_elapsed_s`: `worker.call` duration including worker execution, transport and its existing heartbeat processing; not pure planner or inference CPU time.

Totals contain subfields; subtracting all subfields does not isolate a single operation, and the fields must not be summed as independent contributions. No goal timing, motion or ROS process is started by this change. Backup files and exact unified diff are retained alongside this file.
