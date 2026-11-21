# NSO physical evidence, 4 November 2026

This snapshot records one completed onboard joint-exploration episode and its negative semantic-effect diagnosis. It does not replace the repository's historical method or present an efficiency benchmark.

- [Episode audit](episode_audit.json): 40 paid actions, 40 verified ROS arrivals, original grid/heading anchor restored; CAN-estimated 6.835 m in 527.663 s. Observed-area proxy 72.84→120.00 m². One scene, no independent reconstruction accuracy reference.
- [Semantic diagnosis](semantic_diagnosis.json): 12 planning events have equal S/G scores and top choices; adjacent observation overlap exists, but background/object mask support can enter the same track. Excluding floor/wall from the offline object index retains 469 rather than143 of483 marked points at action26; candidate S/G scores remain equal.
- [Manuscript excerpt](manuscript_robot_evidence.md): updated physical evidence, retaining earlier failed sessions and the failed Nov03 standard-ROS return.

The two-filter candidate relaxation and object-only grouping are OFFLINE proposals. Production candidate selection, scoring, original circular research footprint and return contract were not changed. Common ROS collision protection and hardware protection remained active. Automatic reverse authorization remains limited to same-target endpoint correction.

`native_driver_fault:8` was recorded near episode shutdown. Its cause remains unestablished; fresh zero chassis feedback was confirmed afterward. Do not label the run fault-free.

The archived raw observations, masks and source implementation remain on the controller; the source hashes in [manifest.json](manifest.json) identify them. This lightweight publication is a reviewable evidence snapshot, not yet a standalone reproduction package. Large raw sensor data and detector weights are intentionally not duplicated here. The local 25-page PDF preview is not included in this text-only upload.

Next work is to validate a geometrically accessible semantic comparison and prepare the matching reproducible inputs, without tuning semantic weights to manufacture a winner. Unknown-space coverage remains the primary exploration objective.
