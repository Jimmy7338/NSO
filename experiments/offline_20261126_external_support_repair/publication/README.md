# External-scan floor-support repair evidence

Run the following command from this study's root after the sealed jobs and evaluator have finished.

```sh
python3 publication/stage_publication.py
```

The staging script uses Python 3.9 or newer and writes `publication/payload/` and `publication/upload.json`. The manifest follows the existing draft PR uploader format and gives the Git blob ID, SHA-256, and byte count for each staged file. The intended repository directory is `experiments/offline_20261126_external_support_repair/`. Review the manifest and the final metrics before publishing. Staging is local and preserves preceding study files.

The stage requires terminal records for all 12 combinations of the frozen four starts and three policy arms at 192 actions. A terminal policy failure is part of the matrix. The unmodified protocol, run receipt, validation evidence, final summary, common-mask sensitivity, and aggregate CSVs stay readable in the tree. The ZIP holds compact native decision, action, checkpoint, launch, and exit records for the retained attempts. SHA-256 metadata and selected scalar diagnostics record how the source evidence was examined.

The published evidence contains no scene assets, downloaded navigation meshes, textures, sensor frames, reference geometry, or reconstructed meshes. Reproduction requires obtaining the original datasets independently under their respective licenses. The previous external-scene study and its results remain in their original record. [Original draft PR evidence](https://github.com/Jimmy7338/NSO/pull/1)
