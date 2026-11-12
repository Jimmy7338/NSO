# Bounded paid36 entry review

No native paid36 belief/state snapshot is retained by the original graph audit.
The saved input navigation grid is NOT the native mapper belief: SHA256
59b1f4952b7e4426b9ee1f91c05230166f4d3119a7b9238e4d12af21ca8098f7 versus
required native 43786e87f6c6d78b9a6cb2fcc223e4a342c2ffaac35680e219db2c1f978414a7.
The mapper fuses original RGBD/laser history and supplied navigation using
np.maximum, so substituting the navigation grid would change the graph.

`diagnose_saved.py` ran without replay/inference/ROS. It extracts original pure
camera_pose/aperture_support/orientation_graph functions using AST, avoiding
Open3D/model imports. `result.json` records all ten group5 entry-target cells
within the original .35m projection tolerance. All ten have positive original
wanted-heading aperture. Seven are necessarily blocked even using only supplied
occupied cells, which remain occupied in the native fusion. The other three
([203,236], [204,236], [205,236]) are unknown in supplied navigation and are not
occupied-inflation-blocked by that input alone. Their native belief/safety and
connectivity cannot be inferred. Therefore a complete hard geometric exclusion
of the entry neighborhood is NOT established, nor is entry feasibility.

The minimum-cost witnesses remain cost78/error2.8738715m and
cost88/error2.9426412m; these are not minimum-projection witnesses.
Nearest connected wanted-heading aperture and its costs remain unavailable
without the native belief.

Prepared but NOT executed: `restore_paid36.py --run-offline` restores the original
37-frame state using the existing exact-baseline reconstruction and cached
semantic masks, saving only into this directory. It requires the production
science312 Python environment and should run only after robot work stops.
Then run `diagnose_saved.py --belief paid36_native_belief.npy` with an absolute
snapshot path to compute the nearest connected wanted-heading aperture without
budget filtering. Hash assertions prevent substitution of another belief.

No policy, threshold, candidate, score, guard or production file changed.
