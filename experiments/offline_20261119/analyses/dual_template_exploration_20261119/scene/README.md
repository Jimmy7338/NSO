# Controlled unknown-exploration scenes

`controlled_world.py` implements two fixed 10 × 8 m three-branch layouts. The south entrance initially observes 42.606% of each layout's 1,366 start-reachable cells. Side-room targets are initially occluded. `shelf_left` and `shelf_right` swap the same two unscaled source objects, with building geometry, start `(34,25)`, north heading `0` and all sensor settings unchanged. This is controlled synthetic architecture and placement, not an official apartment or a measured natural semantic distribution. The original apartment's negative results remain retained separately.

```python
from controlled_world import ControlledConfig, ControlledFunctionalWorld
world = ControlledFunctionalWorld(ControlledConfig(max_steps=256), seed=20261119,
                                 layout='shelf_left')
frame = world.sense()                 # original RGBDFrame, visible class2=open/class3=closed
scan = world.scan()                   # original PlanarScan
owner = world.sense_visible_instances() # current first-hit ideal instance support only
frame, collision, done = world.step('forward') # or left/right/stop
```

Do not pass the world object to the policy. Pass only the validated packet and, for a declared common-instance frontend, its current visible owner array. Owner IDs are observation support, not a global object list or hidden geometry. The original full V3 two-template module uses a different category dictionary (3=shelf, 2=solid); its caller must explicitly adapt the sensor's existing v10 dictionary. This world does not change the research algorithm.

Evaluator-only APIs are `world.to_evaluator_truth()` (`occupancy`, `reachable`, `robot_valid`, `edge_valid`) and `world.source_scene_for_evaluator()` (trimesh scene, source object records, node metadata). `build_scene(layout)` provides the same assembly without starting a world. The assembled scene is already metric Z-up; `world_from_source` is identity. Each furniture part retains its original mesh and texture; its original Y-up asset/link transform followed by the explicit rigid relocation is recorded in metadata.

Mesh geometry determines full-circle and full-forward-sweep collision in the 0.15–1.2 m band. The entire circle/sweep must have source-floor support. Ground truth uses the identical swept-edge graph; no prefilled known map is exposed to the actor. The camera is horizontal at1.2 m, 120 × 90, 90° horizontal FOV, ideal opaque depth/laser range8 m, zero noise and exact pose. Original source mesh centimetre-to-metre import scaling is preserved; no new object resizing is applied.

Original source31 is a complete open shelving mesh, including open rear bays. Original source118 contains its whole cabinet body, backboard and both sliding doors at zero joint displacement. Fixed interior probes distinguish open/closed geometry; every object has independently fixed reachable front, rear and side views. The shelf is1.982 m tall and the cabinet0.778 m, while the original V3 estimated-template height is bounded0.9–1.5 m and includes a hypothesized backboard. These are coarse structural hypotheses with explicit model mismatch, not exact ground-truth templates.

`PROTOCOL.json` fixes geometry and acceptance criteria before any policy run; `acceptance.json` records mesh probes, first-packet validation, topology and fixed-view visibility. `validate_scene.py` executes geometry/sensor checks only. `--help` never executes a batch; `controlled_world.py --prepare --layout shelf_left` constructs one world and its initial sensor frame. No NSO policy, robot, ROS node, video or training is invoked by these entry points.
