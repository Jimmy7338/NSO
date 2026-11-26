# ARiADNE with the shared NSO simulator

The comparison runs the official ARiADNE graph builder and released policy weights in a persistent CPU process. The policy input contains the observed occupancy grid, the common footprint inflation, and the current robot cell. The frozen NSO simulator supplies observations and evaluates the surfaces fused along each executed trajectory.

The adapter executes the selected graph waypoint through the common four-connected driver. It takes the shortest observed-free route to the exact selected cell and keeps the heading of the final translation. The common driver handles collision checks, the return, and the 192-action budget. If the graph provides no usable waypoint, the run requests the common observed-map return. This implements the ARiADNE policy with shared execution under accurate simulated poses. The upstream ROS waypoint scheduling and its local controller are outside the treatment.

Grid spacing and sensor range follow the frozen simulation, at 0.20 m and 8 m. The comparison uses the upstream default 2 m graph spacing and one common configuration for every floor. A CPU smoke check established that the native checkpoint produced finite, nonzero-utility proposals on all four original observed start maps at the selected spacing. No formal trajectory outcome was used to set the spacing.

## Reproduction on the Jetson

Keep the layout of the sibling experiments and use the two existing Python environments. System Python 3.8 provides ARM PyTorch, rospy, NumPy, SciPy, matplotlib and scikit-image. The frozen experiment runs with science312. From this directory, restore source at the recorded revision if it is missing.

```sh
git init source
git -C source fetch --depth 1 https://github.com/marmotlab/ARiADNE-ROS-Planner.git 773ebcf60334b2a16a865f1207938f53fee92031
git -C source checkout --detach FETCH_HEAD
```

Run each case once in a new output directory. The driver reads the published official actor and its checkpoint and verifies every file listed in `PROTOCOL.json` before running.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nso_python=/home/ouc/nso_robot/envs/science312/bin/python
output=execution/reproduction
mkdir -p "$output"
for floor in ind26_vertical_spine ind26_bent_corridor ind26_offset_ring ind26_double_loop; do
    "$nso_python" run.py "$floor" --out "$output/$floor" > "$output/$floor.log" 2>&1
    printf '{"exit_code":%d}\n' "$?" > "$output/$floor.exit.json"
done
```

The upstream laboratory publishes the actor and weights under the MIT license. [Official ARiADNE implementation](https://github.com/marmotlab/ARiADNE-ROS-Planner/tree/773ebcf60334b2a16a865f1207938f53fee92031) · [ICRA 2023 paper](https://arxiv.org/abs/2301.11575)
