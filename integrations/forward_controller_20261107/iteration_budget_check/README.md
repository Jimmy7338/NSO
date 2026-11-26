# Native ICP iteration-budget negative result

Only the existing `max_iterations` value differs (10 versus 50); the actual 06 ref096/previous098/target099 state and original scan, prediction and guard audit are otherwise unchanged. The fixed-state jump is 50.106 mm at 10 and 50.159 mm at 50. Default-reference recursive replay still exceeds the original bound at indices 99 and 100 with either budget. More iterations do not resolve this captured failure; no accuracy or convergence guarantee follows. Original results and summary are retained; timing measurements are not a benchmark.

For a local ROS Noetic environment with the already-published probe06 ring and recursive native source, first build `instrumented_csm_candidate/sequence_abc07/native_sequence.cpp` using its `build.py` and export `SEQUENCE_OUTPUT_DIR`. Then from this integration directory:

```sh
export ITERATION_OUTPUT_DIR="$(mktemp -d)"
g++ -std=c++14 -O2 -fno-access-control iteration_budget_check/state_fixture.cpp -o "$ITERATION_OUTPUT_DIR/state_fixture" -L/opt/ros/noetic/lib -llaser_scan_matcher $(pkg-config --cflags --libs roscpp tf pcl_ros gsl)
/usr/bin/python3 -B iteration_budget_check/run.py
/usr/bin/python3 -B iteration_budget_check/summarize.py
```

The publication copy reads the package-relative probe06 and stored matcher parameters, writes only to the explicit output directory, and uses isolated localhost master 11404. The originating-machine rerun reproduced the fixed jumps and both recursive violations. Binary libraries and a clean ROS environment are not supplied; no live robot is involved.
