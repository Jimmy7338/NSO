# Single CAN-sample constant-velocity prior: negative result

For the original probe06 reference096/previous098/target099 state, this isolated fixture compares the unchanged zero prior with the installed matcher's constant-velocity `use_vel` prediction from the actual current valid CAN sample. The zero prior reproduced a 50.105966 mm jump; the constant-velocity prior still produced 50.104664 mm. This holds one sampled velocity constant across the interval as an explicit model, not full-rate integrated motion or independent ground truth; it does not validate or deploy a repair.

First execute the sibling `iteration_budget_check/README.md` procedure to generate `ITERATION_OUTPUT_DIR/fixed_jobs.txt`. Then from this integration directory, with the existing ROS Noetic environment and libraries:

```sh
export PREDICTION_OUTPUT_DIR="$(mktemp -d)"
g++ -std=c++14 -O2 -fno-access-control current_CAN_velocity_prediction06/state_fixture.cpp -o "$PREDICTION_OUTPUT_DIR/state_fixture" -L/opt/ros/noetic/lib -llaser_scan_matcher $(pkg-config --cflags --libs roscpp tf pcl_ros gsl)
/usr/bin/python3 -B current_CAN_velocity_prediction06/run.py
```

The relocated copy reads the retained observed CAN sample and package-relative archived matcher parameters, writes only to the explicit output directory and uses isolated localhost master 11405. It reuses original probe06 input scans via the iteration fixture rather than duplicating them. The originating-machine rerun reproduced both archived output poses, jumps and priors. No live runtime was changed.
