# External dependency source map

Observed source locations on the recorded machine (not included; exact revisions not established):

- `/home/ouc/scout_nav/src/lslidar_n301`: lslidar driver package 5.1.2; license metadata says TODO. Upstream/revision not established.
- `/home/ouc/scout_nav/src/scout/scout_ros/scout_description`: package 0.4.2; exact upstream/revision not established.
- `/home/ouc/scout_nav/src/zed-ros/zed-ros-wrapper`: ZED wrapper package 3.5.0, https://github.com/stereolabs/zed-ros-wrapper; proprietary ZED SDK 5.0.0 separately installed.
- `source path not established`: https://github.com/stereolabs/zed-ros-interfaces; exact revision not established.
- Local navigation and gmapping overlays: `/home/ouc/scout_nav/src/nav/src/pakage/navigation`, `/home/ouc/scout_nav/src/nav/src/pakage/slam_gmapping`, and `/home/ouc/scout_nav/src/nav/src/pakage/openslam_gmapping`; exact revisions not established. ROS distribution packages and exact observed versions are listed in README.md. Upstream https://github.com/ros-planning/navigation and https://github.com/ros-perception/slam_gmapping.
- CSM and laser_scan_matcher sources were not located; ROS Noetic package versions are in README.md.

Paths describe the recorded environment and are not portable installation instructions. Hardware calibration, the ZED SDK, CUDA and robot-specific configuration are external.
