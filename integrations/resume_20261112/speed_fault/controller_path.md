Offline source trace: stop/rotate bypass of DWA configured speed cap

Source tree /home/ouc/scout_nav/src/nav/src/pakage/navigation.

- dwa_local_planner/src/dwa_planner_ros.cpp:286-300 dispatches position-reached execution to LatchedStopRotateController.
- base_local_planner/src/latched_stop_rotate_controller.cpp:121 computes vx = sign(measured_vx)*max(0,abs(measured_vx)-acc_lim_x*sim_period). This function has no max_vel_x argument/clamp. It submits candidate to obstacle_check.
- dwa_local_planner/src/dwa_planner.cpp:230 ignores the bool return from generator_.generateTrajectory.
- base_local_planner/src/simple_trajectory_generator.cpp:198 rejects candidates above max_vel_trans and returns false before trajectory integration; trajectory is already reset empty.
- checkTrajectory proceeds to score that empty trajectory and returns true for nonnegative cost. Scoring initializes total cost0 and obstacle/map grid loops iterate trajectory points, allowing this rejection to be lost.

Navigation config controller_frequency10Hz, acc_lim_x0.25 => stop deceleration subtracts0.025m/s. Postfault observed proposal0.120 is numerically consistent with measured0.145 minus0.025. CAN ring has0.145 at kernel1794478296011620617ns. Proposal3994 raw0.120 received117741.333586602. CAN control odometry pairs measured feedback at/before scan, so exact input for that invocation is not established merely by proximity. Actual branch execution and loaded binary correspondence remain unverified; this source trace explains a feasible mechanism, not proof of unique episode causation.

Repairs proposed offline only: first capture stoprotate branch, input velocity/source stamp, sim_period, generated bool, and raw/applied commands; reproduce max_vel_trans rejection and ensure checkTrajectory honors false. Separately, a named execution consistency adapter profile capping positive command at declared0.05m/s is grounded in existing navigation speed setting, while explicitly changing adapter clipping from0.1 to0.05. It leaves all chassis stop predicates unchanged but is not behavior-equivalent execution. Do not claim existing S96 method unchanged under this adaptation. Threshold relaxation/debounce/glitch rejection remains unsupported.
