# Official TARE ARM64 feasibility probe

The pinned source is `caochao39/tare_planner@44500592b86138257273e0cab264e6a847ccefc7`. The GitHub source archive matches the previous audited SHA-256 `a3adfcab09a669a2a1b9bf0aaec734171cc46ce0fa5cc16a3b54fc51fdb36024`. The earlier V39 restore script expected x86_64 system libraries and an old binary. This probe builds the original planner source against the Jetson's ROS Noetic and the official Google OR-Tools Linux AArch64 C++ package for the exact same upstream version, v9.8.3296. The latter's archive SHA-256 is `f1cbe084697087e057e815dfe13e3f8ace0f39114b9ee2bbe7a4baba338d769e`. The native shared library is AArch64, requires at most GLIBC 2.25 and GLIBCXX 3.4.22, and `ldd` resolves every dependency on this host.

Only the upstream `src/tare_planner` package is staged. Exclude its packaged x86 OR-Tools tree and all links to other workspaces. Link `catkin/src/tare_planner/or-tools/include` and `catkin/src/tare_planner/or-tools/lib` to `include` and `lib64` from the official ARM64 release. The source code and CMake settings remain unchanged.

```sh
curl -fLsS https://codeload.github.com/caochao39/tare_planner/tar.gz/44500592b86138257273e0cab264e6a847ccefc7 -o tare_4450059.tar.gz
curl -fLsS https://github.com/google/or-tools/releases/download/v9.8/or-tools_aarch64_AlmaLinux-8.8_cpp_v9.8.3296.tar.gz -o ortools_arm64_v9.8.tar.gz
source /opt/ros/noetic/setup.bash
export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10
export LD_LIBRARY_PATH="$PWD/or-tools_aarch64_AlmaLinux-8.8_cpp_v9.8.3296/lib64:${LD_LIBRARY_PATH:-}"
timeout 220 catkin_make -C catkin -j3 -l4 -DPYTHON_EXECUTABLE=/usr/bin/python3 tare_planner_node > catkin_build_first_attempt.log 2>&1
timeout 120 catkin_make -C catkin -j3 -l4 -DPYTHON_EXECUTABLE=/usr/bin/python3 tare_planner_node > catkin_build.log 2>&1
timeout 35 /usr/bin/python3 -B smoke_arm64.py > smoke_arm64.log 2>&1
```

The first build reached the main coverage planner and stopped at its 220-second timeout. The incremental continuation linked the unchanged upstream source successfully. The native executable is AArch64 and its SHA-256 is `a41fd3967455594eed3f3eafbb5e130d89b5d9077ea0a19b1178c2af7435258e`. All libraries resolved (see `node.ldd.txt`).

The sensor smoke used five archived RGB-D plus planar laser pairs from an earlier simulation prefix. It kept upstream indoor parameters, fixed native visibility and the five-scan keypose cadence. Each acquired frame contributed 3,119–6,829 valid RGB-D points plus 95 planar-lidar hits. The node emitted its first post-sensing `/way_point` after scan five, namely `(3.67953, 8.68894, 0.8) m` in `map`. The private loopback ROS master and native node exited after the probe. The packets, selected upstream parameters and native path messages are listed in [the smoke receipt](smoke_receipt.json). This result verifies transport and planning output. It records zero new simulator actions and provides no four-floor coverage, reconstruction or execution result.

Formal comparison requires routing native waypoints through the same acquired-map action and return guard. The archive's old public route graph cannot be used for unknown-space exploration. A small separate adapter can inject a waypoint policy at the frozen runner's `frontier_adapter` hook and feed the native bridge from the current packet, once per action ID. Convert the native waypoint to the shared frame and keep its execution result, next-sensing packet, collision checks and explicit stop reason in the native ledger. A failed or rejected waypoint must remain in the record.

The upstream five-scan keypose cadence and its 360°×24° LiDAR coverage model are separate constraints. In the earlier V39 transfer, the wrapper took up to four paid turns to gather distinct startup views. Any continuation of that strategy must record those real actions explicitly. The current 90° camera plus planar lidar cannot supply native 3D coverage in all directions. A result from unchanged native TARE with transferred sensor data would describe that specific sensor transfer. A future native-sensing comparison would need to add the same physically simulated 3D lidar to all compared methods, then freeze a distinct protocol.
