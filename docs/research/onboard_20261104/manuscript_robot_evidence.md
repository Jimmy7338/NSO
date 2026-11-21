# Physical evidence excerpt — 4 November 2026

This is the updated physical-evidence section of the local manuscript, not a new main-method definition. The main controlled study uses a supplied graph and two templates; the robot episode uses the declared V10.3.1 joint-exploration deployment variant. Figures for earlier sessions are omitted from this standalone excerpt.

## 6 Physical Integration and the Remaining Transfer Gap

The project has progressed from a prospective ROS interface to measured robot integration. The platform uses ROS 1 navigation, a ZED camera, visual mapping, a bounded velocity interface, and actual chassis feedback. The standard laser-mapping/navigation chain has been exercised, and the visual chain produces maps and navigation inputs. These engineering records do not yet supply a matched physical comparison with the simulation policies: the simulation assumes exact poses and a supplied safe graph, whereas the deployed perception and local navigation can fail before an observation goal is reached.

**Table 7. Recorded robot execution by implementation stage; these are separate sessions, not matched performance trials.**

| Configuration / session | Recorded motion and goal outcome | Measured processing | Evidential scope |
|---|---|---|---|
| Geometry-only bridge, 28 October | 5 arrivals / 7 goals; CAN-estimated path 2.878 m; 146.219 s | 18 acquired frames and 18 TSDF integrations | A geometric observation loop can issue goals, move, and acquire new data; last two goals fail local planning |
| Full structural stack, subsequent session | 0 arrivals / 8 goals; 0 m; 94.003 s | 7 local-planner failures and 1 cancellation | No arrival was achieved in this session; later deployments differ |
| Full structural stack, second session | 0 feasible candidates / 0 issued goals and 0 arrivals; 0 m; 22.678 s | Candidate feasibility fails before motion | Candidate generation and navigation feasibility remain limiting |
| Three-frame recorded-image diagnostic, 29 October | No physical motion trial | 3 detector calls, 3 TSDF integrations, 8 forecasts; one person association | Perception/planning modules execute on real data, but no qualified semantic activation or continuous deployment is established |
| Standard ROS viewpoint diagnostic, 3 November | Return goal failed after 0.261 m of CAN-estimated travel; zero feedback confirmed after shutdown | Local controller reported invalid control | A separately commanded setup movement, not an NSO policy action; failure and incomplete return are retained |
| Joint exploration integration, 4 November | 40 verified ROS arrivals and 40 paid actions; returned to the starting grid state; 6.835 m CAN-estimated travel; 527.663 s | 41 accepted observations including the initial frame; 12 planning events | One partially known physical scene; feasibility evidence for a declared deployment variant, with a native driver fault recorded near the end |



The travel distances in Table 7 come from chassis feedback, not an external trajectory reference. Goal attempts, frames, and planning events within an episode are not independent robot experiments. On 4 November, the recorded two-dimensional observed-area proxy increased from 72.84 to 120.00 square metres. This is an accumulated map proxy in a partially known scene, without an independent ground-truth map or a matched baseline; it does not establish exploration efficiency or mapping accuracy. The final verified ROS goal was 0.049 m and 0.054 rad from its commanded pose. After the last action, the adapter recorded `native_driver_fault:8`; its cause and physical effect were not established. Zero chassis feedback was confirmed after shutdown, without an independent physical stop measurement.

The deployed exploration variant uses the V10.3.1 decision framework with revised unknown-space candidate generation, real-scene semantic mapping, and declared sensor-range and measured-extent adapters. Its CSM localization, controller odometry projection, and scoped endpoint-retry capability are physical execution adaptations. This configuration differs from the supplied-graph, two-template controlled study in Sections 3–5; a completed robot episode does not establish equivalence to that study or transfer its simulated benefit. Across the 12 distinct planning events in the 4 November episode, the recorded S and G scores were equal for every shared candidate, so their within-state top choices were also equal. This diagnostic is not a separately executed G trajectory and cannot measure a physical semantic advantage.

An offline analysis of the 41 accepted frames found measured-voxel overlap in 28 of 40 adjacent frame pairs and none in the other 12. These are repeated observations within one episode, and the predicted masks and voxel overlap do not establish object identity or recognition accuracy. The 3 November viewpoint diagnostic also retained a failed return after 0.261 m: a standard ROS setup goal was rejected after local-control failure. It was separately commanded and is not credited as an NSO action. Together with the earlier failed sessions, this result shows why a completed later episode should not be read as universal deployment reliability.

A physical comparison still requires matched semantic and geometry-only runs under the same scene, initial information, sensor rules, protection settings, and time budget. It should retain all failures and distinguish measured new coverage and travel cost from surface accuracy against an independent reference. The present physical evidence supports a bounded closed-loop integration case, not semantic efficacy, reconstruction accuracy, or superiority over standard ROS exploration.

