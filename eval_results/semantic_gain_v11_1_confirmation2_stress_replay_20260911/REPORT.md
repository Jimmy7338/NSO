# V11.1 confirmation independent physical and metric replay

Passed 40/40 frozen development branches. The post-hoc verifier did not call the planner, runtime integration, option generator, or learned gain model. It regenerated every prefix and future sensor packet from saved actions, rebuilt each TSDF map, recomputed the primary 2D-area-times-3D-F1 gain and all recorded secondary metrics, and checked collision-free return to the exact prefix anchor.
