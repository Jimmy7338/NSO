# V10.3 independent replay

Passed 40/40 compact branches. The post-hoc verifier did not call the planner or runtime. It regenerated the physical prefix and future sensor packets from saved actions, rebuilt every TSDF map, recomputed surface area and F1, and checked collision-free return to the exact prefix anchor.
