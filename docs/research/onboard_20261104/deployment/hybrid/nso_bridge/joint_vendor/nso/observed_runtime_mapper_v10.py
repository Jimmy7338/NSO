"""V10 sensor mapper preserving measured obstacles at the robot's own cell.

The older mapper marks its sensor origin free after inserting obstacles. That
convention can erase a short laser/depth hit, making an unsafe footprint pass
the execution guard. This adapter preserves that observed conflict; it changes
neither the historical mapper nor TSDF reconstruction or quality evidence.
"""
import numpy as np

from nso.semantic_completion_v3 import SemanticHistoryMapperV3


class ObservedRuntimeMapperV10(SemanticHistoryMapperV3):
    def __init__(self, shape, config, truncation_m=.12):
        super().__init__(shape, config, truncation_m)
        self.current_footprint_conflict = False
        self.current_footprint_conflict_details = None
        self.current_footprint_conflict_count = 0

    def update(self, frame, scan=None):
        frame.validate()
        origin = self.grid_cell(frame.world_from_camera[:3, 3])
        inside = all(0 <= value < limit for value, limit in zip(origin, self.shape))
        reasons = []
        if inside and self.belief[origin] == 1:
            reasons.append("previously_observed_occupied_current_cell")

        if inside and scan is not None:
            ranges = np.asarray(scan.ranges_m)
            hit_ids = np.flatnonzero(np.isfinite(ranges) & (ranges > 0)
                                    & (ranges < scan.range_max_m - 1e-3))
            for index in hit_ids:
                angle = scan.angle_min_rad + int(index) * scan.angle_increment_rad
                distance = float(ranges[index])
                local = np.array([np.cos(angle) * distance,
                                  np.sin(angle) * distance, 0.])
                point = scan.world_from_laser[:3, :3] @ local + scan.world_from_laser[:3, 3]
                if self.grid_cell(point) == origin:
                    reasons.append("current_scan_hit_in_current_cell")
                    break

        # Match the legacy mapper's actual obstacle insertion support exactly.
        # Height is world z; this is independent of the optical depth value.
        if inside:
            points, _ = frame.points(stride=2)
            obstacle_points = points[(points[:, 2] > .15) & (points[:, 2] < 1.2)]
            if any(self.grid_cell(point) == origin for point in obstacle_points):
                reasons.append("current_depth_obstacle_in_current_cell")

        super().update(frame, scan)
        self.current_footprint_conflict = bool(reasons)
        self.current_footprint_conflict_details = dict(
            map_version=int(self.frames), timestamp_s=float(frame.timestamp_s),
            current_cell=list(origin), conflict=bool(reasons), reasons=reasons,
        )
        if reasons:
            self.belief[origin] = 1
            self.visible[origin] = True
            self.current_footprint_conflict_count += 1
