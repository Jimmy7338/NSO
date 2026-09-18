"""V25 candidate family: physical, class-independent service partitions.

Only simulator geometry changes. V24.1 sensor configuration, cabinet bodies,
attachments, markers, task bounds and proposed front/outer-side prefix stay
unchanged. Truth and legacy service metadata are evaluator-only; no planner
may use owner IDs or these geometry declarations for instance separation.

This first revision is an unvalidated candidate in the graduation 14-day
window. Construction checks physical spacing/safety, not paid coverage or Q.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.ndimage import label

from env.facility_choice_v24 import ASSIGNMENTS_V24, LAYOUTS_V24, prefix_spec_v24
from env.facility_choice_v24_1 import FacilityChoiceWorldV24_1
from env.virtual3d_inspection_v4 import union_surface_from_boxes
from utils.grid_geometry import inflated_obstacles


VERSION_V25 = 'two-facility-physical-partitions-v25-r0'
PARENT_SOURCE_V25 = {'D25-P00': 'D24-P00', 'D25-P01': 'D24-P01'}
PARENTS_V25 = tuple(PARENT_SOURCE_V25)
ASSIGNMENTS_V25 = tuple(ASSIGNMENTS_V24)
LAYOUTS_V25 = {name: deepcopy(LAYOUTS_V24[source]) for name,source in PARENT_SOURCE_V25.items()}
MIN_PARTITION_ASSET_GAP_M = .2
PARTITION_HEIGHT_M = 2.4


def prefix_spec_v25(parent='D25-P00'):
    value = deepcopy(prefix_spec_v24(PARENT_SOURCE_V25[parent]))
    value.update(parent=parent, source_parent=PARENT_SOURCE_V25[parent],
                 world_version=VERSION_V25, static_safety_revalidation_required=True)
    return value


def partition_boxes_v25(parent='D25-P00'):
    """Literal single-design physical panels; dimensions never depend on class."""
    panels = []
    for asset_id,(cx,front) in enumerate(LAYOUTS_V25[parent]['fronts']):
        inward = 1 if asset_id == 0 else -1
        def panel(role, low_x, high_x, low_y, high_y):
            # Local x is positive towards this cabinet's inward attachment side.
            a,b = sorted((cx + inward*low_x, cx + inward*high_x))
            panels.append(dict(asset_id=asset_id, role=role,
                primitive=(a,front+low_y,0.,b-a,high_y-low_y,PARTITION_HEIGHT_M,1)))
        panel('inner_long', 1.4,1.6,-.8,2.4)
        panel('inner_front_return', .8,1.6,-.4,-.2)
        # The body rear edge is at +.8: keep that outer-side edge in front of
        # the panel and provide a farther real background, rather than placing
        # a near occluder across the body's observed 0.8 m depth.
        panel('outer_long', -1.4,-1.2,1.,2.4)
        panel('outer_front_return', -1.4,-.8,1.,1.2)
        panel('rear_offset_screen', -.2,0.,1.,2.4)
    return panels


def box_separation_m(first, second):
    """Euclidean distance between two closed axis-aligned physical boxes."""
    first,second = np.asarray(first[:6],float),np.asarray(second[:6],float)
    gap = np.maximum(np.maximum(first[:3]-(second[:3]+second[3:]),
                                second[:3]-(first[:3]+first[3:])),0.)
    return float(np.linalg.norm(gap))


class StaticGeometryErrorV25(ValueError):
    def __init__(self, audit):
        self.audit = deepcopy(audit)
        super().__init__('V25 first-design physical safety/spacing failed: '+str(audit))


class FacilityChoiceWorldV25(FacilityChoiceWorldV24_1):
    """One family, two parent layouts; inherited physical sensors unchanged."""
    def __init__(self, parent='D25-P00', assignment='A_complex_B_simple',
                 sensor_model='iid_025px', noise_seed=1901, semantic_condition='aligned'):
        if parent not in PARENTS_V25 or assignment not in ASSIGNMENTS_V25:
            raise ValueError('Undeclared V25 candidate parent or class assignment')
        source_parent = PARENT_SOURCE_V25[parent]
        super().__init__(parent=source_parent, assignment=assignment,
            sensor_model=sensor_model, noise_seed=noise_seed, semantic_condition=semantic_condition)
        original_config = self.config
        original_assets = deepcopy(self.objects)
        original_markers = deepcopy(self._physical_markers)
        old_regions = deepcopy(self.service_regions)
        task_primitives = [p for p in self._solid_primitives if p[6] in (100,101)]
        panels = partition_boxes_v25(parent)
        spacing = []
        for row in panels:
            primitive = row['primitive']
            distance = min(box_separation_m(primitive,p) for p in task_primitives)
            spacing.append(dict(asset_id=row['asset_id'],role=row['role'],minimum_asset_distance_m=distance))
            self._solid_primitives.append(primitive)
            x,y,_,sx,sy,_,_ = primitive
            self.boxes.append((x,y,sx,sy))
        self.mesh,self.triangle_owners,self.surface_measure_audit = union_surface_from_boxes(self._solid_primitives)
        self.triangle_classes = self.triangle_owners.copy()
        for item in self.objects:
            self.triangle_classes[self.triangle_owners == item['owner']] = item['category']
        self.occupancy = np.zeros(self.shape,bool)
        resolution = self.config.resolution_m
        for x,y,sx,sy in self.boxes:
            c0,c1 = math.floor(x/resolution+1e-8),math.ceil((x+sx)/resolution-1e-8)
            r0,r1 = math.floor((self.height-y-sy)/resolution+1e-8),math.ceil((self.height-y)/resolution-1e-8)
            self.occupancy[max(0,r0):min(self.shape[0],r1),max(0,c0):min(self.shape[1],c1)] = True
        self._blocked = inflated_obstacles(self.occupancy,self.config.robot_radius_m/resolution)
        groups,_ = label(~self._blocked)
        anchor_safe = not bool(self._blocked[self.start])
        self.reachable = groups == groups[self.start] if anchor_safe else np.zeros(self.shape,bool)
        self.parent = parent
        self.prefix_proposal = prefix_spec_v25(parent)
        blocked_prefix = [i for i,(r,c,_) in enumerate(self.prefix_proposal['states']) if self._blocked[r,c]]
        self.static_geometry_audit = dict(world_version=VERSION_V25,partition_count=len(panels),
            minimum_partition_asset_distance_m=min(r['minimum_asset_distance_m'] for r in spacing),
            minimum_required_distance_m=MIN_PARTITION_ASSET_GAP_M,
            reference_08m_voxel_26_neighbor_diagonal_m=.08*math.sqrt(3),partition_spacing=spacing,
            spacing_passed=all(r['minimum_asset_distance_m'] >= MIN_PARTITION_ASSET_GAP_M-1e-9 for r in spacing),
            anchor_safe=anchor_safe,safe_floor_cells=int((~self._blocked).sum()),
            reachable_safe_floor_cells=int(self.reachable.sum()),
            disconnected_safe_floor_cells=int(np.count_nonzero((~self._blocked)&~self.reachable)),
            blocked_proposed_prefix_action_indices=blocked_prefix,
            sensor_body_marker_and_task_bounds_unchanged=(self.config==original_config
                and self.objects==original_assets and self._physical_markers==original_markers),
            actual_noisy_instance_component_separation_verified=False,
            online_coverage_or_shape_quality_verified=False)
        # Former target poses can remain physically safe yet lose their intended
        # sight lines. Never silently reuse their old service-cost proof.
        self.legacy_service_region_audit = []
        for region in old_regions:
            safe_states = [p for p in region['states'] if self.reachable[tuple(p[:2])]]
            self.legacy_service_region_audit.append(dict(asset_id=region['asset_id'],view=region['view'],
                original_pose_count=len(region['states']),currently_safe_pose_count=len(safe_states),
                original_states=region['states'],visibility_revalidated=False,
                old_service_cost_or_quality_claim_reusable=False))
        self.service_regions = []
        self.service_contract = dict(status='not_declared_for_v25',truth_only=True,
            reason='Old V24 service poses/visibility/cost have not been revalidated after physical partitions')
        self.sensor_metadata.update(world_version=VERSION_V25,source_parent=source_parent,
            partition_primitives=[dict(r) for r in panels],class_independent_partitions=True,
            sensor_and_original_task_geometry_unchanged=True,task_asset_count=2,
            old_service_proofs_reused=False,static_only_development_candidate=True)
        if (not self.static_geometry_audit['spacing_passed'] or not anchor_safe
                or self.static_geometry_audit['disconnected_safe_floor_cells'] or blocked_prefix):
            raise StaticGeometryErrorV25(self.static_geometry_audit)
