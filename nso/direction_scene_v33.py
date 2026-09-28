"""Pure frozen V33 geometry assembly; no observation tables or reward solver."""
import math
from collections import deque

from nso.box_union_geometry_v30 import BoxV30, union_exterior_faces_v30


EPS = 1e-7
HEADING_VECTORS = ((0, 1), (1, 0), (0, -1), (-1, 0))
PREFIX_ACTIONS = ('left', 'forward', 'forward', 'forward', 'right', 'right',
    'forward', 'forward', 'forward', 'forward', 'forward', 'forward',
    'left', 'left', 'forward', 'forward', 'forward', 'right')


def rotate_xy_v33(point, quarter_turns_ccw):
    x, y = point
    for _ in range(quarter_turns_ccw % 4):
        x, y = -y, x
    return (x, y)


def rotate_box_v33(bounds, quarter_turns_ccw):
    corners = [rotate_xy_v33((x, y), quarter_turns_ccw)
               for x in bounds[:2] for y in bounds[2:4]]
    return [min(x for x, _ in corners), max(x for x, _ in corners),
            min(y for _, y in corners), max(y for _, y in corners), bounds[4], bounds[5]]


def swept_safe_v33(start, end, boxes, radius=.2):
    """V29-style exact axis-aligned segment/rectangle distance; tangency allowed.

    Conservative footprint: every solid's full XY projection, even above laser.
    This computes geometric safety only, not sensing or a visibility payoff.
    """
    if start[0] != end[0] and start[1] != end[1]:
        raise ValueError('only axis-aligned translation is declared')
    for b in boxes:
        dx = max(b[0]-max(start[0], end[0]), min(start[0], end[0])-b[1], 0.)
        dy = max(b[2]-max(start[1], end[1]), min(start[1], end[1])-b[3], 0.)
        if dx*dx+dy*dy < radius*radius-EPS:
            return False
    return True


def complete_safe_cells_v33(local_grid, boxes, quarter_turns_ccw, radius=.2):
    candidates = [rotate_xy_v33((x, y), quarter_turns_ccw)
        for x in range(local_grid['x_min'], local_grid['x_max']+1)
        for y in range(local_grid['y_min'], local_grid['y_max']+1)]
    return tuple(sorted(p for p in candidates if swept_safe_v33(p, p, boxes, radius)))


def graph_v33(parent):
    """All safe forward/left/right edges on the full declared integer grid."""
    cells = set(map(tuple, parent['nav_cells']))
    poses = tuple((x, y, h) for x, y in sorted(cells) for h in range(4))
    indices = {pose: i for i, pose in enumerate(poses)}
    boxes = [b for a in parent['hypotheses'][0]['assets'] for b in a['boxes']]
    boxes += parent['background_boxes']
    edges = []
    for x, y, heading in poses:
        dx, dy = HEADING_VECTORS[heading]
        target = (x+dx, y+dy)
        row = []
        if target in cells and swept_safe_v33((x, y), target, boxes, parent['robot_radius_m']):
            row.append(('forward', indices[(*target, heading)]))
        row.extend((('left', indices[(x, y, (heading-1) % 4)]),
                    ('right', indices[(x, y, (heading+1) % 4)])))
        edges.append(tuple(row))
    return poses, tuple(edges), indices[tuple(parent['anchor'])]


def prefix_states_v33(parent):
    poses, edges, state = graph_v33(parent)
    output = [poses[state]]
    for action in parent['prefix_actions']:
        if action not in dict(edges[state]):
            raise ValueError('declared prefix contains an unsafe action')
        state = dict(edges[state])[action]
        output.append(poses[state])
    return tuple(output)


def exact_asset_faces_v33(asset, *, vertical_only=False, split_planes=None):
    return union_exterior_faces_v30([BoxV30(tuple(b), asset['id']) for b in asset['boxes']],
        vertical_only=vertical_only, split_planes=split_planes)


def _parent(identifier, rear_y, body_end_y, ribs, turns):
    local_grid = dict(x_min=-3, x_max=3, y_min=0, y_max=rear_y, spacing_m=1.)
    common = [(-2.75, 2.75, 1.1, 1.5, 0., 2.),
              (-1.8, 1.8, 1.4, body_end_y, 0., .4),
              (-.6, .6, 1.4, body_end_y, .4, 1.8)]
    hypotheses = []
    for h, side in enumerate((-1, 1)):
        extent = (-1.8, -.6) if side == -1 else (.6, 1.8)
        local_boxes = [*common, *[(extent[0], extent[1], y0, y1, .4, 1.8) for y0, y1 in ribs]]
        seed = (*rotate_xy_v33((0., 1.1), turns), .9)
        hypotheses.append(dict(id=h, semantic_cue=dict(type='type_A' if h == 0 else 'type_B'),
            assets=[dict(id=0, boxes=[rotate_box_v33(b, turns) for b in local_boxes],
                         front_seed_xyz=list(seed))]))
    safe = [complete_safe_cells_v33(local_grid, hypothesis['assets'][0]['boxes'], turns)
            for hypothesis in hypotheses]
    if safe[0] != safe[1]:
        raise ValueError('hidden structure changes the complete legal navigation grid')
    return dict(id=identifier, local_grid=local_grid, nav_cells=[list(p) for p in safe[0]],
        robot_radius_m=.2, anchor=[0, 0, (-turns) % 4], prefix_actions=list(PREFIX_ACTIONS),
        total_action_budget=42, prefix_paid_actions=18, remaining_action_budget=24,
        device_frame=dict(origin_xyz=[0., 0., 0.], quarter_turns_ccw=turns,
            local_u='front-view horizontal axis; positive toward the viewer right',
            local_v='front-to-rear equipment axis'),
        geometry_facts=dict(body_end_local_y=body_end_y, ribs_local_y=[list(r) for r in ribs],
            all_front_guard_surface_in_asset_denominator=True),
        background_boxes=[], hypotheses=hypotheses)


def build_direction_scene_v33():
    """Deterministic configuration construction, never search over dimensions."""
    return dict(version='v33-direction-service-scene-r0',
        purpose='one finite analytic directional-information family; not measured SLAM or 3D quality',
        actions=['forward', 'left', 'right'], heading_vectors=[list(v) for v in HEADING_VECTORS],
        graph_contract=dict(domain='all safe integer centres in each full declared local rectangle, then rotated',
            forward_translation_m=1., turn_degrees=90., paid_per_primitive=1,
            exact_return_pose_required=True, allow_tangent_footprints=True,
            swept_safety_epsilon=EPS, no_ring_only_action_restriction=True,
            continuous_free_space_optimum_not_claimed=True),
        geometric_observation=dict(camera_height_m=.9, range_m=4., width=96, height=72,
            fx=48., fy=48., horizontal_fov_deg=90.,
            vertical_fov_deg=math.degrees(2*math.atan(72/(2*48))), ray_epsilon_m=EPS,
            surface_horizontal_spacing_max_m=.5, surface_vertical_spacing_max_m=.5,
            common_split_planes_across_hypotheses_required=True,
            finite_visible_surface_proxy_not_depth_accuracy=True),
        radar_observation=dict(height_m=.25, range_m=8., horizontal_fov_deg=360., rays=180,
            independent_of_camera_range=True, include_in_geometric_signature=True),
        coverage_contract=dict(denominator='all declared reachable safe integer ground centres; none selected by benefit',
            line_of_sight_height_m=.25, visibility_range_m=8., field_of_view_degrees=360.,
            include_visible_ground_in_geometric_signature=True, target_fraction=.80,
            finite_ground_proxy_not_measured_2d_coverage=True),
        signature_contract=dict(visible_geometry='physical coordinates, normal and ideal depth only',
            include_radar_and_ground=True,
            forbidden_fields=['asset_id', 'owner', 'class', 'hidden_patch_id', 'reference_area']),
        public_configuration_model=dict(type_A=dict(service_side_local_u=-1),
            type_B=dict(service_side_local_u=1), direction_is_not_in_observed_cue=True,
            equipment_frame_is_common_public_information=True,
            natural_semantic_network_or_learned_mapping_claimed=False),
        hypothesis_prior=[.5, .5],
        reward_contract=dict(main='C_grid times equal-asset visible vertical exterior area fraction, conditional on C_grid>=0.80 and return',
            owner_is_reference_bookkeeping_only=True, category_weighting=False,
            repeated_observations_use_set_union=True, all_external_vertical_faces_in_denominator=True,
            neither_surface_precision_nor_3d_Q=True),
        solver_limit=dict(wall_seconds=300, maximum_memo_states=1500000),
        geometry_freeze_scope='single r0; no reward or budget sweep; geometry failures retained',
        parents=[_parent('P00', 4, 3.4, ((1.6, 1.8), (2.3, 2.5), (3., 3.2)), 0),
                 _parent('P01', 5, 4.4, ((1.6, 1.8), (2.8, 3.), (4., 4.2)), 1)])
