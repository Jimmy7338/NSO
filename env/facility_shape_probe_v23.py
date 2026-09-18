"""Paid single-facility observation fixture for V23 shape-metric validation.

The two variants share one executable, declared route. This is a sensor and
task-mechanism bench, not an autonomous planner or a semantic efficacy result.
Every movement and turn is a normal world.step(); stage markers grant no
observations and no free poses. Existing V19 RGB-D, disparity noise, lidar,
mesh interfaces and exact discrete pose behavior are inherited unchanged.
"""
import math

import numpy as np
from scipy.ndimage import label

from env.facility_documentation_v19 import (
    FacilityConfigV19, FacilityWorldV19, SENSOR_MODELS_V19,
)
from env.virtual3d_inspection_v4 import union_surface_from_boxes
from utils.grid_geometry import DIRECTIONS, inflated_obstacles


SHAPE_PROBE_VERSION_V23 = 'single-facility-paid-shape-probe-v23-1'
SHAPE_KINDS_V23 = ('simple', 'complex')
# All evaluation windows are simulator/evaluator metadata, never planner input.
EVALUATION_BOUNDS_V23 = ((3.08, 3.08, .01), (5.20, 4.72, 1.72))


def route_spec_v23():
    """Return the predeclared common 96-action route, independent of kind.

    Each segment ends after its final paid action has produced a sensor frame.
    `coarse` and `extra` are the primary pre/post observation stages; `front`
    and `rear` are useful descriptive checkpoints. Return is also paid.
    """
    definitions = (
        ('front', [('forward', 3)]),
        ('coarse', [('left', 1), ('forward', 10), ('right', 1),
                    ('forward', 7), ('right', 1)]),
        ('rear', [('left', 1), ('forward', 12), ('right', 1),
                  ('forward', 10), ('right', 1)]),
        ('extra', [('left', 1), ('forward', 10), ('right', 1),
                   ('forward', 12), ('right', 1)]),
        ('returned', [('left', 1), ('forward', 10), ('right', 1),
                      ('forward', 10), ('right', 1)]),
    )
    actions = []
    segments = []
    stage_indices = {'initial': 0}
    for name, runs in definitions:
        start = len(actions)
        paid = [action for action, count in runs for _ in range(count)]
        actions.extend(paid)
        segments.append(dict(name=name, start_action_id=start,
                             end_action_id=len(actions), actions=paid))
        stage_indices[name] = len(actions)
    return dict(version=SHAPE_PROBE_VERSION_V23, actions=actions,
                segments=segments, stage_indices=stage_indices,
                stage_groups=dict(common_coarse=['front', 'coarse'],
                                  additional_side_rear=['rear', 'extra'],
                                  return_to_anchor=['returned']),
                total_paid_actions=len(actions), identical_across_kinds=True,
                action_units='0.2 metre forward or 90 degree turn; one frame per paid action',
                initial_frame_is_common_bootstrap=True,
                stage_evaluation_acquires_no_new_frame=True,
                planner=False, semantic_policy=False,
                full_mission_budget_witness=False)


class ShapeProbeWorldV23(FacilityWorldV19):
    """An 8x8m single closed cabinet, optionally with real exposed attachments.

    The broad front is the actual cabinet body, not an extra masking plate.
    The complex attachments overlap the body/each other to form one connected
    solid, and lie above the lidar plane. Initial occlusion follows the same
    unmodified renderer as all later viewpoints. No visibility override exists.
    """

    def __init__(self, kind='simple', sensor_model='iid_025px', noise_seed=1901,
                 semantic_condition='aligned'):
        if kind not in SHAPE_KINDS_V23:
            raise ValueError('V23 shape kind must be simple or complex')
        if sensor_model not in SENSOR_MODELS_V19:
            raise ValueError('undeclared V19 stereo sensitivity model')
        if semantic_condition not in ('aligned', 'shuffled', 'absent'):
            raise ValueError('invalid semantic sensor intervention')
        if isinstance(noise_seed, bool) or not isinstance(noise_seed, (int, np.integer)) or noise_seed < 0:
            raise ValueError('nonnegative integer sensor seed required')
        self.kind = kind
        self.parent = 'D23-SHAPE'
        self.parent_index = 0  # Same parent/time/noise stream for both shapes.
        self.assignment = kind
        self.semantic_condition = semantic_condition
        self.noise_seed = int(noise_seed)
        self.seed = 23000
        self.config = FacilityConfigV19(width_m=8., height_m=8.,
                                        stereo_model=sensor_model, max_steps=100)
        self.width = self.height = 8.
        self.shape = (40, 40)
        self._solid_primitives = []
        self.boxes = []
        self._physical_markers = []
        self.objects = []
        self.inspection_truth = []

        def box(x, y, z, sx, sy, sz, owner=1):
            self._solid_primitives.append((x, y, z, sx, sy, sz, owner))
            if z < 1.2 and z+sz > .15:
                self.boxes.append((x, y, sx, sy))

        box(0, 0, -.12, 8., 8., .12)
        box(0, 0, 0, 8., .2, 2.4)
        box(0, 7.8, 0, 8., .2, 2.4)
        box(0, .2, 0, .2, 7.6, 2.4)
        box(7.8, .2, 0, .2, 7.6, 2.4)
        owner = 100
        box(3.2, 3.2, 0, 1.6, .8, 1.6, owner)
        if kind == 'complex':
            # Lower rear block intersects the main body by 4cm. Its dimensions
            # are 60x64x52cm; only the attachment overlap is narrow.
            box(4.2, 3.96, .42, .60, .64, .52, owner)
            # Upper/right block is joined to the first. Its 28cm protrusion
            # beyond the body affects the actual exterior occupancy boundary.
            box(4.68, 4.02, .82, .40, .40, .50, owner)
        category = 3 if kind == 'complex' else 2
        self._physical_markers.append(dict(x0=3.64, x1=4.36, y=3.2,
                                           z0=.88, z1=1.24, physical_class=category))
        self.objects.append(dict(id=0, owner=owner, category=category,
            name='cabinet', center=[4., 3.6, .8], front_center=[4., 3.2, .8],
            evaluation_bounds=[list(v) for v in EVALUATION_BOUNDS_V23],
            external_geometry='closed_body_with_external_attachments' if kind == 'complex' else 'closed_body'))
        self.mesh, self.triangle_owners, self.surface_measure_audit = union_surface_from_boxes(self._solid_primitives)
        self.triangle_classes = self.triangle_owners.copy()
        self.triangle_classes[self.triangle_owners == owner] = category
        self.occupancy = np.zeros(self.shape, bool)
        for x, y, sx, sy in self.boxes:
            c0 = math.floor(x/.2+1e-8)
            c1 = math.ceil((x+sx)/.2-1e-8)
            r0 = math.floor((self.height-y-sy)/.2+1e-8)
            r1 = math.ceil((self.height-y)/.2-1e-8)
            self.occupancy[max(0, r0):min(self.shape[0], r1),
                           max(0, c0):min(self.shape[1], c1)] = True
        self._blocked = inflated_obstacles(self.occupancy, self.config.robot_radius_m/.2)
        self.start = self._cell(4.1, 1.3)
        groups, _ = label(~self._blocked)
        if groups[self.start] == 0:
            raise ValueError('unsafe initial shape-probe pose')
        self.reachable = groups == groups[self.start]
        if np.any((~self._blocked) & ~self.reachable):
            raise ValueError('disconnected safe floor in shape probe')
        self.position = self.start
        self.heading = 0
        self.step_count = self.collisions = self.moves = 0
        c = self.config
        focal = c.width_px/(2*np.tan(np.deg2rad(c.fov_deg/2)))
        self.intrinsic = np.asarray([[focal, 0., (c.width_px-1)/2],
                                     [0., focal, (c.height_px-1)/2], [0., 0., 1.]])
        self.sensor_metadata = dict(model=sensor_model, noise_seed=self.noise_seed,
            seed_fields=['parent_index', 'action_step', 'noise_seed', '73919'],
            reference_fx_px=c.stereo_reference_fx_px, baseline_m=c.stereo_baseline_m,
            output_fx_px=focal, pose='perfect centred discrete CPU contract',
            scope='same generic V19 disparity sensitivity, not calibrated ZED',
            shape_probe_version=SHAPE_PROBE_VERSION_V23)
        self.route = route_spec_v23()
        self.static_route_audit = self.check_declared_route()

    def check_declared_route(self):
        """Check declared grid moves on truth without moving or sensing.

        A static feasibility check is not evidence that a planner completed
        the route or that any stage meets a reconstruction quality criterion.
        """
        state = (*self.start, 0)
        states = [state]
        for index, action in enumerate(self.route['actions'], start=1):
            r, col, heading = state
            if action == 'forward':
                dr, dc = DIRECTIONS[heading]
                r += dr
                col += dc
            elif action in ('left', 'right'):
                heading = (heading + (1 if action == 'right' else -1)) % 4
            else:
                raise ValueError('declared route contains non-motion action')
            if not (0 <= r < self.shape[0] and 0 <= col < self.shape[1]) or self._blocked[r, col]:
                raise ValueError(f'declared route blocked at paid action {index}')
            state = (r, col, heading)
            states.append(state)
        if state != (*self.start, 0):
            raise ValueError('declared route fails to return to exact anchor pose')
        if len(states)-1 > 100:
            raise ValueError('shape probe exceeds its 100-action bound')
        return dict(static_only=True, no_sense_or_step_called=True,
            paid_actions=len(states)-1, translations=self.route['actions'].count('forward'),
            turns=len(states)-1-self.route['actions'].count('forward'),
            stage_poses={name: list(states[index]) for name, index in self.route['stage_indices'].items()},
            states=[list(state) for state in states],
            all_centres_have_safe_footprints=True, returned_to_exact_anchor=True,
            quality_pass_not_tested=True)
