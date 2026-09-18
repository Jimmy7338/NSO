"""Two-facility V24 choice fixtures; world truth is evaluator-only.

Geometry, static prefix and declared service regions support offline task
feasibility. They are NOT permitted inputs to online candidate generation.
The public prefix is a proposal pending paid sensor-history verification.
"""
import math

import numpy as np
from scipy.ndimage import label

from env.facility_documentation_v19 import FacilityConfigV19, FacilityWorldV19, SENSOR_MODELS_V19
from env.virtual3d_inspection_v4 import union_surface_from_boxes
from utils.grid_geometry import DIRECTIONS, inflated_obstacles


VERSION_V24 = 'two-facility-choice-world-v24-3'
ASSIGNMENTS_V24 = ('A_complex_B_simple', 'A_simple_B_complex')
LAYOUTS_V24 = {
    'D24-P00': dict(width=16., height=10., fronts=((4.6, 5.6), (11.6, 5.6)),
                   divider=(8.0, 3.2, .2, 6.6), anchor=(8.1, 1.3)),
    'D24-P01': dict(width=18., height=12., fronts=((4.8, 5.8), (13.4, 5.8)),
                   divider=(9.0, 3.4, .2, 8.4), anchor=(9.1, 1.3)),
}
PARENTS_V24 = tuple(LAYOUTS_V24)


def _cell_v24(x, y, height):
    return (round(height/.2)-1-int(math.floor(y/.2+1e-9)), int(math.floor(x/.2+1e-9)))


def prefix_spec_v24(parent='D24-P00'):
    """Paid front+outer-side basic scans for BOTH objects, then anchor.

    Axis order is declared, not chosen using a hidden occupancy map. The low
    transverse corridor and frontal approaches are physical camera geometry,
    not an instruction to suppress visible attachments. Actual paired history
    and basic-shape usefulness remain untested by this static specification.
    """
    spec = LAYOUTS_V24[parent]
    start = (*_cell_v24(*spec['anchor'], spec['height']), 2)
    state = start
    actions, states, stages, waypoints = [], [list(start)], {'initial': 0}, []

    def rotate(heading):
        nonlocal state
        difference = (heading-state[2]) % 4
        turns = ['left'] if difference == 3 else ['right']*difference
        for action in turns:
            state = (*state[:2], (state[2]+(1 if action == 'right' else -1)) % 4)
            actions.append(action); states.append(list(state))

    def move_to(name, x, y, heading):
        nonlocal state
        destination = _cell_v24(x, y, spec['height'])
        for axis in (1, 0):  # First translate east/west, then north/south.
            delta = destination[axis]-state[axis]
            if delta:
                direction = (1 if delta > 0 else 3) if axis == 1 else (2 if delta > 0 else 0)
                rotate(direction)
                for _ in range(abs(delta)):
                    dr, dc = DIRECTIONS[direction]
                    state = (state[0]+dr, state[1]+dc, direction)
                    actions.append('forward'); states.append(list(state))
        rotate(heading)
        stages[name] = len(actions)
        waypoints.append(dict(stage=name, xy_m=[x,y], heading=heading, state=list(state), action_id=len(actions)))

    for i,(cx,front) in enumerate(spec['fronts']):
        role='AB'[i]
        sign=1 if i==0 else -1
        # A front approach from the outer side keeps rear protrusions
        # physically occluded at long range. Move centrally only when near.
        move_to(role+'_approach', cx-sign*.9, front-1.3, 0)
        move_to(role+'_front', cx+sign*.1, front-1.3, 0)
        move_to(role+('_left' if i==0 else '_right'), cx-sign*1.9, front+.1, 1 if i==0 else 3)
        move_to(role+'_exit', cx-sign*1.9, spec['anchor'][1], 2)
    move_to('decision', *spec['anchor'], 2)
    assert state == start
    return dict(parent=parent, actions=actions, states=states, stage_indices=stages,
        waypoints=waypoints, paid_actions=len(actions), decision_anchor=list(start),
        includes_turns=True, complete_history_pairing_not_yet_verified=True,
        basic_shape_quality_not_yet_verified=True, online_planner=False)


class FacilityChoiceWorldV24(FacilityWorldV19):
    """Two closed cabinets in side branches, using inherited V19 sensors."""
    def __init__(self, parent='D24-P00', assignment='A_complex_B_simple',
                 sensor_model='iid_025px', noise_seed=1901, semantic_condition='aligned'):
        if parent not in PARENTS_V24 or assignment not in ASSIGNMENTS_V24:
            raise ValueError('undeclared V24 layout or paired assignment')
        if sensor_model not in SENSOR_MODELS_V19:
            raise ValueError('undeclared V19 stereo sensitivity')
        if semantic_condition not in ('aligned','shuffled','absent'):
            raise ValueError('invalid semantic sensor condition')
        if isinstance(noise_seed,bool) or not isinstance(noise_seed,(int,np.integer)) or noise_seed < 0:
            raise ValueError('nonnegative integer noise seed required')
        spec=LAYOUTS_V24[parent]
        self.parent=parent; self.parent_index=PARENTS_V24.index(parent)
        self.assignment=assignment; self.semantic_condition=semantic_condition; self.noise_seed=int(noise_seed)
        self.seed=24000+self.parent_index
        self.width=spec['width']; self.height=spec['height']
        self.config=FacilityConfigV19(width_m=self.width,height_m=self.height,
                                      stereo_model=sensor_model,max_steps=1000)
        self.shape=(round(self.height/.2),round(self.width/.2))
        self._solid_primitives=[]; self.boxes=[]; self._physical_markers=[]
        self.objects=[]; self.inspection_truth=[]

        def box(x,y,z,sx,sy,sz,owner=1):
            self._solid_primitives.append((x,y,z,sx,sy,sz,owner))
            if z < 1.2 and z+sz > .15:
                self.boxes.append((x,y,sx,sy))

        w,h=self.width,self.height
        box(0,0,-.12,w,h,.12)
        box(0,0,0,w,.2,2.4); box(0,h-.2,0,w,.2,2.4)
        box(0,.2,0,.2,h-.4,2.4); box(w-.2,.2,0,.2,h-.4,2.4)
        x,y,sx,sy=spec['divider']; box(x,y,0,sx,sy,2.4)
        for i,(cx,front) in enumerate(spec['fronts']):
            complex_shape=(assignment==ASSIGNMENTS_V24[i])
            owner=100+i; category=3 if complex_shape else 2
            # Real L-shaped work-station backstops. Visible background around
            # front/outer-side edges lies within the unchanged 5 m sensor range.
            # Both the rear and inner-side viewing aisle stay physically open.
            sign=1 if i==0 else -1
            box(cx-2.8 if i==0 else cx-3.1,front+3.2,0,5.9,.2,2.4)
            box(cx+2.9 if i==0 else cx-3.1,front-.6,0,.2,3.8,2.4)
            box(cx-.8,front,0,1.6,.8,1.6,owner)
            if complex_shape:
                # B is a physical left/right mirror, with the same front y.
                # Inward supplemental views then have matched geometric cost.
                box(cx+(.2 if i==0 else -.8),front+.76,.42,.60,.64,.52,owner)
                box(cx+(.68 if i==0 else -1.08),front+.82,.82,.40,.40,.50,owner)
            self._physical_markers.append(dict(x0=cx-.36,x1=cx+.36,y=front,
                z0=.88,z1=1.24,physical_class=category))
            self.objects.append(dict(id=i,owner=owner,category=category,name='AB'[i],
                center=[cx,front+.4,.8],front_center=[cx,front,.8],
                evaluation_bounds=[[cx-(.92 if i==0 else 1.2),front-.12,.01],
                                   [cx+(1.2 if i==0 else .92),front+1.52,1.72]],
                lateral_mount_sign=1 if i==0 else -1,
                external_geometry='closed_body_with_external_attachments' if complex_shape else 'closed_body'))
        self.mesh,self.triangle_owners,self.surface_measure_audit=union_surface_from_boxes(self._solid_primitives)
        self.triangle_classes=self.triangle_owners.copy()
        for item in self.objects:
            self.triangle_classes[self.triangle_owners==item['owner']]=item['category']
        self.occupancy=np.zeros(self.shape,bool)
        for x,y,sx,sy in self.boxes:
            c0,c1=math.floor(x/.2+1e-8),math.ceil((x+sx)/.2-1e-8)
            r0,r1=math.floor((h-y-sy)/.2+1e-8),math.ceil((h-y)/.2-1e-8)
            self.occupancy[max(0,r0):min(self.shape[0],r1),max(0,c0):min(self.shape[1],c1)]=True
        self._blocked=inflated_obstacles(self.occupancy,self.config.robot_radius_m/.2)
        self.start=_cell_v24(*spec['anchor'],h); self.start_heading=2
        groups,_=label(~self._blocked)
        if not groups[self.start]:
            raise ValueError('unsafe anchor')
        self.reachable=groups==groups[self.start]
        if np.any((~self._blocked)&~self.reachable):
            raise ValueError('disconnected safe floor')
        self.position=self.start; self.heading=self.start_heading
        self.step_count=self.collisions=self.moves=0
        c=self.config;focal=c.width_px/(2*np.tan(np.deg2rad(c.fov_deg/2)))
        self.intrinsic=np.asarray([[focal,0,(c.width_px-1)/2],[0,focal,(c.height_px-1)/2],[0,0,1.]])
        self.sensor_metadata=dict(model=sensor_model,noise_seed=self.noise_seed,
            seed_fields=['parent_index','action_step','noise_seed','73919'],
            reference_fx_px=c.stereo_reference_fx_px,baseline_m=c.stereo_baseline_m,
            output_fx_px=focal,pose='perfect centred discrete CPU contract',
            scope='same V19 generic disparity sensitivity; not measured ZED parameters',world_version=VERSION_V24)
        self.prefix_proposal=prefix_spec_v24(parent)
        if any(self._blocked[r,col] for r,col,_ in self.prefix_proposal['states']):
            raise ValueError('declared common prefix is statically blocked')
        self.service_regions=self.service_regions_v24()

    def service_regions_v24(self):
        """GT-side fixed task regions, never online candidate templates.

        Each facility's declared supplemental task visits BOTH its rear and
        inward-side regions at the required heading. Arrival/turn already pays
        for one sensor acquisition; no duplicate or free dwell frame is added.
        These regions do not certify a quality score or minimal quality cost.
        """
        rows=[]
        for i,(cx,front) in enumerate(LAYOUTS_V24[self.parent]['fronts']):
            lateral_bounds=([[cx+1.9,front+.1],[cx+2.3,front+.5]] if i==0
                            else [[cx-2.3,front+.1],[cx-1.9,front+.5]])
            for role,bounds,heading in (
                ('rear',[[cx-.3,front+2.3],[cx+.3,front+2.7]],2),
                ('right' if i==0 else 'left',lateral_bounds,3 if i==0 else 1)):
                states=[]
                for r,col in np.argwhere(self.reachable):
                    x=(col+.5)*.2; y=(self.shape[0]-r-.5)*.2
                    if bounds[0][0]-1e-8 <= x <= bounds[1][0]+1e-8 and bounds[0][1]-1e-8 <= y <= bounds[1][1]+1e-8:
                        states.append([int(r),int(col),heading])
                if not states:
                    raise ValueError('empty declared view region')
                rows.append(dict(bit=len(rows),asset_id=i,asset_name='AB'[i],view=role,
                    xy_bounds_m=bounds,heading=heading,states=states,
                    minimum_paid_observations=1,truth_only=True))
        return rows
