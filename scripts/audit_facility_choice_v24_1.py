#!/usr/bin/env python3
"""Static V24.1 height revision: paid graph, pairing, true top-edge background."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from env.facility_choice_v24_1 import (FacilityChoiceWorldV24_1, VERSION_V24_1,
    PARENTS_V24, ASSIGNMENTS_V24, LAYOUTS_V24)
from env.canonical_rgbd_v15 import render_axial_depth
from env.virtual3d import camera_pose
from scripts.audit_facility_choice_v24 import (StaticWorld, shortest_service_tours,
    static_prefix_projection_check, sha, write)


class TallStaticWorld(FacilityChoiceWorldV24_1):
    def step(self, action):
        raise AssertionError('Static projection audit prohibits world.step')


def top_edge_support(old, new):
    records = []
    for index,(cx,front) in enumerate(LAYOUTS_V24[new.parent]['fronts']):
        for side in (False,True):
            stage = 'AB'[index]+('_left' if index==0 else '_right') if side else 'AB'[index]+'_front'
            action_id = new.prefix_proposal['stage_indices'][stage]
            r,c,h = new.prefix_proposal['states'][action_id]
            pose = camera_pose((r,c),h,new.config,new.shape[0])
            segment = np.asarray([[cx-.4,front,1.6],[cx+.4,front,1.6]] if not side else
                [[cx+(-.8 if index==0 else .8),front+.2,1.6],
                 [cx+(-.8 if index==0 else .8),front+.6,1.6]])
            camera = (segment-pose[:3,3])@pose[:3,:3]
            uvw = camera@new.intrinsic.T
            uv = uvw[:,:2]/uvw[:,2,None]
            row = int(np.floor(uv[:,1].min()))
            cols = np.arange(int(np.ceil(uv[:,0].min())),int(np.floor(uv[:,0].max()))+1)
            assert 0 <= row < new.config.height_px and len(cols)>0
            assert cols.min()>=0 and cols.max()<new.config.width_px
            supports = []
            for world in (old,new):
                depth,points = render_axial_depth(np.asarray(world._solid_primitives,float)[:,:6],
                    world.intrinsic,pose,world.config.height_px,world.config.width_px,world.config.max_depth_m)
                depths = depth[row,cols]; xyz = points[row,cols]
                # These pixel centres are above the true top-edge projection.
                # A valid hit farther than the visible front/side plane is an
                # actual background measurement; zero range is never free.
                valid = depths > float(camera[:,2].max())+.2
                supports.append(dict(valid_background_pixels=int(valid.sum()),tested_pixels=len(cols),
                    axial_depth_min_m=float(depths.min()),axial_depth_max_m=float(depths.max()),
                    hit_height_min_m=float(xyz[valid,2].min()) if valid.any() else None,
                    hit_height_max_m=float(xyz[valid,2].max()) if valid.any() else None))
            records.append(dict(stage=stage,action_id=action_id,pixel_row=row,
                pixel_columns=cols.tolist(),edge_projected_uv=uv.tolist(),
                lower_2p4m_backstop=supports[0],higher_4p8m_backstop=supports[1],
                tested_top_segment_background_valid= supports[1]['valid_background_pixels']==len(cols),
                scope='above a central visible top-edge segment, not every object boundary'))
    return records


def main():
    output = ROOT/'audit_results/facility_choice_v24_1_high_backstops_20260915'
    if output.exists():
        raise ValueError('retain existing static revision evidence')
    old_audit = ROOT/'audit_results/facility_choice_v24_static_backstops_20260915'
    import json
    previous = json.loads((old_audit/'result.json').read_text())
    for name,expected in previous['source_sha256'].items():
        assert sha(ROOT/name)==expected,name
    source_paths = list(previous['source_sha256'])+['env/facility_choice_v24_1.py',
        'scripts/audit_facility_choice_v24_1.py']
    hashes = {name:sha(ROOT/name) for name in source_paths}
    cases = []; parents = []
    for parent in PARENTS_V24:
        worlds = []
        for assignment in ASSIGNMENTS_V24:
            old = StaticWorld(parent,assignment); new = TallStaticWorld(parent,assignment)
            same = all(np.array_equal(getattr(old,name),getattr(new,name))
                       for name in ('occupancy','_blocked','reachable'))
            assert same and old.prefix_proposal==new.prefix_proposal
            proof = shortest_service_tours(new)
            prior = next(c for c in previous['cases'] if (c['parent'],c['assignment'])==(parent,assignment))
            assert proof==prior['proof'],'height changed exact action graph proof'
            top = top_edge_support(old,new)
            cases.append(dict(parent=parent,assignment=assignment,unchanged_floor_and_prefix=same,
                exact_cost_and_attaining_witness_identical_to_v3=True,
                minimum_service_costs={k:v['minimum_paid_actions'] for k,v in proof['tours'].items()},
                top_edge_static_support=top,all_tested_top_segments_have_actual_background=all(x['tested_top_segment_background_valid'] for x in top),
                physical_actions_executed=new.step_count))
            worlds.append(new)
        pairing = static_prefix_projection_check(worlds)
        parents.append(dict(parent=parent,static_prefix_projection=pairing))
        print(parent, 'clean_pair',pairing['all_planned_raster_depths_equal'],
              'top_support',[x['higher_4p8m_backstop']['valid_background_pixels'] for x in cases[-1]['top_edge_static_support']],flush=True)
    for name,expected in hashes.items():
        assert sha(ROOT/name)==expected,name
    result = dict(status='complete_static_height_revision',world_version=VERSION_V24_1,cases=cases,parents=parents,
        source_sha256=hashes,previous_v3_result_sha256=sha(old_audit/'result.json'),
        static_revision_gates_passed=all(c['all_tested_top_segments_have_actual_background'] for c in cases) and
            all(p['static_prefix_projection']['all_planned_raster_depths_equal'] for p in parents),
        actual_sensor_packets=0,physical_actions=0,mapper_or_quality_runs=0,
        actual_prefix_verification_still_required=True,any_semantic_efficacy_proven=False)
    output.mkdir(); write(output/'result.json',result); write(output/'source_sha256.json',hashes)
    write(output/'artifact_hashes.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(output, result['static_revision_gates_passed'],flush=True)


if __name__ == '__main__':
    main()
