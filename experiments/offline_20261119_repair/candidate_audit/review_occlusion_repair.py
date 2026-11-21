"""Independent synthetic ray checks; no actor/world import or source mutation."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys
import numpy as np
import open3d as o3d

BASE = Path('/home/ouc/nso_robot/analyses/dual_template_repair_20261119')
SOURCE = Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('reviewed_occlusion_repair', BASE / 'method/occlusion_repair.py')
repair = importlib.util.module_from_spec(spec); spec.loader.exec_module(repair)
from env.virtual3d import camera_pose
from nso.semantic_completion_v3 import ObjectCompletionModel

config = SimpleNamespace(resolution_m=.2, camera_height_m=1.2, width_px=9, height_px=9, fov_deg=90., max_depth_m=8.)
cell = (20, 20); heading = 0; pose = camera_pose(cell, heading, config, 30)
origin = pose[:3, 3]


def scene(distances):
    vs, fs = [], []
    for d in distances:
        start = len(vs); x,y,z = origin + pose[:3, 2] * d
        vs += [[x-1,y,z-1],[x+1,y,z-1],[x+1,y,z+1],[x-1,y,z+1]]
        fs += [[start,start+1,start+2],[start,start+2,start+3]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vs),o3d.utility.Vector3iVector(fs))
    ray = o3d.t.geometry.RaycastingScene(nthreads=1)
    ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return ray


def frame(depth):
    return SimpleNamespace(world_from_camera=pose.copy(), intrinsic=np.array([[4.5,0,4],[0,4.5,4],[0,0,1.]]),
                           depth_m=np.full((9,9),depth,dtype=float))


def model(depths):
    return SimpleNamespace(points=np.array([origin+pose[:3,2]*4.]), weights=np.array([.125]),
        hypothesis_ids=np.array([0]), hypothesis_rays=[scene([1.,3.])],
        mapper=SimpleNamespace(config=config,shape=(30,40),keyframes=[frame(d)for d in depths]),
        audit={'history_rejection':False})


rows=[]
for name,depths,real_ray,wanted in [
    ('all_preceding_intersections_strict_free_can_recover',[5.],None,.125),
    ('later_unresolved_intersection_still_blocks',[2.],None,0.),
    ('measured_first_surface_still_blocks',[1.],None,0.),
    ('another_frame_measured_surface_vetoes_free',[5.,1.],None,0.),
    ('zero_depth_cannot_prove_free',[0.],None,0.),
    ('nan_depth_cannot_prove_free',[float('nan')],None,0.),
    ('infinite_depth_cannot_prove_free',[float('inf')],None,0.),
    ('real_TSDF_occluder_is_not_removed',[5.],scene([.5]),0.),
]:
    m=model(depths)
    source=ObjectCompletionModel.gain(m,cell,heading,real_ray)
    old,oa=repair.trace_gain(m,cell,heading,real_ray,False)
    new,na=repair.trace_gain(m,cell,heading,real_ray,True)
    np.testing.assert_array_equal(old,source)
    assert float(new.sum())==wanted,(name,new,wanted)
    assert np.all(new>=old)
    rows.append(dict(name=name,source_gain=float(source.sum()),repaired_gain=float(new.sum()),
                     original_exact=True,expected_matched=True))

m=model([5.]);m.hypothesis_rays=[scene([5.])]
expected=ObjectCompletionModel.gain(m,cell,heading,None)
off,_=repair.trace_gain(m,cell,heading,None,False)
on,_=repair.trace_gain(m,cell,heading,None,True)
assert float(expected.sum())==.125
np.testing.assert_array_equal(off,expected);np.testing.assert_array_equal(on,expected)
rows.append(dict(name='already_visible_positive_gain_preserved',source_gain=.125,repaired_gain=.125,
                 original_exact=True,expected_matched=True))

# Measured-band and invalid-pixel boundaries are independently probed, not inferred from rendered outcomes.
probes=np.array([origin+pose[:3,2]*d for d in [.10,1.8,1.95,2.,2.05,2.2]])
cl=repair.classify_history(probes,[frame(2.)])
assert cl['strict_free'].tolist()==[False,True,False,False,False,False]
oob=np.array([origin+pose[:3,0]*10+pose[:3,2]])
assert not repair.classify_history(oob,[frame(5.)])['strict_free'].any()
boundary_frame=frame(1.1);boundary_frame.world_from_camera=np.eye(4)
free_frame=frame(2.);free_frame.world_from_camera=np.eye(4)
boundary=repair.classify_history(np.array([[0.,0.,1.]]),[boundary_frame,free_frame])
assert boundary['raw_free_evidence'][0] and boundary['measured_surface'][0] and not boundary['strict_free'][0]
result=dict(schema='independent_occlusion_repair_review_v1',all_passed=True,
            reviewed_source_sha256=hashlib.sha256((BASE/'method/occlusion_repair.py').read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),ray_cases=rows,
            classification_checks=['minimum_depth','strict_free_vs_surface_tolerance','out_of_image','closed_tolerance_boundary_vetoes_other_frame_free'],
            test_fixture_correction='Initial fixture used nominal .15m exactly; world-to-camera subtraction produced .15000000000000013, so the strict >.15 check accepted it. Corrected the lower-bound test to .10m, not the implementation; exact-boundary binary rounding is not claimed to be eliminated.',
            no_actor_replay=True,no_hardware=True,
            limits=['Synthetic ray checks are not evidence of exploration or reconstruction improvement.',
                    'Nearest-pixel free evidence inherits sensor rasterization and the declared depth tolerance.',
                    'S_no_history retains this visibility repair; it disables reward-history filtering only.',
                    'Fixed-state measured-depth replay must separately verify repaired scoring on actual retained frames.'])
with (BASE/f"candidate_audit/occlusion_review_{result['reviewed_source_sha256'][:12]}.json").open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,ensure_ascii=False))
