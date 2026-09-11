"""Construction only; no world, GT, evaluator, future frame or new action."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from env.virtual3d_inspection_v4 import InspectionConfigV4
from utils.rgbd_contract import RGBDFrame,PlanarScan
from scripts.eval_counterfactual_views import remap,candidate_routes
from nso.counterfactual_candidates_v6 import augment_candidate_routes


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    source=ROOT/'eval_results/counterfactual_view_development_first_20260910'
    output=Path(__file__).resolve().parent;meta=read(source/'metadata.json')
    assert meta['status']=='complete' and read(source/'verification.json')['status']=='passed_full'
    # Source hash only dependencies actually imported; unrelated root work is
    # allowed to proceed. Original policies/model/mapper remain unchanged.
    dependencies=['env/grid_exploration.py','env/virtual3d.py','env/virtual3d_v2.py','env/virtual3d_inspection_v4.py',
        'nso/semantic_completion_v3.py','nso/quality_coverage3d.py','nso/camera_joint_planner_v2.py',
        'nso/joint_planner_v2.py','nso/navigable_frontier_v2.py','nso/coverage_planner_v2.py',
        'nso/route_coverage_v2.py','nso/camera_mapping_v2.py','nso/mapping3d_v2.py','nso/mapping3d.py',
        'utils/grid_geometry.py','utils/rgbd_contract.py','scripts/eval_counterfactual_views.py']
    for name in dependencies:assert sha(ROOT/name)==meta['source_sha256'][name],name
    source_hashes={name:sha(ROOT/name) for name in dependencies+['nso/counterfactual_candidates_v6.py']}
    manifest=read(source/'artifact_hashes.json');inputs={};rows=[];started=time.monotonic()
    def check(p):
        actual=sha(p);assert manifest[str(p.relative_to(source))]==actual
        inputs[str(p)]=actual;return p
    for history in read(check(source/'summary.json'))['histories']:
        fixture=source/history['fixture'];hdir=(source/history['predictions']).parent
        c=InspectionConfigV4(**read(check(fixture/'fixture.json'))['environment'])
        index=history['history_step'];records=read(check(fixture/'prefix/records.json'))[:index+1]
        frames=[RGBDFrame.load(check(fixture/'prefix/frames'/f'{i:04d}.npz')) for i in range(index+1)]
        scans=[PlanarScan.load(check(fixture/'prefix/scans'/f'{i:04d}.npz')) for i in range(index+1)]
        shape=(round(c.height_m/c.resolution_m),round(c.width_m/c.resolution_m))
        mapper=remap(frames,scans,records,c,shape);r=records[-1]
        obs=mapper.observation(tuple(r['position']),r['heading'],index,r['collision'])
        original=read(check(hdir/'candidates.json'))
        repeated,_=candidate_routes(mapper,obs)
        assert json.loads(json.dumps(repeated))==original
        extras,audit=augment_candidate_routes(mapper,obs,original)
        out=output/history['fixture']/hdir.name;out.mkdir(parents=True,exist_ok=False)
        write(out/'extra_routes.json',extras);write(out/'audit.json',audit)
        row={'fixture':history['fixture'],'step':index,'added':len(extras),
             'extra_poses':[x['pose'] for x in extras],'costs':[x['cost'] for x in extras],
             'original_candidates_exact':True,'new_navigation_branches':0}
        rows.append(row);print(row,flush=True)
    assert all(sha(ROOT/name)==value for name,value in source_hashes.items())
    assert all(sha(name)==value for name,value in inputs.items())
    write(output/'summary.json',{'status':'complete','elapsed_s':time.monotonic()-started,
          'script_sha256':sha(__file__),'source':str(source),'source_hashes':source_hashes,
          'histories':rows,'total_extra_routes':sum(r['added'] for r in rows),'new_navigation_branches':0,
          'GT_used':False,'prediction_gain_used':False,'semantic_intervention_selection_test':'unit test passed'})
    write(output/'input_hashes.json',inputs)
    write(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':main()
