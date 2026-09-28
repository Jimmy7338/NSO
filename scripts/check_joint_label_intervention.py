#!/usr/bin/env python3
"""Verify cross-run intervention invariance and missing-label fallback."""
import argparse,json
from pathlib import Path
import numpy as np


def check(aligned,intervention):
    load=lambda d:[json.loads(s) for s in (d/'episodes.jsonl').read_text().splitlines()]
    a,b=load(aligned),load(intervention)
    key=lambda r:(r['layout'],r['seed'],r['occluded_objects'])
    geo={key(r):r for r in a if r['method']=='geometry_v2'}
    full={key(r):r for r in a if r['method']=='full_v2'}
    assert set(geo)==set(full)
    assert {(key(r),r['semantic_condition']) for r in b}=={(k,condition) for k in full for condition in ('absent','shuffled')}
    pairs=[]
    for row in b:
        original=full[key(row)]
        assert original['truth_sha256']==row['truth_sha256']
        x=np.load(aligned/original['artifact_dir']/'frames/0000.npz')
        y=np.load(intervention/row['artifact_dir']/'frames/0000.npz')
        for field in ('depth_m','color_rgb','intrinsic','world_from_camera'):
            np.testing.assert_array_equal(x[field],y[field])
        x=np.load(aligned/original['artifact_dir']/'scans/0000.npz')
        y=np.load(intervention/row['artifact_dir']/'scans/0000.npz')
        assert set(x.files)==set(y.files)
        for field in x.files:np.testing.assert_array_equal(x[field],y[field])
        if row['semantic_condition']=='absent':
            baseline=geo[key(row)]
            for file,fields in [('maps.npz',('poses','known_packed')),('final_mesh.npz',('vertices','triangles'))]:
                x=np.load(aligned/baseline['artifact_dir']/file);y=np.load(intervention/row['artifact_dir']/file)
                for field in fields:np.testing.assert_array_equal(x[field],y[field])
            pairs.append(dict(layout=row['layout'],seed=row['seed'],occluded_objects=row['occluded_objects'],exact_trajectory_and_mesh_match=True))
    result=dict(status='passed',absent_vs_geometry_pairs=pairs,truth_and_initial_sensor_invariance_pairs=len(b))
    (intervention/'label_intervention_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('passed:',len(pairs),'missing-label trajectory/mesh matches;',len(b),'truth/initial sensor invariant pairs')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--aligned',type=Path,required=True);p.add_argument('--intervention',type=Path,required=True)
    a=p.parse_args();check(a.aligned,a.intervention)
