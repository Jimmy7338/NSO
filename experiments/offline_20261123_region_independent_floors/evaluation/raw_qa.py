import argparse
import csv
import hashlib
import json
from math import isclose
from pathlib import Path

import numpy as np
import open3d as o3d


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def group(records, fields):
    sets={}
    for item in records:
        key=tuple(item[k] for k in fields)
        sets.setdefault(key,[]).append(item)
    return sets


def exact(num, expected):
    return bool(num) and isclose(float(num),expected,abs_tol=1e-12,rel_tol=0)


def ray(mesh):
    scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def distance(mesh_scene, points):
    return mesh_scene.compute_distance(o3d.core.Tensor(np.asarray(points,dtype=np.float32)),nthreads=1).numpy()


def direct_metrics(source_scene, fixed_points, mesh_path):
    native=o3d.io.read_triangle_mesh(str(mesh_path))
    faces=np.asarray(native.triangles)
    vertices=np.asarray(native.vertices)
    tris=vertices[faces]
    if not len(tris):
        return 0.,0.,0.
    a,b,c=(tris[:,n,:] for n in range(3))
    mass=np.linalg.norm(np.cross(b-a,c-a),axis=1)/2
    if not mass.sum()>0:
        return 0.,0.,0.
    rng=np.random.default_rng(812)
    count=12000
    chosen=rng.choice(len(tris),count,p=mass/mass.sum())
    u=np.sqrt(rng.random(count))
    v=rng.random(count)
    sampled=(1-u)[:,None]*a[chosen]+(u*(1-v))[:,None]*b[chosen]+(u*v)[:,None]*c[chosen]
    p=np.count_nonzero(distance(source_scene,sampled)<=.05)/count
    r=np.count_nonzero(distance(ray(native),fixed_points)<=.05)/len(fixed_points)
    return p,r,2*p*r/(p+r) if p+r else 0.


def main(root, results):
    protocol=read(root/'FORMAL_PROTOCOL.json')
    freeze=read(root/'FORMAL_FREEZE.json')
    receipt=read(root/'execution/formal/phase_receipt.json')
    summary=read(results/'summary.json')
    evidence=read(results/'measurement_audit.json')
    expected={(layout,arm) for layout in protocol['layouts'] for arm in protocol['arms']}
    assert sha(root/'FORMAL_FREEZE.json')==summary['run_freeze_sha256']==receipt['freeze_sha256']
    assert sha(root/'execution/formal/phase_receipt.json')==summary['phase_receipt_sha256']==evidence['phase_receipt_sha256']
    assert all(sha(results/path)==digest for path,digest in summary['results_sha256'].items())
    eps=group(load_csv(results/'episodes.csv'),('layout','arm'))
    curve=group(load_csv(results/'coverage_curve.csv'),('layout','arm'))
    checks=group(load_csv(results/'checkpoints.csv'),('layout','arm'))
    trace_csv=group(load_csv(results/'trajectory.csv'),('layout','arm'))
    workers={(x['layout'],x['arm']):x for x in receipt['episodes']}
    assert set(eps)==set(curve)==set(checks)==set(trace_csv)==set(workers)==expected
    derived={}
    count_states=count_meshes=count_unavailable=0
    for layout in protocol['layouts']:
        base=Path(freeze['scene_root'])/'evaluator_references'/layout
        with np.load(base/'evaluator_truth.npz',allow_pickle=False) as z:
            reachable=z['reachable']
        with np.load(base/'fixed_reference.npz',allow_pickle=False) as z:
            fixed=z['all']
        source=ray(o3d.io.read_triangle_mesh(str(base/'source_mesh.ply')))
        total=int(reachable.sum())
        assert total>0 and len(fixed)>0
        cold=[]
        for arm in protocol['arms']:
            key=(layout,arm)
            ep,=eps[key]
            worker=workers[key]
            path=Path(ep['run_directory'])
            assert worker['exit_code']==0 and path==Path(worker['run_path'])
            assert worker['result_sha256']==sha(path/'manifest.json')
            m=read(path/'manifest.json')
            raw=read(path/'steps.json')
            n=len(raw)
            assert ep['status']=='completed_execution' and n==int(ep['actual_paid'])==m['paid_actions']<=192
            pose=[*raw[-1]['position'],raw[-1]['heading']] if raw else m['initial_pose']
            returned=pose==m['initial_pose']
            collisions=sum(r['collision'] for r in raw)
            assert ep['returned']==str(returned) and returned==m['returned']
            assert m['observed_native_termination']['returned_to_anchor']==returned
            assert int(ep['collisions'])==collisions==m['collisions']
            rows=trace_csv[key]
            assert len(rows)==n
            for k,(raw_step,csv_step) in enumerate(zip(raw,rows,strict=True),1):
                assert int(csv_step['step'])==k==raw_step['step']
                assert csv_step['action']==raw_step['action'] and list(json.loads(csv_step['position']))==raw_step['position']
                assert int(csv_step['heading'])==raw_step['heading'] and csv_step['packet_sha256']==raw_step['packet_sha256']
                assert csv_step['guard_allowed']==str(raw_step['guard_allowed'])=='True'
                r=raw_step['reserved_return_cost']
                assert r is None or k+r<=192
            with np.load(path/'observed_belief_trace.npz',allow_pickle=False) as z:
                beliefs=z['belief']
                assert list(z['paid'])==list(range(n+1))
            # Boolean indexing in raw grid orientation and explicit exact threshold.
            counts=np.count_nonzero(beliefs[:,reachable]==0,axis=1).tolist()
            a95=next((k for k,c in enumerate(counts) if c*20>=19*total),None)
            assert [int(row['paid']) for row in curve[key]]==list(range(n+1))
            for c,row in zip(counts,curve[key],strict=True):
                assert int(row['correct_free_cells'])==c and int(row['reachable_cells'])==total and exact(row['C_free'],c/total)
            assert ep['A95_paid']==('' if a95 is None else str(a95))
            assert ep['A95_status']==('not_reached_within_budget' if a95 is None else 'reached')
            assert exact(ep['C_free'],counts[-1]/total)
            checkpoints={}
            for cp in checks[key]:
                label=cp['checkpoint']
                raw_receipt=Path(cp['checkpoint_receipt_path'])
                raw=read(raw_receipt)
                assert raw_receipt==path/label/'checkpoint.json' and sha(raw_receipt)==cp['source_checkpoint_sha256']
                assert m['files_sha256'][f'{label}/checkpoint.json']==sha(raw_receipt)
                if cp['status']=='unavailable':
                    assert not cp['actual_paid'] and int(cp['requested_paid'])>n
                    assert cp['eligible_for_paired_comparison']=='False'
                    count_unavailable+=1
                    continue
                assert cp['eligible_for_paired_comparison']=='True'
                clock=int(cp['actual_paid'])
                assert raw['actual_paid']==clock
                if label=='terminal':
                    assert clock==n and raw['pose']==pose
                elif label=='start':
                    assert clock==0
                    cold.append((read(path/'initial_raw_rgbd_scan_sha256.json'),raw['belief_sha256'],raw['files_sha256']['native_mesh.ply']))
                else:
                    assert clock==int(label.removeprefix('paid_'))
                mesh=Path(cp['fused_mesh_path'])
                belief=Path(cp['observed_belief_path'])
                assert sha(mesh)==raw['files_sha256']['native_mesh.ply']
                assert sha(belief)==raw['files_sha256']['belief.npz']
                with np.load(belief,allow_pickle=False) as state:
                    assert np.array_equal(beliefs[clock],state['belief'])
                p,r,f=direct_metrics(source,fixed,mesh)
                c=counts[clock]/total
                metrics={'p_scene_05cm':p,'r_scene_05cm':r,'F1_scene_05cm':f,'C_free':c,'J':c*f}
                assert all(exact(cp[name],val) for name,val in metrics.items()),(key,label,metrics)
                if label=='terminal':
                    assert all(exact(ep[name],val) for name,val in metrics.items()),key
                checkpoints[label]=(clock,metrics)
                count_meshes+=1
            count_states+=len(counts)
            derived[key]=(a95,returned,checkpoints)
        assert len(cold)==3 and cold[0]==cold[1]==cold[2]
    pairs=load_csv(results/'pairs.csv')
    pair_tables=load_csv(results/'paired_checkpoints.csv')
    assert len(pairs)==2*len(protocol['layouts'])
    for pair in pairs:
        L, R=(derived[pair['layout'],pair[key]] for key in ('left_arm','right_arm'))
        saved=R[0]-L[0] if L[0] is not None and R[0] is not None else None
        assert pair['revised_actions_saved']==('' if saved is None else str(saved))
        if all('paid_96' in q[2] for q in (L,R)):
            assert exact(pair['A96_delta_J_pp'],100*(L[2]['paid_96'][1]['J']-R[2]['paid_96'][1]['J']))
        if L[1] and R[1]:
            assert exact(pair['return_delta_r_scene_05cm_pp'],100*(L[2]['terminal'][1]['r_scene_05cm']-R[2]['terminal'][1]['r_scene_05cm']))
        else:
            assert not pair['return_delta_r_scene_05cm_pp']
    assert len(pair_tables)==6*len(pairs)
    for item in pair_tables:
        L, R=(derived[item['layout'],item[key]] for key in ('left_arm','right_arm'))
        left,right=(q[2].get(item['checkpoint']) for q in (L,R))
        both=bool(left and right)
        assert item['paired_available']==str(both)
        assert item['equal_actual_paid']==str(bool(both and left[0]==right[0]))
        for name in ('C_free','p_scene_05cm','r_scene_05cm','F1_scene_05cm','J'):
            val=item[f'delta_{name}_pp']
            assert exact(val,100*(left[1][name]-right[1][name])) if both else not val
    print(json.dumps({'n_runs':len(expected),'n_states':count_states,'n_full_raw_meshes':count_meshes,
                      'n_unavailable_clock_receipts':count_unavailable,
                      'paired_checkpoint_rows':len(pair_tables),'crosscheck_output_sha256':sha(results/'summary.json')},indent=2))


if __name__=='__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('root',type=Path)
    arg.add_argument('results',type=Path)
    args=arg.parse_args()
    main(args.root.resolve(),args.results.resolve())
