#!/usr/bin/env python3
"""Post-hoc common bootstrap support diagnostic; native mapper, no policy/ROS."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse, csv, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--code',type=Path,required=True,help='directory containing nso_bridge and joint_vendor')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    result_path=a.output/'result.json'
    if result_path.exists():raise RuntimeError('Use a new output directory; do not overwrite an existing result')
    protocol=dict(schema='nso.posthoc_common_bootstrap_quality.protocol.v1',created_unix_s=time.time(),
        selection='All intersection keys of the two bootstrap supported sets, in the identical planning coordinate frame; no outcome-based subset',
        support_cell_m=.15,metric='native I/(I+.25+100*residual)',
        common_action_checkpoint=32,endpoint_reporting='Both actual endpoints also retained: C40 and S32',
        decomposition='Q(I_t,R_0)-Q(I_0,R_0) and Q(I_t,R_t)-Q(I_t,R_0)',
        claim_scope='Post-hoc internal reliability diagnostic, not a matched causal test or measured reconstruction accuracy',
        source_configuration_sha256={m:digest(a.package/m/'source/joint_configuration.json') for m in ['C','S']})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    sys.path.insert(0,str(a.code.resolve()))
    from nso_bridge.joint_engine import _MeasuredMapMapper
    from nso_bridge.sensor_range_adapter import adapt_scan
    from env.virtual3d_v2 import VirtualConfigV2
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    from nso.observed_feedback_v10 import _map_evidence
    native=Path(sys.modules[_MeasuredMapMapper.__module__].__file__).resolve()
    code_files=[native,Path(sys.modules['nso.observed_feedback_v10'].__file__),
                Path(sys.modules['nso.mapping3d_v2'].__file__),Path(sys.modules['nso.mapping3d'].__file__),
                Path(sys.modules['nso.observed_runtime_mapper_v10'].__file__),
                Path(sys.modules['nso_bridge.sensor_range_adapter'].__file__)]
    started=time.monotonic();histories={};initial_keys={};checks={};configs={}
    for mode in ['C','S']:
        p=a.package/mode;config=json.loads((p/'source/joint_configuration.json').read_text())['configuration'];configs[mode]=config
        manifest=json.loads((p/'frames_manifest.json').read_text())['frames']
        log=[json.loads(x) for x in (p/'decisions.jsonl').read_text().splitlines()]
        obs={x['request']['params']['action_id']:x['response']['result'] for x in log if x['request']['method']=='observe'}
        source={x['path']:x['sha256'] for x in json.loads((p/'source/source_manifest.json').read_text())['files']}
        for file in code_files:
            rel='nso_bridge/'+str(file.resolve().relative_to(a.code.resolve()))
            assert digest(file)==source[rel],rel
        sensor=dict(config['sensor_config']);sensor.update(resolution_m=.2,robot_radius_m=config['robot_radius_m'])
        mapper=_MeasuredMapMapper(tuple(config['shape']),VirtualConfigV2(**sensor));T=np.asarray(config['planning_from_odom'])
        histories[mode]={};checks[mode]=[]
        for item in manifest:
            aid=item['action_id'];path=p/item['file'];assert digest(path)==item['sha256']
            with np.load(path,allow_pickle=False) as z:
                frame=RGBDFrame(float(z['rgbd_timestamp_s']),z['depth'].copy(),z['color_rgb'].copy(),z['intrinsic'].copy(),T@z['world_from_camera'],z['semantic'].copy())
                scan=PlanarScan(float(z['scan_timestamp_s']),z['ranges'].copy(),float(z['angle_min']),float(z['angle_increment']),float(z['range_max']),T@z['world_from_laser'])
                scan,_=adapt_scan(scan,config['sensor_range_profile']);mapper.input_observed_grid=z['joint_belief_grid'].copy()
            mapper.update(frame,scan);ev=_map_evidence(mapper)
            if aid==0:initial_keys[mode]=set(ev['supported'])
            support=initial_keys[mode];q=sum(ev['quality'].get(k,0.) for k in sorted(support))/len(support)
            fb=obs[aid]['summary']['runtime']['modules']['feedback']
            assert len(support)==fb['fixed_old_support_count']
            assert hashlib.sha256(json.dumps(sorted(support)).encode()).hexdigest()==fb['fixed_old_support_sha256']
            assert math.isclose(q,fb['old_quality_proxy'],abs_tol=1e-12,rel_tol=1e-12),(mode,aid,q,fb['old_quality_proxy'])
            checks[mode].append(dict(action=aid,native_own_support_q=q,logged_q=fb['old_quality_proxy']))
            histories[mode][aid]={k:dict(I=float(mapper.quality[k]['information']),R=float(mapper.quality[k]['residual']),n=int(mapper.quality[k]['n']),point=mapper.quality[k]['point'].tolist()) for k in support}
        print(mode,'native mapper observations verified',len(histories[mode]),flush=True)
    assert configs['C']['planning_from_odom']==configs['S']['planning_from_odom'],'common coordinate frame required'
    keys=sorted(initial_keys['C']&initial_keys['S']);assert keys,'No common bootstrap support; diagnostic unavailable'
    assert all(32 in histories[m] for m in ['C','S'])
    def quality(i,r):return i/(i+.25+100*r)
    curves=[];summaries={};initial_point_separation=[]
    for key in keys:
        initial_point_separation.append(float(np.linalg.norm(np.array(histories['C'][0][key]['point'])-histories['S'][0][key]['point'])))
    for mode,history in histories.items():
        initial=history[0];q0=float(np.mean([quality(initial[k]['I'],initial[k]['R']) for k in keys]))
        for aid,rows in history.items():
            q=float(np.mean([quality(rows[k]['I'],rows[k]['R']) for k in keys]))
            only_i=float(np.mean([quality(rows[k]['I'],initial[k]['R']) for k in keys]))
            row=dict(arm=mode,action=aid,common_support_count=len(keys),quality=q,change_from_own_initial=q-q0,
                information_component=only_i-q0,residual_component=q-only_i,
                observed_again_count=sum(rows[k]['n']>initial[k]['n'] for k in keys),
                mean_information=float(np.mean([rows[k]['I'] for k in keys])),
                mean_residual_m2=float(np.mean([rows[k]['R'] for k in keys])))
            assert math.isclose(row['change_from_own_initial'],row['information_component']+row['residual_component'],abs_tol=1e-14)
            curves.append(row)
        summaries[mode]={'initial':next(r for r in curves if r['arm']==mode and r['action']==0),
            'action32':next(r for r in curves if r['arm']==mode and r['action']==32),
            'actual_terminal':curves[-1]}
    with (a.output/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=curves[0]);w.writeheader();w.writerows(curves)
    (a.output/'common_keys.json').write_text(json.dumps(keys)+'\n')
    result=dict(schema='nso.posthoc_common_bootstrap_quality.v1',analysis_type='posthoc_descriptive_internal_proxy',
        original_sources_and_frame_hashes_checked=True,native_own_support_checks=checks,
        native_source_sha256={str(p):digest(p) for p in code_files},
        common_support_count=len(keys),own_bootstrap_support_count={m:len(k) for m,k in initial_keys.items()},
        shared_keys_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),planning_transforms_identical=True,
        initial_same_key_point_separation_m=dict(min=min(initial_point_separation),median=float(np.median(initial_point_separation)),max=max(initial_point_separation)),
        arms=summaries,elapsed_s=time.monotonic()-started,
        limits=['Same15cm cell is not independently verified identity of a physical surface.',
        'Different actual initial sensor inputs, a single ordered pair, retained unverified camera extrinsics and S failure remain confounds.',
        'Action32 comparison does not turn S failure into a successful episode or erase C40 endpoint.',
        'Native reliability proxy is not TSDF accuracy, geometric ground truth or causal semantic benefit.',
        'This diagnostic was chosen after seeing the unequal-support published results; no protocol-level confirmatory inference.'])
    result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'common_support_count':len(keys),'arms':summaries,'elapsed_s':result['elapsed_s']}),flush=True)

if __name__=='__main__':main()
