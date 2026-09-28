#!/usr/bin/env python3
"""Fixed-alpha leave-one-seed-out ranking probe on already viewed development data."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.counterfactual_view_scoring import _mapper_snapshot
from nso.semantic_view_response_v6 import response_features,RelativeResponseRidge,FEATURE_NAMES
from scripts.eval_counterfactual_views import remap,write_json
from utils.rgbd_contract import RGBDFrame,PlanarScan
from utils.cpu_protocol import file_hash


def read(path):return json.loads(path.read_text())


def run(source,output):
    meta=read(source/'metadata.json');summary=read(source/'summary.json')
    assert meta['status']=='complete' and read(source/'verification.json')['status']=='passed_full'
    assert {h['seed'] for h in summary['histories']}=={751,752}
    output.mkdir(parents=True,exist_ok=False);started=time.time();original_hashes=read(source/'artifact_hashes.json')
    names=sorted(set(meta['source_sha256'])|{'nso/semantic_view_response_v6.py','scripts/probe_semantic_view_response_v6.py'})
    hashes={name:file_hash(ROOT/name) for name in names}
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
    protocol={'status':'running','source':str(source.resolve()),'scope':'already viewed development data; leave-one-geometry-seed-out diagnostic, not independent confirmation',
        'fixed_alpha':1.,'targets':['area_per_action','f1_gain_05cm'],'primary_selection_target':'area_per_action',
        'history_centering':'train X and y; test X only; no test y or intercept supplied to prediction',
        'G_O_N':'separately retrained strong ablations','X':'train aligned S, swap test labels only',
        'M':'actual absent-label features and explicit fitted-G fallback',
        'G_shared_O_shared':'frozen fitted-S parameters with geometric/objectness feature priors; separate mechanism controls',
        'new_physical_branches':0,
        'source_sha256':hashes,'feature_names':list(FEATURE_NAMES),'original_gates_changed':False,
        'checked_seeds_used':False,'independent_seeds_used':False}
    write_json(output/'protocol.json',protocol);inputs={};histories=[]
    def checked(path):
        sha=file_hash(path);key=str(path.relative_to(source));assert original_hashes[key]==sha,key
        inputs[str(path.resolve())]=sha;return path
    for h in summary['histories']:
        fixture=source/h['fixture'];entry=read(checked(fixture/'fixture.json'));c=InspectionConfigV4(**entry['environment'])
        index=h['history_step'];rows=read(checked(fixture/'prefix/records.json'))[:index+1]
        frames=[RGBDFrame.load(checked(fixture/'prefix/frames'/f'{i:04d}.npz')) for i in range(index+1)]
        scans=[PlanarScan.load(checked(fixture/'prefix/scans'/f'{i:04d}.npz')) for i in range(index+1)]
        pred=read(checked(source/h['predictions']));folder=(source/h['predictions']).parent
        routes=read(checked(folder/'candidates.json'))
        mapper=remap(frames,scans,rows,c,(round(c.height_m/c.resolution_m),round(c.width_m/c.resolution_m)))
        assert _mapper_snapshot(mapper)['hashes']==pred['invariants']['mapper_hashes']['aligned']
        features,audit=response_features(mapper,routes,pred)
        target=output/h['fixture']/f'history_{index:04d}';target.mkdir(parents=True)
        np.savez_compressed(target/'features.npz',**features)
        write_json(target/'feature_audit.json',audit)
        # Features are fully computed before this history's response labels load.
        outcomes={r['candidate_id']:r for r in read(checked(source/h['outcomes']))}
        y=np.array([[outcomes[r['candidate_id']][key] for key in protocol['targets']] for r in routes])
        histories.append({'history':h,'features':features,'y':y,'routes':routes,'outcomes':outcomes})
        print('features',h['seed'],c.depth_sigma_m,index,flush=True)
        del mapper
    choices=[];models=[]
    for held_seed in (751,752):
        train=[h for h in histories if h['history']['seed']!=held_seed]
        test=[h for h in histories if h['history']['seed']==held_seed]
        y=np.concatenate([h['y'] for h in train]);ids=np.concatenate([np.full(len(h['y']),i) for i,h in enumerate(train)])
        fitted={name:RelativeResponseRidge.fit(np.concatenate([h['features'][name] for h in train]),y,ids,alpha=1.)
                for name in ('G','O','S','N')}
        for name,model in fitted.items():
            models.append({'held_out_seed':held_seed,'name':name,'train_seeds':[s for s in (751,752) if s!=held_seed],
                           **{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in asdict(model).items()}})
        for h in test:
            output_scores={}
            for name in ('G','O','S','X','M','N','G_shared','O_shared'):
                feature_name=name.split('_')[0]
                model=fitted['S'] if name in ('X','G_shared','O_shared') else fitted['G'] if name=='M' else fitted[name]
                prediction=model.predict(h['features'][feature_name]);output_scores[name]=prediction
                ranks=sorted(range(len(prediction)),key=lambda i:(-prediction[i,0],h['routes'][i]['cost'],i))
                chosen=ranks[0];cid=h['routes'][chosen]['candidate_id'];outcome=h['outcomes'][cid]
                actual=h['y'][:,0];pairs=[(i,j) for i in range(len(actual)) for j in range(i+1,len(actual)) if actual[i]!=actual[j]]
                accuracy=float(np.mean([1 if (prediction[i,0]-prediction[j,0])*(actual[i]-actual[j])>0 else
                        .5 if prediction[i,0]==prediction[j,0] else 0 for i,j in pairs])) if pairs else None
                correlation=float(spearmanr(prediction[:,0],actual).statistic) if np.ptp(prediction[:,0]) and np.ptp(actual) else None
                choices.append({**{k:h['history'][k] for k in ('fixture','history_step','seed','depth_sigma_m')},
                    'scorer':name,'chosen':cid,'train_seed':751 if held_seed==752 else 752,
                    'predicted_relative_area_rate':float(prediction[chosen,0]),
                    'predicted_relative_f1_gain':float(prediction[chosen,1]),
                    'area_per_action':outcome['area_per_action'],'f1_gain_05cm':outcome['f1_gain_05cm'],
                    'coverage_gain_m2':outcome['coverage_gain_m2'],'new_area_m2':outcome['new_area_m2'],
                    'spearman':correlation,'pairwise_accuracy':accuracy,
                    'area_rate_regret':float(actual.max()-actual[chosen])})
            np.testing.assert_array_equal(output_scores['G'],output_scores['M'])
            target=output/h['history']['fixture']/f"history_{h['history']['history_step']:04d}"
            write_json(target/'predictions.json',{name:v.tolist() for name,v in output_scores.items()})
    metrics={}
    for name in ('G','O','S','X','M','N','G_shared','O_shared'):
        selected=[row for row in choices if row['scorer']==name]
        metrics[name]={key:float(np.mean([r[key] for r in selected])) for key in (
            'area_per_action','f1_gain_05cm','coverage_gain_m2','area_rate_regret')}
        metrics[name]['per_seed']={str(seed):{key:float(np.mean([r[key] for r in selected if r['seed']==seed]))
            for key in ('area_per_action','f1_gain_05cm','area_rate_regret')} for seed in (751,752)}
    assert all(file_hash(ROOT/name)==sha for name,sha in hashes.items()),'sources changed during probe'
    assert all(file_hash(Path(name))==sha for name,sha in inputs.items())
    write_json(output/'models.json',models);write_json(output/'choices.json',choices)
    write_json(output/'summary.json',{'metrics':metrics,'histories':6,'physical_branches_reused':72,
               'no_absolute_calibration_claim':True,'independent_confirmation':False,'gates_passed':False})
    write_json(output/'input_hashes.json',inputs)
    protocol.update(status='complete',elapsed_s=time.time()-started,source_inputs_checked=len(inputs))
    write_json(output/'protocol.json',protocol)
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
