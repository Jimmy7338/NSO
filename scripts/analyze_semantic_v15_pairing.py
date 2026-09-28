#!/usr/bin/env python3
"""Exact pairing of freshly executed canonical-sensor histories, no rewards."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nso.cpu_sensor_contract_v10 import digest
from scripts.audit_semantic_gain_v13_pairing import prefix_geometry_hashes
from scripts.collect_semantic_gain_v13_history import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();source=args.source
    m=json.loads((source/'manifest.json').read_text())
    if m['status']!='complete':raise ValueError('completed and replayed histories required')
    for name,h in m['source_sha256'].items():assert sha(Path(name))==h,name
    inv=json.loads((source/'artifact_hashes.json').read_text())
    for name,h in inv.items():assert sha(source/name)==h,name
    args.output.mkdir(parents=True,exist_ok=False)
    seeds=m['protocol']['structure_seeds'];histories={};decisions={}
    for seed in seeds:
        folder=source/f'structure_{seed}'
        assert json.loads((folder/'verification.json').read_text())['status']=='passed'
        histories[seed]=prefix_geometry_hashes(folder)
        rows=json.loads((folder/'all_decisions.json').read_text())
        if len({r['action_id'] for r in rows})!=len(rows):raise ValueError('multiple decisions at same step require ordinal pairing')
        decisions[seed]={r['action_id']:r for r in rows}
    base=seeds[0];rows=[]
    for seed in seeds[1:]:
        for step in sorted(set(decisions[base])|set(decisions[seed])):
            row=dict(base_seed=base,alternative_seed=seed,action_id=step,parent='D12-00')
            if step not in decisions[base] or step not in decisions[seed]:
                rows.append({**row,'matched':False,'eligible_semantic_pair':False,'reason':'no_common_natural_decision'});continue
            a,b=decisions[base][step],decisions[seed][step]
            gh=histories[base][0][step]==histories[seed][0][step]
            sh=histories[base][1][step]!=histories[seed][1][step]
            pool=a['candidate_sha256']==b['candidate_sha256']
            features=[f['geometry'] for f in a['features']]==[f['geometry'] for f in b['features']]
            # Explicit complete geometry feedback module state, as in V13.
            feedback=digest(a['state_before']['evidence']['modules'])==digest(b['state_before']['evidence']['modules'])
            votes=lambda r:[[v['class_vote'],v['semantic_confidence']] for v in r['descriptors'][0]['observed_assets']] if r['descriptors'] else []
            changed=votes(a)!=votes(b)
            matched=gh and pool and features and feedback
            rows.append({**row,'matched':matched,'depth_scan_pose_history_equal':gh,'semantic_history_changed':sh,
                'pool_equal':pool,'geometry_features_equal':features,'feedback_equal':feedback,
                'observed_class_evidence_changed':changed,'candidate_count_base':len(a['candidates']),
                'candidate_count_alternative':len(b['candidates']),
                'eligible_semantic_pair':matched and sh and changed and len(a['candidates'])>1})
    result=dict(status='complete',rows=rows,parent_groups=1,
        matched_pairs=sum(r['matched'] for r in rows),
        eligible_semantic_pairs=sum(r['eligible_semantic_pair'] for r in rows),
        future_candidate_outcomes=0,training_allowed=False,semantic_efficacy_proven=False,
        boundary='Shared baseline variants are dependent; all belong to one already-viewed layout. RGB shape equivalence not certified.')
    write(args.output/'result.json',result)
    write(args.output/'manifest.json',dict(status='complete',input_inventory_sha256=sha(source/'artifact_hashes.json'),
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                       'scripts/audit_semantic_gain_v13_pairing.py':sha(Path('scripts/audit_semantic_gain_v13_pairing.py'))}))
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    write(args.output/'artifact_hashes.json',{str(p.relative_to(args.output)):sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
