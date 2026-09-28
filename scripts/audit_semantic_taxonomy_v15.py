#!/usr/bin/env python3
"""Observation-only taxonomy conversion audit on sealed Q and D12 features."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from nso.semantic_taxonomy_v15 import canonical_feature_record,canonical_observed_records
from scripts.collect_semantic_gain_v13_history import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sources={
        'competition_v9':Path('audit_results/semantic_opportunities_v14_1_observed_inputs_20260914'),
        'inspection_v4':Path('eval_results/semantic_v15_canonical_histories_20260914')}
    inventory={}
    for name,folder in sources.items():
        inv=json.loads((folder/'artifact_hashes.json').read_text())
        for path,h in inv.items():
            if sha(folder/path)!=h:raise ValueError('input artifact changed')
        inventory[str(folder)]=sha(folder/'artifact_hashes.json')
    a.output.mkdir(parents=True,exist_ok=False)
    names=['nso/semantic_taxonomy_v15.py',str(Path(__file__).relative_to(ROOT)),
           'tests/virtual3d/test_semantic_taxonomy_v15.py','scripts/collect_semantic_gain_v13_history.py']
    frozen={n:sha(Path(n)) for n in names}
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(n,n)
    write(a.output/'manifest.json',dict(status='running',source_sha256=frozen,input_inventories=inventory,
        future_rewards_read=False,training_allowed=False,runtime_policy_modified=False))
    states=[]
    for folder in sorted(sources['competition_v9'].glob('*_step_*')):
        observed=json.loads((folder/'inputs.json').read_text())
        states.append(('competition_v9',folder.name,json.loads((folder/'features.json').read_text()),observed['descriptors']))
    for path in sorted(sources['inspection_v4'].glob('structure_*/all_decisions.json')):
        for row in json.loads(path.read_text()):
            states.append(('inspection_v4',path.parent.name+f"_step_{row['action_id']}",row['features'],row['descriptors']))
    summaries=[];converted=[];max_error=0.
    for source,state,features,descriptors in states:
        if len(features)!=len(descriptors):raise ValueError('candidate descriptor count differs')
        changed=0
        for feature,desc in zip(features,descriptors):
            row=canonical_feature_record(feature,source)
            observed=canonical_observed_records(desc['observed_assets'],source)
            kernels=np.asarray(feature['instance_kernels'],dtype=float).reshape(len(observed),8)
            signed=np.array([r['class_vote']*r['semantic_confidence'] for r in observed])
            # Independently rebuild the signed moments from observed instances.
            expected=np.array([float(np.sum(np.sort(signed*kernels[:,i]),dtype=np.float64))/len(observed)
                if observed else 0. for i in range(8)])
            actual=np.asarray(row['semantic'][18:]);error=float(np.max(np.abs(expected-actual)))
            max_error=max(max_error,error)
            if error>1e-12:raise ValueError('encoded feature does not match observed class moments')
            if row['geometry']!=feature['geometry'] or row['semantic'][:18]!=feature['semantic'][:18]:
                raise ValueError('geometry or common capacity changed')
            if row['residual_confidence']!=feature['residual_confidence']:raise ValueError('confidence changed')
            changed+=row['semantic']!=feature['semantic']
            converted.append(dict(source=source,state=state,feature=row))
        summaries.append(dict(source=source,state=state,candidates=len(features),changed_signed_tails=changed))
    for n,h in frozen.items():
        if sha(Path(n))!=h:raise ValueError('audit source changed')
    write(a.output/'canonical_features.json',converted)
    write(a.output/'result.json',dict(status='passed',states=len(states),candidate_features=len(converted),
        summaries=summaries,geometry_and_common_inputs_exact=True,confidence_unchanged=True,
        dimension_unchanged=26,maximum_signed_moment_reconstruction_error=max_error,
        training_allowed=False,runtime_policy_modified=False,semantic_efficacy_proven=False,
        boundary='Semantic encoding repair only; source metadata is not a model input and layouts remain development data.'))
    write(a.output/'manifest.json',dict(status='complete',source_sha256=frozen,input_inventories=inventory,
        future_rewards_read=False,training_allowed=False,runtime_policy_modified=False))
    write(a.output/'artifact_hashes.json',{str(f.relative_to(a.output)):sha(f) for f in sorted(a.output.iterdir()) if f.is_file()})
    print(json.dumps(dict(states=len(states),candidate_features=len(converted),maximum_error=max_error)),flush=True)


if __name__=='__main__':main()
