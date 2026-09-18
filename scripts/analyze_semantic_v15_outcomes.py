#!/usr/bin/env python3
"""Post-execution observed-group information screen for the sealed V15 pilot."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nso.information_value_v15 import observed_group_information
from scripts.collect_semantic_gain_v13_history import sha,write
from scripts.collect_semantic_v15_outcomes import branch_name,verify_sources,verify_inputs,seal


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.source/'manifest.json').read_text())
    if m['status']!='complete':raise ValueError('complete acquisition and replay required')
    verify_sources(m);protocol=m['protocol'];verify_inputs(protocol)
    for name,h in json.loads((a.source/'artifact_hashes.json').read_text()).items():
        if sha(a.source/name)!=h:raise ValueError('outcome artifact changed')
    rows=json.loads((a.source/'partial.json').read_text())
    expected=[(c['structure_seed'],s['action_id'],cid) for s in protocol['states']
              for c in s['certificates'] for cid in c['candidate_ids']]
    lookup={(r['structure_seed'],r['action_id'],r['candidate_id']):r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!=set(expected) or len(rows)!=46:
        raise ValueError('missing, duplicate or extra outcomes')
    checks=[]
    for r in rows:
        folder=a.source/branch_name(r)
        if json.loads((folder/'verification.json').read_text())['status']!='passed':
            raise ValueError('physical replay missing')
        if (r['total_budget']!=protocol['total_task_budget'] or r['future_rewards_used_by_planner']
                or r['intervention']['extra_global_selection_calls']!=0):
            raise ValueError('execution contract differs')
        archive=json.loads((folder/'archive_index.json').read_text())
        for name,record in archive['files'].items():
            stored=folder/record['stored_file']
            if sha(stored)!=record['compressed_sha256']:raise ValueError('compressed log changed')
            raw=gzip.decompress(stored.read_bytes())
            if hashlib.sha256(raw).hexdigest()!=record['raw_sha256']:raise ValueError('log recovery differs')
        decisions=json.loads(gzip.decompress((folder/'all_decisions.json.gz').read_bytes()))
        subsequent=[d for d in decisions if d['ordinal']>r['intervention']['decision_ordinal']]
        checks.append(dict(structure_seed=r['structure_seed'],action_id=r['action_id'],candidate_id=r['candidate_id'],
            first_target_reached=r['first_target_reached'],
            subsequent_global_decisions=len(subsequent),
            first_subsequent_action_id=subsequent[0]['action_id'] if subsequent else None))
    results=[]
    for ref in protocol['evaluation']['reference_seeds']:
        states=[]
        for s in protocol['states']:
            ids=s['certificates'][0]['candidate_ids']
            if any(c['candidate_ids']!=ids for c in s['certificates']):raise ValueError('pool IDs mismatch')
            groups=[]
            for members in s['observed_semantic_groups']:
                hashes={c['observed_semantic_features_sha256'] for c in s['certificates'] if c['structure_seed'] in members}
                if len(hashes)!=1:raise ValueError('declared observation group mixes features')
            for seed in s['structure_seeds']:
                indices=[i for i,g in enumerate(s['observed_semantic_groups']) if seed in g]
                if len(indices)!=1:raise ValueError('observation partition is not unique')
                groups.append(str(indices[0]))
            rewards=[[lookup[(seed,s['action_id'],cid)]['after'][str(ref)]['joint_05cm']
                      -lookup[(seed,s['action_id'],cid)]['before'][str(ref)]['joint_05cm']
                      for cid in ids] for seed in s['structure_seeds']]
            values=observed_group_information(rewards,groups)
            states.append(dict(action_id=s['action_id'],structure_seeds=s['structure_seeds'],
                candidate_ids=ids,rewards=rewards,**values))
        info=sum(s['information_value'] for s in states)/len(states)
        blind=sum(s['geometry_blind_oracle'] for s in states)/len(states)
        relative=info/blind if blind>protocol['pilot_screen']['minimum_blind_mean'] else None
        results.append(dict(reference_seed=ref,states=states,mean_information_value=info,
            mean_geometry_blind_oracle=blind,relative_information_value=relative,
            numerical_single_parent_screen_passed=relative is not None and
                relative>=protocol['pilot_screen']['relative_observed_information_minimum']))
    physical=all(r['collisions']==0 and not r['termination']['failed']
        and r['termination']['returned_to_anchor'] and r['total_paid_actions']<=r['total_budget'] for r in rows)
    result=dict(status='complete',candidate_outcomes=len(rows),physically_replayed=len(rows),
        physical_screen_passed=physical,reference_results=results,
        single_parent_necessary_screen_passed=physical and all(r['numerical_single_parent_screen_passed'] for r in results),
        next_decision_checks=checks,parent_groups=1,training_allowed=False,
        semantic_model_efficacy_proven=False,multi_decision_semantic_efficacy_proven=False,
        independent_confirmation=False)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'result.json',result)
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
             'nso/information_value_v15.py':sha(ROOT/'nso/information_value_v15.py')}
    write(a.output/'manifest.json',dict(status='complete',source_sha256=sources,
        input_inventory_sha256=sha(a.source/'artifact_hashes.json')))
    for name in sources:(a.output/Path(name).name).write_bytes((ROOT/name).read_bytes())
    seal(a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('reference_results','next_decision_checks')}),flush=True)
    for r in results:print(r['reference_seed'],r['relative_information_value'],flush=True)


if __name__=='__main__':main()
