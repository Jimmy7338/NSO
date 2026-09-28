#!/usr/bin/env python3
"""Summarize activation, candidate reranking, labels and paired ablations."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np


def analyze(folder):
    folder=Path(folder)
    episodes=[json.loads(s) for s in (folder/'episodes.jsonl').read_text().splitlines()]
    if any(e['status'] != 'complete' for e in episodes):
        raise ValueError('mechanism report requires complete episodes')
    summary=json.loads((folder/'summary.json').read_text())
    if not summary['run_complete']:
        raise ValueError('incomplete configured matrix')
    report={'scope':'development synthetic mechanism verification; no held-out claims',
            'activation':{},'paired_ablations':{},'label_status':{},'decision_reranking':{}}
    for method in summary['methods']:
        runs=[e for e in episodes if e['method']==method]
        report['activation'][method]={k:sum(e.get(k,0) for e in runs)
                                      for k in ('semantic_updates','topo_updates','rpn_calls','goal_changes')}
        statuses=Counter();changed=total=0
        for e in runs:
            path=folder/e['artifact_dir']
            if not (path/'candidates.jsonl').exists():continue
            choices=defaultdict(list)
            for line in (path/'candidates.jsonl').read_text().splitlines():
                row=json.loads(line)
                if not row['excluded']:choices[row['step']].append(row)
            for rows in choices.values():
                chosen=next((r for r in rows if r['selected']),None)
                if chosen is None:continue
                baseline=min(rows,key=lambda r:(-r['gain_per_cost'],r['features'][0],r['goal']))
                total+=1;changed+=chosen['goal'] != baseline['goal']
            for line in (path/'goal_attempts.jsonl').read_text().splitlines():
                statuses[json.loads(line)['status']]+=1
        report['label_status'][method]=dict(statuses)
        report['decision_reranking'][method]={'selections':total,'different_from_gain_on_same_candidates':changed}
    pairs=[('gain_per_cost','gain_control'),('gain_control','gain_semantic'),
           ('gain_control','gain_structure'),('gain_control','gain_semantic_structure'),
           ('gain_semantic_structure','gain_semantic_structure_rpn'),
           ('gain_control','gain_budget_rule'),
           ('gain_semantic_structure','gain_semantic_structure_budget_rule'),
           ('gain_semantic_structure_budget_rule','gain_semantic_structure_rpn'),
           ('gain_semantic_structure','gain_semantic_structure_budget_soft'),
           ('gain_semantic_structure_budget_soft','gain_semantic_structure_rpn')]
    for a,b in pairs:
        left={(e['map_id'],e['start_seed']):e for e in episodes if e['method']==a}
        right={(e['map_id'],e['start_seed']):e for e in episodes if e['method']==b}
        if not left or not right:continue
        if left.keys()!=right.keys():raise ValueError('unpaired ablation')
        differences={}
        for field in ('coverage_ratio','coverage_auc','revisit_ratio','collisions'):
            bymap=defaultdict(list)
            for key in left:bymap[key[0]].append(right[key][field]-left[key][field])
            means={m:float(np.mean(v)) for m,v in bymap.items()}
            differences[field]={'mean':float(np.mean(list(means.values()))),'by_map':means}
        changed=0
        for key in left:
            with np.load(folder/left[key]['artifact_dir']/'observations.npz') as x, np.load(folder/right[key]['artifact_dir']/'observations.npz') as y:
                changed+=not np.array_equal(x['poses'],y['poses'])
        report['paired_ablations'][b+' minus '+a]={'differences':differences,'different_trajectories':changed,'paired_episodes':len(left)}
    (folder/'mechanism_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_directory',type=Path)
    analyze(p.parse_args().run_directory)
