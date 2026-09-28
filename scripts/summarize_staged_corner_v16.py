#!/usr/bin/env python3
"""Summarize complete proposal checks and freeze an all-new-role smoke scope."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.source/'manifest.json').read_text())
    if m['status']!='complete':raise ValueError('complete preflight required')
    for n,v in json.loads((a.source/'artifact_hashes.json').read_text()).items():
        if sha(a.source/n)!=v:raise ValueError('input changed')
    rows={s:json.loads((a.source/f'structure_{s}_decisions.json').read_text()) for s in range(27101,27105)}
    paired=[]
    # These exact geometry-paired states were fixed in the earlier V15 protocol.
    for aid,seeds in [(26,list(rows)),(53,[27101,27104])]:
        chosen=[next(x for x in rows[s] if x['action_id']==aid) for s in seeds]
        pools=len({digest(x['candidates']) for x in chosen})==1
        geometry=len({digest([f['geometry'] for f in x['features']]) for x in chosen})==1
        if not pools or not geometry:raise ValueError('previous matched state pairing lost')
        paired.append(dict(action_id=aid,structure_seeds=seeds,candidate_pools_equal=pools,
            feature_geometry_equal=geometry,candidate_count=len(chosen[0]['candidates'])))
    flat=[x for group in rows.values() for x in group]
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'summary.json',dict(status='passed',parent_layouts=1,decisions=len(flat),
        candidates=sum(x['candidate_count'] for x in flat),
        base_candidates_preserved=sum(x['base_candidates_preserved'] for x in flat),
        corner_candidates=sum(x['corner_candidates'] for x in flat),
        states_with_corner_candidates=sum(x['corner_candidates']>0 for x in flat),paired_states=paired,
        input_inventory_sha256=sha(a.source/'artifact_hashes.json'),new_physical_outcomes=0,
        training_allowed=False,semantic_efficacy_proven=False,source_sha256=sha(Path(__file__))))
    row=next(x for x in rows[27101] if x['action_id']==26)
    write(a.output/'next_physical_smoke_scope.json',dict(schema='staged_corner_smoke_scope/1',
        status='declared_not_executed',input_root=str(a.source),structure_seed=27101,action_id=26,
        state_selection='first existing strictly geometry-paired semantic-eligible decision; first declared structure',
        candidate_pool_sha256=row['candidate_sha256'],
        candidates=[c for c in row['candidates'] if '_corner_' in c['group']],
        selection_rule='all newly appended corner roles at this state; no outcome selection',
        total_task_budget=192,paid_prefix_actions=26,
        execution='recorded prefix then guarded outbound; record actual arrival or denial; guarded return',
        candidate_reached_not_assumed=True,record_all_failed_attempts=True,
        future_replanning='capture measured state and shared pool at actual arrival; do not execute a second inspection in this smoke',
        validation='independent physical replay, packet identity, map/mesh, observed 3D support and directions; no semantic efficacy claim',
        pending='implement collector and replay, then freeze their source hashes before execution'))
    write(a.output/'artifact_hashes.json',{x.name:sha(x) for x in sorted(a.output.iterdir()) if x.is_file()})


if __name__=='__main__':main()
