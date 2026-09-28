#!/usr/bin/env python3
"""Observed-region continuation preflight; no motion or policy rollout."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from copy import deepcopy
import gzip,json
from pathlib import Path
import sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.observed_asset_axes_v16 import measured_assets_v16
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.semantic_taxonomy_v15 import canonical_observed_records
from nso.decision_replay_v13 import load_packet
from nso.visibility_corrected_scores_v17 import correct_scores
from nso.observed_region_evidence_v17 import observed_regions
from nso.observed_continuation_v17 import select_continuation
from scripts.audit_observed_region_evidence_v17 import check_map
from scripts.collect_semantic_gain_v13_history import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.source/'manifest.json').read_text());protocol=m['protocol']
    for folder in (a.source,a.preflight):
        if json.loads((folder/'manifest.json').read_text())['status']!='complete':raise ValueError('complete evidence required')
        for n,v in json.loads((folder/'artifact_hashes.json').read_text()).items():
            if sha(folder/n)!=v:raise ValueError('input changed')
    for n,v in m['source_sha256'].items():
        if sha(ROOT/n)!=v:raise ValueError('frozen dependency changed')
    files=set(m['source_sha256'])|{str(Path(__file__).relative_to(ROOT)),
        'nso/visibility_corrected_scores_v17.py','nso/observed_region_evidence_v17.py',
        'scripts/audit_observed_region_evidence_v17.py','tests/virtual3d/test_visibility_corrected_scores_v17.py',
        'nso/observed_continuation_v17.py','tests/virtual3d/test_observed_continuation_v17.py'}
    sources={n:sha(ROOT/n) for n in sorted(files)};a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',source_sha256=sources,training_allowed=False,policy_executed=False,
        input_inventory_sha256={str(p):sha(p/'artifact_hashes.json') for p in (a.source,a.preflight)})
    write(a.output/'manifest.json',manifest)
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in sources:z.write(ROOT/n,n)
    try:
        history=ROOT/protocol['history'];hp=json.loads((history/'manifest.json').read_text())['protocol']
        doc=json.loads((ROOT/hp['scene_protocol']).read_text());c=next(x for x in doc['contexts'] if x['id']==hp['context'])
        config=InspectionConfigV4(**{**doc['shared_conditions'],**{k:v for k,v in c.items() if k not in ('id','seed')}})
        hf=history/f"structure_{protocol['structure_seed']}";bounds=json.loads((hf/'all_decisions.json').read_text())[0]['bounds']
        shape=(bounds[1],bounds[3]);seen=set();rows=[]
        for d in protocol['declarations']:
            first=d['first_candidate']['candidate_id'];second=d['second_candidate']['candidate_id']
            if first in seen:continue
            seen.add(first);folder=a.source/f'first_{first}_second_{second}'
            saved=json.loads(gzip.decompress((folder/'deterministic.json.gz').read_bytes()))
            mapper=ObservedRuntimeMapperV10(shape,config);frames=[]
            for aid in range(d['arrival_action_id']+1):
                packet=load_packet((hf if aid<=protocol['action_id'] else folder)/f'packets/{aid:04d}.npz')
                mapper.update(packet.frame,packet.scan);frames.append(packet.frame)
                if aid==protocol['action_id']:
                    initial_assets=measured_assets_v16(AxisHistoryViewV16(mapper,frames))
                    initial_regions=observed_regions(mapper,initial_assets)
                    initial_anchors=[r for r in initial_regions if r['asset_index'] is not None]
                    anchor=next(r for r in initial_anchors if r['asset_index']==d['first_candidate']['asset_index'])
                    peers=[r for r in initial_anchors if r is not anchor]
            check_map(mapper,saved['first_arrival_state']['evidence'])
            assets=canonical_observed_records(measured_assets_v16(AxisHistoryViewV16(mapper,frames)),'inspection_v4')
            before=json.loads((a.preflight/f'after_{first}.json').read_text())['selection']
            routes=before['candidates'];support=before['score_audit'][0]['measured_direction_support']
            scores,audit=correct_scores(mapper,assets,support,routes,before['scores'],before['score_audit'])
            regions=observed_regions(mapper,assets)
            continuation=select_continuation(anchor,regions,peers,routes,scores['G'],audit)
            for intervention in ('swap','zero'):
                altered=deepcopy(assets)
                for asset in altered:
                    if intervention=='swap':asset['class_vote']=-asset['class_vote']
                    else:asset['marked_points']=0
                controls,_=correct_scores(mapper,altered,support,routes,before['scores'],before['score_audit'])
                if any(controls[x]!=scores[x] for x in ('N','G','M')):raise ValueError('geometry score changed under labels')
                if controls['S']!=scores['X' if intervention=='swap' else 'G']:raise ValueError('semantic score contract')
                if select_continuation(anchor,regions,peers,routes,controls['G'],audit)!=continuation:
                    raise ValueError('common continuation changed under labels')
            check_map(mapper,saved['first_arrival_state']['evidence'])
            def choice(values):return min(range(len(routes)),key=lambda i:(-values[i],routes[i]['cost'],routes[i]['candidate_id']))
            row=dict(continuation=continuation,anchor=anchor,peers=peers,current_regions=regions,first_candidate_id=first,candidates=len(routes),old_scores=before['scores'],corrected_scores=scores,
                audit=audit,old_choices={k:choice(v) for k,v in before['scores'].items()},
                corrected_choices={k:choice(v) for k,v in scores.items()},candidate_pool_unchanged=True,
                geometry_and_zero_confidence_controls_passed=True,map_mesh_unchanged=True,
                common_coverage_quality_term_unchanged=True,executed_policy=False,learned_feature_contract_updated=False)
            rows.append(row);write(a.output/f'after_{first}.json',row)
        for n,v in sources.items():
            if sha(ROOT/n)!=v:raise ValueError('source changed during audit')
        write(a.output/'result.json',dict(status='passed',states=len(rows),candidates=sum(x['candidates'] for x in rows),
            continuations=sum(x['continuation']['status']=='continue_observed_region' for x in rows),
            common_fallbacks=sum(x['continuation']['status']=='common_geometry_fallback' for x in rows),
            changed_geometry_choices=sum(x['old_choices']['G']!=x['corrected_choices']['G'] for x in rows),
            executed_new_policy=False,semantic_efficacy_proven=False,training_allowed=False))
        manifest['status']='complete'
    except Exception as error:manifest.update(status='failed',error=repr(error));raise
    finally:
        write(a.output/'manifest.json',manifest)
        write(a.output/'artifact_hashes.json',{p.name:sha(p) for p in sorted(a.output.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':main()
