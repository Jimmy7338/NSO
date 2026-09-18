#!/usr/bin/env python3
"""Observed-only region/occlusion diagnosis of every completed second view."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,gzip,json
from pathlib import Path
import sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from env.virtual3d import camera_pose
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.observed_asset_axes_v16 import measured_assets_v16
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.observed_region_evidence_v17 import observed_regions,link_region,actual_depth_evidence,predicted_mesh_evidence
from nso.decision_replay_v13 import load_packet,array_hash
from scripts.collect_semantic_gain_v13_history import sha,write


def check_map(mapper,expected):
    arrays={k:array_hash(v) for k,v in vars(mapper).items() if isinstance(v,np.ndarray)}
    mesh=mapper.mesh();hashes={k:array_hash(np.asarray(getattr(mesh,k))) for k in ('vertices','triangles','vertex_colors')}
    if arrays!=expected['map_arrays'] or hashes!=expected['mesh']:raise ValueError('recorded map/mesh differs')
    return mesh


def plane_points(asset):
    offsets,heights=np.meshgrid(np.linspace(-.5,.5,5)*asset['measured_width_m'],
        np.linspace(asset['observed_low'][2],asset['observed_high'][2],5))
    return np.column_stack([asset['rear_boundary_xy'][0]+offsets.ravel()*asset['side_axis'][0],
        asset['rear_boundary_xy'][1]+offsets.ravel()*asset['side_axis'][1],heights.ravel()])


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.source/'manifest.json').read_text());protocol=m['protocol']
    if m['status']!='complete':raise ValueError('complete physical probes required')
    for n,v in json.loads((a.source/'artifact_hashes.json').read_text()).items():
        if sha(a.source/n)!=v:raise ValueError('input changed')
    for n,v in m['source_sha256'].items():
        if sha(ROOT/n)!=v:raise ValueError('frozen source changed')
    files=set(m['source_sha256'])|{str(Path(__file__).relative_to(ROOT)),
        'nso/observed_region_evidence_v17.py','tests/virtual3d/test_observed_region_evidence_v17.py'}
    sources={n:sha(ROOT/n) for n in sorted(files)}
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',source_sha256=sources,input_inventory_sha256=sha(a.source/'artifact_hashes.json'),
        new_policy_executed=False,planner_changed=False,training_allowed=False)
    write(a.output/'manifest.json',manifest)
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    try:
        history=ROOT/protocol['history'];hp=json.loads((history/'manifest.json').read_text())['protocol']
        doc=json.loads((ROOT/hp['scene_protocol']).read_text());c=next(x for x in doc['contexts'] if x['id']==hp['context'])
        config=InspectionConfigV4(**{**doc['shared_conditions'],**{k:v for k,v in c.items() if k not in ('id','seed')}})
        hf=history/f"structure_{protocol['structure_seed']}";bounds=json.loads((hf/'all_decisions.json').read_text())[0]['bounds']
        shape=(bounds[1],bounds[3]);results=[]
        for d in protocol['declarations']:
            first,second=d['first_candidate'],d['second_candidate']
            folder=a.source/f"first_{first['candidate_id']}_second_{second['candidate_id']}"
            saved=json.loads(gzip.decompress((folder/'deterministic.json.gz').read_bytes()))
            outcome=saved['outcome'];arrival=d['arrival_action_id']+outcome['second_view_actions']
            mapper=ObservedRuntimeMapperV10(shape,config);frames=[]
            for aid in range(arrival+1):
                path=(hf if aid<=protocol['action_id'] else folder)/f'packets/{aid:04d}.npz'
                packet=load_packet(path);mapper.update(packet.frame,packet.scan);frames.append(packet.frame)
                if aid==protocol['action_id']:
                    check_map(mapper,saved['prefix_state']['evidence'])
                    initial_assets=measured_assets_v16(AxisHistoryViewV16(mapper,frames))
                    initial_regions=observed_regions(mapper,initial_assets)
                    initial_anchors=[r for r in initial_regions if r['asset_index'] is not None]
                    initial_anchor=next(r for r in initial_anchors if r['asset_index']==first['asset_index'])
                if aid==d['arrival_action_id']:
                    first_mesh=check_map(mapper,saved['first_arrival_state']['evidence'])
                    first_assets=measured_assets_v16(AxisHistoryViewV16(mapper,frames))
                    first_regions=observed_regions(mapper,first_assets)
                    first_link=link_region(initial_anchor,first_regions,[r for r in initial_anchors if r is not initial_anchor])
                    first_anchors=[r for r in first_regions if r['asset_index'] is not None]
                    target=first_assets[second['asset_index']]
                    anchor=next(r for r in first_anchors if r['asset_index']==second['asset_index'])
                    planned_pose=camera_pose(second['pose'][:2],second['pose'][2],config,shape[0])
                    predictions={name:predicted_mesh_evidence(points,planned_pose,packet.frame.intrinsic,
                        packet.frame.depth_m.shape,config,first_mesh,anchor['keys'])
                        for name,points in [('measured_points',target['points']),('hypothetical_plane',plane_points(target))]}
            check_map(mapper,saved['arrival_state']['evidence'])
            last_assets=measured_assets_v16(AxisHistoryViewV16(mapper,frames));last_regions=observed_regions(mapper,last_assets)
            second_link=link_region(anchor,last_regions,[r for r in first_anchors if r is not anchor])
            actual={name:actual_depth_evidence(points,packet.frame,config)
                for name,points in [('measured_points',target['points']),('hypothetical_plane',plane_points(target))]}
            if actual['measured_points']['consistent']!=outcome['actual_second_arrival_depth_support_on_initial_face']:
                raise ValueError('actual depth support count differs from physical probe')
            row=dict(first_candidate_id=first['candidate_id'],second_candidate_id=second['candidate_id'],
                first_link=first_link,second_link=second_link,
                second_view_matches_first_observed_region=(first_link['matched_asset_index']==second['asset_index']
                    if first_link['status']=='unique_observed_region' else None),
                original_first_asset_index=first['asset_index'],original_second_asset_index=second['asset_index'],
                target_observed_class_vote=target['class_vote'],raw_inspection_encoding=True,
                target_marked_points=target['marked_points'],prediction=predictions,actual=actual,
                anchor_key_count=len(anchor['keys']),descriptor_point_count=len(target['points']),
                first_arrival_action=d['arrival_action_id'],second_arrival_action=arrival,
                map_mesh_exact=True,truth_used=False,semantic_labels_used_for_matching=False)
            results.append(row);write(a.output/f"first_{first['candidate_id']}_second_{second['candidate_id']}.json",row)
        for n,v in sources.items():
            if sha(ROOT/n)!=v:raise ValueError('source changed while running')
        write(a.output/'result.json',dict(status='passed',records=results,new_physical_outcomes=0,
            semantic_efficacy_proven=False,planner_updated=False,training_allowed=False))
        manifest['status']='complete'
    except Exception as error:manifest.update(status='failed',error=repr(error));raise
    finally:
        write(a.output/'manifest.json',manifest)
        write(a.output/'artifact_hashes.json',{x.name:sha(x) for x in sorted(a.output.iterdir()) if x.is_file() and x.name!='artifact_hashes.json'})


if __name__=='__main__':main()
