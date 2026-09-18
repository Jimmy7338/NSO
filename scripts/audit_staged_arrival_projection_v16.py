#!/usr/bin/env python3
"""Explain rejected rear targets using recorded arrival observations only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import gzip
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.sparse.csgraph import dijkstra
from env.virtual3d import camera_pose
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.observed_asset_axes_v16 import measured_assets_v16
from nso.competition_candidates_v8_1 import aperture_support,navigation_rear_boundary,XY_DIRECTIONS
from nso.route_coverage_v2 import orientation_graph
from nso.decision_replay_v13 import load_packet,array_hash
from utils.grid_geometry import inflated_obstacles
from scripts.collect_semantic_gain_v13_history import sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.source/'manifest.json').read_text())
    if m['status']!='complete':raise ValueError('complete independently replayed probes required')
    for n,v in json.loads((a.source/'artifact_hashes.json').read_text()).items():
        if sha(a.source/n)!=v:raise ValueError('input changed')
    for n,v in m['source_sha256'].items():
        if sha(ROOT/n)!=v:raise ValueError('frozen dependency changed: '+n)
    protocol=m['protocol'];history=ROOT/protocol['history'];hp=json.loads((history/'manifest.json').read_text())['protocol']
    doc=json.loads((ROOT/hp['scene_protocol']).read_text())
    c=next(x for x in doc['contexts'] if x['id']==hp['context'])
    config=InspectionConfigV4(**{**doc['shared_conditions'],**{k:v for k,v in c.items() if k not in ('id','seed')}})
    a.output.mkdir(parents=True,exist_ok=False)
    names={**m['source_sha256'],str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
    rows=[]
    for candidate in protocol['candidates']:
        folder=a.source/f"candidate_{candidate['candidate_id']}"
        d=json.loads(gzip.decompress((folder/'deterministic.json.gz').read_bytes()))
        state=d['arrival_state']['evidence'];out=d['outcome'];pool=d['arrival_pool'];frames=[]
        shape=tuple(pool['candidate_audit']['current_pose']) # mapper shape comes from fixed scene dimensions below
        prefix=history/f"structure_{protocol['structure_seed']}"
        old=json.loads((prefix/'all_decisions.json').read_text())[0]['bounds'];shape=(old[1],old[3])
        mapper=ObservedRuntimeMapperV10(shape,config)
        arrival=protocol['action_id']+out['outbound_actions']
        for aid in range(arrival+1):
            source=prefix if aid<=protocol['action_id'] else folder
            packet=load_packet(source/f'packets/{aid:04d}.npz');frames.append(packet.frame);mapper.update(packet.frame,packet.scan)
        arrays={n:array_hash(v) for n,v in vars(mapper).items() if isinstance(v,np.ndarray)}
        mesh=mapper.mesh();hashes={n:array_hash(np.asarray(getattr(mesh,n))) for n in ('vertices','triangles','vertex_colors')}
        if arrays!=state['map_arrays'] or hashes!=state['mesh']:raise ValueError('arrival reconstruction differs')
        assets=measured_assets_v16(AxisHistoryViewV16(mapper,frames))
        safe=~inflated_obstacles(mapper.belief!=0,config.robot_radius_m/config.resolution_m)
        graph,cells,ids=orientation_graph(safe)
        start=int(ids[packet.position])*4+packet.heading
        anchor=pool['candidate_audit']['return_anchor'];home=int(ids[tuple(anchor[:2])])*4+anchor[2]
        outward=dijkstra(graph,directed=True,indices=start)
        backward=dijkstra(graph.T.tocsr(),directed=True,indices=home)
        costs=outward+backward;budget=protocol['total_task_budget']-arrival
        feasible=np.flatnonzero(np.isfinite(costs)&(outward>0)&(costs<=budget))
        used={tuple(x['pose']) for x in pool['candidates'] if x['asset_index'] is None}
        for ai,asset in enumerate(assets):
            wanted=int(np.argmax(XY_DIRECTIONS@asset['front_axis']))
            desired=navigation_rear_boundary(mapper,asset)+.4*asset['back_axis'];options=[]
            for node in feasible:
                cell=tuple(map(int,cells[node//4]));heading=int(node%4);pose=(*cell,heading)
                if pose in used or heading!=wanted:continue
                camera=camera_pose(cell,heading,config,shape[0])
                depth=float((camera[:2,3]-asset['rear_boundary_xy'])@asset['back_axis'])
                if depth<=.15:continue
                aperture=aperture_support(asset,camera,config)
                if aperture.max()<=0:continue
                options.append(dict(pose=list(pose),xy=camera[:2,3].tolist(),total_reserved_cost=int(round(costs[node])),
                    outbound_cost=int(round(outward[node])),return_cost=int(round(backward[node])),
                    error_to_center_target_m=float(np.linalg.norm(camera[:2,3]-desired)),
                    hypothetical_aperture_fraction=float(aperture.mean()),back_axis_offset_m=depth))
            closest=min(options,key=lambda r:(r['error_to_center_target_m'],r['total_reserved_cost'],r['pose'])) if options else None
            audit=next(x for x in pool['candidate_audit']['target_audit'] if x['role']==f'asset_{ai}_entry')
            expected=audit['nearest_error_m']
            if (closest is None)!=(expected is None):raise ValueError('candidate filter diagnostic differs')
            if closest is not None and abs(closest['error_to_center_target_m']-expected)>1e-10:
                raise ValueError('nearest feasible original target differs')
            rows.append(dict(probe_candidate_id=candidate['candidate_id'],arrival_action_id=arrival,asset_index=ai,
                observed_geometry_group=asset['group'],original_entry_reason=audit['reason'],
                original_target_error_limit_m=.35,desired_xy=desired.tolist(),feasible_aperture_poses=len(options),
                feasible_within_original_limit=sum(x['error_to_center_target_m']<=.35 for x in options),
                nearest_feasible_aperture_pose=closest,all_feasible_aperture_poses=options,
                candidate_is_not_actual_rear_visibility=True,semantic_labels_used=False,truth_used=False,map_mesh_exact=True))
    write(a.output/'records.json',rows)
    write(a.output/'result.json',dict(status='passed',arrival_states=3,observed_asset_states=len(rows),
        center_projection_is_only_rejection=sum(x['feasible_aperture_poses']>0 and x['feasible_within_original_limit']==0 for x in rows),
        no_feasible_aperture_pose=sum(x['feasible_aperture_poses']==0 for x in rows),
        original_policy_changed=False,new_physical_outcomes=0,new_rear_visibility_proven=False,
        source_sha256=names,input_inventory_sha256=sha(a.source/'artifact_hashes.json')))
    write(a.output/'artifact_hashes.json',{x.name:sha(x) for x in sorted(a.output.iterdir()) if x.is_file()})


if __name__=='__main__':main()
